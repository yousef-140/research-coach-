import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import pipeline
from coach_service import coach
from digest import build_and_save_digest
from scheduler import start_scheduler
from schemas.source import Subscription
from tools.metadata_store import MetadataStore

load_dotenv()
logging.basicConfig(level=logging.INFO)

API_KEYS = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]
if not API_KEYS:
    raise RuntimeError("API_KEYS is not set in .env")

UI_KEY = os.getenv("UI_API_KEY", "").strip()
if not UI_KEY or UI_KEY not in API_KEYS:
    raise RuntimeError("UI_API_KEY must be set and also listed in API_KEYS")

INVITE_CODES = [c.strip() for c in os.getenv("INVITE_CODES", "").split(",") if c.strip()]
if not INVITE_CODES:
    raise RuntimeError("INVITE_CODES is not set in .env")

COOKIE_SECURE = os.getenv("COOKIE_SECURE") == "1"
SESSION_TTL = 7 * 24 * 3600
LOGIN_ATTEMPTS_PER_MIN = 5

STATIC_DIR = Path(__file__).parent / "static"

RATE_LIMITS = {"ask": 10, "research": 2, "subscribe": 5, "digest": 30, "jobs": 60}
RATE_WINDOW_SECONDS = 60

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

stale = MetadataStore().fail_stale_jobs()
if stale:
    logging.warning(f"Marked {stale} stale job(s) as failed")

start_scheduler()

_hits = defaultdict(deque)
_hits_lock = threading.Lock()
_job_lock = threading.Lock()
_sessions = {}


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("cf-connecting-ip")
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def hit(bucket, limit):
    now = time.monotonic()
    with _hits_lock:
        hits = _hits[bucket]
        while hits and now - hits[0] > RATE_WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = int(RATE_WINDOW_SECONDS - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail="Too many requests",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


def get_session(token):
    if not token:
        return None
    session = _sessions.get(token)
    if not session:
        return None
    code, expires = session
    if expires < time.time():
        _sessions.pop(token, None)
        return None
    return code


def check_api_key(x_api_key: str = Header(default="")) -> str:
    for key in API_KEYS:
        if hmac.compare_digest(x_api_key.encode(), key.encode()):
            return key
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def rate_limit(name: str):
    limit = RATE_LIMITS[name]

    def dependency(request: Request, api_key: str = Depends(check_api_key)) -> str:
        who = request.scope.get("session_user") or client_ip(request)
        hit((who, name), limit)
        return api_key

    return dependency


class LoginRequest(BaseModel):
    code: str = Field(min_length=1, max_length=100)


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=100)


class AskRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    auto_research: bool = True
    topic: str | None = Field(default=None, max_length=100)


def run_and_digest(topic: str):
    try:
        result = pipeline.run_pipeline(topic)
        build_and_save_digest(result)
    except Exception as e:
        logging.error(f"Background research failed for '{topic}': {type(e).__name__}: {e}")


def run_job(job_id: str, topic: str):
    store = MetadataStore()
    try:
        result = pipeline.run_pipeline(topic)
        digest = build_and_save_digest(result)
        payload = {"chunks_stored": result["chunks_stored"], "digest": digest}
        store.finish_job(job_id, "done", result=json.dumps(payload, default=str))
    except Exception as e:
        logging.error(f"Job {job_id} failed for '{topic}': {type(e).__name__}: {e}")
        store.finish_job(job_id, "failed", error=type(e).__name__)


@app.middleware("http")
async def attach_session(request: Request, call_next):
    user = get_session(request.cookies.get("session"))
    if user:
        request.scope["session_user"] = user
        if "x-api-key" not in request.headers:
            request.scope["headers"].append((b"x-api-key", UI_KEY.encode()))
    return await call_next(request)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(body: LoginRequest, request: Request, response: Response):
    hit((client_ip(request), "login"), LOGIN_ATTEMPTS_PER_MIN)
    matched = next(
        (c for c in INVITE_CODES if hmac.compare_digest(body.code.encode(), c.encode())),
        None,
    )
    if not matched:
        raise HTTPException(status_code=401, detail="Invalid invite code")
    token = secrets.token_urlsafe(32)
    _sessions[token] = (matched, time.time() + SESSION_TTL)
    response.set_cookie(
        "session",
        token,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        max_age=SESSION_TTL,
    )
    return {"status": "ok"}


@app.get("/me", dependencies=[Depends(check_api_key)])
def me():
    return {"ok": True}


@app.post("/research", status_code=202, dependencies=[Depends(rate_limit("research"))])
def research(request: ResearchRequest, background_tasks: BackgroundTasks):
    topic = " ".join(request.topic.split())
    store = MetadataStore()

    with _job_lock:
        running = store.get_running_job(topic)
        if running:
            return {"status": "running", "job_id": running, "already_running": True}
        job_id = uuid.uuid4().hex
        store.create_job(job_id, topic)

    background_tasks.add_task(run_job, job_id, topic)
    return {"status": "running", "job_id": job_id, "already_running": False}


@app.get("/jobs/{job_id}", dependencies=[Depends(rate_limit("jobs"))])
def get_job(job_id: str):
    job = MetadataStore().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job_id, topic, status, result, error, created_at, finished_at = job

    response = {
        "job_id": job_id,
        "topic": topic,
        "status": status,
        "created_at": created_at,
        "finished_at": finished_at,
    }
    if status == "done" and result:
        response.update(json.loads(result))
    if status == "failed":
        response["error"] = error
    return response


@app.post("/ask", dependencies=[Depends(rate_limit("ask"))])
def ask(request: AskRequest):
    try:
        return coach.answer_question(request.query, request.auto_research, request.topic)
    except Exception as e:
        logging.error(f"/ask failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail="The answer service is temporarily unavailable. Please try again.")


@app.post("/subscribe", dependencies=[Depends(rate_limit("subscribe"))])
def subscribe(request: Subscription, background_tasks: BackgroundTasks):
    metadata_store = MetadataStore()
    existing_subscription = metadata_store.get_subscription(request.topic)
    if existing_subscription:
        topic, frequency, last_run = existing_subscription
        return {"status": "already_subscribed", "topic": topic, "frequency": frequency}

    metadata_store.add_subscription(request.topic, request.frequency)
    background_tasks.add_task(run_and_digest, request.topic)
    topic, frequency, last_run = metadata_store.get_subscription(request.topic)
    return {"status": "subscribed", "topic": topic, "frequency": frequency}


@app.delete("/subscribe/{topic}", dependencies=[Depends(rate_limit("subscribe"))])
def unsubscribe(topic: str):
    removed = MetadataStore().remove_subscription(topic)
    if not removed:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"status": "unsubscribed", "topic": topic}


@app.get("/digest/{topic}", dependencies=[Depends(rate_limit("digest"))])
def get_digest(topic: str):
    metadata_store = MetadataStore()
    digest = metadata_store.get_latest_digest(topic)
    if not digest:
        raise HTTPException(status_code=404, detail="No digest found for this topic yet")
    content, created_at = digest
    return {"topic": topic, "digest": content, "created_at": created_at}