import json
import logging

from dotenv import load_dotenv

from schemas.source import CriticOutput, SourceEvaluation
from tools.llm import call_llm
from tools.metadata_store import MetadataStore

load_dotenv()

MAX_CHUNKS_PER_RUN = 30
MAX_CHARS_PER_CHUNK = 500

SYSTEM_PROMPT = (
    "You are a strict source credibility critic. You receive a research topic and a "
    "list of sources, each with chunk_id, url, title and text. "
    "Score every source from 0 to 5, combining two things: how reliable the source is "
    "(peer-reviewed or official sources and specific, evidenced claims score higher; "
    "marketing, anecdotes and unsupported claims score lower) and how relevant it is "
    "to the topic. A source that is off-topic, such as a dictionary definition or "
    "website navigation text, must score 0 or 1 no matter how reputable the site is. "
    "Compare the sources with each other, and when claims conflict, say so in the reason. "
    "Use only the text provided and never outside knowledge. "
    "Reply with a single JSON object and nothing else, in this shape: "
    '{"evaluations": [{"chunk_id": "...", "score": 3.5, "reason": "..."}]}. '
    "Include exactly one evaluation per source, copy each chunk_id exactly, and keep "
    "each reason under 25 words."
)


def build_context(chunks) -> str:
    blocks = []
    for chunk_id, text, url, title in chunks:
        blocks.append(
            f"chunk_id: {chunk_id}\n"
            f"url: {url}\n"
            f"title: {title}\n"
            f"text: {text[:MAX_CHARS_PER_CHUNK]}"
        )
    return "\n\n".join(blocks)


def parse_evaluations(text: str, valid_ids: set) -> list:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the model reply")
    data = json.loads(text[start:end + 1])

    evaluations = []
    seen = set()
    for item in data.get("evaluations", []):
        chunk_id = item.get("chunk_id")
        if chunk_id not in valid_ids or chunk_id in seen:
            continue
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            continue
        seen.add(chunk_id)
        evaluations.append(SourceEvaluation(
            chunk_id=chunk_id,
            score=max(0.0, min(5.0, score)),
            reason=str(item.get("reason", "")).strip(),
        ))
    return evaluations


def run_critic(topic: str) -> CriticOutput:
    metadata_store = MetadataStore()
    chunks = metadata_store.get_unscored_chunks(topic)[:MAX_CHUNKS_PER_RUN]

    if not chunks:
        return CriticOutput(evaluations=[])

    valid_ids = {chunk[0] for chunk in chunks}

    try:
        reply = call_llm(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Research topic: {topic}\n\nSources:\n{build_context(chunks)}"},
            ],
            json_mode=True,
        )
        evaluations = parse_evaluations(reply, valid_ids)
    except Exception as e:
        logging.error(f"Critic failed for '{topic}', chunks stay unscored for the next run: {type(e).__name__}: {e}")
        return CriticOutput(evaluations=[])

    for evaluation in evaluations:
        metadata_store.update_score(evaluation.chunk_id, evaluation.score)

    missing = len(valid_ids) - len(evaluations)
    if missing:
        logging.warning(f"Critic returned no valid score for {missing} chunk(s); they will be retried next run")

    return CriticOutput(evaluations=evaluations)