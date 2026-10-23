import logging
from functools import lru_cache
import numpy as np

from dotenv import load_dotenv

from tools.llm import call_llm
from tools.metadata_store import MetadataStore
from tools.embedder import Embedder
from tools.faiss_index import FaissIndex

load_dotenv()

EMBEDDING_DIMENSION = 384
TOP_K = 5
MIN_SIMILARITY = 0.80
MIN_GOOD_RESULTS = 2
MAX_CONTEXT_CHARS_PER_SOURCE = 400
TOPIC_MIN_WORDS = 2
TOPIC_MAX_WORDS = 8
TOPIC_ATTEMPTS = 2
CLIENT_TOPIC_MAX_CHARS = 100

NO_ANSWER_MARKER = "NO_ANSWER"
NO_ANSWER_PHRASES = (
    "do not contain",
    "does not contain",
    "do not answer",
    "does not answer",
    "not covered in the sources",
    "no information on",
)
NO_ANSWER_MAX_CHARS = 250

ANSWER_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "numbered sources provided. "
    "First judge whether the sources actually address the question asked, not "
    "merely the same general field. "
    f"If they do not address it at all, reply with exactly {NO_ANSWER_MARKER} and nothing else. "
    "Reply in the same language as the question. "
    "Start directly with the answer, with no preamble such as 'Based on the context'. "
    "If the question asks for advice, start with the most important advice and make "
    "every point an action the reader can take, not a description of a concept. "
    "Be practical and concise, using short paragraphs or a short list. "
    "Cite sources inline as [1], [2] matching the source numbers. "
    "Prefer higher-credibility sources, and mention credibility only when sources "
    "disagree or a claim rests on a weak source. "
    "If the sources answer only part of the question, answer that part and add one "
    "sentence saying what is missing. "
    "Never add facts that are not in the sources. "
    "For health or medical topics, add one short sentence recommending a "
    "professional for individual decisions."
)

TOPIC_SYSTEM_PROMPT = (
    "Turn the user's question into a short English research topic for searching "
    "academic papers and the web. "
    "The question may be in any language; the topic must always be in English. "
    "Rules: 2 to 8 words; a noun phrase, not a question or a command; remove "
    "first-person words and request wording such as 'my', 'give me', 'how can I', "
    "'advice'; keep the key subject and the field or population it concerns. "
    "Reply with the topic only, on one line, with no quotes and no punctuation at the end. "
    "Examples: "
    "'how can i treat my child if he is starting teenage years' -> "
    "'parenting strategies for early adolescence'; "
    "'give me some advices to be a master in making relationships' -> "
    "'interpersonal relationship building skills'."
)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def is_valid_topic(topic: str) -> bool:
    words = topic.split()
    return TOPIC_MIN_WORDS <= len(words) <= TOPIC_MAX_WORDS and "?" not in topic


def extract_topic(query: str):
    for attempt in range(TOPIC_ATTEMPTS):
        try:
            text = call_llm(
                [
                    {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ]
            )
        except Exception as e:
            logging.warning(f"Topic extraction failed: {type(e).__name__}: {e}")
            return None
        lines = text.strip().splitlines()
        topic = lines[0].strip().strip("\"'").rstrip(".") if lines else ""
        if is_valid_topic(topic):
            return topic
        logging.warning(f"Rejected extracted topic (attempt {attempt + 1}/{TOPIC_ATTEMPTS}): {topic!r}")
    return None


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve(queries: list[str], top_k: int = TOP_K) -> list:
    faiss_index = FaissIndex(dimension=EMBEDDING_DIMENSION)
    if faiss_index.index.ntotal == 0:
        return []

    metadata_store = MetadataStore()
    vectors = get_embedder().encode(queries, is_query=True)
    question_vector = vectors[0]
    distances, indices = faiss_index.search(vectors, top_k=top_k)

    positions = {int(p) for row in indices for p in row if int(p) != -1}

    results = []
    for position in positions:
        chunk_data = metadata_store.get_chunk_by_position(position)
        if not chunk_data:
            continue
        chunk_id, text, url, title, credibility_score = chunk_data
        chunk_vector = faiss_index.index.reconstruct(position)
        results.append({
            "chunk_id": chunk_id,
            "text": text,
            "url": url,
            "title": title,
            "credibility_score": credibility_score,
            "similarity": round(cosine(question_vector, chunk_vector), 3),
        })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:top_k]


def keep_relevant(results: list) -> list:
    return [r for r in results if r["similarity"] >= MIN_SIMILARITY]


def research_topic(topic: str) -> bool:
    from pipeline import run_pipeline
    from digest import build_and_save_digest

    try:
        result = run_pipeline(topic)
        build_and_save_digest(result)
        return True
    except Exception as e:
        logging.error(f"On-demand research failed for '{topic}': {type(e).__name__}: {e}")
        return False


def build_answer(query: str, results: list) -> str:
    blocks = []
    for number, r in enumerate(results, start=1):
        score = r["credibility_score"]
        credibility = f"{score}/5" if score is not None else "not scored yet"
        blocks.append(
            f"[{number}] {r['title']} ({r['url']})\n"
            f"Credibility: {credibility}\n"
            f"Content: {r['text'][:MAX_CONTEXT_CHARS_PER_SOURCE]}"
        )
    context = "\n\n".join(blocks)

    return call_llm([
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nSources:\n{context}"},
    ])


def is_no_answer(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    if text.upper().startswith(NO_ANSWER_MARKER):
        return True
    lowered = text.lower()
    return len(text) <= NO_ANSWER_MAX_CHARS and any(p in lowered for p in NO_ANSWER_PHRASES)


def attempt_research(query: str, topic: str = None):
    client_topic = (topic or "").strip()[:CLIENT_TOPIC_MAX_CHARS]
    if client_topic and not is_valid_topic(client_topic):
        client_topic = ""
    used_topic = client_topic or extract_topic(query)

    if not used_topic:
        notice = (
            "I couldn't turn your question into a clear search topic, so I did not "
            "search for new sources. Try rephrasing, or send an explicit topic."
        )
        return None, "topic_unclear", notice, False

    if research_topic(used_topic):
        return used_topic, "researched", f"I searched for new sources on: {used_topic}", True

    notice = f"I tried to research '{used_topic}' but the search failed."
    return used_topic, "research_failed", notice, False


def answer_question(query: str, auto_research: bool = True, topic: str = None) -> dict:
    results = keep_relevant(retrieve([query]))
    used_topic = None
    status = "not_needed"
    notice = None
    searched = False

    if len(results) < MIN_GOOD_RESULTS and auto_research:
        used_topic, status, notice, ok = attempt_research(query, topic)
        searched = True
        if ok:
            results = keep_relevant(retrieve([query, used_topic]))

    answer = build_answer(query, results) if results else None

    if answer is not None and is_no_answer(answer):
        answer = None
        if auto_research and not searched:
            used_topic, status, notice, ok = attempt_research(query, topic)
            searched = True
            if ok:
                results = keep_relevant(retrieve([query, used_topic]))
                if results:
                    retry = build_answer(query, results)
                    if not is_no_answer(retry):
                        answer = retry

    if answer is None:
        if status == "researched":
            status = "searched_no_results"
            notice = f"I searched for new sources on: {used_topic}, but found nothing that answers your question."
        return {
            "answer": "I couldn't find reliable sources that answer this yet.",
            "sources": [],
            "topic": used_topic,
            "research_status": status,
            "notice": notice,
        }

    return {
        "answer": answer,
        "sources": results,
        "topic": used_topic,
        "research_status": status,
        "notice": notice,
    }


if __name__ == "__main__":
    output = answer_question("give me some advices to be a master in making relationships")
    print(output["answer"])
    print("topic:", output["topic"])
    print("status:", output["research_status"])
    print("notice:", output["notice"])
    for s in output["sources"]:
        print(f"  similarity={s['similarity']}  {s['title']}")
