import os
import time
import logging
import unicodedata
import requests
from dotenv import load_dotenv
from schemas.source import Source, SourceList, SourceType
from tools.llm import call_llm

load_dotenv()

MAX_SOURCES_PER_QUERY = 3
MAX_CHARS_PER_SOURCE = 400
REQUEST_TIMEOUT = 20

ACADEMIC_DOMAINS = [
    "pmc.ncbi.nlm.nih.gov", "semanticscholar.org", "frontiersin.org",
    "springer.com", "nih.gov", ".edu", "sciencedirect.com", "pubmed"
]
NEWS_DOMAINS = ["bbc.", "cnn.", "nytimes.", "reuters.", "forbes."]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def classify_source_type(url: str) -> SourceType:
    if any(domain in url for domain in ACADEMIC_DOMAINS):
        return SourceType.ACADEMIC
    if any(domain in url for domain in NEWS_DOMAINS):
        return SourceType.NEWS
    return SourceType.BLOG


def fetch_semantic_scholar(query: str) -> list[Source]:
    headers = {}
    api_key = os.getenv("SEM_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={"query": query, "limit": MAX_SOURCES_PER_QUERY, "fields": "title,url,abstract"},
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    sources = []
    for paper in response.json().get("data", [])[:MAX_SOURCES_PER_QUERY]:
        title = paper.get("title")
        url = paper.get("url")
        abstract = paper.get("abstract")
        if not (title and url and abstract):
            continue
        sources.append(Source(
            title=clean_text(title),
            url=url,
            content=clean_text(abstract)[:MAX_CHARS_PER_SOURCE],
            source_type=SourceType.ACADEMIC,
        ))
    return sources


def fetch_tavily(query: str) -> list[Source]:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "query": query,
            "max_results": MAX_SOURCES_PER_QUERY,
            "api_key": os.getenv("TAVILY_API_KEY"),
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    sources = []
    for article in response.json().get("results", [])[:MAX_SOURCES_PER_QUERY]:
        title = article.get("title")
        url = article.get("url")
        content = article.get("content")
        if not (title and url and content):
            continue
        sources.append(Source(
            title=clean_text(title),
            url=url,
            content=clean_text(content)[:MAX_CHARS_PER_SOURCE],
            source_type=classify_source_type(url),
        ))
    return sources


def generate_queries(topic: str) -> list[str]:
    fallback = [f"{topic} systematic review"]
    try:
        # نداء غير حرج: محاولة واحدة لكل موديل وtimeout قصير، ولو فشل بنستخدم الـ fallback
        text = call_llm(
            [
                {"role": "system", "content": (
                    "You write web and academic search queries. Reply with exactly 2 "
                    "different search queries for the given topic, one per line, with "
                    "no numbering, no quotes and no extra text."
                )},
                {"role": "user", "content": topic},
            ],
            retries=1,
            timeout=REQUEST_TIMEOUT,
        )
        queries = [line.strip().strip('"').lstrip("-• ").strip() for line in text.splitlines()]
        queries = [q for q in queries if q and q.lower() != topic.lower()][:2]
        return queries or fallback
    except Exception as e:
        logging.warning(f"Query generation failed, using fallback: {type(e).__name__}: {e}")
        return fallback


def run_scout(topic: str) -> SourceList:
    queries = [topic] + generate_queries(topic)
    sources = []
    seen_urls = set()
    attempts = 0
    failures = 0

    for query in queries:
        for fetch in (fetch_semantic_scholar, fetch_tavily):
            attempts += 1
            try:
                found = fetch(query)
            except Exception as e:
                failures += 1
                logging.warning(f"{fetch.__name__} failed for '{query}': {type(e).__name__}: {e}")
                found = []
            for source in found:
                if source.url in seen_urls:
                    continue
                seen_urls.add(source.url)
                sources.append(source)
            if fetch is fetch_semantic_scholar:
                time.sleep(1.1)

    if failures == attempts:
        raise RuntimeError("All search requests failed")

    return SourceList(sources=sources)


if __name__ == "__main__":
    result = run_scout("adolescent menstrual cycle hormonal changes")
    for source in result.sources:
        print(source.source_type.value, "|", source.title, "|", source.url)