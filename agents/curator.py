import hashlib
import logging

import numpy as np

from schemas.source import SourceList
from tools.embedder import Embedder
from tools.faiss_index import FaissIndex
from tools.metadata_store import MetadataStore

MIN_TOPIC_RELEVANCE = .80

def run_curator(source_list: SourceList, topic: str) -> int:
    metadata_store = MetadataStore()

    candidates = []
    seen_ids = set()
    for source in source_list.sources:
        chunk_id = hashlib.md5(source.content.encode("utf-8")).hexdigest()
        if chunk_id in seen_ids:
            continue
        if metadata_store.url_exists(source.url) or metadata_store.get_chunk_by_id(chunk_id):
            continue
        seen_ids.add(chunk_id)
        candidates.append((source, chunk_id))

    if not candidates:
        return 0

    embedder = Embedder()
    topic_vector = embedder.encode([topic], is_query=True)[0]
    chunk_vectors = embedder.encode([source.content for source, _ in candidates])

    chunk_vectors = chunk_vectors / np.clip(
        np.linalg.norm(chunk_vectors, axis=1, keepdims=True), 1e-9, None
    )
    topic_vector = topic_vector / max(np.linalg.norm(topic_vector), 1e-9)
    similarities = chunk_vectors @ topic_vector

    kept = []
    for (source, chunk_id), similarity, vector in zip(candidates, similarities, chunk_vectors):
        if similarity < MIN_TOPIC_RELEVANCE:
            logging.info(f"Dropped off-topic source ({similarity:.2f}): {source.title}")
            continue
        kept.append((source, chunk_id, vector))

    if not kept:
        return 0

    faiss_index = FaissIndex(dimension=chunk_vectors.shape[1])
    starting_position = faiss_index.index.ntotal
    faiss_index.add(np.array([vector for _, _, vector in kept], dtype=np.float32))

    for offset, (source, chunk_id, _) in enumerate(kept):
        metadata_store.insert_chunk(
            chunk_id=chunk_id,
            text=source.content,
            url=source.url,
            title=source.title,
            topic=topic,
            faiss_position=starting_position + offset,
        )

    return len(kept)