from tools.metadata_store import MetadataStore


def generate_digest(results, metadata_store):
    topic = results["topic"]
    evaluations = results["evaluations"]
    chunks_stored = results["chunks_stored"]

    if not evaluations:
        return None

    top_evaluations = sorted(evaluations, key=lambda x: x.score, reverse=True)[:3]
    average_score = sum(e.score for e in evaluations) / len(evaluations)

    lines = [
        f"Research topic: {topic}",
        f"New sources stored: {chunks_stored}",
        f"Evaluated chunks: {len(evaluations)}",
        f"Average credibility: {average_score:.2f}/5",
        "",
        "Top findings:",
    ]
    for evaluation in top_evaluations:
        chunk = metadata_store.get_chunk_by_id(evaluation.chunk_id)
        if chunk:
            _, _, url, title, _ = chunk
        else:
            title, url = "Unknown source", ""
        lines.append(f"- {title} ({url}) | {evaluation.score}/5 | {evaluation.reason}")
    return "\n".join(lines)


def build_and_save_digest(results):
    metadata_store = MetadataStore()
    digest = generate_digest(results, metadata_store)
    if digest:
        metadata_store.save_digest(results["topic"], digest)
    return digest