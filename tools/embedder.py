import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDER_MODEL", "intfloat/multilingual-e5-small")


class Embedder:
    def __init__(self, model: str = MODEL_NAME):
        self.model = SentenceTransformer(model)
        self.use_e5_prefix = "e5" in model.lower()

    def encode(self, texts: list[str], is_query: bool = False):
        if self.use_e5_prefix:
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t for t in texts]
        return self.model.encode(texts, normalize_embeddings=True)