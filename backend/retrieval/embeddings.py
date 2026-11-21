from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def generate_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype="float32")
    if any(not text or not text.strip() for text in texts):
        raise ValueError("Embedding text cannot be empty.")

    model = load_embedding_model(model_name)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype="float32")
