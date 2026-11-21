from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_index: int
    similarity: float


class VectorStore:
    def __init__(self, embedding_dimension: int):
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")
        self.embedding_dimension = embedding_dimension
        self.index = faiss.IndexFlatIP(embedding_dimension)

    def add(self, embeddings: np.ndarray) -> None:
        vectors = self._validate_vectors(embeddings)
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension} values per vector, "
                f"received {vectors.shape[1]}."
            )
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.index.ntotal == 0:
            return []

        query = self._validate_vectors(query_embedding)
        if query.shape != (1, self.embedding_dimension):
            raise ValueError(
                f"Query must have shape (1, {self.embedding_dimension})."
            )
        faiss.normalize_L2(query)

        count = min(top_k, self.index.ntotal)
        similarities, indices = self.index.search(query, count)
        return [
            VectorSearchResult(
                chunk_index=int(index),
                similarity=float(similarity),
            )
            for similarity, index in zip(similarities[0], indices[0])
            if index >= 0
        ]

    def save(self, directory: str | Path, name: str = "document") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / f"{name}.faiss"))

    @classmethod
    def load(
        cls,
        directory: str | Path,
        name: str = "document",
    ) -> "VectorStore":
        index_path = Path(directory) / f"{name}.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index does not exist: {index_path}")

        index = faiss.read_index(str(index_path))
        if not isinstance(index, faiss.IndexFlatIP):
            raise ValueError("Only IndexFlatIP FAISS indexes are supported.")

        store = cls(index.d)
        store.index = index
        return store

    def _validate_vectors(self, embeddings: np.ndarray) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 1D or 2D array.")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")
        return vectors.copy()
