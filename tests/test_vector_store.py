from pathlib import Path

import numpy as np
import pytest

from backend.retrieval.vector_store import VectorStore


def test_add_and_search_vectors():
    store = VectorStore(embedding_dimension=2)
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        dtype="float32",
    )
    store.add(vectors)

    results = store.search(np.asarray([1.0, 0.0], dtype="float32"), top_k=2)

    assert len(results) == 2
    assert results[0].chunk_index == 0
    assert results[0].similarity > results[1].similarity


def test_empty_store_search_returns_empty_list():
    store = VectorStore(embedding_dimension=3)

    assert store.search(np.asarray([1.0, 0.0, 0.0]), top_k=5) == []


def test_save_and_load_index(tmp_path: Path):
    store = VectorStore(embedding_dimension=2)
    store.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    store.save(tmp_path, name="test")

    loaded = VectorStore.load(tmp_path, name="test")

    assert loaded.embedding_dimension == 2
    assert loaded.index.ntotal == 2


def test_dimension_mismatch_is_rejected():
    store = VectorStore(embedding_dimension=2)

    with pytest.raises(ValueError, match="Expected 2"):
        store.add(np.asarray([[1.0, 0.0, 1.0]], dtype="float32"))


def test_invalid_top_k_is_rejected():
    store = VectorStore(embedding_dimension=2)

    with pytest.raises(ValueError, match="top_k"):
        store.search(np.asarray([1.0, 0.0]), top_k=0)
