from unittest.mock import patch

import numpy as np
import pytest

from backend.retrieval.embeddings import generate_embeddings


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        convert_to_numpy,
        normalize_embeddings,
        show_progress_bar,
    ):
        return np.asarray(
            [[1.0, 0.0] if "first" in text else [0.0, 1.0] for text in texts],
            dtype="float32",
        )


def test_generate_embeddings_creates_one_vector_per_text():
    with patch(
        "backend.retrieval.embeddings.load_embedding_model",
        return_value=FakeEmbeddingModel(),
    ):
        vectors = generate_embeddings(["first chunk", "second chunk"])

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(vectors[0], [1.0, 0.0])
    np.testing.assert_allclose(vectors[1], [0.0, 1.0])


def test_generate_embeddings_for_empty_list():
    vectors = generate_embeddings([])

    assert vectors.shape == (0, 0)


def test_generate_embeddings_rejects_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        generate_embeddings(["valid text", ""])
