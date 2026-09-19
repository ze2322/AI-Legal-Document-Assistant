"""Tests for EmbeddingService.

The real Sentence-Transformers model is several hundred MB, so the unit tests
patch it out. The tests that exercise the genuine model are marked ``integration``
and are deselected by default (see pyproject.toml).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.embeddings import EmbeddingService

DIM = 8


@pytest.fixture
def fake_model() -> MagicMock:
    model = MagicMock()
    # sentence-transformers 5.x renamed this method; the service supports both.
    model.get_embedding_dimension.return_value = DIM
    model.get_sentence_embedding_dimension.return_value = DIM
    model.encode.side_effect = lambda texts, **_: (
        np.ones((len(texts), DIM), dtype="float32") / np.sqrt(DIM)
        if isinstance(texts, list)
        else np.ones(DIM, dtype="float32") / np.sqrt(DIM)
    )
    return model


@pytest.fixture
def service(fake_model: MagicMock) -> EmbeddingService:
    with patch("services.embeddings.SentenceTransformer", return_value=fake_model):
        service = EmbeddingService(model_name="fake-model")
        _ = service.model  # force the lazy load while the patch is active
    return service


def test_model_is_loaded_lazily(fake_model: MagicMock) -> None:
    """Constructing the service must not download or load weights."""
    with patch("services.embeddings.SentenceTransformer", return_value=fake_model) as loader:
        service = EmbeddingService(model_name="fake-model")
        loader.assert_not_called()

        _ = service.model
        loader.assert_called_once_with("fake-model")


def test_model_is_loaded_only_once(fake_model: MagicMock) -> None:
    with patch("services.embeddings.SentenceTransformer", return_value=fake_model) as loader:
        service = EmbeddingService(model_name="fake-model")
        _ = service.model
        _ = service.model

        assert loader.call_count == 1


def test_embed_documents_returns_one_row_per_chunk(service: EmbeddingService) -> None:
    embeddings = service.embed_documents(["first chunk", "second chunk"])

    assert embeddings.shape == (2, DIM)
    assert embeddings.dtype == np.float32


def test_embed_documents_handles_empty_input(service: EmbeddingService) -> None:
    """An empty list must still yield a 2-D array so shape[1] stays valid."""
    embeddings = service.embed_documents([])

    assert embeddings.shape == (0, DIM)
    assert embeddings.ndim == 2


def test_embed_query_returns_a_single_vector(service: EmbeddingService) -> None:
    embedding = service.embed_query("What must partners retain?")

    assert embedding.shape == (DIM,)
    assert embedding.dtype == np.float32


@pytest.mark.parametrize("query", ["", "   "])
def test_embed_query_rejects_blank_input(service: EmbeddingService, query: str) -> None:
    with pytest.raises(ValueError):
        service.embed_query(query)


def test_defaults_to_the_configured_model() -> None:
    from utils.config import settings

    assert EmbeddingService().model_name == settings.embedding_model


@pytest.mark.integration
def test_real_model_produces_normalised_vectors() -> None:
    """Sanity-check the genuine model; downloads weights on first run."""
    service = EmbeddingService()
    embeddings = service.embed_documents(["a legal clause", "another clause"])

    assert embeddings.shape == (2, service.dimension)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)


@pytest.mark.integration
def test_real_model_ranks_similar_text_higher() -> None:
    service = EmbeddingService()
    query = service.embed_query("How long must records be kept?")
    related, unrelated = service.embed_documents(
        ["Records must be retained for five years.", "The kitchen is on the third floor."]
    )

    assert float(query @ related) > float(query @ unrelated)
