"""Tests for VectorStore."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from services.exceptions import IndexNotReadyError
from services.vector_store import VectorStore

from .conftest import FakeEmbeddingService


@pytest.fixture
def chunks() -> list[str]:
    return [
        "Implementing partners deliver project outputs.",
        "Financial records must be retained for five years.",
        "The organisation may audit any partner.",
    ]


@pytest.fixture
def store(tmp_path: Path, embedding_service: FakeEmbeddingService, chunks: list[str]) -> VectorStore:
    store = VectorStore(index_dir=tmp_path / "index")
    store.create_index(embedding_service.embed_documents(chunks), chunks)
    return store


def test_new_store_is_not_ready(tmp_path: Path) -> None:
    assert VectorStore(index_dir=tmp_path).is_ready is False


def test_create_index_makes_store_ready(store: VectorStore, chunks: list[str]) -> None:
    assert store.is_ready
    assert store.index.ntotal == len(chunks)


def test_search_returns_the_matching_chunk(
    store: VectorStore, embedding_service: FakeEmbeddingService, chunks: list[str]
) -> None:
    """Querying with a chunk's own text should rank that chunk first."""
    query = embedding_service.embed_query(chunks[1])
    results = store.search(query, top_k=1)

    assert results[0]["chunk"] == chunks[1]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_search_results_are_sorted_by_score(
    store: VectorStore, embedding_service: FakeEmbeddingService
) -> None:
    results = store.search(embedding_service.embed_query("audit"), top_k=3)
    scores = [r["score"] for r in results]

    assert scores == sorted(scores, reverse=True)


def test_search_before_indexing_raises(tmp_path: Path) -> None:
    """Previously this crashed with AttributeError on a None index."""
    store = VectorStore(index_dir=tmp_path)

    with pytest.raises(IndexNotReadyError):
        store.search(np.zeros(8, dtype="float32"))


def test_search_caps_top_k_at_index_size(
    store: VectorStore, embedding_service: FakeEmbeddingService, chunks: list[str]
) -> None:
    results = store.search(embedding_service.embed_query("records"), top_k=50)

    assert len(results) == len(chunks)
    assert all(0 <= r["id"] < len(chunks) for r in results)


def test_search_rejects_mismatched_dimensions(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="dimension"):
        store.search(np.zeros(4, dtype="float32"))


def test_create_index_rejects_empty_input(tmp_path: Path) -> None:
    store = VectorStore(index_dir=tmp_path)

    with pytest.raises(ValueError):
        store.create_index(np.empty((0, 8), dtype="float32"), [])


def test_create_index_rejects_length_mismatch(
    tmp_path: Path, embedding_service: FakeEmbeddingService
) -> None:
    store = VectorStore(index_dir=tmp_path)
    embeddings = embedding_service.embed_documents(["a", "b", "c"])

    with pytest.raises(ValueError, match="does not match"):
        store.create_index(embeddings, ["a", "b"])


def test_save_and_load_round_trip(
    store: VectorStore, tmp_path: Path, embedding_service: FakeEmbeddingService, chunks: list[str]
) -> None:
    store.metadata = {"document_name": "policy.pdf"}
    store.save()

    reloaded = VectorStore(index_dir=tmp_path / "index")

    assert reloaded.load() is True
    assert reloaded.chunks == chunks
    assert reloaded.metadata["document_name"] == "policy.pdf"

    query = embedding_service.embed_query(chunks[0])
    assert reloaded.search(query, top_k=1)[0]["chunk"] == chunks[0]


def test_load_without_saved_index_returns_false(tmp_path: Path) -> None:
    assert VectorStore(index_dir=tmp_path / "missing").load() is False


def test_load_ignores_inconsistent_index(store: VectorStore, tmp_path: Path) -> None:
    """A stale chunks file must not produce out-of-range lookups."""
    store.save()
    folder = tmp_path / "index"
    (folder / "chunks.json").write_text('{"chunks": ["only one"]}', encoding="utf-8")

    reloaded = VectorStore(index_dir=folder)

    assert reloaded.load() is False
    assert reloaded.is_ready is False


def test_save_without_index_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexNotReadyError):
        VectorStore(index_dir=tmp_path).save()


def test_clear_resets_the_store(store: VectorStore) -> None:
    store.clear()

    assert store.is_ready is False
    assert store.chunks == []
