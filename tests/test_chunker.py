"""Tests for TextChunker."""

from __future__ import annotations

import pytest

from services.chunker import TextChunker
from utils.config import settings


def test_splits_long_text_into_multiple_chunks(sample_text: str) -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_text(sample_text)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_respects_chunk_size_argument(sample_text: str) -> None:
    """The constructor arguments used to be ignored in favour of the config."""
    chunker = TextChunker(chunk_size=120, chunk_overlap=10)

    assert chunker.chunk_size == 120
    assert chunker.chunk_overlap == 10
    assert all(len(chunk) <= 120 for chunk in chunker.split_text(sample_text))


def test_falls_back_to_configured_defaults() -> None:
    chunker = TextChunker()

    assert chunker.chunk_size == settings.chunk_size
    assert chunker.chunk_overlap == settings.chunk_overlap


def test_smaller_chunk_size_produces_more_chunks(sample_text: str) -> None:
    few = TextChunker(chunk_size=300, chunk_overlap=20).split_text(sample_text)
    many = TextChunker(chunk_size=80, chunk_overlap=20).split_text(sample_text)

    assert len(many) > len(few)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_blank_text_produces_no_chunks(text: str) -> None:
    assert TextChunker().split_text(text) == []


def test_short_text_produces_single_chunk() -> None:
    chunks = TextChunker(chunk_size=500, chunk_overlap=50).split_text("A short clause.")
    assert chunks == ["A short clause."]
