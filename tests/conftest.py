"""Shared pytest fixtures and helpers.

The suite runs without a GPU, without Ollama and without downloading model
weights: the embedding model and the LLM are replaced by deterministic fakes.
That keeps the tests fast and runnable in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from docx import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDING_DIM = 8


class FakeEmbeddingService:
    """Deterministic stand-in for EmbeddingService.

    Each text is hashed into a small vector and normalised, so identical texts
    embed identically and similarity search behaves predictably.
    """

    def __init__(self, model_name: str = "fake-embedding-model") -> None:
        self.model_name = model_name
        self.dimension = EMBEDDING_DIM

    def _vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.standard_normal(EMBEDDING_DIM).astype("float32")
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype="float32")
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("Cannot embed an empty query")
        return self._vector(query)


class FakeLLMService:
    """Stand-in for LLMService that echoes a canned answer."""

    def __init__(self, answer: str = "A fake answer.") -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def generate(self, context: str, question: str) -> str:
        self.calls.append({"context": context, "question": question})
        return self.answer

    def generate_stream(self, context: str, question: str):
        self.calls.append({"context": context, "question": question})
        for word in self.answer.split():
            yield word + " "


@pytest.fixture
def embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def llm_service() -> FakeLLMService:
    return FakeLLMService()


@pytest.fixture
def sample_text() -> str:
    """A short document body long enough to produce several chunks."""
    return (
        "Implementing partners are organisations responsible for delivering "
        "project outputs under Grant-out Agreements.\n\n"
        "Partners must retain all financial records for a period of five years "
        "following the completion of the project.\n\n"
        "The organisation reserves the right to audit any implementing partner "
        "at any point during the agreement period."
    )


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    """A minimal DOCX file containing a paragraph and a table."""
    document = Document()
    document.add_paragraph("Confidentiality obligations survive termination.")
    document.add_paragraph("   ")  # blank paragraphs should be skipped

    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Clause"
    table.rows[0].cells[1].text = "Retention period"

    path = tmp_path / "agreement.docx"
    document.save(path)
    return path


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """A minimal single-page PDF with a line of text."""
    fitz = pytest.importorskip("fitz")

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Implementing partners must retain financial records.")

    path = tmp_path / "policy.pdf"
    document.save(path)
    document.close()
    return path
