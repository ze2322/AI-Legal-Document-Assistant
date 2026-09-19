"""Tests for the end-to-end RAGPipeline, wired with fake models."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.chunker import TextChunker
from services.exceptions import EmptyDocumentError, IndexNotReadyError
from services.parser import DocumentParser
from services.rag import RAGPipeline
from services.vector_store import VectorStore

from .conftest import FakeEmbeddingService, FakeLLMService


@pytest.fixture
def pipeline(
    tmp_path: Path, embedding_service: FakeEmbeddingService, llm_service: FakeLLMService
) -> RAGPipeline:
    return RAGPipeline(
        parser=DocumentParser(),
        chunker=TextChunker(chunk_size=120, chunk_overlap=20),
        embedding_service=embedding_service,
        vector_store=VectorStore(index_dir=tmp_path / "index"),
        llm=llm_service,
    )


def test_pipeline_starts_empty(pipeline: RAGPipeline) -> None:
    assert pipeline.is_ready is False
    assert pipeline.chunk_count == 0


def test_ingest_indexes_the_document(pipeline: RAGPipeline, docx_file: Path) -> None:
    result = pipeline.ingest_document(docx_file)

    assert pipeline.is_ready
    assert result.chunk_count > 0
    assert result.document_name == docx_file.name
    assert pipeline.chunk_count == result.chunk_count


def test_ingest_records_the_display_name(pipeline: RAGPipeline, docx_file: Path) -> None:
    pipeline.ingest_document(docx_file, document_name="Original Upload.docx")

    assert pipeline.document_name == "Original Upload.docx"


def test_ingest_replaces_the_previous_document(
    pipeline: RAGPipeline, docx_file: Path, pdf_file: Path
) -> None:
    """Answers must never mix content from two different uploads."""
    pipeline.ingest_document(docx_file)
    pipeline.ingest_document(pdf_file)

    assert pipeline.document_name == pdf_file.name
    assert all("Confidentiality" not in chunk for chunk in pipeline.vector_store.chunks)


def test_ingest_persists_the_index(pipeline: RAGPipeline, docx_file: Path, tmp_path: Path) -> None:
    pipeline.ingest_document(docx_file)

    reloaded = VectorStore(index_dir=tmp_path / "index")

    assert reloaded.load() is True
    assert reloaded.chunks == pipeline.vector_store.chunks


def test_empty_document_raises(pipeline: RAGPipeline, tmp_path: Path) -> None:
    """A scanned PDF used to crash on embeddings.shape[1]."""
    fitz = pytest.importorskip("fitz")

    document = fitz.open()
    document.new_page()
    path = tmp_path / "scanned.pdf"
    document.save(path)
    document.close()

    with pytest.raises(EmptyDocumentError):
        pipeline.ingest_document(path)

    assert pipeline.is_ready is False


def test_ask_before_ingesting_raises(pipeline: RAGPipeline) -> None:
    with pytest.raises(IndexNotReadyError):
        pipeline.ask("What must partners retain?")


@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_raises(pipeline: RAGPipeline, docx_file: Path, question: str) -> None:
    pipeline.ingest_document(docx_file)

    with pytest.raises(ValueError):
        pipeline.ask(question)


def test_ask_returns_answer_and_sources(
    pipeline: RAGPipeline, docx_file: Path, llm_service: FakeLLMService
) -> None:
    pipeline.ingest_document(docx_file)
    response = pipeline.ask("What survives termination?")

    assert response["answer"] == llm_service.answer
    assert response["sources"]
    assert {"id", "chunk", "score"} <= set(response["sources"][0])


def test_ask_passes_retrieved_context_to_the_llm(
    pipeline: RAGPipeline, docx_file: Path, llm_service: FakeLLMService
) -> None:
    pipeline.ingest_document(docx_file)
    pipeline.ask("What survives termination?")

    context = llm_service.calls[-1]["context"]

    assert "[Chunk " in context
    assert "Confidentiality" in context


def test_retrieve_respects_top_k(pipeline: RAGPipeline, sample_text: str, tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text(sample_text, encoding="utf-8")

    # Index the sample text directly rather than round-tripping through a file
    # format the parser supports.
    chunks = pipeline.chunker.split_text(sample_text)
    pipeline.vector_store.create_index(
        pipeline.embedding_service.embed_documents(chunks), chunks
    )

    assert len(pipeline.retrieve("audit", top_k=2)) == 2


def test_ask_stream_yields_tokens_and_sources(pipeline: RAGPipeline, docx_file: Path) -> None:
    pipeline.ingest_document(docx_file)
    stream, sources = pipeline.ask_stream("What survives termination?")

    assert sources, "sources must be available before generation completes"
    assert "".join(stream).strip() == "A fake answer."


def test_load_existing_index_restores_state(
    pipeline: RAGPipeline, docx_file: Path, tmp_path: Path, embedding_service: FakeEmbeddingService
) -> None:
    pipeline.ingest_document(docx_file)

    fresh = RAGPipeline(
        embedding_service=embedding_service,
        vector_store=VectorStore(index_dir=tmp_path / "index"),
        llm=FakeLLMService(),
    )

    assert fresh.load_existing_index() is True
    assert fresh.is_ready
    assert fresh.document_name == docx_file.name


def test_build_context_labels_each_chunk() -> None:
    context = RAGPipeline.build_context(
        [{"id": 0, "chunk": "first", "score": 0.9}, {"id": 7, "chunk": "second", "score": 0.5}]
    )

    assert context == "[Chunk 0]\nfirst\n\n[Chunk 7]\nsecond"
