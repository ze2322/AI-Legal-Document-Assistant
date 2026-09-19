"""The end-to-end Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.exceptions import EmptyDocumentError, IndexNotReadyError
from services.llm import LLMService
from services.parser import DocumentParser
from services.vector_store import VectorStore
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of what was indexed, for display in the UI."""

    document_name: str
    chunk_count: int
    character_count: int


class RAGPipeline:
    """Wires parsing, chunking, embedding, retrieval and generation together.

    The collaborators are injectable so tests can substitute fakes instead of
    loading a real embedding model or talking to Ollama.
    """

    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunker: TextChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        llm: LLMService | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.chunker = chunker or TextChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.llm = llm or LLMService()

    @property
    def is_ready(self) -> bool:
        """True when a document has been indexed and questions can be answered."""
        return self.vector_store.is_ready

    @property
    def document_name(self) -> str | None:
        """Name of the currently indexed document, if known."""
        return self.vector_store.metadata.get("document_name")

    @property
    def chunk_count(self) -> int:
        """How many chunks are currently indexed."""
        return len(self.vector_store.chunks)

    def load_existing_index(self) -> bool:
        """Restore the index saved by a previous run, if there is one."""
        return self.vector_store.load()

    def ingest_document(
        self,
        file_path: str | Path,
        document_name: str | None = None,
    ) -> IngestionResult:
        """Parse, chunk, embed and index a document.

        Any previously indexed document is replaced, so answers can never mix
        content from two different files.

        Args:
            file_path: Path to the document on disk.
            document_name: Display name to record; defaults to the file name.

        Returns:
            An IngestionResult describing what was indexed.

        Raises:
            EmptyDocumentError: If the document produced no usable chunks.
        """
        path = Path(file_path)
        name = document_name or path.name

        text = self.parser.parse(path)
        chunks = self.chunker.split_text(text)

        if not chunks:
            raise EmptyDocumentError(f"No usable text chunks were produced from {name}.")

        embeddings = self.embedding_service.embed_documents(chunks)

        self.vector_store.create_index(
            embeddings,
            chunks,
            metadata={"document_name": name, "embedding_model": self.embedding_service.model_name},
        )
        self.vector_store.save()

        logger.info("Ingested %s into %d chunks", name, len(chunks))

        return IngestionResult(
            document_name=name,
            chunk_count=len(chunks),
            character_count=len(text),
        )

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Return the chunks most relevant to a question.

        Raises:
            IndexNotReadyError: If no document has been indexed.
            ValueError: If the question is empty.
        """
        if not question or not question.strip():
            raise ValueError("The question cannot be empty")
        if not self.is_ready:
            raise IndexNotReadyError("Upload a document before asking questions.")

        query_embedding = self.embedding_service.embed_query(question)
        return self.vector_store.search(query_embedding, top_k=top_k or settings.top_k)

    @staticmethod
    def build_context(results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into the context block sent to the model."""
        return "\n\n".join(f"[Chunk {r['id']}]\n{r['chunk']}" for r in results)

    def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """Answer a question against the indexed document.

        Returns:
            A dict with the generated "answer" and the "sources" it was based on.
        """
        results = self.retrieve(question, top_k=top_k)
        answer = self.llm.generate(context=self.build_context(results), question=question)
        return {"answer": answer, "sources": results}

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[dict[str, Any]]]:
        """Answer a question, streaming the tokens as they are generated.

        Returns:
            A (token_iterator, sources) tuple. Retrieval happens eagerly so the
            caller can render the sources without waiting for generation.
        """
        results = self.retrieve(question, top_k=top_k)
        stream = self.llm.generate_stream(context=self.build_context(results), question=question)
        return stream, results
