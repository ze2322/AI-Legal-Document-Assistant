"""Splitting extracted document text into overlapping chunks."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Splits document text into overlapping, retrieval-sized chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Args:
            chunk_size: Maximum characters per chunk. Defaults to the configured value.
            chunk_overlap: Characters shared between neighbouring chunks, which keeps
                sentences that straddle a boundary retrievable. Defaults to the
                configured value.
        """
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks, dropping any that are blank.

        Args:
            text: Extracted document text.

        Returns:
            A list of non-empty text chunks.
        """
        if not text or not text.strip():
            return []

        chunks = [chunk.strip() for chunk in self.splitter.split_text(text)]
        chunks = [chunk for chunk in chunks if chunk]

        logger.info("Split %d characters into %d chunks", len(text), len(chunks))
        return chunks
