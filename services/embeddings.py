"""Sentence-Transformers embedding service."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Turns text into normalised dense vectors.

    The model weights are loaded lazily on first use. Loading takes a few
    seconds and several hundred MB of RAM, so deferring it keeps application
    start-up (and the test suite) fast.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """The underlying model, loaded on first access."""
        if self._model is None:
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensionality of the vectors this service produces.

        sentence-transformers 5.x renamed ``get_sentence_embedding_dimension``
        to ``get_embedding_dimension``; both spellings are supported so the
        service works across versions.
        """
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of text chunks.

        Args:
            texts: Text chunks to embed.

        Returns:
            A (len(texts), dimension) float32 array of unit-norm vectors. An
            empty input yields an empty array with the correct number of
            columns, so callers can rely on shape[1] either way.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("Embedded %d chunks", len(texts))
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single user question.

        Args:
            query: The question text.

        Returns:
            A 1-D float32 unit-norm vector.

        Raises:
            ValueError: If the query is empty.
        """
        if not query or not query.strip():
            raise ValueError("Cannot embed an empty query")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype("float32")
