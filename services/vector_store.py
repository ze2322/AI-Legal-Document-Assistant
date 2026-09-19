"""FAISS-backed storage and similarity search over document chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from services.exceptions import IndexNotReadyError
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.json"


class VectorStore:
    """Stores chunk embeddings in a FAISS index and searches them by similarity.

    Chunks are persisted as JSON rather than pickle: the payload is only plain
    text, and JSON avoids executing arbitrary code when loading an index file
    that came from somewhere else.
    """

    def __init__(self, index_dir: str | Path | None = None) -> None:
        self.index_dir = Path(index_dir) if index_dir else settings.index_dir
        self.index: faiss.Index | None = None
        self.chunks: list[str] = []
        self.metadata: dict[str, Any] = {}

    @property
    def is_ready(self) -> bool:
        """True when the store holds a searchable index."""
        return self.index is not None and self.index.ntotal > 0

    def create_index(
        self,
        embeddings: np.ndarray,
        chunks: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Build a fresh index, replacing anything currently held.

        Args:
            embeddings: An (n_chunks, dimension) array of unit-norm vectors.
            chunks: The chunk texts, aligned with embeddings row for row.
            metadata: Optional extra information to persist alongside the index.

        Raises:
            ValueError: If the inputs are empty or their lengths disagree.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2-D embedding array, got shape {embeddings.shape}")
        if len(chunks) == 0 or embeddings.shape[0] == 0:
            raise ValueError("Cannot build an index from zero chunks")
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match "
                f"chunk count ({len(chunks)})"
            )

        dimension = embeddings.shape[1]

        # Inner product on unit-norm vectors is cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))

        self.chunks = list(chunks)
        self.metadata = dict(metadata or {})

        logger.info("Built FAISS index with %d vectors (dim=%d)", self.index.ntotal, dimension)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the chunks most similar to a query vector.

        Args:
            query_embedding: A 1-D unit-norm query vector.
            top_k: How many chunks to return. Defaults to the configured value
                and is capped at the number of indexed chunks.

        Returns:
            A list of {"id", "chunk", "score"} dicts, best match first.

        Raises:
            IndexNotReadyError: If no document has been indexed yet.
            ValueError: If the query dimension does not match the index.
        """
        if not self.is_ready:
            raise IndexNotReadyError("No document has been indexed yet.")

        top_k = top_k if top_k is not None else settings.top_k
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension "
                f"{self.index.d}. The index was probably built with a different "
                "embedding model."
            )

        # Asking FAISS for more neighbours than it holds just returns -1 padding.
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({"id": int(idx), "chunk": self.chunks[idx], "score": float(score)})

        return results

    def clear(self) -> None:
        """Drop the in-memory index and chunks."""
        self.index = None
        self.chunks = []
        self.metadata = {}

    def save(self, folder: str | Path | None = None) -> None:
        """Persist the index and chunks to disk.

        Raises:
            IndexNotReadyError: If there is nothing to save.
        """
        if not self.is_ready:
            raise IndexNotReadyError("There is no index to save.")

        folder = Path(folder) if folder else self.index_dir
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(folder / INDEX_FILENAME))

        payload = {"chunks": self.chunks, "metadata": self.metadata}
        (folder / CHUNKS_FILENAME).write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Saved index with %d chunks to %s", len(self.chunks), folder)

    def load(self, folder: str | Path | None = None) -> bool:
        """Load a previously saved index.

        Returns:
            True if an index was loaded, False if no saved index was found or it
            was inconsistent. A corrupt cache is never fatal: the caller can
            simply ask the user to upload the document again.
        """
        folder = Path(folder) if folder else self.index_dir

        index_path = folder / INDEX_FILENAME
        chunks_path = folder / CHUNKS_FILENAME

        if not index_path.is_file() or not chunks_path.is_file():
            logger.info("No saved index found in %s", folder)
            return False

        try:
            index = faiss.read_index(str(index_path))
            payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = payload["chunks"]
            metadata = payload.get("metadata", {})
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logger.warning("Could not load the saved index from %s: %s", folder, exc)
            return False

        if index.ntotal != len(chunks):
            logger.warning(
                "Saved index is inconsistent (%d vectors vs %d chunks); ignoring it",
                index.ntotal,
                len(chunks),
            )
            return False

        self.index = index
        self.chunks = chunks
        self.metadata = metadata

        logger.info("Loaded index with %d chunks from %s", len(chunks), folder)
        return True
