from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL
import numpy as np


class EmbeddingService:
    """
    Handles generating text embeddings using Sentence Transformers.
    """

    def __init__(self):
        """
        Load the embedding model once when the service is created.
        """
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple text chunks.

        Args:
            texts: List of text chunks.

        Returns:
            Numpy array of embeddings.
        """
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a user question.

        Args:
            query: User question.

        Returns:
            Query embedding.
        """
        return self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )