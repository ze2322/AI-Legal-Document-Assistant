import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Handles storing and retrieving document embeddings using FAISS.
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    def create_index(self, embeddings: np.ndarray, chunks: list[str]):
        """
        Create a FAISS index from embeddings.
        """

        dimension = embeddings.shape[1]

        # IndexFlatIP works well with normalized embeddings
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings.astype("float32"))

        self.chunks = chunks

    def search(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Search for the most relevant chunks.
        """

        distances, indices = self.index.search(
            query_embedding.astype("float32").reshape(1, -1),
            top_k
        )

        results = []

        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            results.append({
            "id": int(idx),
            "chunk": self.chunks[idx],
            "score": float(score)
            })

        return results

    def save(self, folder="data/faiss_index"):
        """
        Save the FAISS index and chunks.
        """

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(folder / "index.faiss")
        )

        with open(folder / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, folder="data/faiss_index"):
        """
        Load a previously saved FAISS index.
        """

        folder = Path(folder)

        self.index = faiss.read_index(
            str(folder / "index.faiss")
        )

        with open(folder / "chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)