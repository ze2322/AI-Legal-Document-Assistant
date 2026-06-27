from pathlib import Path

from services.parser import DocumentParser
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore
from services.llm import LLMService

from utils.config import TOP_K


class RAGPipeline:
    """
    Coordinates the complete Retrieval-Augmented Generation pipeline.
    """

    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm = LLMService()

        index_path = Path("data/faiss_index/index.faiss")

        if index_path.exists():
            self.vector_store.load()

    def ingest_document(self, file_path: Path):
        """
        Parse, chunk, embed, and index a document.
        """

        text = self.parser.parse(file_path)

        chunks = self.chunker.split_text(text)

        embeddings = self.embedding_service.embed_documents(chunks)

        self.vector_store.create_index(
            embeddings,
            chunks
        )

        self.vector_store.save()

    def ask(self, question: str):
        """
        Answer a user's question.
        """

        query_embedding = self.embedding_service.embed_query(question)

        results = self.vector_store.search(
            query_embedding,
            top_k=TOP_K
        )

        context = ""

        for result in results:
           context += (
             f"[Chunk {result['id']}]\n"
             f"{result['chunk']}\n\n"
            )

        answer = self.llm.generate(
            context=context,
            question=question
        )

        return {
    "answer": answer,
    "sources": results
        }