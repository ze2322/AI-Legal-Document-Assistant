from pathlib import Path

from services.parser import DocumentParser
from services.chunker import TextChunker
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore


parser = DocumentParser()
chunker = TextChunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()

text = parser.parse(
    Path("data/documents/Policy Guidelines for Agreements with Implementing Partners.pdf")
)

chunks = chunker.split_text(text)

embeddings = embedding_service.embed_documents(chunks)

vector_store.create_index(
    embeddings,
    chunks
)

query = "What are the responsibilities of implementing partners?"

query_embedding = embedding_service.embed_query(query)

results = vector_store.search(
    query_embedding,
    top_k=3
)

print("\nTop Results\n")

for i, result in enumerate(results, 1):
    print("=" * 80)
    print(f"Result {i}")
    print("=" * 80)
    print(f"Chunk ID: {result['id']}")
    print(f"Similarity Score: {result['score']:.4f}")
    print()
    print(result["chunk"])
    print()

vector_store.save()


from services.vector_store import VectorStore

vector_store = VectorStore()

vector_store.load()

print(len(vector_store.chunks))