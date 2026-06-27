from pathlib import Path

from services.rag import RAGPipeline

rag = RAGPipeline()

rag.ingest_document(
    Path("data/documents/Policy Guidelines for Agreements with Implementing Partners.pdf")
)

question = input("Ask a question: ")

response = rag.ask(question)

print("\n" + "=" * 80)
print("AI Answer")
print("=" * 80)
print(response["answer"])

print("\n" + "=" * 80)
print("Retrieved Sources")
print("=" * 80)

for source in response["sources"]:
    print(f"\nChunk ID: {source['id']}")
    print(f"Similarity Score: {source['score']:.4f}")
    print("-" * 80)
    print(source["chunk"][:250] + "...")