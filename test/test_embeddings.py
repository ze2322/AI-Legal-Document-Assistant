from services.parser import DocumentParser
from services.chunker import TextChunker
from services.embeddings import EmbeddingService

parser = DocumentParser()
chunker = TextChunker()
embedding_service = EmbeddingService()

text = parser.parse("data\\documents\\Policy Guidelines for Agreements with Implementing Partners.pdf")

chunks = chunker.split_text(text)

embeddings = embedding_service.embed_documents(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)