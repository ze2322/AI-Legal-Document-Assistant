from services.parser import DocumentParser
from services.chunker import TextChunker
from pathlib import Path

parser = DocumentParser()
chunker = TextChunker()

text = parser.parse(
    Path("data/documents/Policy Guidelines for Agreements with Implementing Partners.pdf")
)

chunks = chunker.split_text(text)

print(f"Number of chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[3])