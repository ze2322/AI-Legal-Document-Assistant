from pathlib import Path
import fitz  # PyMuPDF
from docx import Document


class DocumentParser:
    """Handles extracting text from PDF and DOCX files."""

    def parse(self, file_path: str) -> str:
        """
        Extract text from a supported document.

        Args:
            file_path (str): Path to the uploaded file.

        Returns:
            str: Extracted document text.

        Raises:
            ValueError: If the file type is unsupported.
        """

        extension = Path(file_path).suffix.lower()

        if extension == ".pdf":
            return self._parse_pdf(file_path)

        elif extension == ".docx":
            return self._parse_docx(file_path)

        raise ValueError(f"Unsupported file type: {extension}")

    def _parse_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""

        text = ""

        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()

        return text

    def _parse_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""

        document = Document(file_path)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)
    

    