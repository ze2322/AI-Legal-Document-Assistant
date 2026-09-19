"""Tests for DocumentParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.exceptions import (
    DocumentParsingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from services.parser import DocumentParser


@pytest.fixture
def parser() -> DocumentParser:
    return DocumentParser()


def test_parses_pdf(parser: DocumentParser, pdf_file: Path) -> None:
    text = parser.parse(pdf_file)
    assert "Implementing partners" in text


def test_parses_docx_paragraphs(parser: DocumentParser, docx_file: Path) -> None:
    text = parser.parse(docx_file)
    assert "Confidentiality obligations survive termination." in text


def test_parses_docx_tables(parser: DocumentParser, docx_file: Path) -> None:
    """Legal documents keep obligations in tables, so cells must be extracted."""
    text = parser.parse(docx_file)
    assert "Clause | Retention period" in text


def test_skips_blank_docx_paragraphs(parser: DocumentParser, docx_file: Path) -> None:
    text = parser.parse(docx_file)
    assert not any(not line.strip() for line in text.splitlines())


def test_accepts_string_paths(parser: DocumentParser, pdf_file: Path) -> None:
    assert parser.parse(str(pdf_file)) == parser.parse(pdf_file)


def test_missing_file_raises(parser: DocumentParser, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "nope.pdf")


def test_unsupported_extension_raises(parser: DocumentParser, tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("some text", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        parser.parse(path)


def test_extension_check_is_case_insensitive(parser: DocumentParser, pdf_file: Path) -> None:
    upper = pdf_file.with_suffix(".PDF")
    upper.write_bytes(pdf_file.read_bytes())

    assert "Implementing partners" in parser.parse(upper)


def test_corrupt_docx_raises_parsing_error(parser: DocumentParser, tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a zip archive")

    with pytest.raises(DocumentParsingError):
        parser.parse(path)


def test_empty_document_raises(parser: DocumentParser, tmp_path: Path) -> None:
    """A PDF with no text layer behaves like a scanned document."""
    fitz = pytest.importorskip("fitz")

    document = fitz.open()
    document.new_page()
    path = tmp_path / "scanned.pdf"
    document.save(path)
    document.close()

    with pytest.raises(EmptyDocumentError):
        parser.parse(path)
