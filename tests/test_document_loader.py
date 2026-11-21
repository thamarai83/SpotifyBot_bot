from pathlib import Path

import pymupdf
import pytest
from docx import Document

from backend.ingestion.document_loader import DocumentLoadError, load_document


def test_load_txt(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Spotify uses microservices.", encoding="utf-8")

    pages = load_document(file_path)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].source_name == "notes.txt"
    assert "microservices" in pages[0].text


def test_load_docx(tmp_path: Path):
    file_path = tmp_path / "architecture.docx"
    document = Document()
    document.add_paragraph("The recommendation service receives events.")
    document.save(file_path)

    pages = load_document(file_path)

    assert len(pages) == 1
    assert "recommendation service" in pages[0].text
    assert pages[0].source_name == "architecture.docx"


def test_load_pdf_preserves_page_numbers(tmp_path: Path):
    file_path = tmp_path / "spotify.pdf"
    pdf = pymupdf.open()
    first_page = pdf.new_page()
    first_page.insert_text((72, 72), "Spotify architecture overview")
    second_page = pdf.new_page()
    second_page.insert_text((72, 72), "Event delivery pipeline")
    pdf.save(file_path)
    pdf.close()

    pages = load_document(file_path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Spotify architecture" in pages[0].text
    assert pages[1].page_number == 2
    assert "Event delivery" in pages[1].text


def test_empty_txt_is_rejected(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="empty"):
        load_document(file_path)


def test_unsupported_extension_is_rejected(tmp_path: Path):
    file_path = tmp_path / "document.csv"
    file_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="Unsupported"):
        load_document(file_path)


def test_missing_file_is_rejected(tmp_path: Path):
    with pytest.raises(DocumentLoadError, match="does not exist"):
        load_document(tmp_path / "missing.pdf")
