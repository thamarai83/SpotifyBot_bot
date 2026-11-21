from pathlib import Path

import pytest

from backend.ingestion.chunker import create_chunks
from backend.ingestion.document_loader import DocumentPage, load_document
from backend.ingestion.text_cleaner import clean_text


def test_clean_text_normalizes_whitespace():
    dirty = "  Spotify\r\n\r\n\r\n architecture   uses\tmicroservices.  "

    result = clean_text(dirty)

    assert result == "Spotify\n\narchitecture uses microservices."


def test_create_chunks_preserves_page_metadata(tmp_path: Path):
    file_path = tmp_path / "spotify.txt"
    file_path.write_text(
        "Spotify architecture uses event delivery and recommendation services.",
        encoding="utf-8",
    )
    pages = load_document(file_path)

    chunks = create_chunks(pages, chunk_size=30, chunk_overlap=5)

    assert chunks
    assert all(chunk.source_name == "spotify.txt" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_overlap_must_be_valid():
    with pytest.raises(ValueError, match="chunk_overlap"):
        create_chunks([], chunk_size=20, chunk_overlap=20)


def test_empty_page_does_not_create_chunk():
    chunks = create_chunks(
        [DocumentPage(text="   ", page_number=1, source_name="empty.txt")],
    )

    assert chunks == []


def test_chunks_are_split_at_frontend_and_backend_headings():
    page = DocumentPage(
        text=(
            "Overview of the application.\n"
            "Frontend:\n- Next.js and React.\n- Tailwind CSS.\n"
            "Backend:\n- Node.js and NestJS.\n"
            "Databases & Storage:\n- PostgreSQL and Redis."
        ),
        page_number=1,
        source_name="architecture.txt",
    )

    chunks = create_chunks([page])
    by_section = {chunk.section: chunk.text for chunk in chunks}

    assert "Next.js" in by_section["Frontend"]
    assert "Node.js" not in by_section["Frontend"]
    assert "Node.js" in by_section["Backend"]
    assert "Next.js" not in by_section["Backend"]
    assert "PostgreSQL" in by_section["Databases & Storage"]


def test_unsectioned_text_is_not_assigned_to_a_section():
    page = DocumentPage(
        text="Overview before heading.\nFrontend:\n- React.",
        page_number=1,
        source_name="architecture.txt",
    )

    chunks = create_chunks([page])

    assert chunks[0].section is None
    assert chunks[1].section == "Frontend"
