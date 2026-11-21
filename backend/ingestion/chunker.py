from dataclasses import dataclass
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.ingestion.document_loader import DocumentPage
from backend.ingestion.text_cleaner import clean_text


SECTION_HEADINGS = {
    "frontend": "Frontend",
    "front-end": "Frontend",
    "backend": "Backend",
    "back-end": "Backend",
    "databases & storage": "Databases & Storage",
    "infrastructure & devops": "Infrastructure & DevOps",
}


@dataclass(frozen=True)
class DocumentChunk:
    text: str
    chunk_index: int
    source_name: str
    page_number: int
    section: str | None = None


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_section: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = re.fullmatch(r"\s*([^:]{1,60}):\s*", line)
        normalized_heading = (
            re.sub(r"\s+", " ", heading.group(1).strip().casefold())
            if heading
            else ""
        )
        matched_section = SECTION_HEADINGS.get(normalized_heading)

        if matched_section:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((current_section, section_text))
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    section_text = "\n".join(current_lines).strip()
    if section_text:
        sections.append((current_section, section_text))
    return sections


def create_chunks(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be at least zero and less than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for page in pages:
        cleaned_text = clean_text(page.text)
        if not cleaned_text:
            continue

        for section, section_text in _split_sections(cleaned_text):
            for chunk_text in splitter.split_text(section_text):
                chunks.append(
                    DocumentChunk(
                        text=chunk_text.strip(),
                        chunk_index=chunk_index,
                        source_name=page.source_name,
                        page_number=page.page_number,
                        section=section,
                    )
                )
                chunk_index += 1

    return chunks
