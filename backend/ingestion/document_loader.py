from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@dataclass(frozen=True)
class DocumentPage:
    text: str
    page_number: int
    source_name: str


class DocumentLoadError(Exception):
    pass


def load_document(file_path: str | Path) -> list[DocumentPage]:
    path = Path(file_path)
    if not path.exists():
        raise DocumentLoadError(f"Document does not exist: {path}")
    if not path.is_file():
        raise DocumentLoadError(f"Path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{extension}'. Use PDF, TXT, or DOCX."
        )
    if path.stat().st_size == 0:
        raise DocumentLoadError("The document is empty.")

    if extension == ".pdf":
        return _load_pdf(path)
    if extension == ".docx":
        return _load_docx(path)
    return _load_text(path)


def _load_pdf(path: Path) -> list[DocumentPage]:
    try:
        with pymupdf.open(path) as pdf:
            pages = [
                DocumentPage(
                    text=page.get_text("text").strip(),
                    page_number=page_number,
                    source_name=path.name,
                )
                for page_number, page in enumerate(pdf, start=1)
            ]
    except (pymupdf.FileDataError, RuntimeError, OSError) as error:
        raise DocumentLoadError(f"Could not read PDF '{path.name}'.") from error

    if not any(page.text for page in pages):
        raise DocumentLoadError("The PDF contains no extractable text.")
    return pages


def _load_docx(path: Path) -> list[DocumentPage]:
    try:
        document = Document(path)
        parts = [paragraph.text.strip() for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text:
                    parts.append(row_text)
    except (OSError, ValueError) as error:
        raise DocumentLoadError(f"Could not read DOCX '{path.name}'.") from error

    text = "\n".join(part for part in parts if part).strip()
    if not text:
        raise DocumentLoadError("The DOCX document contains no text.")
    return [DocumentPage(text=text, page_number=1, source_name=path.name)]


def _load_text(path: Path) -> list[DocumentPage]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as error:
        raise DocumentLoadError("The TXT file must be UTF-8 encoded.") from error
    except OSError as error:
        raise DocumentLoadError(f"Could not read TXT '{path.name}'.") from error

    if not text:
        raise DocumentLoadError("The TXT document contains no text.")
    return [DocumentPage(text=text, page_number=1, source_name=path.name)]
