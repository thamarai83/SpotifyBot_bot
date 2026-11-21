import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.ingestion.chunker import DocumentChunk, create_chunks
from backend.ingestion.document_loader import (
    DocumentLoadError,
    DocumentPage,
    load_document,
)
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.query_matching import (
    contains_exact_term,
    exact_search_term,
    sections_in_query,
)
from backend.retrieval.vector_store import VectorStore


logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
MAX_SEARCH_LENGTH = 1000
RESULT_LIMIT = 5


@dataclass
class ProcessedDocument:
    name: str
    pages: list[DocumentPage]
    chunks: list[DocumentChunk]
    vector_store: VectorStore


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_SEARCH_LENGTH)


class ProcessResponse(BaseModel):
    filename: str
    page_count: int
    character_count: int
    chunk_count: int
    embedding_count: int
    status: str


app = FastAPI(title="Hybrid RAG Backend")
app.state.processed_document = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/process", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...)) -> ProcessResponse:
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if not filename or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Upload a PDF, TXT, or DOCX document.",
        )

    contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded document is empty.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The maximum document size is 20 MB.",
        )

    app.state.processed_document = None
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIR / filename
    try:
        destination.write_bytes(contents)
    except OSError as error:
        logger.exception("Could not save uploaded document %s", filename)
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded document.",
        ) from error

    try:
        pages = load_document(destination)
        chunks = create_chunks(pages)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No searchable text was found in the document.",
            )

        embeddings = generate_embeddings([chunk.text for chunk in chunks])
        vector_store = VectorStore(embeddings.shape[1])
        vector_store.add(embeddings)
        vector_store.save(INDEX_DIR)
    except DocumentLoadError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, RuntimeError) as error:
        logger.exception("Document processing failed for %s", filename)
        raise HTTPException(
            status_code=503,
            detail="Document processing failed. Check the backend logs.",
        ) from error

    app.state.processed_document = ProcessedDocument(
        name=filename,
        pages=pages,
        chunks=chunks,
        vector_store=vector_store,
    )
    return ProcessResponse(
        filename=filename,
        page_count=len(pages),
        character_count=sum(len(page.text) for page in pages),
        chunk_count=len(chunks),
        embedding_count=len(embeddings),
        status="Document processed and indexed",
    )


@app.get("/documents/details")
def document_details() -> dict[str, list[dict[str, object]]]:
    document: ProcessedDocument | None = app.state.processed_document
    if document is None:
        raise HTTPException(
            status_code=409,
            detail="Upload and process a document first.",
        )

    return {
        "pages": [
            {"page_number": page.page_number, "text": page.text}
            for page in document.pages
        ],
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "source_name": chunk.source_name,
                "page_number": chunk.page_number,
                "section": chunk.section,
                "text": chunk.text,
            }
            for chunk in document.chunks
        ],
    }


@app.post("/search")
def search_document(request: SearchRequest) -> dict[str, object]:
    document: ProcessedDocument | None = app.state.processed_document
    if document is None:
        raise HTTPException(
            status_code=409,
            detail="Upload and process a document before searching.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="Enter a search question.")

    try:
        query_embedding = generate_embeddings([query])
    except (OSError, RuntimeError, ValueError) as error:
        logger.exception("Could not create query embedding")
        raise HTTPException(
            status_code=503,
            detail="Search failed while creating the query embedding.",
        ) from error

    selected_sections = sections_in_query(query)
    exact_term = exact_search_term(query) if not selected_sections else None
    results = document.vector_store.search(
        query_embedding[0],
        top_k=(
            document.vector_store.index.ntotal
            if exact_term or selected_sections
            else RESULT_LIMIT
        ),
    )

    if selected_sections:
        results = [
            result
            for result in results
            if document.chunks[result.chunk_index].section in selected_sections
        ][:RESULT_LIMIT]
    elif exact_term:
        results = [
            result
            for result in results
            if contains_exact_term(
                document.chunks[result.chunk_index].text,
                exact_term,
            )
        ][:RESULT_LIMIT]

    return {
        "results": [
            {
                "chunk_index": result.chunk_index,
                "similarity": result.similarity,
                "text": document.chunks[result.chunk_index].text,
                "source_name": document.chunks[result.chunk_index].source_name,
                "page_number": document.chunks[result.chunk_index].page_number,
                "section": document.chunks[result.chunk_index].section,
            }
            for result in results
        ]
    }
