from pathlib import Path

import streamlit as st
from streamlit.typing import UploadedFile

from backend.ingestion.chunker import create_chunks
from backend.ingestion.document_loader import DocumentLoadError, load_document
from backend.retrieval.embeddings import generate_embeddings
from backend.retrieval.query_matching import (
    contains_exact_term,
    exact_search_term,
    sections_in_query,
)
from backend.retrieval.vector_store import VectorStore


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_SIZE_MB = 20
PROJECT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_DIR / "data" / "uploads"
INDEX_DIR = PROJECT_DIR / "data" / "indexes"


def validate_upload(uploaded_file: UploadedFile) -> str | None:
    if uploaded_file is None:
        return None

    file_name = Path(uploaded_file.name).name
    extension = Path(file_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return "Unsupported file type. Upload a PDF, TXT, or DOCX file."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return f"File is too large. The maximum size is {MAX_UPLOAD_SIZE_MB} MB."

    if not uploaded_file.getvalue():
        return "The uploaded document is empty."

    return None


st.set_page_config(
    page_title="Hybrid RAG Document Chatbot",
    page_icon=":material/description:",
    layout="wide",
)

st.session_state.setdefault("uploaded_document", None)
st.session_state.setdefault("processing_status", "")
st.session_state.setdefault("extracted_pages", [])
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("embeddings", None)
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("vector_search_results", [])
st.session_state.setdefault("vector_search_submitted", False)

st.title("Hybrid RAG Document Chatbot")
st.caption("Upload a PDF, TXT, or DOCX document before document processing begins.")

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=False,
    max_upload_size=MAX_UPLOAD_SIZE_MB,
    key="document_upload",
    help=f"Supported formats: PDF, TXT, and DOCX. Maximum size: {MAX_UPLOAD_SIZE_MB} MB.",
)

if uploaded_file is not None:
    validation_error = validate_upload(uploaded_file)
    if validation_error:
        st.error(validation_error)
        st.session_state.uploaded_document = None
    else:
        st.success(f"Selected: {uploaded_file.name}")

        left, right = st.columns(2)
        left.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
        right.metric("File type", Path(uploaded_file.name).suffix.lower().replace(".", "").upper())

        if st.button(
            "Process document",
            icon=":material/upload:",
            type="primary",
            width="stretch",
        ):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = Path(uploaded_file.name).name
            destination = UPLOAD_DIR / safe_name

            if destination.exists():
                st.info("This document is already uploaded. Processing the existing file.")
                st.session_state.uploaded_document = {
                    "name": safe_name,
                    "size": uploaded_file.size,
                    "path": str(destination),
                }
            else:
                destination.write_bytes(uploaded_file.getvalue())
                st.session_state.uploaded_document = {
                    "name": safe_name,
                    "size": uploaded_file.size,
                    "path": str(destination),
                }
                st.session_state.processing_status = "Document upload complete"
                st.success(f"Saved: {safe_name}")

            destination_document = st.session_state.uploaded_document
            if destination_document is None:
                destination_document = {
                    "name": safe_name,
                    "size": uploaded_file.size,
                    "path": str(destination),
                }
                st.session_state.uploaded_document = destination_document

            try:
                pages = load_document(destination_document["path"])
                st.session_state.extracted_pages = pages
                chunks = create_chunks(pages)
                st.session_state.chunks = chunks
                st.session_state.embeddings = generate_embeddings(
                    [chunk.text for chunk in chunks]
                )
                if len(st.session_state.embeddings) > 0:
                    embedding_dimension = st.session_state.embeddings.shape[1]
                    vector_store = VectorStore(embedding_dimension)
                    vector_store.add(st.session_state.embeddings)
                    vector_store.save(INDEX_DIR)
                    st.session_state.vector_store = vector_store
                st.session_state.processing_status = (
                    "Text extraction, chunking, embedding, and FAISS indexing complete"
                )
                st.success(f"Extracted text from {len(pages)} page/section.")
                st.success(f"Created {len(chunks)} text chunks.")
                st.success(
                    f"Generated {len(st.session_state.embeddings)} embeddings."
                )
            except DocumentLoadError as error:
                st.session_state.extracted_pages = []
                st.session_state.chunks = []
                st.session_state.embeddings = None
                st.session_state.vector_store = None
                st.session_state.processing_status = "Text extraction failed"
                st.error(str(error))
            except ValueError as error:
                st.session_state.chunks = []
                st.session_state.embeddings = None
                st.session_state.vector_store = None
                st.session_state.processing_status = "Chunking failed"
                st.error(str(error))
            except (OSError, RuntimeError, ValueError) as error:
                st.session_state.embeddings = None
                st.session_state.processing_status = "Embedding generation failed"
                st.error(f"Could not generate embeddings: {error}")

if st.session_state.uploaded_document:
    document = st.session_state.uploaded_document
    with st.container(border=True):
        st.subheader("Ready document")
        st.write(f"**Name:** {document['name']}")
        st.write(f"**Status:** {st.session_state.processing_status}")

        extracted_pages = st.session_state.extracted_pages
        if extracted_pages:
            page_count = len(extracted_pages)
            total_characters = sum(len(page.text) for page in extracted_pages)
            chunk_count = len(st.session_state.chunks)
            chunks = st.session_state.chunks
            embedding_count = (
                len(st.session_state.embeddings)
                if st.session_state.embeddings is not None
                else 0
            )
            page_metric, character_metric, chunk_metric, embedding_metric = st.columns(4)
            page_metric.metric("Pages/sections", page_count)
            character_metric.metric("Extracted characters", total_characters)
            chunk_metric.metric("Text chunks", chunk_count)
            embedding_metric.metric("Embeddings", embedding_count)

            if st.session_state.vector_store is not None:
                indexed_vectors = st.session_state.vector_store.index.ntotal
                st.caption(f"FAISS vectors indexed: {indexed_vectors}")

                query = st.text_input(
                    "Search the document",
                    key="vector_search_query",
                    placeholder="Example: How does Spotify deliver events?",
                )
                st.caption(
                    "Search for frontend or backend to see only that section. "
                    "Full questions use semantic search."
                )
                if st.button(
                    "Search",
                    icon=":material/search:",
                ):
                    if not query.strip():
                        st.warning("Enter a search question.")
                    else:
                        query_embedding = generate_embeddings([query])
                        selected_sections = sections_in_query(query)
                        exact_term = (
                            exact_search_term(query) if not selected_sections else None
                        )
                        results = st.session_state.vector_store.search(
                            query_embedding[0],
                            top_k=(
                                st.session_state.vector_store.index.ntotal
                                if exact_term or selected_sections
                                else 5
                            ),
                        )
                        if selected_sections:
                            results = [
                                result
                                for result in results
                                if chunks[result.chunk_index].section
                                in selected_sections
                            ][:5]
                        elif exact_term:
                            results = [
                                result
                                for result in results
                                if contains_exact_term(
                                    chunks[result.chunk_index].text,
                                    exact_term,
                                )
                            ][:5]
                        st.session_state.vector_search_results = results
                        st.session_state.vector_search_submitted = True

                if st.session_state.vector_search_results:
                    st.subheader("Document search results")
                    for rank, result in enumerate(
                        st.session_state.vector_search_results,
                        start=1,
                    ):
                        chunk = chunks[result.chunk_index]
                        with st.expander(
                            f"{rank}. Similarity {result.similarity:.4f} - "
                            f"{chunk.section or 'Uncategorized'}, "
                            f"{chunk.source_name}, page {chunk.page_number}"
                        ):
                            st.text(chunk.text)
                elif st.session_state.vector_search_submitted:
                    st.info(f"No matching document chunks found for '{query.strip()}'.")

            with st.expander("Ingestion details (full text and all chunks)"):
                st.subheader("Extracted text")
                for page in extracted_pages:
                    with st.expander(
                        f"Page/section {page.page_number}"
                    ):
                        st.text(page.text)

                if chunks:
                    st.subheader("Text chunks")
                    for chunk in chunks:
                        section_name = chunk.section or "Uncategorized"
                        with st.expander(
                            f"Chunk {chunk.chunk_index + 1} - {section_name} - "
                            f"{chunk.source_name}, page {chunk.page_number}"
                        ):
                            st.write(f"**Characters:** {len(chunk.text)}")
                            st.text(chunk.text)
