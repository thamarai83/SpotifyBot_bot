import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from streamlit.typing import UploadedFile


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_UPLOAD_SIZE_MB = 20
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def validate_upload(uploaded_file: UploadedFile) -> str | None:
    if uploaded_file is None:
        return None

    extension = Path(uploaded_file.name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return "Unsupported file type. Upload a PDF, TXT, or DOCX file."
    if uploaded_file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return f"File is too large. The maximum size is {MAX_UPLOAD_SIZE_MB} MB."
    if not uploaded_file.getvalue():
        return "The uploaded document is empty."

    return None


def call_backend(
    method: str,
    endpoint: str,
    timeout: tuple[int, int] = (5, 60),
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.request(
            method,
            f"{BACKEND_URL}{endpoint}",
            timeout=timeout,
            **kwargs,
        )
    except requests.RequestException as error:
        return None, f"Could not connect to the backend at {BACKEND_URL}: {error}"

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if not response.ok:
        detail = (
            payload.get("detail", response.text)
            if isinstance(payload, dict)
            else response.text
        )
        return None, f"Backend error ({response.status_code}): {detail}"
    if not isinstance(payload, dict):
        return None, "The backend returned an unexpected response."

    return payload, None


st.set_page_config(
    page_title="Hybrid RAG Document Chatbot",
    page_icon=":material/description:",
    layout="wide",
)

st.session_state.setdefault("uploaded_document", None)
st.session_state.setdefault("processing_status", "")
st.session_state.setdefault("document_metrics", None)
st.session_state.setdefault("vector_search_results", [])
st.session_state.setdefault("vector_search_submitted", False)
st.session_state.setdefault("search_error", None)
st.session_state.setdefault("ingestion_details", None)

st.title("Hybrid RAG Document Chatbot")
st.caption("Upload a PDF, TXT, or DOCX document, then search its content.")

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
    else:
        st.success(f"Selected: {uploaded_file.name}")
        file_size_column, file_type_column = st.columns(2)
        file_size_column.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")
        file_type_column.metric(
            "File type",
            Path(uploaded_file.name).suffix.lower().replace(".", "").upper(),
        )

        if st.button(
            "Process document",
            icon=":material/upload:",
            type="primary",
            width="stretch",
        ):
            st.session_state.vector_search_results = []
            st.session_state.vector_search_submitted = False
            st.session_state.search_error = None
            st.session_state.ingestion_details = None

            upload = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }
            with st.spinner("Backend is extracting, chunking, and indexing the document..."):
                result, error = call_backend(
                    "POST",
                    "/documents/process",
                    files=upload,
                    timeout=(10, 600),
                )

            if error:
                st.session_state.uploaded_document = None
                st.session_state.document_metrics = None
                st.session_state.processing_status = "Document processing failed"
                st.error(error)
            else:
                st.session_state.uploaded_document = result["filename"]
                st.session_state.document_metrics = result
                st.session_state.processing_status = result["status"]
                st.success(f"Processed {result['filename']}.")

if st.session_state.uploaded_document:
    with st.container(border=True):
        st.subheader("Processed document")
        st.write(f"**Name:** {st.session_state.uploaded_document}")
        st.write(f"**Status:** {st.session_state.processing_status}")

        metrics = st.session_state.document_metrics
        if metrics:
            page_column, character_column, chunk_column, embedding_column = st.columns(4)
            page_column.metric("Pages/sections", metrics["page_count"])
            character_column.metric("Extracted characters", metrics["character_count"])
            chunk_column.metric("Text chunks", metrics["chunk_count"])
            embedding_column.metric("Embeddings", metrics["embedding_count"])

        query = st.text_input(
            "Search the document",
            key="vector_search_query",
            placeholder="Example: frontend",
        )
        st.caption(
            "Search for frontend or backend to see only that section. "
            "Full questions use semantic search."
        )
        if st.button("Search", icon=":material/search:"):
            st.session_state.vector_search_submitted = True
            st.session_state.search_error = None
            if not query.strip():
                st.session_state.vector_search_results = []
                st.warning("Enter a search question.")
            else:
                with st.spinner("Searching document..."):
                    result, error = call_backend(
                        "POST",
                        "/search",
                        json={"query": query},
                    )

                if error:
                    st.session_state.vector_search_results = []
                    st.session_state.search_error = error
                else:
                    st.session_state.vector_search_results = result["results"]

        if st.session_state.search_error:
            st.error(st.session_state.search_error)
        elif st.session_state.vector_search_results:
            st.subheader("Document search results")
            for rank, result in enumerate(st.session_state.vector_search_results, start=1):
                section = result["section"] or "Uncategorized"
                with st.expander(
                    f"{rank}. Similarity {result['similarity']:.4f} - "
                    f"{section}, {result['source_name']}, "
                    f"page {result['page_number']}"
                ):
                    st.text(result["text"])
        elif st.session_state.vector_search_submitted and query.strip():
            st.info(f"No matching document chunks found for '{query.strip()}'.")

        with st.expander("Ingestion details (full text and all chunks)"):
            if st.button("Load ingestion details", key="load_ingestion_details"):
                details, error = call_backend("GET", "/documents/details")
                if error:
                    st.error(error)
                else:
                    st.session_state.ingestion_details = details

            details = st.session_state.ingestion_details
            if details:
                st.subheader("Extracted text")
                for page in details["pages"]:
                    with st.expander(f"Page/section {page['page_number']}"):
                        st.text(page["text"])

                st.subheader("Text chunks")
                for chunk in details["chunks"]:
                    section = chunk["section"] or "Uncategorized"
                    with st.expander(
                        f"Chunk {chunk['chunk_index'] + 1} - {section} - "
                        f"{chunk['source_name']}, page {chunk['page_number']}"
                    ):
                        st.write(f"**Characters:** {len(chunk['text'])}")
                        st.text(chunk["text"])
