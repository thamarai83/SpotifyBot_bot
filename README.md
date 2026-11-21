# Hybrid RAG Document Chatbot

An educational, Windows-friendly chatbot for asking questions about uploaded PDF, TXT, and DOCX documents. It uses local Ollama, FAISS, BM25, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.

The current version runs a Dockerized Streamlit frontend and FastAPI backend. It supports document upload, extraction, section-aware chunking, embeddings, and FAISS vector search. BM25, reranking, and Ollama answer generation are planned next.

## What the application will do

1. Upload a PDF, TXT, or DOCX file.
2. Extract text and preserve page numbers for PDF documents.
3. Clean and split text into smaller chunks.
4. Generate document embeddings.
5. Search chunks using both semantic vector search and keyword search.
6. Combine both rankings with RRF.
7. Rerank the strongest matches.
8. Send only relevant context to a local LLM.
9. Generate an answer with source document and page citations.
10. Return "I couldn't find this information in the uploaded document" when the answer is not supported.

## Architecture

```text
Browser
   |
   v
Streamlit frontend (port 8501)
   | HTTP requests on Docker's internal network
   v
FastAPI backend (port 8000)
   |
   +--> Upload, extraction, section chunking
   +--> Sentence Transformer embeddings
   +--> FAISS vector search
   |
   v
Persistent upload and index folders
```

This is the ingestion and retrieval pipeline:

- **Embedding:** a numeric representation of text meaning.
- **FAISS:** stores and searches embedding vectors quickly.
- **BM25:** finds exact or close keyword matches.
- **RRF:** combines the vector and keyword rankings.
- **Cross-Encoder:** compares each retrieved chunk against the question again to improve ordering.
- **Ollama:** runs the answer-generation model locally.

The Streamlit interface lives in `frontend/`; document processing and retrieval code plus the HTTP API live in `backend/`. Neo4j is not enabled in the baseline. It can be added later for graph questions about relationships between components in an architecture document.

For architecture PDFs with recognized headings, search terms such as `frontend` and `backend` are scoped to their matching sections. Full-text and all-chunk views remain available under the collapsed ingestion details panel.

## Final folder structure

```text
bot_offline/
|-- frontend/
|   |-- app.py
|   |-- Dockerfile
|   `-- requirements.txt
|-- backend/
|   |-- api.py
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- ingestion/
|   |-- retrieval/
|   |-- generation/   (later phase)
|   `-- pipeline/     (later phase)
|-- compose.yaml
|-- .dockerignore
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
|-- data/
|   |-- uploads/
|   `-- indexes/
|-- tests/
`-- logs/
```

The `generation/` and `pipeline/` backend folders are reserved for later phases; they will receive code when those components are implemented.

## Windows prerequisites

For Docker:

- Docker Desktop for Windows with Linux containers enabled

For local development, Python 3.12, Visual Studio Code, and Git are also useful. Ollama is only needed when answer generation is implemented.

## Run with Docker

Docker Desktop for Windows should be running with Linux containers enabled.

Build and start both containers from the project root:

```text
docker compose up --build
```

Open the frontend at [http://localhost:8501](http://localhost:8501). The backend API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

The Compose setup keeps uploaded documents and FAISS index files in `data/uploads/` and `data/indexes/`, and caches the embedding model between runs. The backend is available to the frontend over the internal Docker network; both published ports bind to localhost.

The currently processed document is kept in backend memory. After restarting the backend container, upload and process the document again; its uploaded file remains in `data/uploads/`.

Stop the containers with `Ctrl+C`, or run:

```text
docker compose down
```

## Run locally without Docker

Install the development dependencies and activate the virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the backend in one terminal:

```powershell
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Run the frontend in a second terminal:

```powershell
python -m streamlit run frontend/app.py
```

## Ollama (later phase)

Ollama answer generation is not connected yet. When it is added, keep Ollama on Windows initially and configure the backend to use `http://host.docker.internal:11434` from Docker.

## Implementation roadmap

The frontend and backend Docker containers currently support document processing and FAISS vector search. Next development stages are BM25 retrieval, RRF hybrid search, Cross-Encoder reranking, Ollama answer generation, citations, and a chat interface.

## Quick testing commands

Run the tests from the project root:

```powershell
python -m pytest -q
```

## Common setup errors

### `py -3` is not recognized

Install Python from python.org or confirm that the Python Launcher is selected during installation. Then reopen VS Code.

### PowerShell activation is blocked

Run activation with the process-level execution policy shown above.

### Ollama cannot be reached

Confirm that Ollama is running:

```powershell
ollama serve
```

Then confirm that the model exists:

```powershell
ollama list
```

### The virtual environment uses the wrong Python

Delete `.venv` only if necessary, recreate it with `py -3`, and activate it again.
