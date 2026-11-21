# Hybrid RAG Document Chatbot

An educational, Windows-friendly chatbot for asking questions about uploaded PDF, TXT, and DOCX documents. It uses local Ollama, FAISS, BM25, Reciprocal Rank Fusion (RRF), and Cross-Encoder reranking.

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
Document Upload
      |
      v
PDF / TXT / DOCX Loader
      |
      v
Text Cleaning -> Chunking -> Metadata
      |
      +--> FAISS Vector Search
      |
      +--> BM25 Keyword Search
      |
      v
RRF Hybrid Merge -> Cross-Encoder Rerank
      |
      v
Relevant Context -> Ollama LLM
      |
      v
Grounded Answer + Source and Page
```

This is the ingestion and retrieval pipeline:

- **Embedding:** a numeric representation of text meaning.
- **FAISS:** stores and searches embedding vectors quickly.
- **BM25:** finds exact or close keyword matches.
- **RRF:** combines the vector and keyword rankings.
- **Cross-Encoder:** compares each retrieved chunk against the question again to improve ordering.
- **Ollama:** runs the answer-generation model locally.

The Streamlit interface lives in `frontend/`; document processing and retrieval code live in `backend/`. Neo4j is not enabled in the baseline. It can be added later for graph questions about relationships between components in an architecture document; FAISS and BM25 remain the baseline search.

For architecture PDFs with recognized headings, search terms such as `frontend` and `backend` are scoped to their matching sections. Full-text and all-chunk views remain available under the collapsed ingestion details panel.

## Final folder structure

```text
bot_offline/
|-- frontend/
|   `-- app.py
|-- backend/
|   |-- ingestion/
|   |-- retrieval/
|   |-- generation/   (later phase)
|   `-- pipeline/     (later phase)
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

The folder structure will be created only when each implementation phase requires it.

The `generation/` and `pipeline/` backend folders are reserved for later phases; they will receive code when those components are implemented.

## Windows prerequisites

Install:

- Python 3.11 or 3.12
- Visual Studio Code
- Git
- Ollama

Check the tools:

```powershell
py -3 --version
git --version
ollama --version
```

## Phase 1: Environment setup

Open VS Code in this project folder, then open a PowerShell terminal.

### 1. Create a virtual environment

```powershell
py -3 -m venv .venv
```

### 2. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The terminal should now begin with `(.venv)`.

### 3. Confirm Python

```powershell
python --version
python -m pip --version
```

Both commands must run successfully.

### 4. Install Ollama and its model

Download and install Ollama for Windows, then run:

```powershell
ollama pull llama3.2
ollama list
```

Start Ollama:

```powershell
ollama serve
```

Keep this terminal running while using the chatbot.

### 5. Configure environment variables

Copy [.env.example](.env.example) to `.env`:

```powershell
Copy-Item .env.example .env
```

Do not commit `.env`; it is intended for local configuration.

## Implementation roadmap

The project is developed in separate confirmed stages:

1. Environment and project setup
2. Streamlit document upload
3. PDF, TXT, and DOCX ingestion
4. Text cleaning and chunking
5. Embeddings
6. FAISS vector retrieval
7. BM25 keyword retrieval
8. RRF hybrid retrieval
9. Cross-Encoder reranking
10. Ollama answer generation
11. Source citations and no-answer handling
12. Chat interface
13. Automated tests
14. Final review

No application code is created until the setup phase is confirmed.

## Quick testing commands

After dependencies and tests are added in later phases:

```powershell
pytest -q
```

Run the Streamlit application from the project root:

```powershell
python -m streamlit run frontend/app.py
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
