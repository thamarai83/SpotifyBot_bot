import io

import numpy as np
import pymupdf
from fastapi.testclient import TestClient

from backend.api import app


def create_pdf() -> bytes:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Frontend:\n- Next.js and React.\nBackend:\n- Node.js and NestJS.",
    )
    contents = pdf.tobytes()
    pdf.close()
    return contents


def test_health_endpoint():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_requires_a_processed_document():
    app.state.processed_document = None

    response = TestClient(app).post("/search", json={"query": "frontend"})

    assert response.status_code == 409


def test_process_and_section_search(monkeypatch, tmp_path):
    app.state.processed_document = None
    monkeypatch.setattr("backend.api.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("backend.api.INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setattr(
        "backend.api.generate_embeddings",
        lambda texts: np.tile(
            np.asarray([[1.0, 0.0]], dtype="float32"),
            (len(texts), 1),
        ),
    )

    client = TestClient(app)
    process_response = client.post(
        "/documents/process",
        files={"file": ("architecture.pdf", io.BytesIO(create_pdf()), "application/pdf")},
    )

    assert process_response.status_code == 200
    assert process_response.json()["chunk_count"] == 2

    frontend_response = client.post("/search", json={"query": "frontend"})
    backend_response = client.post("/search", json={"query": "backend"})

    assert frontend_response.status_code == 200
    assert [r["section"] for r in frontend_response.json()["results"]] == ["Frontend"]
    assert "Next.js" in frontend_response.json()["results"][0]["text"]
    assert backend_response.status_code == 200
    assert [r["section"] for r in backend_response.json()["results"]] == ["Backend"]
    assert "NestJS" in backend_response.json()["results"][0]["text"]
