from streamlit.testing.v1 import AppTest


def test_app_starts_without_error():
    at = AppTest.from_file("../frontend/app.py").run()

    assert not at.exception
    assert at.title[0].value == "Hybrid RAG Document Chatbot"
    assert at.file_uploader


def test_pdf_upload_is_accepted():
    at = AppTest.from_file("../frontend/app.py").run()

    at.file_uploader(key="document_upload").upload(
        "spotify.pdf",
        b"%PDF-1.4\n%%EOF",
    ).run()

    assert not at.exception
    assert at.success
