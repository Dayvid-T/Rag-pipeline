"""
Tests for the /documents endpoints. Indexing (embed/delete/list) is patched
out at the routes-module level; PDF parsing is exercised through the real
loader with a fake PyMuPDF.
"""

import pytest
from fastapi.testclient import TestClient

import src.api.routes as routes_module
from src.api.routes import app

client = TestClient(app)


class IndexSpy:
    def __init__(self, existing=None):
        self.embedded = []
        self.deleted = []
        self._existing = existing or {}

    def embed_chunks(self, chunks):
        self.embedded.append(chunks)

    def delete_source(self, source):
        self.deleted.append(source)
        return self._existing.get(source, 0)

    def list_sources(self):
        return [{"source": s, "chunks": n} for s, n in sorted(self._existing.items())]


@pytest.fixture
def spy(monkeypatch):
    spy = IndexSpy(existing={"old.pdf": 4})
    monkeypatch.setattr(routes_module, "embed_chunks", spy.embed_chunks)
    monkeypatch.setattr(routes_module, "delete_source", spy.delete_source)
    monkeypatch.setattr(routes_module, "list_sources", spy.list_sources)
    return spy


def test_list_documents(spy):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": [{"source": "old.pdf", "chunks": 4}]}


def test_upload_txt_replaces_then_embeds(spy):
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"hello there rag pipeline", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {"source": "notes.txt", "chunks": 1}
    assert spy.deleted == ["notes.txt"]
    assert len(spy.embedded) == 1
    assert spy.embedded[0][0]["chunk_id"] == "notes.txt::0"
    assert spy.embedded[0][0]["text"] == "hello there rag pipeline"


def test_upload_strips_path_from_filename(spy):
    response = client.post(
        "/documents",
        files={"file": ("../../etc/notes.txt", b"content", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["source"] == "notes.txt"


def test_upload_rejects_unsupported_type(spy):
    response = client.post("/documents", files={"file": ("photo.png", b"\x89PNG", "image/png")})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
    assert spy.embedded == []


def test_upload_rejects_empty_text(spy):
    response = client.post("/documents", files={"file": ("blank.txt", b"   \n", "text/plain")})

    assert response.status_code == 400
    assert "No text could be extracted" in response.json()["detail"]
    assert spy.embedded == []


def test_upload_rejects_oversized_file(spy, monkeypatch):
    monkeypatch.setattr(routes_module, "MAX_UPLOAD_BYTES", 10)

    response = client.post("/documents", files={"file": ("big.txt", b"x" * 11, "text/plain")})

    assert response.status_code == 413


def test_upload_reports_indexing_failure(spy, monkeypatch):
    def boom(chunks):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(routes_module, "embed_chunks", boom)

    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 500
    assert "Indexing failed" in response.json()["detail"]


def test_delete_document(spy):
    response = client.delete("/documents/old.pdf")

    assert response.status_code == 200
    assert response.json() == {"source": "old.pdf", "chunks_deleted": 4}


def test_delete_unknown_document_is_404(spy):
    response = client.delete("/documents/missing.pdf")

    assert response.status_code == 404


CITATION = {
    "source": "old.pdf",
    "metadata": {
        "title": "Old Paper",
        "authors": [{"given": "A", "family": "B"}],
        "year": 2020,
        "venue": None,
        "document_type": "other",
        "url": None,
        "doi": None,
    },
    "ieee": '[1] A. B, "Old Paper," 2020.',
    "apa": "B, A. (2020). Old Paper.",
}


def test_cite_document(monkeypatch):
    monkeypatch.setattr(routes_module.citation, "cite", lambda source: CITATION)

    response = client.get("/documents/old.pdf/citation")

    assert response.status_code == 200
    assert response.json() == CITATION


def test_cite_unknown_document_is_404(monkeypatch):
    monkeypatch.setattr(routes_module.citation, "cite", lambda source: None)

    assert client.get("/documents/missing.pdf/citation").status_code == 404


def test_cite_reports_failure(monkeypatch):
    def boom(source):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(routes_module.citation, "cite", boom)

    response = client.get("/documents/old.pdf/citation")

    assert response.status_code == 500
    assert "Citation failed" in response.json()["detail"]
