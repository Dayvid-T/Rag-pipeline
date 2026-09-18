"""
Tests for POST /query. Retrieval and generation are both patched out at the
routes-module level, so this covers only the HTTP contract.
"""

from fastapi.testclient import TestClient

import src.api.routes as routes_module
from src.api.routes import app

client = TestClient(app)

CHUNKS = [
    {"text": "chunk one", "source": "b.txt", "score": 0.9},
    {"text": "chunk two", "source": "a.txt", "score": 0.8},
]


def test_query_returns_answer_sources_and_contexts(monkeypatch):
    monkeypatch.setattr(routes_module, "hybrid_search", lambda q, top_k: CHUNKS)
    monkeypatch.setattr(
        routes_module,
        "generate_answer",
        lambda q, chunks: {"answer": "the answer", "sources": ["a.txt", "b.txt"]},
    )

    response = client.post("/query", json={"question": "anything?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "the answer",
        "sources": ["a.txt", "b.txt"],
        "contexts": ["chunk one", "chunk two"],
    }


def test_query_reports_retrieval_failure(monkeypatch):
    def boom(q, top_k):
        raise RuntimeError("pinecone down")

    monkeypatch.setattr(routes_module, "hybrid_search", boom)

    response = client.post("/query", json={"question": "anything?"})

    assert response.status_code == 500
    assert "Retrieval failed" in response.json()["detail"]


def test_query_reports_generation_failure(monkeypatch):
    def boom(q, chunks):
        raise RuntimeError("gemini down")

    monkeypatch.setattr(routes_module, "hybrid_search", lambda q, top_k: CHUNKS)
    monkeypatch.setattr(routes_module, "generate_answer", boom)

    response = client.post("/query", json={"question": "anything?"})

    assert response.status_code == 500
    assert "Generation failed" in response.json()["detail"]


def test_query_rejects_missing_question():
    response = client.post("/query", json={})
    assert response.status_code == 422
