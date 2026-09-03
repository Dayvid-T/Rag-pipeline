"""
Tests for src.retrieval.hybrid_search.

Pinecone is mocked entirely here - no real API key, no real network call.
That means this suite runs identically on your machine and in CI, and
never touches your actual Pinecone index or costs anything to run.
"""

import src.retrieval.hybrid_search as hybrid_search_module
from src.retrieval.hybrid_search import embed_chunks, hybrid_search


class FakeEmbeddingResult:
    def __init__(self, vectors):
        self.data = [{"values": v} for v in vectors]


class FakeIndex:
    def __init__(self):
        self.upserted = None

    def upsert(self, vectors):
        self.upserted = vectors

    def query(self, vector, top_k, include_metadata):
        return {
            "matches": [
                {
                    "id": "a.txt::0",
                    "score": 0.9,
                    "metadata": {"text": "matched text", "source": "a.txt"},
                }
            ]
        }


class FakePinecone:
    class inference:
        @staticmethod
        def embed(model, inputs, parameters):
            # one throwaway 3-number vector per input - real vectors are
            # 1024 numbers, but nothing here checks the actual length
            return FakeEmbeddingResult([[0.1, 0.2, 0.3] for _ in inputs])


def _patch_pinecone(monkeypatch, fake_index):
    monkeypatch.setattr(hybrid_search_module, "_get_index", lambda: fake_index)
    monkeypatch.setattr(hybrid_search_module, "_pc", FakePinecone())


def test_embed_chunks_empty_list_does_nothing(monkeypatch):
    fake_index = FakeIndex()
    _patch_pinecone(monkeypatch, fake_index)

    embed_chunks([])

    assert fake_index.upserted is None


def test_embed_chunks_upserts_correct_vectors(monkeypatch):
    fake_index = FakeIndex()
    _patch_pinecone(monkeypatch, fake_index)

    chunks = [
        {"text": "hello", "source": "a.txt", "chunk_id": "a.txt::0"},
        {"text": "world", "source": "a.txt", "chunk_id": "a.txt::1"},
    ]
    embed_chunks(chunks)

    assert len(fake_index.upserted) == 2
    assert fake_index.upserted[0]["id"] == "a.txt::0"
    assert fake_index.upserted[0]["values"] == [0.1, 0.2, 0.3]
    assert fake_index.upserted[0]["metadata"] == {"text": "hello", "source": "a.txt"}


def test_hybrid_search_returns_correct_shape(monkeypatch):
    fake_index = FakeIndex()
    _patch_pinecone(monkeypatch, fake_index)

    results = hybrid_search("some question", top_k=1)

    assert results == [{"text": "matched text", "source": "a.txt", "score": 0.9}]
