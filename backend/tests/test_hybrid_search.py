"""
Tests for src.retrieval.hybrid_search.

Pinecone and the local BM25 index are both mocked entirely here - no real
API key, no real network call, no dependency on backend/data/. That means
this suite runs identically on your machine and in CI.
"""

import src.retrieval.hybrid_search as hybrid_search_module
from src.retrieval.hybrid_search import embed_chunks, dense_search, sparse_search, hybrid_search


class FakeEmbeddingResult:
    def __init__(self, vectors):
        self.data = [{"values": v} for v in vectors]


class FakeIndex:
    def __init__(self, matches=None):
        self.upserted = None
        self._matches = matches or []

    def upsert(self, vectors):
        self.upserted = vectors

    def query(self, vector, top_k, include_metadata):
        return {"matches": self._matches[:top_k]}


class FakePinecone:
    class inference:
        @staticmethod
        def embed(model, inputs, parameters):
            return FakeEmbeddingResult([[0.1, 0.2, 0.3] for _ in inputs])


def _patch_pinecone(monkeypatch, fake_index):
    monkeypatch.setattr(hybrid_search_module, "_get_index", lambda: fake_index)
    monkeypatch.setattr(hybrid_search_module, "_pc", FakePinecone())


# --- embed_chunks ---------------------------------------------------------

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


# --- dense_search -----------------------------------------------------------

def test_dense_search_returns_correct_shape(monkeypatch):
    fake_index = FakeIndex(matches=[
        {"id": "a.txt::0", "score": 0.9, "metadata": {"text": "matched text", "source": "a.txt"}}
    ])
    _patch_pinecone(monkeypatch, fake_index)

    results = dense_search("some question", top_k=1)

    assert results == [
        {"chunk_id": "a.txt::0", "text": "matched text", "source": "a.txt", "score": 0.9}
    ]


# --- sparse_search ------------------------------------------------------

def test_sparse_search_ranks_by_keyword_overlap(monkeypatch):
    # 3 documents, not 2 - with only 2 docs where every word is unique to one
    # of them, BM25's IDF formula legitimately evaluates to exactly 0 for
    # every term (n/N = 0.5 is a real edge case), which isn't representative
    # of a normal corpus.
    chunks = [
        {"chunk_id": "a.txt::0", "text": "cats and dogs are common household pets", "source": "a.txt"},
        {"chunk_id": "b.txt::0", "text": "the stock market rose sharply today", "source": "b.txt"},
        {"chunk_id": "c.txt::0", "text": "quarterly earnings reports show growth", "source": "c.txt"},
    ]

    def fake_get_bm25():
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in chunks]
        return BM25Okapi(tokenized), chunks

    monkeypatch.setattr(hybrid_search_module, "_get_bm25", fake_get_bm25)

    results = sparse_search("cats and dogs", top_k=2)

    assert results[0]["chunk_id"] == "a.txt::0"
    assert results[0]["score"] > results[1]["score"]


# --- hybrid_search (RRF fusion) ------------------------------------------

def test_hybrid_search_fuses_dense_and_sparse(monkeypatch):
    dense_results = [
        {"chunk_id": "x::0", "text": "dense top match", "source": "x.txt", "score": 0.95},
        {"chunk_id": "shared::0", "text": "shared chunk", "source": "s.txt", "score": 0.80},
    ]
    sparse_results = [
        {"chunk_id": "shared::0", "text": "shared chunk", "source": "s.txt", "score": 5.0},
        {"chunk_id": "y::0", "text": "sparse top match", "source": "y.txt", "score": 3.0},
    ]

    monkeypatch.setattr(hybrid_search_module, "dense_search", lambda q, top_k: dense_results)
    monkeypatch.setattr(hybrid_search_module, "sparse_search", lambda q, top_k: sparse_results)

    results = hybrid_search("some question", top_k=3)

    # the chunk both rankers agree on should come out on top
    assert results[0]["source"] == "s.txt"
    sources = {r["source"] for r in results}
    assert sources == {"x.txt", "s.txt", "y.txt"}
