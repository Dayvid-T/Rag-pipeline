"""
Tests for src.retrieval.hybrid_search.

Pinecone is mocked entirely here - no real API key, no real network call.
That means this suite runs identically on your machine and in CI.
"""

import pytest
from rank_bm25 import BM25Okapi

import src.retrieval.hybrid_search as hybrid_search_module
from src.retrieval.hybrid_search import (
    delete_source,
    dense_search,
    embed_chunks,
    get_source_text,
    hybrid_search,
    list_sources,
    sparse_search,
)


class FakeEmbeddingResult:
    def __init__(self, vectors):
        self.data = [{"values": v} for v in vectors]


class FakeVector:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeFetchResponse:
    def __init__(self, vectors):
        self.vectors = vectors


class FakeIndex:
    """Stores {id: metadata} so list/fetch/upsert/delete behave like Pinecone."""

    def __init__(self, matches=None, stored=None):
        self.upserted = []
        self.deleted = []
        self._matches = matches or []
        self._stored = dict(stored or {})

    def upsert(self, vectors):
        self.upserted.extend(vectors)
        for v in vectors:
            self._stored[v["id"]] = v["metadata"]

    def delete(self, ids):
        self.deleted.extend(ids)
        for vector_id in ids:
            self._stored.pop(vector_id, None)

    def list(self, prefix=None):
        ids = [i for i in self._stored if prefix is None or i.startswith(prefix)]
        if ids:
            yield ids

    def fetch(self, ids):
        return FakeFetchResponse({i: FakeVector(self._stored[i]) for i in ids})

    def query(self, vector, top_k, include_metadata):
        return {"matches": self._matches[:top_k]}


class FakePinecone:
    class inference:
        @staticmethod
        def embed(model, inputs, parameters):
            return FakeEmbeddingResult([[0.1, 0.2, 0.3] for _ in inputs])


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch):
    monkeypatch.setattr(hybrid_search_module, "_corpus", None)
    monkeypatch.setattr(hybrid_search_module, "_bm25", None)


def _patch_pinecone(monkeypatch, fake_index):
    monkeypatch.setattr(hybrid_search_module, "_get_index", lambda: fake_index)
    monkeypatch.setattr(hybrid_search_module, "_pc", FakePinecone())


# --- embed_chunks ---------------------------------------------------------

def test_embed_chunks_empty_list_does_nothing(monkeypatch):
    fake_index = FakeIndex()
    _patch_pinecone(monkeypatch, fake_index)

    embed_chunks([])

    assert fake_index.upserted == []


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


def test_embed_chunks_batches_large_inputs(monkeypatch):
    fake_index = FakeIndex()
    _patch_pinecone(monkeypatch, fake_index)
    monkeypatch.setattr(hybrid_search_module, "EMBED_BATCH", 2)

    chunks = [{"text": f"t{i}", "source": "a.txt", "chunk_id": f"a.txt::{i}"} for i in range(5)]
    embed_chunks(chunks)

    assert [v["id"] for v in fake_index.upserted] == [f"a.txt::{i}" for i in range(5)]


def test_embed_chunks_makes_new_chunks_searchable_immediately(monkeypatch):
    fake_index = FakeIndex()
    _patch_pinecone(monkeypatch, fake_index)

    embed_chunks([
        {"text": "cats and dogs are common household pets", "source": "pets.txt", "chunk_id": "pets.txt::0"},
        {"text": "the stock market rose sharply today", "source": "money.txt", "chunk_id": "money.txt::0"},
        {"text": "quarterly earnings reports show growth", "source": "money.txt", "chunk_id": "money.txt::1"},
    ])

    results = sparse_search("cats and dogs", top_k=1)

    assert results[0]["chunk_id"] == "pets.txt::0"
    assert list_sources() == [{"source": "money.txt", "chunks": 2}, {"source": "pets.txt", "chunks": 1}]


# --- corpus cache / list_sources / delete_source ---------------------------

STORED = {
    "a.txt::0": {"text": "alpha one", "source": "a.txt"},
    "a.txt::1": {"text": "alpha two", "source": "a.txt"},
    "b.pdf::0": {"text": "bravo", "source": "b.pdf"},
}


def test_list_sources_loads_corpus_from_pinecone(monkeypatch):
    _patch_pinecone(monkeypatch, FakeIndex(stored=STORED))

    assert list_sources() == [{"source": "a.txt", "chunks": 2}, {"source": "b.pdf", "chunks": 1}]


def test_list_sources_empty_index(monkeypatch):
    _patch_pinecone(monkeypatch, FakeIndex())

    assert list_sources() == []
    assert sparse_search("anything") == []


def test_get_source_text_joins_chunks_in_order(monkeypatch):
    stored = {
        "a.txt::10": {"text": "eleventh", "source": "a.txt"},
        "a.txt::0": {"text": "first", "source": "a.txt"},
        "a.txt::2": {"text": "third", "source": "a.txt"},
        "b.pdf::0": {"text": "other", "source": "b.pdf"},
    }
    _patch_pinecone(monkeypatch, FakeIndex(stored=stored))

    assert get_source_text("a.txt") == "first\nthird\neleventh"
    assert get_source_text("a.txt", max_chars=5) == "first"
    assert get_source_text("missing.txt") == ""


def test_delete_source_removes_vectors_and_cache(monkeypatch):
    fake_index = FakeIndex(stored=STORED)
    _patch_pinecone(monkeypatch, fake_index)

    deleted = delete_source("a.txt")

    assert deleted == 2
    assert sorted(fake_index.deleted) == ["a.txt::0", "a.txt::1"]
    assert list_sources() == [{"source": "b.pdf", "chunks": 1}]
    assert all(r["source"] != "a.txt" for r in sparse_search("alpha"))


def test_delete_source_unknown_returns_zero(monkeypatch):
    _patch_pinecone(monkeypatch, FakeIndex(stored=STORED))

    assert delete_source("nope.txt") == 0
    assert len(list_sources()) == 2


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
    monkeypatch.setattr(hybrid_search_module, "_get_corpus", lambda: chunks)
    monkeypatch.setattr(
        hybrid_search_module, "_bm25", BM25Okapi([c["text"].lower().split() for c in chunks])
    )

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
