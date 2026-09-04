from typing import List, Dict

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.loader import load_documents, chunk_documents

# Pinecone's hosted embedding model. Outputs 1024-dimension vectors - the
# index was created with dimension=1024 specifically to match this.
EMBEDDING_MODEL = "multilingual-e5-large"

# Where sparse_search rebuilds its local keyword index from. Must be the
# same folder embed_chunks was last pointed at, so dense and sparse search
# are searching the same corpus.
DATA_DIR = "data"

# How many results RRF pulls from each ranker before fusing - wider than the
# final top_k so a chunk that's #1 in one ranker but absent from the other's
# top_k still gets a fair shot at the merged result.
FETCH_MULTIPLIER = 3

# Standard constant for Reciprocal Rank Fusion (RRF). Not sensitive to tuning
# for a project this size - 60 is the commonly-cited default in the RRF
# literature.
RRF_K = 60

# Lazily-created connections/indexes, shared across calls in this process.
# Created on first use (not at import time) so importing this module never
# makes a network call or reads the local data/ folder on its own.
_pc = None
_index = None
_bm25 = None
_bm25_chunks = None


def _get_index():
    """Return a connected Pinecone Index handle, connecting on first use."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        index_info = _pc.describe_index(settings.pinecone_index_name)
        _index = _pc.Index(host=index_info.host)
    return _index


def _get_bm25():
    """
    Build (once, lazily) a local BM25 index over the same chunks
    embed_chunks was last pointed at, for sparse_search to query.
    """
    global _bm25, _bm25_chunks
    if _bm25 is None:
        _bm25_chunks = chunk_documents(load_documents(DATA_DIR))
        tokenized = [chunk["text"].lower().split() for chunk in _bm25_chunks]
        _bm25 = BM25Okapi(tokenized)
    return _bm25, _bm25_chunks


def embed_chunks(chunks: List[Dict]) -> None:
    """Embed each chunk's text and upsert it into the Pinecone index."""
    if not chunks:
        return

    index = _get_index()

    texts = [chunk["text"] for chunk in chunks]
    result = _pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"},
    )

    vectors = [
        {
            "id": chunk["chunk_id"],
            "values": embedding["values"],
            "metadata": {"text": chunk["text"], "source": chunk["source"]},
        }
        for chunk, embedding in zip(chunks, result.data)
    ]
    index.upsert(vectors=vectors)


def dense_search(query: str, top_k: int = 5) -> List[Dict]:
    """Semantic similarity search: embed `query`, search Pinecone."""
    index = _get_index()

    result = _pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=[query],
        parameters={"input_type": "query"},
    )
    query_vector = result.data[0]["values"]

    matches = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    return [
        {
            "chunk_id": match["id"],
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "score": match["score"],
        }
        for match in matches["matches"]
    ]


def sparse_search(query: str, top_k: int = 5) -> List[Dict]:
    """Keyword search: BM25 over the locally-chunked documents."""
    bm25, chunks = _get_bm25()

    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
            "score": float(score),
        }
        for chunk, score in ranked[:top_k]
    ]


def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Return the top_k most relevant chunks for `query`, combining dense
    (semantic) and sparse (keyword) search via Reciprocal Rank Fusion (RRF).

    RRF combines two ranked lists using each result's *position* in its own
    list rather than its raw score - avoids having to normalize dense's
    cosine similarity (0-1) against BM25's unbounded scores, which aren't
    on comparable scales.
    """
    fetch_k = top_k * FETCH_MULTIPLIER
    dense_results = dense_search(query, top_k=fetch_k)
    sparse_results = sparse_search(query, top_k=fetch_k)

    fused_scores: Dict[str, float] = {}
    chunks_by_id: Dict[str, Dict] = {}

    for ranked_list in (dense_results, sparse_results):
        for rank, chunk in enumerate(ranked_list):
            chunk_id = chunk["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            chunks_by_id[chunk_id] = chunk

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)

    return [
        {
            "text": chunks_by_id[cid]["text"],
            "source": chunks_by_id[cid]["source"],
            "score": fused_scores[cid],
        }
        for cid in ranked_ids[:top_k]
    ]
