from typing import List, Dict, Optional

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

from src.config import settings

# Pinecone's hosted embedding model. Outputs 1024-dimension vectors - the
# index was created with dimension=1024 specifically to match this.
EMBEDDING_MODEL = "multilingual-e5-large"

# Pinecone's inference API caps a single embed call at 96 inputs for this
# model; upserts are batched the same way.
EMBED_BATCH = 96

# How many results RRF pulls from each ranker before fusing - wider than the
# final top_k so a chunk that's #1 in one ranker but absent from the other's
# top_k still gets a fair shot at the merged result.
FETCH_MULTIPLIER = 3

# Standard constant for Reciprocal Rank Fusion (RRF). Not sensitive to tuning
# for a project this size - 60 is the commonly-cited default in the RRF
# literature.
RRF_K = 60

# Lazily-created connection, shared across calls in this process. Created on
# first use (not at import time) so importing this module never makes a
# network call on its own.
_pc = None
_index = None

# In-process copy of the corpus (chunk text + source), loaded from Pinecone
# on first use. embed_chunks/delete_source keep it in sync so the BM25 index
# reflects an upload immediately, even though Pinecone's own reads are only
# eventually consistent.
_corpus: Optional[List[Dict]] = None
_bm25: Optional[BM25Okapi] = None


def _get_index():
    """Return a connected Pinecone Index handle, connecting on first use."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        index_info = _pc.describe_index(settings.pinecone_index_name)
        _index = _pc.Index(host=index_info.host)
    return _index


def _fetch_corpus() -> List[Dict]:
    """Pull every stored chunk's text and source out of Pinecone."""
    index = _get_index()
    chunks = []
    for id_batch in index.list():
        fetched = index.fetch(ids=list(id_batch))
        for vector_id, vector in fetched.vectors.items():
            chunks.append({
                "chunk_id": vector_id,
                "text": vector.metadata["text"],
                "source": vector.metadata["source"],
            })
    return chunks


def _rebuild_bm25() -> None:
    global _bm25
    if _corpus:
        _bm25 = BM25Okapi([chunk["text"].lower().split() for chunk in _corpus])
    else:
        _bm25 = None


def _get_corpus() -> List[Dict]:
    global _corpus
    if _corpus is None:
        _corpus = _fetch_corpus()
        _rebuild_bm25()
    return _corpus


def _batches(items: List, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def embed_chunks(chunks: List[Dict]) -> None:
    """Embed each chunk's text and upsert it into the Pinecone index."""
    global _corpus
    if not chunks:
        return

    index = _get_index()
    corpus = _get_corpus()

    for batch in _batches(chunks, EMBED_BATCH):
        result = _pc.inference.embed(
            model=EMBEDDING_MODEL,
            inputs=[chunk["text"] for chunk in batch],
            parameters={"input_type": "passage", "truncate": "END"},
        )
        index.upsert(vectors=[
            {
                "id": chunk["chunk_id"],
                "values": embedding["values"],
                "metadata": {"text": chunk["text"], "source": chunk["source"]},
            }
            for chunk, embedding in zip(batch, result.data)
        ])

    by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    for chunk in chunks:
        by_id[chunk["chunk_id"]] = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "source": chunk["source"],
        }
    _corpus = list(by_id.values())
    _rebuild_bm25()


def delete_source(source: str) -> int:
    """Remove every chunk of `source` from the index; returns how many were removed."""
    global _corpus
    index = _get_index()
    corpus = _get_corpus()

    ids = [vector_id for id_batch in index.list(prefix=f"{source}::") for vector_id in id_batch]
    if ids:
        index.delete(ids=ids)

    remaining = [chunk for chunk in corpus if chunk["source"] != source]
    removed_from_cache = len(corpus) - len(remaining)
    _corpus = remaining
    _rebuild_bm25()
    return max(len(ids), removed_from_cache)


def get_source_text(source: str, max_chars: Optional[int] = None) -> str:
    """Reassemble a document's text from its chunks, in order; '' if unknown."""
    chunks = [c for c in _get_corpus() if c["source"] == source]
    chunks.sort(key=lambda c: int(c["chunk_id"].rsplit("::", 1)[1]))
    text = "\n".join(c["text"] for c in chunks)
    return text[:max_chars] if max_chars else text


def list_sources() -> List[Dict]:
    """Return [{'source', 'chunks'}] for every indexed document, sorted by name."""
    counts: Dict[str, int] = {}
    for chunk in _get_corpus():
        counts[chunk["source"]] = counts.get(chunk["source"], 0) + 1
    return [{"source": source, "chunks": counts[source]} for source in sorted(counts)]


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
    """Keyword search: BM25 over the corpus cached from Pinecone."""
    corpus = _get_corpus()
    if _bm25 is None:
        return []

    scores = _bm25.get_scores(query.lower().split())
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)

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
