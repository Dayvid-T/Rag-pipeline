"""
Retrieval layer: embed chunks, store them, and retrieve the most relevant
ones for a query using hybrid (dense + sparse) search.

Phase: 2 (local retrieval) and Phase 3 (hybrid search)
Why it exists: this is the core of the RAG system. Pure semantic (dense
vector) search alone misses exact terms, codes, and names; combining it
with sparse keyword search is the standard fix. See Project 1 Deep Dive,
Phase 3, for the reasoning.

TODO (build-out steps):
  1. embed_chunks(chunks): call the embedding model and upsert vectors +
     metadata into the vector index (Pinecone).
  2. dense_search(query, top_k): semantic similarity search.
  3. sparse_search(query, top_k): keyword/BM25-style search.
  4. hybrid_search(query, top_k): combine and re-rank results from both.
"""

from typing import List, Dict

from pinecone import Pinecone

from src.config import settings

# Pinecone's hosted embedding model. Outputs 1024-dimension vectors - the
# index was created with dimension=1024 specifically to match this.
EMBEDDING_MODEL = "multilingual-e5-large"

# Lazily-created connection, shared across calls in this process. Created on
# first use (not at import time) so importing this module never makes a
# network call on its own - matters for testing and for anything that
# imports this file without actually needing Pinecone yet.
_pc = None
_index = None


def _get_index():
    """Return a connected Pinecone Index handle, connecting on first use."""
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
        index_info = _pc.describe_index(settings.pinecone_index_name)
        _index = _pc.Index(host=index_info.host)
    return _index


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


def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Return the top_k most relevant chunks for `query`.

    Dense (semantic) search only for now - sparse/keyword search gets added
    on top of this later (see the module TODO above); this is the Phase 2
    "dense search first" milestone from the README build order.
    """
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
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "score": match["score"],
        }
        for match in matches["matches"]
    ]
