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


def embed_chunks(chunks: List[Dict]) -> None:
    """Embed and upsert chunks into the vector index. Not yet implemented."""
    raise NotImplementedError("Phase 2: implement embedding + upsert")


def hybrid_search(query: str, top_k: int = 5) -> List[Dict]:
    """Return the top_k most relevant chunks for `query`. Not yet implemented."""
    raise NotImplementedError("Phase 3: implement hybrid search")
