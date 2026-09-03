"""
Generation layer: turn a user question + retrieved context into a grounded
answer using an LLM.

Phase: 2 (Build Core Retrieval Logic, Locally)
Why it exists: this is the "G" in RAG - it takes what retrieval found and
asks the model to answer using only that material, so answers stay
grounded in your actual documents instead of the model's memorized (and
possibly wrong or outdated) knowledge.

TODO (build-out steps):
  1. build_prompt(query, retrieved_chunks): assemble a prompt that clearly
     separates instructions, retrieved context, and the user's question.
  2. generate_answer(query, retrieved_chunks): call the LLM API and return
     the answer, ideally with the source chunks it used cited alongside it.
"""

from typing import List, Dict

from google import genai

from src.config import settings

GENERATION_MODEL = "gemini-3.6-flash"

# Lazily-created connection, shared across calls in this process - same
# reasoning as hybrid_search.py's _get_index(): importing this module should
# never make a network call on its own.
_client = None


def _get_client():
    """Return a connected Gemini client, connecting on first use."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    """Assemble the grounded prompt sent to the LLM."""
    context = "\n\n".join(
        f"[Source: {chunk['source']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    )
    return (
        "You are a helpful assistant that answers questions using ONLY the "
        "context provided below. If the answer isn't contained in the "
        "context, say you don't know - do not make anything up.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> Dict:
    """Call the LLM and return {'answer': ..., 'sources': [...]}."""
    prompt = build_prompt(query, retrieved_chunks)

    client = _get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    sources = sorted({chunk["source"] for chunk in retrieved_chunks})
    return {"answer": response.text, "sources": sources}
