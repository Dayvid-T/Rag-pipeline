import time
from typing import List, Dict

from google import genai
from google.genai import errors

from src.config import settings
from src.guardrails.safety import extract_text, safety_config

GENERATION_MODEL = "gemini-3.1-flash-lite"

# Gemini's free tier intermittently returns 503 "high demand" / 429 quota
# errors that clear within seconds; retrying with backoff turns most of
# them into successful answers instead of 500s to the caller.
RETRYABLE_CODES = {429, 503}
MAX_ATTEMPTS = 3

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


def generate_with_retry(client, prompt: str, config=None):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return client.models.generate_content(
                model=GENERATION_MODEL, contents=prompt, config=config
            )
        except errors.APIError as e:
            if e.code not in RETRYABLE_CODES or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt)


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Call the LLM and return {'answer': ..., 'sources': [...]}.

    `answer` is None when Gemini's own safety filtering blocked the output
    (see guardrails/safety.py) - the caller decides what to show for that.
    """
    prompt = build_prompt(query, retrieved_chunks)
    response = generate_with_retry(_get_client(), prompt, config=safety_config())

    sources = sorted({chunk["source"] for chunk in retrieved_chunks})
    return {"answer": extract_text(response), "sources": sources}
