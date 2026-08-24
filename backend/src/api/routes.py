"""
API layer: exposes the RAG pipeline over HTTP so it can be containerized
and deployed (Phase 4 & 5).

Why it exists: wrapping the pipeline in a small FastAPI app is what makes
it a *service* rather than a script - something that can be containerized,
deployed behind App Runner, and (in Project 2/3) evaluated and guarded at
its API boundary.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

from src.config import settings

app = FastAPI(title="RAG Pipeline QA Assistant", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []


@app.get("/health")
def health() -> dict:
    """Basic liveness check - useful once this is deployed on App Runner."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Main endpoint: takes a question, retrieves relevant chunks, and returns
    a grounded answer.

    TODO (Phase 2/3): wire this up to
      src.retrieval.hybrid_search.hybrid_search()
      src.generation.generator.generate_answer()
    once those are implemented.
    """
    # Attempt to import retrieval + generation at runtime; if they aren't
    # available in early phases that's fine — surface a clear HTTP 501.
    try:
        from src.retrieval.hybrid_search import hybrid_search  # type: ignore
        from src.generation.generator import generate_answer  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=501, detail=f"Retrieval/generation not available: {e}")

    # Call retrieval. Be defensive about call signatures.
    try:
        results = hybrid_search(request.question, top_k=5)  # preferred signature
    except TypeError:
        # fallback to older signature if implemented differently
        results = hybrid_search(request.question, 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # Normalize retrieval output -> list of chunk texts + source metadata list
    chunks = []
    raw_sources: list[Any] = []
    if isinstance(results, dict):
        chunks = results.get("chunks") or results.get("documents") or []
        raw_sources = results.get("sources") or []
    elif isinstance(results, (list, tuple)):
        if results and isinstance(results[0], dict):
            chunks = [r.get("chunk") or r.get("text") or r for r in results]
            raw_sources = [{"doc_id": r.get("doc_id"), "score": r.get("score")} for r in results]
        else:
            chunks = list(results)
    else:
        chunks = []

    # Call generator
    try:
        gen_out = generate_answer(request.question, chunks)
    except TypeError:
        gen_out = generate_answer(request.question, chunks, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # gen_out may be string, dict, or tuple (answer, sources)
    answer = ""
    gen_sources: list[Any] = []
    if isinstance(gen_out, tuple) and len(gen_out) >= 1:
        answer = gen_out[0] or ""
        if len(gen_out) > 1:
            gen_sources = gen_out[1] or []
    elif isinstance(gen_out, dict):
        answer = gen_out.get("answer") or gen_out.get("text") or ""
        gen_sources = gen_out.get("sources") or []
    else:
        answer = str(gen_out or "")

    chosen_sources = gen_sources or raw_sources

    # Normalize sources to list[str] (QueryResponse expects list[str])
    final_sources: list[str] = []
    for s in chosen_sources:
        if isinstance(s, dict):
            if s.get("doc_id") is not None:
                final_sources.append(str(s.get("doc_id")))
            elif s.get("id") is not None:
                final_sources.append(str(s.get("id")))
            elif s.get("text"):
                txt = s.get("text")
                final_sources.append(txt if isinstance(txt, str) else str(txt))
            else:
                final_sources.append(str(s))
        else:
            final_sources.append(str(s))

    return QueryResponse(answer=answer, sources=final_sources)
