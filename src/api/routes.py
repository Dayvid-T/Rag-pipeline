"""
API layer: exposes the RAG pipeline over HTTP so it can be containerized
and deployed (Phase 4 & 5).

Why it exists: wrapping the pipeline in a small FastAPI app is what makes
it a *service* rather than a script - something that can be containerized,
deployed behind App Runner, and (in Project 2/3) evaluated and guarded at
its API boundary.
"""

from fastapi import FastAPI
from pydantic import BaseModel

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
    raise NotImplementedError("Phase 2: wire retrieval + generation into this endpoint")
