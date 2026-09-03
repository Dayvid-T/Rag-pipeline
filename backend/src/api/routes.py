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

from src.config import settings
from src.retrieval.hybrid_search import hybrid_search
from src.generation.generator import generate_answer

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
    """
    try:
        chunks = hybrid_search(request.question, top_k=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    try:
        result = generate_answer(request.question, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return QueryResponse(answer=result["answer"], sources=result["sources"])
