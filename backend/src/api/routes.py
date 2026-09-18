"""
API layer: exposes the RAG pipeline over HTTP so it can be containerized
and deployed (Phase 4 & 5).

Why it exists: wrapping the pipeline in a small FastAPI app is what makes
it a *service* rather than a script - something that can be containerized,
deployed behind App Runner, and (in Project 2/3) evaluated and guarded at
its API boundary.
"""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.citation import citation
from src.ingestion.loader import chunk_documents, parse_document
from src.retrieval.hybrid_search import delete_source, embed_chunks, hybrid_search, list_sources
from src.generation.generator import generate_answer

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="RAG Pipeline QA Assistant", version="0.2.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []
    # The retrieved passages the answer was grounded in. Exposed so the
    # evaluation suite (Project 2) can judge faithfulness at the API boundary.
    contexts: list[str] = []


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentList(BaseModel):
    documents: list[DocumentInfo]


class Author(BaseModel):
    given: str
    family: str


class CitationMetadata(BaseModel):
    title: str
    authors: list[Author]
    year: int | None
    venue: str | None
    document_type: str
    url: str | None
    doi: str | None


class Citation(BaseModel):
    source: str
    metadata: CitationMetadata
    ieee: str
    apa: str


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

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        contexts=[chunk["text"] for chunk in chunks],
    )


@app.get("/documents", response_model=DocumentList)
def documents() -> DocumentList:
    try:
        return DocumentList(documents=list_sources())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Listing documents failed: {e}")


@app.post("/documents", response_model=DocumentInfo)
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    """Index an uploaded .pdf/.txt; re-uploading the same name replaces it."""
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="Missing filename")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB upload limit")

    try:
        document = parse_document(name, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_documents([document])
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No text could be extracted from '{name}'")

    try:
        delete_source(name)
        embed_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    citation.forget(name)
    return DocumentInfo(source=name, chunks=len(chunks))


@app.delete("/documents/{source}")
def remove_document(source: str) -> dict:
    try:
        deleted = delete_source(source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deleting document failed: {e}")
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document named '{source}' is indexed")
    citation.forget(source)
    return {"source": source, "chunks_deleted": deleted}


@app.get("/documents/{source}/citation", response_model=Citation)
def cite_document(source: str) -> Citation:
    """Auto-detected reference for an indexed document in IEEE and APA style."""
    try:
        result = citation.cite(source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Citation failed: {e}")
    if result is None:
        raise HTTPException(status_code=404, detail=f"No document named '{source}' is indexed")
    return Citation(**result)


# Mounted last so it never shadows the API routes above. Skipped when there's
# no built frontend (tests, backend-only dev) so the API still boots.
_static_dir = Path(settings.static_dir)
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
