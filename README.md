# RAG Pipeline QA Assistant

A containerized Retrieval-Augmented Generation (RAG) pipeline that answers
questions over a set of documents, using hybrid (semantic + keyword) search
and deployed on AWS through managed PaaS/SaaS services.

This repo holds all three projects of a series building toward AI
Security & Governance: **Project 1**, the working system; **Project 2**,
an automated evaluation suite measuring hallucination rate and latency;
and **Project 3**, prompt-injection guardrails and output safety filtering
at the API boundary, measured by the same evaluation suite.

## Repo layout

```
backend/
  src/
    config.py                   # env/config loading
    ingestion/loader.py         # load (PyMuPDF) + chunk documents
    retrieval/hybrid_search.py  # Pinecone dense + BM25 sparse, RRF fusion
    generation/generator.py     # grounded answer generation (Gemini)
    citation/citation.py        # IEEE / APA reference generation per document
    guardrails/                 # Project 3: injection detection, output safety, audit log
    api/routes.py               # FastAPI app: /query, /documents, /health, static UI
  data/                         # local documents for bulk indexing (gitignored)
  tests/
frontend/                       # Vite + TypeScript UI, built into the image
evaluation/                     # Project 2+3: eval suite (see evaluation/README.md)
scripts/deploy.sh               # ECR push + App Runner create/update
Dockerfile                      # multi-stage: build frontend, run API
docs/architecture.md            # flow + service-model choices
docs/deployment.md              # AWS setup and deploy steps
.github/workflows/              # tests.yml (all suites), deploy.yml
```

## Local setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # fill in your API keys
uvicorn src.api.routes:app --reload
```

Then check `GET http://localhost:8000/health`. For the UI during
development run `npm install && npm run dev` in `frontend/` (it proxies to
the backend); for the production layout run `npm run build` and the
backend serves `frontend/dist` at `/`.

Add documents either by uploading them in the UI (`POST /documents`,
`.pdf` or `.txt`), or by bulk-indexing everything in `backend/data/`:

```bash
python -c "from src.ingestion.loader import *; from src.retrieval.hybrid_search import embed_chunks; embed_chunks(chunk_documents(load_documents('data')))"
```

Pinecone holds the only copy of the corpus, so uploads persist across
restarts and redeploys.

Each document also has a **Cite** button (`GET /documents/{source}/citation`)
that detects title/authors/year with Gemini and formats a reference in
IEEE and APA 7 style to copy.

`/query` runs through guardrails before and after generation: a
prompt-injection attempt in the question is blocked outright, a match in a
retrieved passage is dropped from that answer's context, and Gemini's own
safety filtering can still block the output. Every block is written to a
structured audit log. See [docs/architecture.md](docs/architecture.md#guardrails-project-3).

## Tests

```bash
cd backend && pytest
cd frontend && npm test
cd evaluation && pytest
```

All three suites mock external services; CI runs them on every push.

## Docker

```bash
docker build -t rag-pipeline-qa .
docker run --rm -p 8000:8000 --env-file backend/.env rag-pipeline-qa
```

## Deploy

`bash scripts/deploy.sh` (or the *Deploy* GitHub Actions workflow) pushes
the image to ECR and creates/updates the App Runner service. Prerequisites
and secrets are in [docs/deployment.md](docs/deployment.md).

## Evaluate

```bash
cd evaluation && pip install -r requirements.txt
python run_eval.py --base-url http://localhost:8000
```

Reports hallucination rate, accuracy, retrieval hit rate, attack block
rate and latency percentiles; see [evaluation/README.md](evaluation/README.md).

## License

MIT
