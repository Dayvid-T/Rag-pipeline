# Architecture Notes

## Layout
- `backend/` — the RAG pipeline + FastAPI service.
- `frontend/` — the web UI that calls `POST /query`; built to static files
  and served by the same FastAPI app (one container, one URL).
- `evaluation/` — Project 2: black-box evaluation of the running service
  (hallucination rate, accuracy, retrieval hit rate, latency).
- `scripts/deploy.sh` + `.github/workflows/deploy.yml` — ECR + App Runner
  deploy. See `deployment.md`.

## Flow
1. Documents arrive either as local files in `backend/data/` (bulk index
   from a laptop) or as uploads to `POST /documents` from the UI. Either
   way they're parsed (PyMuPDF for PDFs) and chunked
   (`backend/src/ingestion/loader.py`).
2. Chunks are embedded with Pinecone's hosted `multilingual-e5-large` and
   upserted into a Pinecone index (`backend/src/retrieval/hybrid_search.py`).
   Pinecone is the only copy of the corpus; the in-process BM25 index is
   rebuilt from Pinecone's stored chunk text, so uploads persist across
   container restarts.
3. A user question hits `POST /query` (`backend/src/api/routes.py`).
4. Hybrid search runs dense (Pinecone) and sparse (BM25 over the same
   chunks) retrieval and fuses them with Reciprocal Rank Fusion.
5. Gemini generates an answer grounded only in those chunks
   (`backend/src/generation/generator.py`). The response carries the
   answer, the cited source files, and the retrieved passages.

## References
`GET /documents/{source}/citation` (`backend/src/citation/citation.py`)
reassembles the document's text from its chunks, asks Gemini for
structured bibliographic metadata (JSON mode, temperature 0), and formats
IEEE and APA 7 strings with plain rule-based code so the output is
deterministic and unit-tested. Results are cached per document and
invalidated on re-upload or delete.

## Service model choices (IaaS / PaaS / SaaS)
| Component                 | Service            | Layer           |
|---------------------------|--------------------|-----------------|
| LLM                       | Google Gemini API  | SaaS            |
| Embeddings + vector DB    | Pinecone           | SaaS            |
| Container registry        | AWS ECR            | managed storage |
| Compute / hosting         | AWS App Runner     | PaaS            |
| CI/CD                     | GitHub Actions     | SaaS            |

## CI
`.github/workflows/tests.yml` runs the backend, frontend and evaluation
unit suites on every push and PR; all three mock their external services.
`.github/workflows/deploy.yml` builds and deploys on manual dispatch (or on
push to `main` once `DEPLOY_ON_PUSH` is enabled).

## Evaluation (Project 2)
`evaluation/run_eval.py` sends a golden dataset to `/query`, times each
request, and asks a Gemini judge whether each answer is grounded in the
returned passages and matches the reference. Because it only uses the
public API, the same run works against a local container or the deployed
App Runner URL, and it exits non-zero when thresholds are breached.
