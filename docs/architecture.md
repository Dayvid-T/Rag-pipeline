# Architecture Notes

## Layout
- `backend/` — the RAG pipeline + FastAPI service.
- `frontend/` — the web UI that calls `POST /query`; built to static files
  and served by the same FastAPI app (one container, one URL).
- `evaluation/` — Project 2: black-box evaluation of the running service
  (hallucination rate, accuracy, retrieval hit rate, latency, attack block
  rate).
- `backend/src/guardrails/` — Project 3: prompt-injection detection, output
  safety filtering, and an audit log, wired into `POST /query`.
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
3. A user question hits `POST /query` (`backend/src/api/routes.py`). A
   high-severity prompt-injection match on the question blocks the
   request here, before retrieval or generation run (see Guardrails below).
4. Hybrid search runs dense (Pinecone) and sparse (BM25 over the same
   chunks) retrieval and fuses them with Reciprocal Rank Fusion. Any
   retrieved passage that matches an injection pattern is dropped before
   it reaches the prompt.
5. Gemini generates an answer grounded only in the remaining chunks
   (`backend/src/generation/generator.py`), routed through Gemini's own
   safety filtering. The response carries the answer, the cited source
   files, the retrieved passages, and whether a guardrail fired.

## References
`GET /documents/{source}/citation` (`backend/src/citation/citation.py`)
reassembles the document's text from its chunks, asks Gemini for
structured bibliographic metadata (JSON mode, temperature 0), and formats
IEEE and APA 7 strings with plain rule-based code so the output is
deterministic and unit-tested. Results are cached per document and
invalidated on re-upload or delete.

## Guardrails (Project 3)
Three checks sit at the `/query` boundary (`backend/src/guardrails/`):

- **`injection.py`** - a fast, free, regex-based scanner for prompt-injection
  phrasing ("ignore previous instructions", "reveal your system prompt",
  DAN-style jailbreaks, credential exfiltration, role-spoofing). It runs
  against two different inputs with different consequences: a high-severity
  match on the *question* blocks the request before retrieval or generation
  ever run; a match on a *retrieved passage* excludes just that chunk from
  the prompt, so one poisoned document can't deny an answer built from the
  rest of the corpus.
- **`safety.py`** - routes every generation call through Gemini's built-in
  safety filtering (harassment, hate speech, dangerous content, sexually
  explicit, jailbreak) at `BLOCK_MEDIUM_AND_ABOVE`, and turns a blocked
  output into a clear refusal instead of raising or returning empty text.
- **`audit.py`** - every block or filter is logged as one structured JSON
  line with the rule that fired. It owns its logger and attaches its own
  handler directly rather than depending on the app's root-logger setup -
  that turned out to matter: under `uvicorn --reload`, records reaching
  root's handler were being silently dropped in a way a standalone
  `uvicorn` process never showed, so an audit trail can't assume it'll
  inherit working config from somewhere else.

`POST /query` responses carry `blocked` and `guardrail_flags` so a caller -
including the evaluation suite - can tell a refusal from a real answer
without guessing from the prose.

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

Adversarial cases (`"attack": true` in `dataset.jsonl`) skip the judge -
there's no well-formed "correct answer" to a jailbreak attempt - and are
scored on whether the API's `blocked` field came back `true`, rolled up
into an `attack_block_rate` metric with its own `--min-attack-block-rate`
gate. This is what makes Project 3 measured rather than just claimed.
