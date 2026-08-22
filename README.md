# RAG Pipeline QA Assistant

A containerized Retrieval-Augmented Generation (RAG) pipeline that answers
questions over a set of documents, using hybrid (semantic + keyword) search
and deployed on AWS through managed PaaS/SaaS services.

This is **Project 1** of a three-project series building toward AI
Security & Governance: this project builds the working system, Project 2
adds an automated evaluation suite (hallucination rate, latency), and
Project 3 adds prompt-injection guardrails and bias filtering at the API
boundary.

## Status: skeleton

This repo currently defines the structure and interfaces; core logic is
stubbed with `NotImplementedError` and TODO comments pointing at the build
phase each piece belongs to. See `docs/architecture.md` for the full flow
and the service-model (IaaS/PaaS/SaaS) choices behind it.

## Project layout

```
src/
  config.py            # env/config loading
  ingestion/loader.py  # load + chunk documents            (Phase 2)
  retrieval/hybrid_search.py  # embed + hybrid search       (Phase 2/3)
  generation/generator.py     # grounded answer generation  (Phase 2)
  api/routes.py         # FastAPI app, /query and /health   (Phase 2/4)
data/                    # local documents for testing (gitignored)
tests/                   # smoke test now, real eval suite in Project 2
Dockerfile               # containerization                (Phase 4)
docs/architecture.md     # flow + service-model choices
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # fill in your API keys
uvicorn src.api.routes:app --reload
```

Then check `GET http://localhost:8000/health`.

## Build order

1. Implement `src/ingestion/loader.py` - load and chunk documents from `data/`.
2. Implement `src/retrieval/hybrid_search.py` (dense search first).
3. Implement `src/generation/generator.py` - wire retrieval into a grounded prompt.
4. Wire both into the `/query` endpoint in `src/api/routes.py`.
5. Add sparse/keyword search to `hybrid_search.py` for true hybrid retrieval.
6. Build and run the Docker image locally.
7. Deploy: push image to AWS ECR, run it on AWS App Runner.

## License

MIT
