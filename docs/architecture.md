# Architecture Notes

## Flow
1. Documents in `data/` are loaded and chunked (`src/ingestion/loader.py`).
2. Chunks are embedded and stored in Pinecone (`src/retrieval/hybrid_search.py`).
3. A user question hits `POST /query` (`src/api/routes.py`).
4. Hybrid search retrieves the most relevant chunks.
5. The LLM generates a grounded answer from those chunks (`src/generation/generator.py`).

## Service model choices (IaaS / PaaS / SaaS)
| Component            | Service                  | Layer |
|-----------------------|---------------------------|-------|
| LLM                   | Anthropic API              | SaaS  |
| Vector DB + hybrid search | Pinecone                | SaaS  |
| Container registry     | AWS ECR                   | managed storage |
| Compute / hosting       | AWS App Runner            | PaaS  |
| CI/CD                  | GitHub Actions             | SaaS  |

See the "Project 1 Deep Dive" document for the full reasoning behind each
phase and why these choices were made.

## Status
This is a skeleton. Each module raises `NotImplementedError` with a TODO
pointing at the phase it belongs to. Build order: ingestion -> retrieval
(dense only) -> generation -> wire into API -> hybrid search -> Docker ->
deploy.
