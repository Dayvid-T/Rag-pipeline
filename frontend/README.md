# Frontend

Minimal web UI for `POST /query`: a question box, the grounded answer, its
cited sources, and (collapsed by default) the retrieved passages. Vanilla
TypeScript + Vite, no framework - the whole thing is one form.

## Develop

Start the backend first (`cd backend && uvicorn src.api.routes:app --reload`),
then:

```bash
npm install
npm run dev
```

Vite proxies `/query` and `/health` to `http://localhost:8000`, so the
browser stays same-origin and no CORS config is needed.

## Test / build

```bash
npm test
npm run build
```

`npm run build` writes `dist/`. The FastAPI app serves that folder at `/`
when it exists (`STATIC_DIR`, default `../frontend/dist`), and the root
`Dockerfile` builds it in a first stage and bakes it into the image, so
production is a single container with one URL.
