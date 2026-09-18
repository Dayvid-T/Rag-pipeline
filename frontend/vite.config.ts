import { defineConfig } from "vite";

// In development the backend runs separately on :8000; the proxy keeps the
// browser same-origin so no CORS config is needed in either environment.
export default defineConfig({
  server: {
    proxy: {
      "/query": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/documents": "http://localhost:8000",
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
