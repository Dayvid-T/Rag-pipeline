import time
from typing import Dict, Optional

import httpx


class RagClient:
    """Thin HTTP client for the RAG service; treats it as a black box."""

    def __init__(self, base_url: str, timeout: float = 90.0, transport: Optional[httpx.BaseTransport] = None):
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def is_healthy(self) -> bool:
        try:
            response = self._http.get("/health")
        except httpx.HTTPError:
            return False
        return response.status_code == 200 and response.json().get("status") == "ok"

    def query(self, question: str) -> Dict:
        """POST /query and return the response body plus wall-clock latency in ms."""
        started = time.perf_counter()
        response = self._http.post("/query", json={"question": question})
        latency_ms = (time.perf_counter() - started) * 1000

        response.raise_for_status()
        body = response.json()
        return {
            "answer": body["answer"],
            "sources": body.get("sources", []),
            "contexts": body.get("contexts", []),
            "latency_ms": latency_ms,
        }
