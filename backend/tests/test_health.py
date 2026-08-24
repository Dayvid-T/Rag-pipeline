"""
Smoke test: confirms the API app boots and the health endpoint responds.
This is deliberately the only test in the skeleton - Project 2 adds the
real evaluation suite (hallucination rate, latency) on top of this.
"""

from fastapi.testclient import TestClient
from src.api.routes import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
