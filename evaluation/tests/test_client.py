import httpx
import pytest

from evalsuite.client import RagClient


def _client(handler):
    return RagClient("http://rag.test", transport=httpx.MockTransport(handler))


def test_query_returns_body_with_latency():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["json"] = request.read()
        return httpx.Response(200, json={"answer": "42", "sources": ["a.txt"], "contexts": ["ctx"]})

    result = _client(handler).query("meaning?")

    assert seen["path"] == "/query"
    assert b'"question":"meaning?"' in seen["json"].replace(b" ", b"")
    assert result["answer"] == "42"
    assert result["sources"] == ["a.txt"]
    assert result["contexts"] == ["ctx"]
    assert result["latency_ms"] >= 0


def test_query_raises_on_server_error():
    def handler(request):
        return httpx.Response(500, json={"detail": "Retrieval failed"})

    with pytest.raises(httpx.HTTPStatusError):
        _client(handler).query("q")


def test_is_healthy():
    def ok(request):
        return httpx.Response(200, json={"status": "ok"})

    def down(request):
        raise httpx.ConnectError("refused")

    assert _client(ok).is_healthy() is True
    assert _client(down).is_healthy() is False
