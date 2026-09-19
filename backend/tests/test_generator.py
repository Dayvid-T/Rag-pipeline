"""
Tests for src.generation.generator.

Gemini is mocked entirely here - no real API key, no real network call,
no cost. Same reasoning as test_hybrid_search.py's Pinecone mocking.
"""

import pytest
from google.genai import errors, types

import src.generation.generator as generator_module
from src.generation.generator import build_prompt, generate_answer


class FakeCandidate:
    def __init__(self, finish_reason=types.FinishReason.STOP):
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, text, finish_reason=types.FinishReason.STOP):
        self.text = text
        self.candidates = [FakeCandidate(finish_reason)]


class FakeModels:
    def __init__(self, text, fail_with=None, finish_reason=types.FinishReason.STOP):
        self._text = text
        self._finish_reason = finish_reason
        self._fail_with = list(fail_with or [])
        self.last_call = None
        self.calls = 0

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        self.last_call = {"model": model, "contents": contents, "config": config}
        if self._fail_with:
            raise self._fail_with.pop(0)
        return FakeResponse(self._text, self._finish_reason)


class FakeClient:
    def __init__(self, text, fail_with=None, finish_reason=types.FinishReason.STOP):
        self.models = FakeModels(text, fail_with, finish_reason)


def _api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "busy", "status": "x"}})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(generator_module.time, "sleep", lambda s: None)


def test_build_prompt_includes_context_and_question():
    chunks = [{"text": "SVMs separate classes with a hyperplane.", "source": "notes.txt"}]

    prompt = build_prompt("What is an SVM?", chunks)

    assert "SVMs separate classes with a hyperplane." in prompt
    assert "notes.txt" in prompt
    assert "What is an SVM?" in prompt


def test_build_prompt_handles_no_chunks():
    prompt = build_prompt("What is an SVM?", [])
    assert "What is an SVM?" in prompt


def test_generate_answer_returns_correct_shape(monkeypatch):
    fake_client = FakeClient("this is the answer")
    monkeypatch.setattr(generator_module, "_get_client", lambda: fake_client)

    chunks = [
        {"text": "chunk one", "source": "b.txt"},
        {"text": "chunk two", "source": "a.txt"},
        {"text": "chunk three", "source": "a.txt"},  # duplicate source on purpose
    ]
    result = generate_answer("some question", chunks)

    assert result == {"answer": "this is the answer", "sources": ["a.txt", "b.txt"]}
    assert fake_client.models.last_call["model"] == generator_module.GENERATION_MODEL
    assert "some question" in fake_client.models.last_call["contents"]


def test_generate_answer_passes_safety_settings(monkeypatch):
    fake_client = FakeClient("this is the answer")
    monkeypatch.setattr(generator_module, "_get_client", lambda: fake_client)

    generate_answer("some question", [])

    config = fake_client.models.last_call["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert len(config.safety_settings) == 5


def test_generate_answer_returns_none_when_output_blocked(monkeypatch):
    fake_client = FakeClient("ignored", finish_reason=types.FinishReason.SAFETY)
    monkeypatch.setattr(generator_module, "_get_client", lambda: fake_client)

    result = generate_answer("some question", [{"text": "x", "source": "a.txt"}])

    assert result["answer"] is None
    assert result["sources"] == ["a.txt"]


def test_generate_answer_retries_transient_errors(monkeypatch):
    fake_client = FakeClient("eventually", fail_with=[_api_error(503), _api_error(429)])
    monkeypatch.setattr(generator_module, "_get_client", lambda: fake_client)

    result = generate_answer("q", [])

    assert result["answer"] == "eventually"
    assert fake_client.models.calls == 3


def test_generate_answer_gives_up_after_max_attempts(monkeypatch):
    fake_client = FakeClient("never", fail_with=[_api_error(503)] * 3)
    monkeypatch.setattr(generator_module, "_get_client", lambda: fake_client)

    with pytest.raises(errors.APIError):
        generate_answer("q", [])
    assert fake_client.models.calls == 3


def test_generate_answer_does_not_retry_other_errors(monkeypatch):
    fake_client = FakeClient("no", fail_with=[_api_error(400)])
    monkeypatch.setattr(generator_module, "_get_client", lambda: fake_client)

    with pytest.raises(errors.APIError):
        generate_answer("q", [])
    assert fake_client.models.calls == 1
