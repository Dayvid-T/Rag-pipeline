"""
Tests for src.generation.generator.

Gemini is mocked entirely here - no real API key, no real network call,
no cost. Same reasoning as test_hybrid_search.py's Pinecone mocking.
"""

import src.generation.generator as generator_module
from src.generation.generator import build_prompt, generate_answer


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text):
        self._text = text
        self.last_call = None

    def generate_content(self, model, contents):
        self.last_call = {"model": model, "contents": contents}
        return FakeResponse(self._text)


class FakeClient:
    def __init__(self, text):
        self.models = FakeModels(text)


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
