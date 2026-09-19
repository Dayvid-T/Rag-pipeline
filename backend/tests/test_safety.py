"""
Tests for src.guardrails.safety.
"""

from google.genai import types

from src.guardrails.safety import extract_text, safety_config


class FakeCandidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, text=None, candidates=None):
        self.text = text
        self.candidates = candidates


def test_safety_config_covers_five_categories():
    config = safety_config()
    assert len(config.safety_settings) == 5
    categories = {s.category for s in config.safety_settings}
    assert types.HarmCategory.HARM_CATEGORY_JAILBREAK in categories
    assert all(
        s.threshold == types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        for s in config.safety_settings
    )


def test_extract_text_returns_text_on_normal_completion():
    response = FakeResponse("the answer", [FakeCandidate(types.FinishReason.STOP)])
    assert extract_text(response) == "the answer"


def test_extract_text_returns_none_when_no_candidates():
    response = FakeResponse("ignored", [])
    assert extract_text(response) is None


def test_extract_text_returns_none_on_safety_block():
    response = FakeResponse("ignored", [FakeCandidate(types.FinishReason.SAFETY)])
    assert extract_text(response) is None


def test_extract_text_returns_none_on_prohibited_content():
    response = FakeResponse("ignored", [FakeCandidate(types.FinishReason.PROHIBITED_CONTENT)])
    assert extract_text(response) is None


def test_extract_text_passes_through_max_tokens():
    # MAX_TOKENS means the output was truncated, not blocked - still real text.
    response = FakeResponse("partial answer", [FakeCandidate(types.FinishReason.MAX_TOKENS)])
    assert extract_text(response) == "partial answer"
