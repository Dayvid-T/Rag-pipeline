import pytest
from google.genai import errors

import evalsuite.judge as judge_module
from evalsuite.judge import GeminiJudge, Verdict, build_judge_prompt, parse_verdict


def test_build_prompt_includes_all_parts():
    prompt = build_judge_prompt(
        question="When is it due?",
        answer="March 18.",
        contexts=["Due March 18, 2026"],
        reference_answer="March 18, 2026",
        expect_abstain=False,
    )
    assert "When is it due?" in prompt
    assert "March 18." in prompt
    assert "[1] Due March 18, 2026" in prompt
    assert "IS contained" in prompt


def test_build_prompt_flags_abstention_expectation():
    prompt = build_judge_prompt("q", "a", [], "ref", expect_abstain=True)
    assert "abstains" in prompt
    assert "(none)" in prompt


def test_parse_verdict_plain_json():
    verdict = parse_verdict('{"grounded": true, "correct": false, "abstained": false, "reason": "partial"}')
    assert verdict == Verdict(grounded=True, correct=False, abstained=False, reason="partial")


def test_parse_verdict_strips_code_fence():
    text = '```json\n{"grounded": false, "correct": false, "abstained": false, "reason": "made up"}\n```'
    verdict = parse_verdict(text)
    assert verdict.grounded is False
    assert verdict.reason == "made up"


def test_parse_verdict_rejects_non_json():
    with pytest.raises(ValueError):
        parse_verdict("The answer looks fine to me.")


def test_parse_verdict_rejects_missing_keys():
    with pytest.raises(ValueError):
        parse_verdict('{"grounded": true}')


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, text, fail_with=None):
        self._text = text
        self._fail_with = list(fail_with or [])
        self.last_call = None
        self.calls = 0

    def generate_content(self, model, contents, config=None):
        self.calls += 1
        self.last_call = {"model": model, "contents": contents, "config": config}
        if self._fail_with:
            raise self._fail_with.pop(0)
        return FakeResponse(self._text)


class FakeClient:
    def __init__(self, text, fail_with=None):
        self.models = FakeModels(text, fail_with)


def _api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "busy", "status": "x"}})


def test_gemini_judge_retries_transient_errors(monkeypatch):
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: None)
    fake = FakeClient(
        '{"grounded": true, "correct": true, "abstained": false, "reason": "ok"}',
        fail_with=[_api_error(503)],
    )
    judge = GeminiJudge(api_key="x", model="test-model")
    monkeypatch.setattr(judge, "_get_client", lambda: fake)

    verdict = judge.judge("q", "a", ["ctx"], "ref")

    assert verdict.grounded is True
    assert fake.models.calls == 2


def test_gemini_judge_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr(judge_module.time, "sleep", lambda s: None)
    fake = FakeClient("{}", fail_with=[_api_error(400)])
    judge = GeminiJudge(api_key="x", model="test-model")
    monkeypatch.setattr(judge, "_get_client", lambda: fake)

    with pytest.raises(errors.APIError):
        judge.judge("q", "a", [], "ref")
    assert fake.models.calls == 1


def test_gemini_judge_calls_model_and_parses(monkeypatch):
    fake = FakeClient('{"grounded": true, "correct": true, "abstained": false, "reason": "ok"}')
    judge = GeminiJudge(api_key="x", model="test-model")
    monkeypatch.setattr(judge, "_get_client", lambda: fake)

    verdict = judge.judge("q", "a", ["ctx"], "ref")

    assert verdict.correct is True
    assert fake.models.last_call["model"] == "test-model"
    assert "ctx" in fake.models.last_call["contents"]
    assert fake.models.last_call["config"].response_mime_type == "application/json"
