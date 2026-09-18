import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from google import genai
from google.genai import errors, types

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gemini-3.1-flash-lite")

# Gemini's free tier intermittently returns 503 "high demand" / 429 quota
# errors; a judge failure would otherwise throw away a perfectly good case.
RETRYABLE_CODES = {429, 503}
MAX_ATTEMPTS = 4


@dataclass
class Verdict:
    grounded: bool
    correct: bool
    abstained: bool
    reason: str


def build_judge_prompt(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: str,
    expect_abstain: bool,
) -> str:
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) or "(none)"
    expectation = (
        "The reference answer is NOT in the retrieved context, so the ideal "
        "response abstains (says it doesn't know / the information isn't available)."
        if expect_abstain
        else "The reference answer IS contained in the retrieved context."
    )
    return (
        "You are grading the output of a retrieval-augmented question answering "
        "system. Judge strictly and return ONLY a JSON object with these keys:\n"
        '  "grounded": true if every factual claim in the answer is supported by '
        "the retrieved context (an abstention counts as grounded); false if the "
        "answer states anything the context does not support.\n"
        '  "correct": true if the answer conveys the same information as the '
        "reference answer (wording may differ). If the reference expects an "
        "abstention, correct is true only when the answer abstains.\n"
        '  "abstained": true if the answer declines to answer or says the '
        "information is not available.\n"
        '  "reason": one sentence explaining the grades.\n\n'
        f"Expectation: {expectation}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"System answer:\n{answer}\n\n"
        "JSON:"
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_verdict(text: str) -> Verdict:
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Judge returned non-JSON output: {text!r}") from e

    missing = {"grounded", "correct", "abstained"} - data.keys()
    if missing:
        raise ValueError(f"Judge output missing keys {sorted(missing)}: {text!r}")

    return Verdict(
        grounded=bool(data["grounded"]),
        correct=bool(data["correct"]),
        abstained=bool(data["abstained"]),
        reason=str(data.get("reason", "")),
    )


class GeminiJudge:
    """LLM-as-judge backed by Gemini, asked for a strict JSON verdict."""

    def __init__(self, api_key: Optional[str] = None, model: str = JUDGE_MODEL):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _generate(self, prompt: str):
        client = self._get_client()
        config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
        for attempt in range(MAX_ATTEMPTS):
            try:
                return client.models.generate_content(model=self._model, contents=prompt, config=config)
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == MAX_ATTEMPTS - 1:
                    raise
                time.sleep(2 ** attempt)

    def judge(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference_answer: str,
        expect_abstain: bool = False,
    ) -> Verdict:
        prompt = build_judge_prompt(question, answer, contexts, reference_answer, expect_abstain)
        return parse_verdict(self._generate(prompt).text)
