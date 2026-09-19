"""
Output-side safety: routes generation through Gemini's built-in safety
filters (harassment, hate speech, dangerous content, sexually explicit,
jailbreak) and turns a blocked response into `None` instead of raising or
silently returning empty text.
"""

from typing import Optional

from google.genai import types

_CATEGORIES = [
    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    types.HarmCategory.HARM_CATEGORY_JAILBREAK,
]

BLOCKED_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
}

REFUSAL_MESSAGE = (
    "I can't provide that response - it was flagged by content safety "
    "filtering. Try rephrasing the question."
)


def safety_config() -> types.GenerateContentConfig:
    """Safety settings applied to every generation call."""
    return types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
            for c in _CATEGORIES
        ]
    )


def extract_text(response) -> Optional[str]:
    """Return the response text, or None if Gemini blocked the output."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return None
    if candidates[0].finish_reason in BLOCKED_FINISH_REASONS:
        return None
    return response.text
