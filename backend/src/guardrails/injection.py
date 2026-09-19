"""
Heuristic prompt-injection detection.

Runs on two different inputs with different consequences:
- the user's question: a high-severity match blocks the request before any
  retrieval or LLM call, so an attempted jailbreak costs nothing.
- each retrieved passage: any match excludes just that chunk from the
  prompt instead of failing the whole request, so a single poisoned
  document can't deny an answer built from the rest of the corpus.

Deliberately a fast, free, deterministic first line of defense - not a
substitute for Gemini's own safety filtering on the output side
(see guardrails/safety.py), just the layer that runs before any token is
spent.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# (label, pattern, is_strong) - a single "strong" match is enough on its
# own to flag something high severity; weak ones need to co-occur.
_PATTERNS: List[Tuple[str, "re.Pattern[str]", bool]] = [
    ("ignore_instructions", re.compile(
        r"\bignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+instructions?\b", re.I), True),
    ("disregard_instructions", re.compile(
        r"\bdisregard\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompt)\b", re.I), True),
    ("reveal_system_prompt", re.compile(
        r"\b(reveal|print|show|repeat|output)\s+(your|the)\s+(system\s+prompt|instructions|rules)\b", re.I), True),
    ("new_instructions", re.compile(r"\bnew\s+instructions\s*:", re.I), True),
    ("role_override", re.compile(r"\byou\s+are\s+now\s+(a|an)\b", re.I), False),
    ("pretend_to_be", re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.I), False),
    ("dan_jailbreak", re.compile(r"\bDAN\b|\bdo\s+anything\s+now\b", re.I), True),
    ("fake_role_marker", re.compile(r"^\s*(system|assistant)\s*:\s*\S", re.I | re.M), False),
    ("override_safety", re.compile(r"\boverride\s+(your\s+)?(safety|instructions|rules)\b", re.I), True),
    ("exfiltrate_secrets", re.compile(
        r"\b(reveal|leak|output|print|show)\b.{0,40}\b(api\s*key|password|credentials?|secret\s*key)\b"
        r"|\b(api\s*key|password|credentials?|secret\s*key)\b.{0,40}\b(reveal|leak|output|print|show)\b",
        re.I | re.S), True),
]


@dataclass
class InjectionVerdict:
    flagged: bool
    matches: List[str] = field(default_factory=list)
    severity: str = "none"  # none | medium | high


def scan(text: str) -> InjectionVerdict:
    """Scan `text` for prompt-injection patterns."""
    if not text:
        return InjectionVerdict(flagged=False)

    matched = []
    strong_hits = 0
    for label, pattern, is_strong in _PATTERNS:
        if pattern.search(text):
            matched.append(label)
            if is_strong:
                strong_hits += 1

    if not matched:
        return InjectionVerdict(flagged=False)

    severity = "high" if strong_hits > 0 or len(matched) >= 2 else "medium"
    return InjectionVerdict(flagged=True, matches=matched, severity=severity)
