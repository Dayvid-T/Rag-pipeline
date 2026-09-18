from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class CaseResult:
    id: str
    question: str
    reference_answer: str
    expected_source: Optional[str] = None
    expect_abstain: bool = False
    answer: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    grounded: Optional[bool] = None
    correct: Optional[bool] = None
    abstained: Optional[bool] = None
    reason: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile; pct in [0, 100]."""
    if not values:
        raise ValueError("percentile of empty list")
    ordered = sorted(values)
    rank = max(1, int(round(pct / 100 * len(ordered) + 0.5)))
    return ordered[min(rank, len(ordered)) - 1]


def _rate(flags: List[bool]) -> Optional[float]:
    return sum(flags) / len(flags) if flags else None


def summarize(results: List[CaseResult]) -> Dict:
    total = len(results)
    completed = [r for r in results if r.error is None]
    judged = [r for r in completed if r.grounded is not None]

    latencies = [r.latency_ms for r in completed if r.latency_ms is not None]
    retrievable = [r for r in completed if r.expected_source]
    abstain_cases = [r for r in judged if r.expect_abstain]

    return {
        "total": total,
        "completed": len(completed),
        "errors": total - len(completed),
        "error_rate": (total - len(completed)) / total if total else None,
        "hallucination_rate": _rate([not r.grounded for r in judged]),
        "accuracy": _rate([bool(r.correct) for r in judged]),
        "abstention_rate_on_unanswerable": _rate([bool(r.abstained) for r in abstain_cases]),
        "retrieval_hit_rate": _rate([r.expected_source in r.sources for r in retrievable]),
        "latency_ms": {
            "mean": sum(latencies) / len(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies),
        } if latencies else None,
    }
