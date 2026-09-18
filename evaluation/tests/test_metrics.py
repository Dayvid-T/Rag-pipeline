import pytest

from evalsuite.metrics import CaseResult, percentile, summarize


def _case(id, **overrides):
    base = dict(
        id=id,
        question="q",
        reference_answer="ref",
        expected_source="a.txt",
        answer="ans",
        sources=["a.txt"],
        latency_ms=100.0,
        grounded=True,
        correct=True,
        abstained=False,
    )
    base.update(overrides)
    return CaseResult(**base)


def test_percentile_nearest_rank():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == 50
    assert percentile(values, 0) == 10


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_rates():
    results = [
        _case("1"),
        _case("2", grounded=False, correct=False),
        _case("3", sources=["b.txt"]),
        _case("4", expected_source=None, expect_abstain=True, abstained=True, latency_ms=300.0),
    ]

    summary = summarize(results)

    assert summary["total"] == 4
    assert summary["errors"] == 0
    assert summary["hallucination_rate"] == 0.25
    assert summary["accuracy"] == 0.75
    assert summary["abstention_rate_on_unanswerable"] == 1.0
    # 3 cases have an expected source; 2 of them retrieved it
    assert summary["retrieval_hit_rate"] == pytest.approx(2 / 3)
    assert summary["latency_ms"]["max"] == 300.0
    assert summary["latency_ms"]["mean"] == 150.0


def test_summarize_excludes_errored_cases_from_rates():
    results = [
        _case("1"),
        _case("2", error="query failed", answer=None, latency_ms=None, grounded=None, correct=None, abstained=None),
    ]

    summary = summarize(results)

    assert summary["errors"] == 1
    assert summary["error_rate"] == 0.5
    assert summary["hallucination_rate"] == 0.0
    assert summary["accuracy"] == 1.0
    assert summary["latency_ms"]["mean"] == 100.0


def test_summarize_empty():
    summary = summarize([])
    assert summary["total"] == 0
    assert summary["hallucination_rate"] is None
    assert summary["latency_ms"] is None
