"""
Run the evaluation suite against a live RAG service.

    python run_eval.py --base-url http://localhost:8000

Each case in the dataset is sent to POST /query; latency is measured on the
wire, and a Gemini judge grades the answer for groundedness (hallucination)
and correctness against the reference. Exit code is non-zero when a
--max-* threshold is breached, so this can gate a deploy.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

from evalsuite.client import RagClient
from evalsuite.judge import GeminiJudge
from evalsuite.metrics import CaseResult, summarize
from evalsuite.report import to_markdown, write_reports

HERE = Path(__file__).resolve().parent


def load_dataset(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_case(case: dict, client: RagClient, judge: GeminiJudge) -> CaseResult:
    result = CaseResult(
        id=case["id"],
        question=case["question"],
        reference_answer=case["reference_answer"],
        expected_source=case.get("expected_source"),
        expect_abstain=case.get("expect_abstain", False),
        attack=case.get("attack", False),
    )

    try:
        response = client.query(case["question"])
    except (httpx.HTTPError, KeyError) as e:
        result.error = f"query failed: {e}"
        return result

    result.answer = response["answer"]
    result.sources = response["sources"]
    result.latency_ms = response["latency_ms"]
    result.blocked = response["blocked"]
    result.guardrail_flags = response["guardrail_flags"]

    if result.attack:
        # Success is the API blocking it, not the judge grading the answer -
        # there's no well-formed "correct" answer to a jailbreak attempt.
        result.correct = result.blocked
        result.reason = (
            f"blocked ({', '.join(result.guardrail_flags)})" if result.blocked
            else "NOT blocked - attack got through"
        )
        return result

    try:
        verdict = judge.judge(
            question=case["question"],
            answer=response["answer"],
            contexts=response["contexts"],
            reference_answer=case["reference_answer"],
            expect_abstain=result.expect_abstain,
        )
    except Exception as e:
        result.error = f"judge failed: {e}"
        return result

    result.grounded = verdict.grounded
    result.correct = verdict.correct
    result.abstained = verdict.abstained
    result.reason = verdict.reason
    return result


def main(argv=None) -> int:
    load_dotenv(HERE.parent / "backend" / ".env")
    load_dotenv(HERE / ".env")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=HERE / "reports")
    parser.add_argument("--limit", type=int, help="only run the first N cases")
    parser.add_argument("--max-hallucination-rate", type=float, help="fail if exceeded (0-1)")
    parser.add_argument("--max-p95-latency-ms", type=float, help="fail if exceeded")
    parser.add_argument("--min-attack-block-rate", type=float, help="fail if not met (0-1)")
    args = parser.parse_args(argv)

    client = RagClient(args.base_url)
    if not client.is_healthy():
        print(f"Service at {args.base_url} is not healthy; aborting.", file=sys.stderr)
        return 2

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    judge = GeminiJudge()
    results = []
    for i, case in enumerate(cases, 1):
        result = run_case(case, client, judge)
        results.append(result)
        if result.error:
            status = f"error: {result.error}"
        elif result.attack:
            status = f"blocked={result.blocked} {result.latency_ms:.0f}ms"
        else:
            status = f"grounded={result.grounded} correct={result.correct} {result.latency_ms:.0f}ms"
        print(f"[{i}/{len(cases)}] {case['id']}: {status}")

    summary = summarize(results)
    json_path = write_reports(args.out_dir, summary, results, args.base_url)
    print()
    print(to_markdown(summary, results, args.base_url, run_at="(see report)").split("## Cases")[0])
    print(f"Full report: {json_path}")

    failed = False
    if args.max_hallucination_rate is not None and summary["hallucination_rate"] is not None:
        if summary["hallucination_rate"] > args.max_hallucination_rate:
            print(f"FAIL: hallucination rate {summary['hallucination_rate']:.2f} > {args.max_hallucination_rate}")
            failed = True
    if args.max_p95_latency_ms is not None and summary["latency_ms"]:
        if summary["latency_ms"]["p95"] > args.max_p95_latency_ms:
            print(f"FAIL: p95 latency {summary['latency_ms']['p95']:.0f}ms > {args.max_p95_latency_ms}")
            failed = True
    if args.min_attack_block_rate is not None and summary["attack_block_rate"] is not None:
        if summary["attack_block_rate"] < args.min_attack_block_rate:
            print(f"FAIL: attack block rate {summary['attack_block_rate']:.2f} < {args.min_attack_block_rate}")
            failed = True
    if summary["errors"]:
        print(f"FAIL: {summary['errors']} case(s) errored")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
