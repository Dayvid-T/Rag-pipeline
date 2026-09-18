import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from evalsuite.metrics import CaseResult


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ms(value) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _mark(value) -> str:
    return "-" if value is None else ("yes" if value else "no")


def to_markdown(summary: Dict, results: List[CaseResult], base_url: str, run_at: str) -> str:
    lat = summary["latency_ms"] or {}
    lines = [
        "# RAG evaluation report",
        "",
        f"- Target: `{base_url}`",
        f"- Run at: {run_at}",
        f"- Cases: {summary['total']} ({summary['errors']} errored)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Hallucination rate | {_pct(summary['hallucination_rate'])} |",
        f"| Accuracy | {_pct(summary['accuracy'])} |",
        f"| Abstention rate on unanswerable | {_pct(summary['abstention_rate_on_unanswerable'])} |",
        f"| Retrieval hit rate | {_pct(summary['retrieval_hit_rate'])} |",
        f"| Latency mean / p50 / p95 / max | {_ms(lat.get('mean'))} / {_ms(lat.get('p50'))} / {_ms(lat.get('p95'))} / {_ms(lat.get('max'))} |",
        "",
        "## Cases",
        "",
        "| ID | Grounded | Correct | Abstained | Latency | Note |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        note = r.error if r.error else r.reason
        note = note.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r.id} | {_mark(r.grounded)} | {_mark(r.correct)} | {_mark(r.abstained)} "
            f"| {_ms(r.latency_ms)} | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(out_dir: Path, summary: Dict, results: List[CaseResult], base_url: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = run_at.replace(":", "").replace("-", "")

    payload = {
        "base_url": base_url,
        "run_at": run_at,
        "summary": summary,
        "cases": [r.to_dict() for r in results],
    }
    json_path = out_dir / f"{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    markdown = to_markdown(summary, results, base_url, run_at)
    (out_dir / "latest.md").write_text(markdown, encoding="utf-8")
    return json_path
