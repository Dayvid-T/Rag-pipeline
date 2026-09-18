# Evaluation suite (Project 2)

Automated evaluation of the RAG service at its API boundary. It treats the
service as a black box: every case in `dataset.jsonl` is sent to
`POST /query`, latency is measured on the wire, and a Gemini judge grades
the answer against the retrieved context and a reference answer.

## Metrics

| Metric | Meaning |
|---|---|
| Hallucination rate | Share of answers containing a claim the retrieved context does not support. An "I don't know" counts as grounded. |
| Accuracy | Share of answers that convey the reference answer (or correctly abstain when the question is unanswerable). |
| Abstention rate on unanswerable | Of the `expect_abstain` cases, how often the system declined to answer. |
| Retrieval hit rate | Of the cases with an `expected_source`, how often that file appeared in the returned `sources`. |
| Latency | Wall-clock `POST /query` time: mean, p50, p95, max. |

## Run

Needs a running service (local or deployed) and `GEMINI_API_KEY` for the
judge (read from `backend/.env` or `evaluation/.env`).

```bash
cd evaluation
python -m venv .venv
.venv\Scripts\activate          # or source .venv/bin/activate
pip install -r requirements.txt
python run_eval.py --base-url http://localhost:8000
```

Reports land in `reports/` (`latest.md` plus a timestamped JSON with every
answer, verdict and reason). Thresholds turn it into a gate:

```bash
python run_eval.py --max-hallucination-rate 0.1 --max-p95-latency-ms 5000
```

Set `EVAL_JUDGE_MODEL` to grade with a stronger model than the pipeline uses.

## Dataset

`dataset.jsonl`, one case per line:

```json
{"id": "a5-01", "question": "...", "reference_answer": "...", "expected_source": "Assignment5.pdf"}
{"id": "oos-01", "question": "...", "reference_answer": "...", "expect_abstain": true}
```

Cases cover each file in `backend/data/` plus out-of-scope questions that
the system should refuse rather than invent.

## Tests

```bash
pytest
```

Unit tests mock the HTTP service and the judge; nothing hits the network.
