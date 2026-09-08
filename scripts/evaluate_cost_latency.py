"""
Research Improvement #5: Cost / Latency Tradeoff.

Turns the previously-qualitative "X-RAG is cheaper but slower" claim (see
generate_comparison_report.py's APPROX_LLM_CALLS table) into real, measured
numbers rather than an assertion. X-RAG's own per-stage latency has always
been captured (DiagnosticEvaluationReport.evaluation_metrics); baseline
wall-clock latency was NOT captured until this session (see
scripts/run_baseline_comparison.py's ragas_latency_ms/ragchecker_latency_ms/
ares_latency_ms columns, added alongside this script) -- so examples run
BEFORE that instrumentation was added have no baseline latency to report.
This script says so explicitly rather than silently omitting or
approximating it.

Usage:
    python -m scripts.evaluate_cost_latency
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

RESULTS_DIR = "artifacts/benchmark_comparison"


def load_results_full(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_report(trace_id: str, reports_dir: str = "artifacts/reports") -> Optional[Dict[str, Any]]:
    path = os.path.join(reports_dir, f"{trace_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 1) if values else None


def compute_xrag_latency(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Real, measured X-RAG per-stage latency (ms), averaged across every
    example that has a report.json -- this data has existed since the
    original pipeline was built, just never formally reported."""
    retrieval, generation, verification = [], [], []
    for row in rows:
        report = load_report(row["trace_id"])
        if report is None:
            continue
        metrics = report.get("evaluation_metrics", {})
        retrieval.append(metrics.get("retrieval_latency_ms"))
        generation.append(metrics.get("generation_latency_ms"))
        verification.append(metrics.get("verification_latency_ms"))

    return {
        "n_examples": len(rows),
        "mean_retrieval_latency_ms": mean(retrieval),
        "mean_generation_latency_ms": mean(generation),
        "mean_verification_latency_ms": mean(verification),
        "mean_total_latency_ms": mean(
            [
                (r or 0) + (g or 0) + (v or 0)
                for r, g, v in zip(retrieval, generation, verification)
                if r is not None and g is not None and v is not None
            ]
        ),
    }


def compute_baseline_latency(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Real, measured baseline latency where available (ragas_latency_ms/
    ragchecker_latency_ms/ares_latency_ms -- added this session). Examples
    run before this instrumentation existed have no data; reported as 0
    instrumented examples, not silently skipped."""
    result = {}
    for baseline in ("ragas", "ragchecker", "ares"):
        key = f"{baseline}_latency_ms"
        values = [row.get(key) for row in rows if row.get(key) is not None]
        result[baseline] = {
            "n_instrumented_examples": len(values),
            "n_total_examples": len(rows),
            "mean_latency_ms": mean(values),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Report real (not approximated) cost/latency for X-RAG vs. baselines.")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "results.json"))
    args = parser.parse_args()

    rows = load_results_full(args.results)
    xrag_latency = compute_xrag_latency(rows)
    baseline_latency = compute_baseline_latency(rows)

    output = {"xrag": xrag_latency, "baselines": baseline_latency}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cost_latency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
