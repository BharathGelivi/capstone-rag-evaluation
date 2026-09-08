"""
Research Improvement #3: Confidence Calibration.

RootCauseAnalysis.diagnosis_confidence is reported for every trace, but
nothing has ever checked whether it's calibrated: when X-RAG reports
confidence=0.93, is it actually right about 93% of the time? Pairs each
example's diagnosis_confidence against whether xrag_primary_cause matched
eval_dataset.csv's expected_failure_type (see
scripts/evaluate_diagnostic_accuracy.py), buckets into confidence bins, and
reports empirical accuracy per bin plus an Expected Calibration Error (ECE)
-- a standard reliability-diagram computation, done in pure Python.

Usage:
    python -m scripts.evaluate_confidence_calibration
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from scripts.evaluate_diagnostic_accuracy import load_eval_dataset

RESULTS_DIR = "artifacts/benchmark_comparison"
DEFAULT_BIN_EDGES = (0.0, 0.5, 0.7, 0.85, 0.95, 1.01)


def load_results_full(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_diagnosis_confidence(trace_id: str, reports_dir: str = "artifacts/reports") -> Optional[float]:
    path = os.path.join(reports_dir, f"{trace_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return report.get("root_cause_analysis", {}).get("diagnosis_confidence")


def build_confidence_correctness_pairs(eval_dataset_path: str, results_path: str) -> List[Tuple[str, float, bool]]:
    """Returns [(eval_id, diagnosis_confidence, was_correct), ...]."""
    expected = load_eval_dataset(eval_dataset_path)
    rows = load_results_full(results_path)

    pairs = []
    for row in rows:
        eval_id = row["eval_id"]
        if eval_id not in expected:
            continue
        confidence = load_diagnosis_confidence(row["trace_id"])
        if confidence is None:
            continue
        correct = expected[eval_id] == row.get("xrag_primary_cause")
        pairs.append((eval_id, confidence, correct))
    return pairs


def bucket_calibration(pairs: List[Tuple[str, float, bool]], bin_edges: Tuple[float, ...] = DEFAULT_BIN_EDGES) -> List[Dict[str, Any]]:
    bins = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = [(c, correct) for _, c, correct in pairs if lo <= c < hi]
        if not in_bin:
            bins.append({"range": f"[{lo:.2f}, {hi:.2f})", "n": 0, "mean_confidence": None, "empirical_accuracy": None})
            continue
        mean_conf = sum(c for c, _ in in_bin) / len(in_bin)
        acc = sum(1 for _, correct in in_bin if correct) / len(in_bin)
        bins.append({
            "range": f"[{lo:.2f}, {hi:.2f})",
            "n": len(in_bin),
            "mean_confidence": round(mean_conf, 3),
            "empirical_accuracy": round(acc, 3),
        })
    return bins


def compute_expected_calibration_error(bins: List[Dict[str, Any]]) -> Optional[float]:
    """ECE = sum over bins of (bin weight) * |mean confidence - empirical accuracy|."""
    total_n = sum(b["n"] for b in bins)
    if total_n == 0:
        return None
    ece = sum(
        (b["n"] / total_n) * abs(b["mean_confidence"] - b["empirical_accuracy"])
        for b in bins if b["n"] > 0
    )
    return round(ece, 4)


def main():
    parser = argparse.ArgumentParser(description="Evaluate calibration of RootCauseAnalysis.diagnosis_confidence.")
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.csv")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "results.json"))
    args = parser.parse_args()

    pairs = build_confidence_correctness_pairs(args.eval_dataset, args.results)
    bins = bucket_calibration(pairs)
    ece = compute_expected_calibration_error(bins)

    output = {
        "n_examples": len(pairs),
        "expected_calibration_error": ece,
        "bins": bins,
        "raw_pairs": [{"eval_id": eid, "confidence": c, "correct": correct} for eid, c, correct in pairs],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "confidence_calibration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
