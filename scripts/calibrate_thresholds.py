"""
Offline threshold calibration script.

Takes a labeled dataset of (claim_text, evidence_text, gold_label) rows and
sweeps NLI threshold values, reporting precision/recall/F1 per threshold so
the constants in configs/thresholds.py can cite a specific calibration run.

Does not run automatically in production -- this is a research/reporting tool.

Usage:
    python scripts/calibrate_thresholds.py --dataset scripts/sample_calibration_data.csv
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List


def load_labeled_dataset(path: str) -> List[Dict[str, Any]]:
    """Loads (claim_text, evidence_text, gold_label) rows from a CSV or JSON file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_precision_recall_f1(preds: List[str], golds: List[str], positive_label: str) -> Dict[str, float]:
    """Computes precision/recall/F1 for a single positive_label against gold labels."""
    if len(preds) != len(golds):
        raise ValueError("preds and golds must be the same length.")

    true_positives = sum(1 for p, g in zip(preds, golds) if p == positive_label and g == positive_label)
    predicted_positives = sum(1 for p in preds if p == positive_label)
    actual_positives = sum(1 for g in golds if g == positive_label)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def sweep_thresholds(dataset: List[Dict[str, Any]], verifier, thresholds: List[float]) -> List[Dict[str, Any]]:
    """
    For each threshold, classifies every row as SUPPORTED if the NLI entailment
    score for (evidence_text, claim_text) clears the threshold, else NOT_SUPPORTED,
    and reports precision/recall/F1 against gold_label.
    """
    golds = [row["gold_label"] for row in dataset]
    results = []

    for threshold in thresholds:
        preds = []
        for row in dataset:
            scores = verifier.run_nli(premise=row["evidence_text"], hypothesis=row["claim_text"])
            preds.append("SUPPORTED" if scores["entailment"] >= threshold else "NOT_SUPPORTED")

        metrics = compute_precision_recall_f1(preds, golds, positive_label="SUPPORTED")
        results.append({"threshold": threshold, **metrics})

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep NLI entailment thresholds against a labeled dataset.")
    parser.add_argument("--dataset", required=True, help="Path to a labeled CSV/JSON dataset.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Threshold values to sweep.",
    )
    args = parser.parse_args()

    dataset = load_labeled_dataset(args.dataset)

    from src.claim_verifier import ClaimVerifier
    verifier = ClaimVerifier()

    results = sweep_thresholds(dataset, verifier, args.thresholds)

    print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 48)
    for row in results:
        print(f"{row['threshold']:>10} | {row['precision']:>10} | {row['recall']:>10} | {row['f1']:>10}")


if __name__ == "__main__":
    main()
