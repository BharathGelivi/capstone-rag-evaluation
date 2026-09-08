"""
Research Improvement #1: Diagnostic Accuracy of Root-Cause Attribution.

Every prior evaluation (RAGAS/RAGChecker/ARES agreement, correlation) has
asked whether X-RAG's *scores* track the baselines' scores. None of them
have asked the more important question for a diagnostic tool: is X-RAG's
*diagnosis* actually correct? eval/eval_dataset.csv already carries an
expected_failure_type label for every injected-failure example (and an
implicit "healthy" label -- empty string -- for the rest); this script is
the first thing to compare that against xrag_primary_cause from
artifacts/benchmark_comparison/results.json.

Produces a confusion matrix and per-category precision/recall/F1, computed
in pure Python (no numpy/sklearn), consistent with scripts/analyze_agreement.py.

Usage:
    python -m scripts.evaluate_diagnostic_accuracy
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

RESULTS_DIR = "artifacts/benchmark_comparison"

# X-RAG reports "UNKNOWN" for a healthy trace (no failure detected); the eval
# dataset represents the same thing as an empty expected_failure_type string.
HEALTHY_LABEL = "UNKNOWN"


def load_eval_dataset(path: str = "eval/eval_dataset.csv") -> Dict[str, str]:
    """Returns {eval_id: normalized_expected_label}."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["id"]: (row.get("expected_failure_type") or HEALTHY_LABEL) for row in rows}


def load_results(path: str) -> Dict[str, str]:
    """Returns {eval_id: xrag_primary_cause}."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {row["eval_id"]: row.get("xrag_primary_cause", HEALTHY_LABEL) for row in rows}


def join_labels(expected: Dict[str, str], predicted: Dict[str, str]) -> List[tuple]:
    """Pairs (expected_label, predicted_label) for every eval_id present in both."""
    return [(expected[eid], predicted[eid]) for eid in expected if eid in predicted]


def compute_confusion_matrix(pairs: List[tuple]) -> Dict[str, Dict[str, int]]:
    """{expected_label: {predicted_label: count}}, only over labels actually observed."""
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for expected, predicted in pairs:
        matrix[expected][predicted] += 1
    return {k: dict(v) for k, v in matrix.items()}


def compute_per_category_metrics(pairs: List[tuple]) -> Dict[str, Dict[str, Optional[float]]]:
    """Precision/recall/F1 for each label that appears as either expected or predicted."""
    categories = sorted({label for pair in pairs for label in pair})
    metrics = {}
    for category in categories:
        tp = sum(1 for e, p in pairs if e == category and p == category)
        fp = sum(1 for e, p in pairs if e != category and p == category)
        fn = sum(1 for e, p in pairs if e == category and p != category)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and (precision + recall) > 0) else None

        support = sum(1 for e, _ in pairs if e == category)
        metrics[category] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return metrics


def compute_overall_accuracy(pairs: List[tuple]) -> Optional[float]:
    if not pairs:
        return None
    correct = sum(1 for e, p in pairs if e == p)
    return correct / len(pairs)


def find_mismatches(expected: Dict[str, str], predicted: Dict[str, str]) -> List[Dict[str, Any]]:
    mismatches = []
    for eid in expected:
        if eid not in predicted:
            continue
        if expected[eid] != predicted[eid]:
            mismatches.append({"eval_id": eid, "expected": expected[eid], "predicted": predicted[eid]})
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Evaluate X-RAG's root-cause diagnostic accuracy against eval_dataset's expected_failure_type.")
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.csv")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "results.json"))
    args = parser.parse_args()

    expected = load_eval_dataset(args.eval_dataset)
    predicted = load_results(args.results)
    pairs = join_labels(expected, predicted)

    confusion_matrix = compute_confusion_matrix(pairs)
    per_category_metrics = compute_per_category_metrics(pairs)
    overall_accuracy = compute_overall_accuracy(pairs)
    mismatches = find_mismatches(
        {eid: label for eid, label in expected.items() if eid in predicted},
        predicted
    )

    output = {
        "n_examples_evaluated": len(pairs),
        "n_total_eval_dataset": len(expected),
        "overall_accuracy": overall_accuracy,
        "confusion_matrix": confusion_matrix,
        "per_category_metrics": per_category_metrics,
        "mismatches": mismatches,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "diagnostic_accuracy.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
