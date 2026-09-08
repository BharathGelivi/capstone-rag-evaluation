"""
Computes cross-framework agreement from scripts/run_baseline_comparison.py's
results: Pearson correlation between X-RAG's own avg_entailment_score and
RAGAS/RAGChecker faithfulness-style scores, a binary failure-agreement
confusion matrix + Cohen's kappa (chance-corrected agreement), and a
disagreements.csv surfacing exactly where X-RAG and the baselines disagree
(the qualitative material for the paper's Discussion section).

Usage:
    python -m scripts.analyze_agreement [--results artifacts/benchmark_comparison/results.json]

Outputs (all under artifacts/benchmark_comparison/):
    correlations.json, agreement.json, disagreements.csv
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

RESULTS_DIR = "artifacts/benchmark_comparison"


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pearson_r(xs: List[Optional[float]], ys: List[Optional[float]]) -> Optional[float]:
    """Pure-python Pearson correlation coefficient. None if <2 paired points or zero variance in either series."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs2, ys2 = zip(*pairs)
    n = len(xs2)
    mean_x = sum(xs2) / n
    mean_y = sum(ys2) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs2, ys2))
    var_x = sum((x - mean_x) ** 2 for x in xs2)
    var_y = sum((y - mean_y) ** 2 for y in ys2)
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def compute_correlations(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    xrag_scores = [r.get("xrag_avg_entailment_score") for r in results]
    return {
        "xrag_vs_ragas_faithfulness": _pearson_r(xrag_scores, [r.get("ragas_faithfulness") for r in results]),
        "xrag_vs_ragchecker_faithfulness": _pearson_r(xrag_scores, [r.get("ragchecker_faithfulness") for r in results]),
        "xrag_vs_ragchecker_precision": _pearson_r(xrag_scores, [r.get("ragchecker_precision") for r in results]),
    }


def _cohens_kappa(a: List[Optional[bool]], b: List[Optional[bool]]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    p_a_true = sum(1 for x, _ in pairs if x) / n
    p_b_true = sum(1 for _, y in pairs if y) / n
    pe = p_a_true * p_b_true + (1 - p_a_true) * (1 - p_b_true)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def _confusion_matrix(a: List[Optional[bool]], b: List[Optional[bool]]) -> Dict[str, int]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return {
        "both_flag_failure": sum(1 for x, y in pairs if x and y),
        "only_first_flags_failure": sum(1 for x, y in pairs if x and not y),
        "only_second_flags_failure": sum(1 for x, y in pairs if not x and y),
        "neither_flags_failure": sum(1 for x, y in pairs if not x and not y),
    }


def _xrag_flags(results: List[Dict[str, Any]]) -> List[bool]:
    return [r.get("xrag_primary_cause") not in (None, "", "UNKNOWN") for r in results]


def _ragas_flags(results: List[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    return [(r.get("ragas_faithfulness") < threshold) if r.get("ragas_faithfulness") is not None else None for r in results]


def _ragchecker_flags(results: List[Dict[str, Any]], threshold: float) -> List[Optional[bool]]:
    return [(r.get("ragchecker_hallucination") > threshold) if r.get("ragchecker_hallucination") is not None else None for r in results]


def compute_agreement(
    results: List[Dict[str, Any]],
    ragas_threshold: float = 0.7,
    ragchecker_hallucination_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Agreement on binary failure/no-failure classification: X-RAG's
    primary_cause != UNKNOWN vs. RAGAS's faithfulness falling below
    ragas_threshold, and vs. RAGChecker's hallucination exceeding
    ragchecker_hallucination_threshold.
    """
    xrag_flags = _xrag_flags(results)
    ragas_flags = _ragas_flags(results, ragas_threshold)
    ragchecker_flags = _ragchecker_flags(results, ragchecker_hallucination_threshold)

    return {
        "xrag_vs_ragas": {
            "confusion_matrix": _confusion_matrix(xrag_flags, ragas_flags),
            "cohens_kappa": _cohens_kappa(xrag_flags, ragas_flags),
        },
        "xrag_vs_ragchecker": {
            "confusion_matrix": _confusion_matrix(xrag_flags, ragchecker_flags),
            "cohens_kappa": _cohens_kappa(xrag_flags, ragchecker_flags),
        },
    }


def load_reasoning_chain(trace_id: str) -> List[str]:
    path = os.path.join("artifacts", "root_cause_analysis", f"TRACE_{trace_id}.json")
    if not trace_id or not os.path.exists(path):
        return []
    from src.root_cause_reasoner import RootCauseAnalysis
    return RootCauseAnalysis.load(path).reasoning_chain


def find_disagreements(
    results: List[Dict[str, Any]],
    ragas_threshold: float = 0.7,
    ragchecker_hallucination_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Surfaces examples where X-RAG's failure/no-failure verdict disagrees with
    RAGAS's or RAGChecker's -- these are exactly the qualitative cases a
    paper's Discussion section should walk through (e.g. "X-RAG attributed
    this to CHUNK_BOUNDARY while RAGAS's aggregate faithfulness score didn't
    localize the failure further").
    """
    xrag_flags = _xrag_flags(results)
    ragas_flags = _ragas_flags(results, ragas_threshold)
    ragchecker_flags = _ragchecker_flags(results, ragchecker_hallucination_threshold)

    disagreements = []
    for r, xrag_flag, ragas_flag, ragchecker_flag in zip(results, xrag_flags, ragas_flags, ragchecker_flags):
        disagrees = (ragas_flag is not None and xrag_flag != ragas_flag) or \
                    (ragchecker_flag is not None and xrag_flag != ragchecker_flag)
        if not disagrees:
            continue

        disagreements.append({
            "eval_id": r.get("eval_id"),
            "question": r.get("question"),
            "expected_failure_type": r.get("expected_failure_type"),
            "xrag_primary_cause": r.get("xrag_primary_cause"),
            "xrag_reasoning_chain": " | ".join(load_reasoning_chain(r.get("trace_id", ""))),
            "ragas_faithfulness": r.get("ragas_faithfulness"),
            "ragchecker_hallucination": r.get("ragchecker_hallucination"),
            "ragchecker_precision": r.get("ragchecker_precision"),
            "ares_context_relevance": r.get("ares_context_relevance"),
            "ares_answer_faithfulness": r.get("ares_answer_faithfulness"),
        })
    return disagreements


def save_disagreements_csv(disagreements: List[Dict[str, Any]], path: str) -> None:
    if not disagreements:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(disagreements[0].keys()))
        writer.writeheader()
        writer.writerows(disagreements)


def main():
    parser = argparse.ArgumentParser(description="Analyze cross-framework agreement over benchmark comparison results.")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "results.json"))
    parser.add_argument("--ragas-threshold", type=float, default=0.7)
    parser.add_argument("--ragchecker-hallucination-threshold", type=float, default=0.5)
    args = parser.parse_args()

    results = load_results(args.results)

    correlations = compute_correlations(results)
    agreement = compute_agreement(results, args.ragas_threshold, args.ragchecker_hallucination_threshold)
    disagreements = find_disagreements(results, args.ragas_threshold, args.ragchecker_hallucination_threshold)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "correlations.json"), "w", encoding="utf-8") as f:
        json.dump(correlations, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "agreement.json"), "w", encoding="utf-8") as f:
        json.dump(agreement, f, indent=2)
    save_disagreements_csv(disagreements, os.path.join(RESULTS_DIR, "disagreements.csv"))

    print(json.dumps({"correlations": correlations, "agreement": agreement, "num_disagreements": len(disagreements)}, indent=2))


if __name__ == "__main__":
    main()
