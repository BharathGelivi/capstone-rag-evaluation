"""
Research Improvement #2: Ablation on evidence-aggregation strategy.

configs/thresholds.py's EVIDENCE_AGGREGATION_STRATEGY has three implemented
options (top1 / max_pool_top3 / concat_top3), added in an earlier session --
but nothing has ever compared them against each other, or checked whether
diagnostic accuracy (see scripts/evaluate_diagnostic_accuracy.py) actually
depends on the choice.

Efficiency note: which strategy is used does NOT change how many sentences
get NLI-scored (that per-sentence scoring is the expensive part) -- it only
changes how the already-computed top-3 scores get combined into a final
verdict. So this script scores each claim's evidence ONCE (via
ClaimVerifier.compute_all_aggregation_strategies) and derives what all three
strategies would have decided from that single pass, rather than re-running
verification three separate times. Reuses PipelineStateAnalyzer and
RootCauseReasoner completely unchanged -- only the VerificationResult
statuses fed into them vary per strategy.

Usage:
    python -m scripts.ablate_aggregation_strategy
"""

import argparse
import json
import os
import uuid
from typing import Any, Dict, List

from src.rag_trace import RAGTrace
from src.chunk_registry import ChunkRegistry
from src.claims import ClaimSet
from src.claim_verifier import ClaimVerifier, VerificationResult, VerificationSummary, VerificationStatus
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.root_cause_reasoner import RootCauseReasoner

from scripts.generate_diagnostic_report import find_trace_path
from scripts.evaluate_diagnostic_accuracy import (
    load_eval_dataset,
    compute_confusion_matrix,
    compute_per_category_metrics,
    compute_overall_accuracy,
)

RESULTS_DIR = "artifacts/benchmark_comparison"
MANIFEST_PATH = os.path.join(RESULTS_DIR, "manifest.json")
STRATEGIES = ["top1", "max_pool_top3", "concat_top3"]


def load_manifest(path: str = MANIFEST_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_trace(trace_id: str) -> RAGTrace:
    path = find_trace_path(trace_id)
    if path is None:
        raise FileNotFoundError(f"No trace file found for trace_id={trace_id}")
    with open(path, encoding="utf-8") as f:
        return RAGTrace(**json.load(f))


def load_report(trace_id: str) -> Dict[str, Any]:
    path = os.path.join("artifacts", "reports", f"{trace_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_verification_result(claim_id: str, claim_text: str, trace_id: str, outcome: Dict[str, Any], best_sentence) -> VerificationResult:
    status = VerificationStatus(outcome["status"])
    confidence = max(outcome["entailment"], outcome["contradiction"])
    return VerificationResult(
        verification_id=str(uuid.uuid4()),
        trace_id=trace_id,
        claim_id=claim_id,
        claim_text=claim_text,
        verification_status=status,
        verification_reason="ablation",
        confidence=confidence,
        best_chunk_id=best_sentence.chunk_id if best_sentence else None,
        best_chunk_rank=best_sentence.chunk_rank if best_sentence else None,
        best_chunk_score=None,
        best_sentence_id=best_sentence.sentence_id if best_sentence else None,
        evidence_text=best_sentence.text if best_sentence else None,
        entailment_score=outcome["entailment"],
        contradiction_score=outcome["contradiction"],
        neutral_score=outcome["neutral"],
    )


def build_summary(trace_id: str, results: List[VerificationResult]) -> VerificationSummary:
    total = len(results)
    avg_ent = (sum(r.entailment_score for r in results) / total) if total else 0.0
    return VerificationSummary(
        trace_id=trace_id,
        total_claims=total,
        supported_claims=sum(1 for r in results if r.verification_status == VerificationStatus.SUPPORTED),
        partially_supported_claims=sum(1 for r in results if r.verification_status == VerificationStatus.PARTIALLY_SUPPORTED),
        contradicted_claims=sum(1 for r in results if r.verification_status == VerificationStatus.CONTRADICTED),
        unsupported_claims=sum(1 for r in results if r.verification_status == VerificationStatus.UNSUPPORTED),
        not_verifiable_claims=sum(1 for r in results if r.verification_status == VerificationStatus.NOT_VERIFIABLE),
        average_entailment_score=avg_ent,
        total_verification_latency_ms=0.0,
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description="Ablate evidence-aggregation strategy's effect on diagnostic accuracy.")
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.csv")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    expected_labels = load_eval_dataset(args.eval_dataset)

    verifier = ClaimVerifier()
    analyzer = PipelineStateAnalyzer()
    reasoner = RootCauseReasoner()
    registry = ChunkRegistry.load_from_json("artifacts/chunk_registry.json")

    predicted_by_strategy: Dict[str, Dict[str, str]] = {s: {} for s in STRATEGIES}

    for eval_id, trace_id in manifest["eval_id_to_trace_id"].items():
        print(f"Ablating eval {eval_id} (trace {trace_id})...")
        trace = load_trace(trace_id)
        report = load_report(trace_id)
        claim_set = ClaimSet(trace_id=trace_id)

        chunks = ClaimVerifier.build_retrieved_chunks_from_trace(trace, registry)
        sources = [(c.chunk_id, c.rank, c.chunk_text) for c in chunks]

        per_strategy_results: Dict[str, List[VerificationResult]] = {s: [] for s in STRATEGIES}
        for evidence in report["evidence_analysis"]:
            outcome = verifier.compute_all_aggregation_strategies(evidence["claim_text"], sources)
            best_sentence = outcome["best_sentence"]
            for strategy in STRATEGIES:
                per_strategy_results[strategy].append(
                    build_verification_result(evidence["claim_id"], evidence["claim_text"], trace_id, outcome[strategy], best_sentence)
                )

        for strategy in STRATEGIES:
            summary = build_summary(trace_id, per_strategy_results[strategy])
            psm = analyzer.analyze(trace, claim_set, summary)
            rca = reasoner.analyze(psm)
            predicted_by_strategy[strategy][eval_id] = rca.primary_cause.value

    output = {}
    for strategy in STRATEGIES:
        pairs = [(expected_labels[eid], predicted_by_strategy[strategy][eid]) for eid in expected_labels if eid in predicted_by_strategy[strategy]]
        output[strategy] = {
            "n_examples_evaluated": len(pairs),
            "overall_accuracy": compute_overall_accuracy(pairs),
            "confusion_matrix": compute_confusion_matrix(pairs),
            "per_category_metrics": compute_per_category_metrics(pairs),
            "predictions": predicted_by_strategy[strategy],
        }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "aggregation_strategy_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps({s: {"accuracy": output[s]["overall_accuracy"], "n": output[s]["n_examples_evaluated"]} for s in STRATEGIES}, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
