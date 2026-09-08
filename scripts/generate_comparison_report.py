"""
Reads everything scripts/run_baseline_comparison.py and scripts/analyze_agreement.py
produced and renders a single, self-contained Markdown report a reader can
understand without opening any of the underlying CSVs/JSON files.

Usage:
    python -m scripts.generate_comparison_report

Output: artifacts/benchmark_comparison/comparison_report.md
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

RESULTS_DIR = "artifacts/benchmark_comparison"

FRAMEWORK_DESCRIPTIONS = {
    "X-RAG": "This project's own diagnostic pipeline: claim-level NLI verification against retrieved context, with root-cause attribution to a *specific pipeline stage* (retrieval / chunking / generation / grounding).",
    "RAGAS": "LLM-as-judge, reference-free by default. Reports aggregate scores per metric (faithfulness, answer relevancy, context precision/recall, answer correctness) with no stage attribution.",
    "RAGChecker": "Claim-level precision/recall/F1 against a gold answer, plus a retriever-vs-generator metric split (claim_recall/context_precision vs. faithfulness/hallucination) -- attributes to *retriever or generator*, not further.",
    "ARES (ues_idp)": "Lightweight LLM-judge: per-retrieved-document context relevance, plus answer relevance and answer faithfulness -- no stage attribution beyond per-document context judgments.",
}

APPROX_LLM_CALLS = {
    "X-RAG": "~1 local NLI model call per claim-sentence pair (no LLM judge at all)",
    "RAGAS": "~2-3 LLM judge calls per example (claim decomposition + per-claim/context judging)",
    "RAGChecker": "~5+ LLM judge calls per example (claim extraction x2 + 4 claim-checking passes, each internally self-consistency-checked)",
    "ARES (ues_idp)": "~1 LLM judge call per retrieved document for context relevance, +2 more per document if relevant",
}


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return round(mean(values), 4) if values else None


def render_summary_table(results: List[Dict[str, Any]]) -> str:
    n = len(results)
    means = {
        "X-RAG": safe_mean([r.get("xrag_avg_entailment_score") for r in results]),
        "RAGAS": safe_mean([r.get("ragas_faithfulness") for r in results]),
        "RAGChecker": safe_mean([r.get("ragchecker_faithfulness") for r in results]),
        "ARES (ues_idp)": safe_mean([r.get("ares_answer_faithfulness") for r in results]),
    }

    lines = [
        f"| Framework | What it measures | Mean faithfulness-style score (n={n}) | Approx. LLM calls / example |",
        "|---|---|---|---|",
    ]
    for name, description in FRAMEWORK_DESCRIPTIONS.items():
        mean_score = means[name]
        mean_str = f"{mean_score}" if mean_score is not None else "N/A (no successful runs)"
        lines.append(f"| {name} | {description} | {mean_str} | {APPROX_LLM_CALLS[name]} |")
    return "\n".join(lines)


def render_correlation_section(correlations: Dict[str, Optional[float]]) -> str:
    if not correlations:
        return "_No correlation data available (run scripts/analyze_agreement.py first)._"
    lines = ["| Comparison | Pearson r |", "|---|---|"]
    labels = {
        "xrag_vs_ragas_faithfulness": "X-RAG avg_entailment_score vs. RAGAS faithfulness",
        "xrag_vs_ragchecker_faithfulness": "X-RAG avg_entailment_score vs. RAGChecker faithfulness",
        "xrag_vs_ragchecker_precision": "X-RAG avg_entailment_score vs. RAGChecker precision",
    }
    for key, label in labels.items():
        value = correlations.get(key)
        lines.append(f"| {label} | {value if value is not None else 'N/A'} |")
    return "\n".join(lines)


def render_agreement_section(agreement: Dict[str, Any]) -> str:
    if not agreement:
        return "_No agreement data available (run scripts/analyze_agreement.py first)._"

    sections = []
    for pair_key, pair_label in [("xrag_vs_ragas", "X-RAG vs. RAGAS"), ("xrag_vs_ragchecker", "X-RAG vs. RAGChecker")]:
        pair = agreement.get(pair_key)
        if not pair:
            continue
        cm = pair["confusion_matrix"]
        kappa = pair["cohens_kappa"]
        sections.append(
            f"**{pair_label}** (binary failure/no-failure classification, Cohen's kappa = {kappa if kappa is not None else 'N/A'}):\n\n"
            f"| | Baseline: failure | Baseline: no failure |\n"
            f"|---|---|---|\n"
            f"| **X-RAG: failure** | {cm['both_flag_failure']} | {cm['only_first_flags_failure']} |\n"
            f"| **X-RAG: no failure** | {cm['only_second_flags_failure']} | {cm['neither_flags_failure']} |"
        )
    return "\n\n".join(sections) if sections else "_No agreement data available._"


def render_value_add_section(disagreements: List[Dict[str, str]], max_examples: int = 3) -> str:
    intro = (
        "X-RAG's differentiating claim is **localization**: where RAGAS and ARES report aggregate "
        "scores per metric with no attribution to *where* in the pipeline a failure occurred, and "
        "RAGChecker attributes only as far as retriever-vs-generator, X-RAG's root cause analysis "
        "names a specific pipeline stage (`MISSING_CORPUS`, `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`, "
        "`UNSUPPORTED_GENERATION`, or `GROUNDING_FAILURE`) with a reasoning chain justifying it.\n"
    )
    if not disagreements:
        return intro + "\nNo disagreement examples were captured to illustrate this (run scripts/analyze_agreement.py)."

    examples = []
    for row in disagreements[:max_examples]:
        examples.append(
            f"- **{row.get('question', '')[:120]}**\n"
            f"  - X-RAG: `{row.get('xrag_primary_cause')}` -- {row.get('xrag_reasoning_chain') or '(no reasoning chain captured)'}\n"
            f"  - RAGAS faithfulness: {row.get('ragas_faithfulness', 'N/A')} | "
            f"RAGChecker hallucination: {row.get('ragchecker_hallucination', 'N/A')} | "
            f"ARES context relevance: {row.get('ares_context_relevance', 'N/A')}"
        )
    return intro + "\n" + "\n".join(examples)


def render_diagnostic_accuracy_section(diagnostic_accuracy: Dict[str, Any]) -> str:
    """
    Research Improvement #1: is X-RAG's root-cause *diagnosis* actually
    correct, not just its scores correlated with baselines'? Compares
    xrag_primary_cause against eval_dataset.csv's expected_failure_type
    (see scripts/evaluate_diagnostic_accuracy.py).
    """
    if not diagnostic_accuracy:
        return "_No diagnostic accuracy data available (run scripts/evaluate_diagnostic_accuracy.py first)._"

    n = diagnostic_accuracy.get("n_examples_evaluated", 0)
    n_total = diagnostic_accuracy.get("n_total_eval_dataset", 0)
    accuracy = diagnostic_accuracy.get("overall_accuracy")
    confusion_matrix = diagnostic_accuracy.get("confusion_matrix", {})
    per_category = diagnostic_accuracy.get("per_category_metrics", {})
    mismatches = diagnostic_accuracy.get("mismatches", [])

    lines = [
        f"Evaluated on {n} of {n_total} labeled examples "
        f"({'a small pilot slice -- treat directionally, not as a powered result' if n < n_total else 'the full labeled set'}).",
        "",
        f"**Overall diagnostic accuracy: {accuracy:.3f}** ({round(accuracy * n) if accuracy is not None else '?'}/{n} correct)" if accuracy is not None else "**Overall diagnostic accuracy: N/A**",
        "",
        "### Confusion Matrix (expected -> predicted)",
        "",
        "| Expected \\ Predicted | " + " | ".join(sorted({p for preds in confusion_matrix.values() for p in preds})) + " |",
        "|---" * (1 + len({p for preds in confusion_matrix.values() for p in preds})) + "|",
    ]
    predicted_labels = sorted({p for preds in confusion_matrix.values() for p in preds})
    for expected_label in sorted(confusion_matrix.keys()):
        row = [str(confusion_matrix[expected_label].get(p, 0)) for p in predicted_labels]
        lines.append(f"| **{expected_label}** | " + " | ".join(row) + " |")

    lines += [
        "",
        "### Per-Category Precision / Recall / F1",
        "",
        "| Category | Precision | Recall | F1 | Support (expected count) |",
        "|---|---|---|---|---|",
    ]
    for category, m in sorted(per_category.items()):
        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"
        lines.append(f"| {category} | {fmt(m.get('precision'))} | {fmt(m.get('recall'))} | {fmt(m.get('f1'))} | {m.get('support', 0)} |")

    lines += ["", "### Mismatches"]
    if not mismatches:
        lines.append("None -- every evaluated example's diagnosis matched the expected label.")
    else:
        for mm in mismatches:
            lines.append(f"- eval_id `{mm['eval_id']}`: expected `{mm['expected']}`, X-RAG predicted `{mm['predicted']}`")

    return "\n".join(lines)


def render_aggregation_ablation_section(ablation: Dict[str, Any]) -> str:
    """
    Research Improvement #2: ablation on evidence-aggregation strategy
    (top1 / max_pool_top3 / concat_top3) and their effect on diagnostic
    accuracy (see scripts/ablate_aggregation_strategy.py).
    """
    if not ablation:
        return "_No ablation data available (run scripts/ablate_aggregation_strategy.py first)._"

    lines = [
        "Scoring each claim's evidence once via `ClaimVerifier.compute_all_aggregation_strategies` "
        "and deriving what each strategy would have decided, rather than re-running verification "
        "three separate times (see `scripts/ablate_aggregation_strategy.py` for methodology).",
        "",
        "| Aggregation Strategy | N evaluated | Overall Accuracy |",
        "|---|---|---|",
    ]
    strategies = ["top1", "max_pool_top3", "concat_top3"]
    for strategy in strategies:
        entry = ablation.get(strategy, {})
        n = entry.get("n_examples_evaluated", "N/A")
        acc = entry.get("overall_accuracy")
        acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "N/A"
        lines.append(f"| `{strategy}` | {n} | {acc_str} |")

    lines += [
        "",
        "### Per-Strategy Per-Category F1",
        "",
    ]
    # collect all categories across strategies
    all_cats = sorted({cat for s in strategies for cat in ablation.get(s, {}).get("per_category_metrics", {}).keys()})
    if all_cats:
        header = "| Category | " + " | ".join(f"`{s}` F1" for s in strategies) + " |"
        sep = "|---" * (1 + len(strategies)) + "|"
        lines += [header, sep]
        for cat in all_cats:
            row_vals = []
            for s in strategies:
                f1 = ablation.get(s, {}).get("per_category_metrics", {}).get(cat, {}).get("f1")
                row_vals.append(f"{f1:.3f}" if isinstance(f1, (int, float)) else "N/A")
            lines.append(f"| {cat} | " + " | ".join(row_vals) + " |")
    else:
        lines.append("_No per-category data found._")

    return "\n".join(lines)


def render_confidence_calibration_section(calibration: Dict[str, Any]) -> str:
    """
    Research Improvement #3: confidence calibration analysis.
    Checks whether RootCauseAnalysis.diagnosis_confidence is well-calibrated
    (see scripts/evaluate_confidence_calibration.py).
    """
    if not calibration:
        return "_No calibration data available (run scripts/evaluate_confidence_calibration.py first)._"

    n = calibration.get("n_examples", 0)
    ece = calibration.get("expected_calibration_error")
    bins = calibration.get("bins", [])

    ece_str = f"{ece:.4f}" if isinstance(ece, (int, float)) else "N/A"
    lines = [
        f"Evaluated on {n} examples. **Expected Calibration Error (ECE): {ece_str}** "
        "(lower = better; 0 = perfectly calibrated).",
        "",
        "A well-calibrated diagnostic tool would show `empirical_accuracy ≈ mean_confidence` "
        "in every bin. A large gap signals over- or under-confidence in the diagnosis.",
        "",
        "| Confidence Bin | N | Mean Confidence | Empirical Accuracy | Gap |",
        "|---|---|---|---|---|",
    ]
    for b in bins:
        if b["n"] == 0:
            lines.append(f"| {b['range']} | 0 | — | — | — |")
            continue
        mc = b["mean_confidence"]
        ea = b["empirical_accuracy"]
        gap = round(abs(mc - ea), 3) if (mc is not None and ea is not None) else None
        mc_str = f"{mc:.3f}" if mc is not None else "N/A"
        ea_str = f"{ea:.3f}" if ea is not None else "N/A"
        gap_str = f"{gap:.3f}" if gap is not None else "N/A"
        lines.append(f"| {b['range']} | {b['n']} | {mc_str} | {ea_str} | {gap_str} |")

    lines += [
        "",
        "> **Interpretation note:** all 9 examples fell in the `[0.85, 0.95)` bin because "
        "the `RootCauseReasoner` emits a fixed confidence of 0.90 or 0.93 depending on "
        "whether evidence is ambiguous -- not a posterior probability learned from data. "
        "True per-example calibration requires a larger, more diverse eval set "
        "and a confidence model that varies more continuously.",
    ]
    return "\n".join(lines)


def render_reasoning_validation_section(consistency: Dict[str, Any]) -> str:
    """
    Research Improvement #4: human validation of reasoning chains.
    Reports automated self-consistency check results and links to the
    human-rating worksheet (see scripts/evaluate_reasoning_consistency.py).
    """
    if not consistency:
        return "_No reasoning consistency data available (run scripts/evaluate_reasoning_consistency.py first)._"

    n = consistency.get("n_examples", 0)
    n_consistent = consistency.get("n_consistent", 0)
    rate = consistency.get("consistency_rate")
    rate_str = f"{rate:.3f}" if isinstance(rate, (int, float)) else "N/A"
    note = consistency.get("note", "")

    lines = [
        f"**Automated self-consistency rate: {rate_str}** ({n_consistent}/{n} examples pass all "
        "automated checks: correct stage → FAIL status mapping, primary cause named in chain, "
        "all FAIL stages mentioned).",
        "",
        f"> ⚠️ **{note}**",
        "",
        "### Per-Example Automated Consistency Checks",
        "",
        "| Eval ID | Consistent | primary_cause_stage_is_fail | chain_names_primary_cause | all_fail_stages_named |",
        "|---|---|---|---|---|",
    ]
    for eid, checks in sorted(consistency.get("per_example", {}).items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        def bool_str(v):
            if v is None: return "N/A"
            return "✓" if v else "✗"
        lines.append(
            f"| {eid} | {bool_str(checks.get('consistent'))} "
            f"| {bool_str(checks.get('primary_cause_stage_is_fail'))} "
            f"| {bool_str(checks.get('chain_names_primary_cause'))} "
            f"| {bool_str(checks.get('all_fail_stages_named_in_chain'))} |"
        )

    lines += [
        "",
        "**Human validation status:** The `reasoning_chain_worksheet.csv` has been generated "
        "at `artifacts/benchmark_comparison/reasoning_chain_worksheet.csv` with one row per "
        "trace, including the full reasoning chain and empty `sound_yn` / `notes` columns. "
        "This worksheet has **not yet been filled in** -- human validation remains a pending step.",
    ]
    return "\n".join(lines)


def render_cost_latency_section(cost_latency: Dict[str, Any]) -> str:
    """
    Research Improvement #5: cost/latency tradeoff formalization.
    Reports real measured latency for X-RAG vs. baselines
    (see scripts/evaluate_cost_latency.py).
    """
    if not cost_latency:
        return "_No cost/latency data available (run scripts/evaluate_cost_latency.py first)._"

    xrag = cost_latency.get("xrag", {})
    baselines = cost_latency.get("baselines", {})

    def ms(v, suffix="ms"):
        return f"{v:,.0f} {suffix}" if isinstance(v, (int, float)) else "N/A"

    lines = [
        f"Measured over {xrag.get('n_examples', 'N/A')} X-RAG traces (all real wall-clock times, "
        "not approximated call counts).",
        "",
        "### X-RAG Per-Stage Latency (mean across examples)",
        "",
        "| Stage | Mean Latency |",
        "|---|---|",
        f"| Retrieval | {ms(xrag.get('mean_retrieval_latency_ms'))} |",
        f"| Generation | {ms(xrag.get('mean_generation_latency_ms'))} |",
        f"| Verification (NLI) | {ms(xrag.get('mean_verification_latency_ms'))} |",
        f"| **Total** | **{ms(xrag.get('mean_total_latency_ms'))}** |",
        "",
        "> **Note:** The verification stage dominates because the local NLI model scores every "
        "claim × sentence pair; retrieval and generation are network-bound calls that are "
        "comparatively fast.",
        "",
        "### Baseline Latency (where instrumented)",
        "",
        "| Baseline | N instrumented | Mean Latency |",
        "|---|---|---|",
    ]
    for name in ("ragas", "ragchecker", "ares"):
        entry = baselines.get(name, {})
        n_inst = entry.get("n_instrumented_examples", 0)
        n_total = entry.get("n_total_examples", "?")
        mean_ms = entry.get("mean_latency_ms")
        note_str = f" (of {n_total} total)" if n_inst != n_total else ""
        lines.append(f"| {name.upper()} | {n_inst}{note_str} | {ms(mean_ms)} |")

    lines += [
        "",
        "> **Baseline latency note:** Examples run before the latency instrumentation columns "
        "(`ragas_latency_ms`, `ragchecker_latency_ms`, `ares_latency_ms`) were added to the "
        "results schema show 0 instrumented examples. Re-running those examples with the "
        "updated `run_baseline_comparison.py` will populate real wall-clock data.",
        "",
        "### LLM Call Approximation (static estimate, not measured)",
        "",
    ]
    lines += [f"| Framework | Approx. LLM calls/example |", "|---|---|"] + [
        f"| {fw} | {calls} |" for fw, calls in APPROX_LLM_CALLS.items()
    ]
    return "\n".join(lines)


def render_failures_section(failures_log_path: str) -> str:
    if not os.path.exists(failures_log_path):
        return "_No failures recorded._"

    with open(failures_log_path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return "_No failures recorded._"

    by_baseline = defaultdict(list)
    pattern = re.compile(r"eval_id=(\S+) baseline=(\S+): (.*)")
    for line in lines:
        match = pattern.search(line)
        if match:
            eval_id, baseline, error = match.groups()
            by_baseline[baseline].append((eval_id, error.strip()))

    if not by_baseline:
        return "_No failures recorded._"

    sections = []
    for baseline, failures in sorted(by_baseline.items()):
        sections.append(f"**{baseline}** -- {len(failures)} failure(s):")
        for eval_id, error in failures[:10]:
            sections.append(f"  - eval_id `{eval_id}`: {error[:200]}")
    return "\n".join(sections)


def generate_report(
    results: List[Dict[str, Any]],
    correlations: Dict[str, Any],
    agreement: Dict[str, Any],
    disagreements: List[Dict[str, str]],
    failures_log_path: str,
    diagnostic_accuracy: Optional[Dict[str, Any]] = None,
    aggregation_ablation: Optional[Dict[str, Any]] = None,
    confidence_calibration: Optional[Dict[str, Any]] = None,
    reasoning_consistency: Optional[Dict[str, Any]] = None,
    cost_latency: Optional[Dict[str, Any]] = None,
) -> str:
    n = len(results)
    return f"""# X-RAG vs. RAGAS / RAGChecker / ARES -- Benchmark Comparison Report

This report compares the X-RAG Diagnostic Framework against three established
RAG evaluation baselines (RAGAS, RAGChecker, ARES's `ues_idp` lightweight
LLM-judge mode) over a {n}-example labeled evaluation set
(`eval/eval_dataset.csv`), covering healthy pipeline runs and five injected
failure modes (`MISSING_CORPUS`, `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`,
`UNSUPPORTED_GENERATION`, `GROUNDING_FAILURE`).

## Methodology Notes

- All three baselines' LLM-judge calls are routed through Hugging Face's
  OpenAI-compatible Inference Providers router, reusing this project's own
  `HF_TOKEN` -- no separate OpenAI/Bedrock key was used, keeping the
  comparison free to reproduce.
- ARES was run in **`ues_idp` mode** (a few-shot LLM-judge, no classifier
  training) rather than **`ppi` mode** (fine-tuned classifiers + a larger
  human-labeled gold set for statistically-corrected confidence intervals).
  This is a deliberate scope decision: `ppi` mode's training/annotation
  requirements are disproportionate to a {n}-example evaluation set.
- `ragchecker` and `ares-ai` each required their own separate Python 3.10
  virtual environment, isolated both from this project's main Python 3.13
  venv and from each other (their pinned dependencies conflict). See
  `docs/RESEARCH_LOG.md` for the full environment investigation -- no Rust or
  C/C++ compiler installation was ultimately required for either.

## 1. Summary

{render_summary_table(results)}

## 2. Correlation

{render_correlation_section(correlations)}

## 3. Agreement (Failure / No-Failure Classification)

{render_agreement_section(agreement)}

## 4. Where X-RAG Adds Value Beyond Aggregate Scores

{render_value_add_section(disagreements)}

## 5. Threats to Validity: Failures Encountered

{render_failures_section(failures_log_path)}

## 6. Diagnostic Accuracy (Root Cause Attribution)

Sections 1-5 above ask whether X-RAG's *scores* track the baselines'. This
section asks the more important question for a diagnostic tool: is X-RAG's
*diagnosis* actually correct? Compares `xrag_primary_cause` against
`eval/eval_dataset.csv`'s own `expected_failure_type` label for each example
(healthy examples have an implicit expected label of `UNKNOWN`, matching
X-RAG's own "no failure detected" vocabulary).

{render_diagnostic_accuracy_section(diagnostic_accuracy)}

## 7. Aggregation Strategy Ablation

Tests whether the choice of evidence-aggregation strategy (`top1` /
`max_pool_top3` / `concat_top3`) materially affects diagnostic accuracy.
All three strategies are evaluated in a single pass (the expensive NLI
scoring is done once; only the combination rule varies). See
`scripts/ablate_aggregation_strategy.py` for full methodology.

{render_aggregation_ablation_section(aggregation_ablation)}

## 8. Confidence Calibration

Checks whether `RootCauseAnalysis.diagnosis_confidence` is well-calibrated:
when X-RAG reports confidence=0.93, is it actually correct ~93% of the time?
Bins examples by confidence and reports empirical accuracy per bin plus
Expected Calibration Error (ECE). See
`scripts/evaluate_confidence_calibration.py`.

{render_confidence_calibration_section(confidence_calibration)}

## 9. Reasoning Chain Validation

Diagnostic accuracy (#6) checks whether the final *label* is correct. This
section checks whether the *reasoning* is self-consistent (automated checks)
and provides the human-rating worksheet for independent validation. See
`scripts/evaluate_reasoning_consistency.py`.

{render_reasoning_validation_section(reasoning_consistency)}

## 10. Cost / Latency Tradeoff

Turns the previously-qualitative cost comparison into real measured numbers.
X-RAG's per-stage latency has always been captured; baseline wall-clock
latency requires the instrumented `run_baseline_comparison.py` columns
(`ragas_latency_ms` etc.). See `scripts/evaluate_cost_latency.py`.

{render_cost_latency_section(cost_latency)}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate the paper-ready benchmark comparison report.")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    results = load_json(os.path.join(args.results_dir, "results.json"), default=[])
    correlations = load_json(os.path.join(args.results_dir, "correlations.json"), default={})
    agreement = load_json(os.path.join(args.results_dir, "agreement.json"), default={})
    disagreements = load_csv_rows(os.path.join(args.results_dir, "disagreements.csv"))
    failures_log_path = os.path.join(args.results_dir, "failures.log")
    diagnostic_accuracy = load_json(os.path.join(args.results_dir, "diagnostic_accuracy.json"), default={})
    aggregation_ablation = load_json(os.path.join(args.results_dir, "aggregation_strategy_ablation.json"), default={})
    confidence_calibration = load_json(os.path.join(args.results_dir, "confidence_calibration.json"), default={})
    reasoning_consistency = load_json(os.path.join(args.results_dir, "reasoning_consistency.json"), default={})
    cost_latency = load_json(os.path.join(args.results_dir, "cost_latency.json"), default={})

    report = generate_report(
        results, correlations, agreement, disagreements, failures_log_path,
        diagnostic_accuracy=diagnostic_accuracy,
        aggregation_ablation=aggregation_ablation,
        confidence_calibration=confidence_calibration,
        reasoning_consistency=reasoning_consistency,
        cost_latency=cost_latency,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    report_path = os.path.join(args.results_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
