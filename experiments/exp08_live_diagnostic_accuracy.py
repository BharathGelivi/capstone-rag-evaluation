"""
E8 -- Live end-to-end diagnostic accuracy against hand-labeled real rows.

The gap this closes
--------------------
E1 (exp01_fault_injection) validates the *attribution layer* only: an
injected fault reaches ``RootCauseReasoner`` via a ``PipelineStateMatrix``
populated by construction, not by a live claim-decomposition-and-NLI-
verification pass over real model output. Section 3.6 / Section 7 of the
paper state this scope limit explicitly and name the open question: how
often does a live run's own upstream signal reach the reasoner as cleanly as
E1's constructed one?

This experiment answers that question directly, using data that already
exists: ``artifacts/benchmark_comparison/results.json`` (produced by E5's
live baseline-comparison run) carries, for all 40 real
``eval/eval_dataset.csv`` questions, both ``expected_failure_type`` (the
hand-authored ground-truth label for the 15 rows that are supposed to fail
in a specific way) and ``xrag_primary_cause`` (X-RAG's actual live diagnosis
on that question, produced by the real pipeline -- generation, claim
decomposition, NLI verification, PSM, RootCauseReasoner -- with no
construction involved). Comparing the two is the live-pipeline analogue of
E1's Table 1: accuracy is scored against a labeled ground truth, but the
label reaches the reasoner by surviving the real upstream stages E1
deliberately bypasses.

No new pipeline runs are performed here. This experiment is a pure
re-scoring of already-collected E5 data, kept as its own experiment (rather
than folded into E5's summary) because it answers a different question --
"is the live diagnosis correct" vs. E5's "do two diagnosers agree" -- and the
paper should be able to cite it independently.

Known constraint, stated rather than hidden
---------------------------------------------
The real-row sample is hard-capped at 40 (the size of
``eval/eval_dataset.csv``), of which only 15 carry a specific failure label
(the other 25 are "healthy" -- expected to pass, used here as the
false-positive check). Both counts are below ``MIN_EXAMPLES`` (50); this
experiment explicitly opts into ``allow_small_sample`` rather than padding
the plan with synthetic rows, for the same reason E5's real-row correlation
is reported at n=40 rather than diluted with injected rows: pooling would
answer a different question. The paper's Limitations section should carry
this n alongside the result.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    accuracy,
    confusion_matrix,
    macro_f1,
    per_label_prf,
    wilson_interval,
)
from experiments.exp05_diagnostic_agreement import load_real_rows
from src.root_cause_reasoner import FailureType


def _ground_truth_cause(row: Dict[str, Any]) -> str:
    """Map eval_dataset's expected_failure_type to a FailureType value.
    Blank/missing means the question is expected to pass cleanly."""
    label = (row.get("expected_failure_type") or "").strip()
    return label if label else FailureType.UNKNOWN.value


class LiveDiagnosticAccuracyExperiment(Experiment):
    key = "exp08_live_diagnostic_accuracy"
    number = 8
    title = "Live end-to-end diagnostic accuracy against hand-labeled real rows"
    claim = (
        "X-RAG's live diagnosis (real generation, claim decomposition, and NLI "
        "verification -- not a constructed PipelineStateMatrix) recovers the "
        "hand-labeled failure cause on real pipeline runs, not only on E1's "
        "synthetic injections."
    )
    #: This experiment re-scores already-collected E5 output; it performs no
    #: new pipeline calls regardless of mode.
    supported_modes = ("offline", "live")

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        rows = load_real_rows()
        if not rows:
            raise RuntimeError(
                "exp08: artifacts/benchmark_comparison/results.json not found or "
                "empty. Run E5's baseline comparison first -- exp08 re-scores its "
                "output rather than calling the pipeline itself."
            )
        return [
            ExampleSpec(example_id=f"real/{row.get('eval_id', i)}", payload={"row": row})
            for i, row in enumerate(rows)
        ]

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        row = spec.payload["row"]
        true_cause = _ground_truth_cause(row)
        xrag_cause = row.get("xrag_primary_cause") or FailureType.UNKNOWN.value
        return {
            "eval_id": row.get("eval_id"),
            "question": row.get("question"),
            "true_cause": true_cause,
            "xrag_cause": xrag_cause,
            "is_labeled_failure": true_cause != FailureType.UNKNOWN.value,
            "correct": true_cause == xrag_cause,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        failures = [r for r in records if r["is_labeled_failure"]]
        healthy = [r for r in records if not r["is_labeled_failure"]]

        truth_all = [r["true_cause"] for r in records]
        pred_all = [r["xrag_cause"] for r in records]

        detection_rate = (
            sum(1 for r in failures if r["xrag_cause"] != FailureType.UNKNOWN.value) / len(failures)
            if failures else None
        )
        attribution_accuracy = accuracy(
            [r["true_cause"] for r in failures], [r["xrag_cause"] for r in failures]
        ) if failures else None

        false_positives = sum(1 for r in healthy if r["xrag_cause"] != FailureType.UNKNOWN.value)
        false_positive_rate = false_positives / len(healthy) if healthy else None
        fpr_ci = wilson_interval(false_positives, len(healthy)) if healthy else (None, None)

        return {
            "headline": {
                "n_real_rows": len(records),
                "n_labeled_failures": len(failures),
                "n_healthy": len(healthy),
                "detection_rate": detection_rate,
                "attribution_accuracy_on_labeled_failures": attribution_accuracy,
                "false_positive_rate_on_healthy": false_positive_rate,
                "false_positive_rate_95ci": fpr_ci,
                "overall_accuracy_all_40_rows": accuracy(truth_all, pred_all),
                "macro_f1": macro_f1(truth_all, pred_all),
            },
            "per_label_prf": per_label_prf(truth_all, pred_all),
            "confusion_truth_rows_vs_xrag_columns": confusion_matrix(truth_all, pred_all),
            "comparison_to_e1": (
                "E1 (synthetic, attribution layer only) recovers 95.6% of 158 "
                "injected faults. This experiment scores the same reasoner against "
                "live pipeline output on the 15 hand-labeled real failure rows -- "
                "compare attribution_accuracy_on_labeled_failures against E1's 1.000 "
                "single-fault figure to see how much accuracy the live "
                "claim-decomposition/NLI stages cost relative to a perfectly "
                "populated synthetic matrix."
            ),
            "interpretation_notes": [
                "n=15 labeled failures / n=25 healthy rows, both far smaller than "
                "E1's n=158 -- report every number here with its small-n caveat, "
                "not as a replacement for E1's headline figure.",
                "This experiment performs no new pipeline calls; it re-scores "
                "artifacts/benchmark_comparison/results.json, which was produced by "
                "E5's live run. If that file is regenerated, re-run this experiment "
                "with --force to refresh the scoring against the new rows.",
            ],
        }


EXPERIMENT = LiveDiagnosticAccuracyExperiment()
