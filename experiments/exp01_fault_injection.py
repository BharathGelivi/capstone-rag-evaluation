"""
E1 -- Stage-attributed failure taxonomy with causal fault injection.

The paper's spine. Inject a fault whose causal stage is known by construction,
run the real diagnosis stack (``PipelineStateAnalyzer`` ->
``RootCauseReasoner``), and ask whether the diagnosis recovers the fault that
was injected. Synthetic ground truth rather than human-correlation, because
the causal question is exactly the one annotators are unreliable about.

Design of the fault library, including what this does and does not validate,
lives in ``experiments/faults.py``. Read that first.

Reported
--------
* Overall recovery accuracy with a Wilson interval.
* Accuracy per arm (SINGLE / COMPOUND_PRESERVED / COMPOUND_MASKED) -- the
  three arms test different things and pooling them would hide which.
* Per-fault precision / recall / F1 and a full confusion matrix, so a
  systematic confusion (e.g. MISSING_CORPUS read as RETRIEVAL_MISS) is
  visible rather than averaged away.
* Control-arm false-positive rate: how often a healthy run is assigned a cause.
* The **causal-precedence lift**: accuracy on compound cases under the real
  reasoner versus a confidence-ranking reasoner over the same matrices. This
  isolates the contribution of walking the causal chain from everything else
  and is the number that justifies the design.
* Near-threshold sensitivity: accuracy split by whether the decisive signal
  sat within a hair of its threshold.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

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
from experiments.faults import (
    ALL_FAULTS,
    COMPOUND_PAIRS,
    Arm,
    FaultType,
    build_case,
    diagnose,
)
from src.pipeline_state_analyzer import PipelineStatus
from src.root_cause_reasoner import FailureType, RootCauseReasoner

logger = logging.getLogger(__name__)

#: Base scenarios per single fault. 6 faults x 17 bases = 102 single-fault cases.
#: Raised from 8 (paper submission requires n>=100 with fault type as a
#: stratified variable rather than n>=100 pooled across a smaller per-type count).
BASES_PER_FAULT = 17
#: Base scenarios per compound pair, per arm. 4 pairs x 7 bases x 2 arms = 56.
#: Raised from 3 for the same reason -- COMPOUND_MASKED (the weak arm) needs
#: enough examples on its own to report a non-degenerate confidence interval.
BASES_PER_COMPOUND = 7
#: Every 4th base scenario is generated in near-threshold mode.
NEAR_THRESHOLD_EVERY = 4


def _confidence_ranked_cause(psm) -> str:
    """
    The ablation baseline: pick the highest-confidence failing stage instead of
    the earliest one in causal order.

    This is what ``RootCauseReasoner`` did before the causal-propagation
    rewrite (see its module docstring). Recomputing it from the same
    ``PipelineStateMatrix`` gives a paired comparison -- identical evidence,
    only the selection rule differs -- so the accuracy gap is attributable to
    the rule and nothing else.
    """
    failing = [s for s in psm.pipeline_states if s.status == PipelineStatus.FAIL]
    if not failing:
        return FailureType.UNKNOWN.value
    best = max(failing, key=lambda s: s.confidence)
    return RootCauseReasoner.STAGE_TO_FAILURE_MAP.get(best.stage, FailureType.UNKNOWN).value


class FaultInjectionExperiment(Experiment):
    key = "exp01_fault_injection"
    number = 1
    title = "Stage-attributed failure taxonomy with causal fault injection"
    claim = (
        "Injected pipeline faults are recovered by stage-attributed diagnosis, "
        "and causal-order selection beats confidence ranking on compound faults."
    )
    supported_modes = ("offline", "live")

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        specs: List[ExampleSpec] = []

        for fault in ALL_FAULTS:
            for base in range(BASES_PER_FAULT):
                specs.append(
                    ExampleSpec(
                        example_id=f"single/{fault.value}/base{base:02d}",
                        payload={
                            "faults": [fault.value],
                            "base": base,
                            "near_threshold": base % NEAR_THRESHOLD_EVERY == 0,
                            "preserve_upstream": True,
                        },
                    )
                )

        for upstream, downstream in COMPOUND_PAIRS:
            for base in range(BASES_PER_COMPOUND):
                for preserve in (True, False):
                    arm = "preserved" if preserve else "masked"
                    specs.append(
                        ExampleSpec(
                            example_id=(
                                f"compound-{arm}/{upstream.value}+{downstream.value}/base{base:02d}"
                            ),
                            payload={
                                "faults": [upstream.value, downstream.value],
                                "base": base,
                                "near_threshold": base % NEAR_THRESHOLD_EVERY == 0,
                                "preserve_upstream": preserve,
                            },
                        )
                    )

        return specs

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload
        # Seeded per example id, so an example's scenario is identical whether
        # it is the first one run or the first one resumed.
        rng = ctx.rng_for(f"e1:{spec.example_id}")

        case = build_case(
            base_id=spec.example_id.replace("/", "-"),
            faults=[FaultType(f) for f in payload["faults"]],
            rng=rng,
            near_threshold=payload["near_threshold"],
            preserve_upstream=payload["preserve_upstream"],
        )

        psm, rca = diagnose(case)
        predicted = rca.primary_cause.value
        expected = case.expected_primary_cause

        return {
            "arm": case.arm,
            "faults_applied": case.faults_applied,
            "injected_upstream_fault": case.faults_applied[0],
            "near_threshold": case.near_threshold,
            "expected_primary_cause": expected,
            "predicted_primary_cause": predicted,
            "correct": predicted == expected,
            "predicted_secondary_effects": [e.value for e in rca.secondary_effects],
            "diagnosis_confidence": rca.confidence,
            "confidence_ranked_baseline_cause": _confidence_ranked_cause(psm),
            "confidence_ranked_baseline_correct": _confidence_ranked_cause(psm) == expected,
            "stage_statuses": psm.summary(),
            "provenance": case.provenance,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        expected = [r["expected_primary_cause"] for r in records]
        predicted = [r["predicted_primary_cause"] for r in records]

        n_correct = sum(1 for r in records if r["correct"])
        lo, hi = wilson_interval(n_correct, len(records))

        by_arm: Dict[str, Any] = {}
        for arm in (a.value for a in Arm):
            subset = [r for r in records if r["arm"] == arm]
            if not subset:
                continue
            correct = sum(1 for r in subset if r["correct"])
            baseline_correct = sum(1 for r in subset if r["confidence_ranked_baseline_correct"])
            arm_lo, arm_hi = wilson_interval(correct, len(subset))
            by_arm[arm] = {
                "n": len(subset),
                "accuracy": correct / len(subset),
                "accuracy_ci95": [arm_lo, arm_hi],
                "confidence_ranked_baseline_accuracy": baseline_correct / len(subset),
                "causal_precedence_lift": (correct - baseline_correct) / len(subset),
            }

        by_fault: Dict[str, Any] = {}
        for fault in (f.value for f in ALL_FAULTS):
            subset = [r for r in records if r["injected_upstream_fault"] == fault]
            if not subset:
                continue
            correct = sum(1 for r in subset if r["correct"])
            by_fault[fault] = {
                "n": len(subset),
                "recovery_rate": correct / len(subset),
                "most_common_confusion": _most_common_wrong(subset),
            }

        controls = [r for r in records if r["expected_primary_cause"] == FailureType.UNKNOWN.value]
        false_positives = sum(1 for r in controls if not r["correct"])

        near = [r for r in records if r["near_threshold"]]
        far = [r for r in records if not r["near_threshold"]]

        compound = [r for r in records if r["arm"] != Arm.SINGLE.value]
        compound_correct = sum(1 for r in compound if r["correct"])
        compound_baseline = sum(1 for r in compound if r["confidence_ranked_baseline_correct"])

        return {
            "headline": {
                "overall_recovery_accuracy": n_correct / len(records) if records else None,
                "accuracy_ci95": [lo, hi],
                "macro_f1": macro_f1(expected, predicted),
                "control_false_positive_rate": (
                    false_positives / len(controls) if controls else None
                ),
                "causal_precedence_lift_on_compound_faults": (
                    (compound_correct - compound_baseline) / len(compound) if compound else None
                ),
            },
            "by_arm": by_arm,
            "by_injected_fault": by_fault,
            "per_cause_prf": per_label_prf(expected, predicted),
            "confusion_matrix": confusion_matrix(expected, predicted),
            "near_threshold_sensitivity": {
                "near_threshold_accuracy": accuracy(
                    [r["expected_primary_cause"] for r in near],
                    [r["predicted_primary_cause"] for r in near],
                ),
                "n_near": len(near),
                "comfortable_margin_accuracy": accuracy(
                    [r["expected_primary_cause"] for r in far],
                    [r["predicted_primary_cause"] for r in far],
                ),
                "n_far": len(far),
            },
            "ablation_confidence_ranked_reasoner": {
                "overall_accuracy": (
                    sum(1 for r in records if r["confidence_ranked_baseline_correct"]) / len(records)
                    if records else None
                ),
                "note": (
                    "Same PipelineStateMatrix, highest-confidence failing stage instead of "
                    "earliest in causal order. The gap is attributable to the selection rule alone."
                ),
            },
            "interpretation_notes": [
                "COMPOUND_MASKED is expected to be the weak arm: when a downstream fault "
                "overwrites an observable the upstream rule reads (a generator fault raising "
                "retrieval-score statistics), threshold-based attribution has no signal left to "
                "recover the upstream cause. Reported separately rather than pooled.",
                "This validates the attribution layer given faithful stage signals. It does not "
                "validate claim decomposition or NLI verification, which produce those signals "
                "in a live run.",
            ],
        }


def _most_common_wrong(records: List[Dict[str, Any]]) -> Any:
    wrong = [r["predicted_primary_cause"] for r in records if not r["correct"]]
    if not wrong:
        return None
    counts: Dict[str, int] = {}
    for cause in wrong:
        counts[cause] = counts.get(cause, 0) + 1
    label, count = max(counts.items(), key=lambda kv: kv[1])
    return {"cause": label, "count": count, "of_errors": len(wrong)}


EXPERIMENT = FaultInjectionExperiment()
