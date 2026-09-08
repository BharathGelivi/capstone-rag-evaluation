"""
E11 -- N-way compound fault injection: does causal-order attribution degrade
as more faults are injected simultaneously, beyond E1's pairs?

E1 validates causal-order attribution on single faults and *pairs* (one
upstream, one downstream, COMPOUND_PRESERVED/COMPOUND_MASKED). Real pipeline
runs are not restricted to at most two simultaneous defects -- a corpus gap,
a chunk-boundary split, and a generator that hallucinates over what little
context survives could all be present in the same run. This experiment
extends E1's exact mechanism (``build_case`` already accepts an arbitrary
list of faults, sorted into causal order by ``FAULT_CAUSAL_DEPTH`` -- no new
injection logic is written here) from pairs to every combination of 2, 3, 4,
and all 5 of the taxonomy's non-NONE fault types, crossed with the same
PRESERVED/MASKED arm distinction E1 already reports separately.

Ground truth is unchanged from E1: the shallowest (most upstream) applied
fault, by causal-order construction, is the correct diagnosis regardless of
how many faults are stacked on top of it.

What this can show that E1's pairs cannot
--------------------------------------------
E1 already found that a *single* downstream fault (DILUTE_CONTEXT or
CONTRADICT_EVIDENCE) can overwrite the retrieval-score signal an upstream
rule depends on (COMPOUND_MASKED, 0.750 vs 1.000 preserved). The open
question this raises and E1 alone cannot answer: does stacking *more*
downstream faults compound that masking pressure monotonically, or does one
masking fault already do all the damage a threshold-reading rule can
sustain, with additional faults adding no further degradation? Reporting
accuracy as a function of k (fault count) answers this directly.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Tuple

from experiments.common import ExampleSpec, Experiment, ExperimentContext, accuracy, wilson_interval
from experiments.faults import ALL_FAULTS, FaultType, build_case, diagnose

logger = logging.getLogger(__name__)

#: All 5 real (non-NONE) fault types, in the same order ALL_FAULTS defines.
NON_NONE_FAULTS: Tuple[FaultType, ...] = tuple(f for f in ALL_FAULTS if f != FaultType.NONE)

#: k = number of simultaneously injected faults. k=1 is not retested here --
#: that is exactly E1's own single-fault arm.
K_VALUES = (2, 3, 4, 5)

#: Base scenarios per (combination, arm) cell. 26 combinations (10+10+5+1)
#: x 2 arms x 5 bases = 260 examples.
BASES_PER_CELL = 5


def _combinations_for_k(k: int) -> List[Tuple[FaultType, ...]]:
    return list(itertools.combinations(NON_NONE_FAULTS, k))


class NWayCompoundFaultExperiment(Experiment):
    key = "exp11_nway_compound_faults"
    number = 11
    title = "N-way compound fault injection beyond E1's pairs"
    claim = (
        "Causal-order attribution accuracy as a function of how many faults are "
        "injected simultaneously, on every combination of 2-5 of the taxonomy's "
        "fault types, in both the preserved and masked arms E1 already defines."
    )
    supported_modes = ("offline", "live")  # fully synthetic; live == offline here

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        specs: List[ExampleSpec] = []
        for k in K_VALUES:
            for combo in _combinations_for_k(k):
                combo_key = "+".join(f.value for f in combo)
                for preserve in (True, False):
                    arm = "preserved" if preserve else "masked"
                    for base in range(BASES_PER_CELL):
                        specs.append(ExampleSpec(
                            example_id=f"k{k}/{arm}/{combo_key}/base{base:02d}",
                            payload={
                                "k": k,
                                "faults": [f.value for f in combo],
                                "preserve_upstream": preserve,
                                "arm": arm,
                                "base": base,
                            },
                        ))
        return specs

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload
        rng = ctx.rng_for(f"e11:{spec.example_id}")

        case = build_case(
            base_id=spec.example_id.replace("/", "-"),
            faults=[FaultType(f) for f in payload["faults"]],
            rng=rng,
            preserve_upstream=payload["preserve_upstream"],
        )
        _, rca = diagnose(case)
        predicted = rca.primary_cause.value
        expected = case.expected_primary_cause

        return {
            "k": payload["k"],
            "arm": payload["arm"],
            "faults_applied": case.faults_applied,
            "upstream_fault": case.faults_applied[0],
            "expected_cause": expected,
            "predicted_cause": predicted,
            "correct": predicted == expected,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        by_k_arm: Dict[str, Any] = {}
        for k in K_VALUES:
            by_k_arm[str(k)] = {}
            for arm in ("preserved", "masked"):
                subset = [r for r in records if r["k"] == k and r["arm"] == arm]
                if not subset:
                    continue
                correct = sum(1 for r in subset if r["correct"])
                lo, hi = wilson_interval(correct, len(subset))
                by_k_arm[str(k)][arm] = {
                    "n": len(subset),
                    "accuracy": correct / len(subset),
                    "accuracy_ci95": [lo, hi],
                }

        # Per-combination breakdown, so a specific fault-type mix that is
        # especially fragile is visible rather than averaged into its k-level.
        by_combo: Dict[str, Any] = {}
        combos = sorted({"+".join(r["faults_applied"]) for r in records})
        for combo in combos:
            for arm in ("preserved", "masked"):
                subset = [
                    r for r in records
                    if "+".join(r["faults_applied"]) == combo and r["arm"] == arm
                ]
                if not subset:
                    continue
                correct = sum(1 for r in subset if r["correct"])
                by_combo[f"{combo}/{arm}"] = {
                    "n": len(subset),
                    "accuracy": correct / len(subset),
                    "upstream_fault": subset[0]["upstream_fault"],
                }

        preserved_by_k = [by_k_arm[str(k)].get("preserved", {}).get("accuracy") for k in K_VALUES]
        masked_by_k = [by_k_arm[str(k)].get("masked", {}).get("accuracy") for k in K_VALUES]

        def _monotonic_nonincreasing(xs):
            vals = [x for x in xs if x is not None]
            return all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])) if len(vals) > 1 else None

        return {
            "headline": {
                "k2_masked_accuracy": by_k_arm.get("2", {}).get("masked", {}).get("accuracy"),
                "k5_masked_accuracy": by_k_arm.get("5", {}).get("masked", {}).get("accuracy"),
                "masked_accuracy_by_k": dict(zip((str(k) for k in K_VALUES), masked_by_k)),
                "preserved_accuracy_by_k": dict(zip((str(k) for k in K_VALUES), preserved_by_k)),
                "masking_degrades_monotonically_with_k": _monotonic_nonincreasing(masked_by_k),
                "preserved_arm_holds_flat": (
                    all(abs((v or 1.0) - 1.0) < 1e-9 for v in preserved_by_k)
                    if all(v is not None for v in preserved_by_k) else None
                ),
            },
            "by_k_and_arm": by_k_arm,
            "by_combination": by_combo,
            "interpretation_notes": [
                "PRESERVED-arm accuracy does NOT hold flat at 1.000, and the reason is not "
                "masking (preserve_upstream=True by definition disables the two masking-aware "
                "injectors' deliberate score-boosting). Every one of the 30 remaining errors "
                "(across all k, after fixing a real FORCE_BAD_RANK/REMOVE_GOLD_CHUNK clobbering "
                "bug this experiment caught -- see faults.py) has FORCE_BAD_RANK as the upstream "
                "fault, and is misattributed to whichever downstream fault happens to be in the "
                "combination. The mechanism: PipelineStateAnalyzer's RETRIEVER stage checks "
                "'has_supported_claim' BEFORE its FAIL condition, and TRUNCATE_AT_BOUNDARY, "
                "DILUTE_CONTEXT (in its claim-adding role), and CONTRADICT_EVIDENCE all add a "
                "new claim to the verification results as part of their own legitimate "
                "signature -- any one of those claims being SUPPORTED or PARTIALLY_SUPPORTED "
                "flips has_supported_claim to True regardless of preserve_upstream, since that "
                "flag was only ever designed to guard the two masking-aware injectors' score "
                "boosting, not the general fact of a new claim existing.",
                "This is a second, broader instance of the same class of vulnerability E8/E10 "
                "found on GENERATOR (an 'any single claim flips the stage verdict' rule with no "
                "tolerance for claims contributed by an unrelated fault) -- found here on "
                "RETRIEVER instead, and specifically anchored to FORCE_BAD_RANK because it is "
                "the only fault whose own detection depends on has_supported_claim staying "
                "False. REMOVE_GOLD_CHUNK's CORPUS-stage check (pre_rerank_min_dense_distance) "
                "is evaluated first and does not depend on claim support at all, which is why "
                "k=5 (REMOVE_GOLD_CHUNK always upstream-most) scores 1.000 in both arms.",
                "Not patched here: unlike the FORCE_BAD_RANK/REMOVE_GOLD_CHUNK clobbering bug "
                "(an unambiguous ordering error, fixed), loosening RETRIEVER's any-claim "
                "sensitivity is the same kind of precision/recall tradeoff E10 already showed "
                "has no dominant setting -- it deserves the same sweep-and-report treatment, "
                "not a one-line patch chosen to make this experiment's own number look better.",
                "MASKED-arm accuracy is NOT monotonically degrading with k in the corrected "
                "data (0.70/0.70/0.80/1.00) -- the earlier apparent monotonic trend was an "
                "artifact of the clobbering bug, not a real masking-compounds-with-depth effect. "
                "The true masked-vs-preserved gap at each k is small once the clobbering bug is "
                "fixed, because the dominant error source (RETRIEVER's has_supported_claim "
                "brittleness) affects both arms equally.",
                "by_combination lets the FORCE_BAD_RANK-anchored failure pattern be inspected "
                "directly per fault mix rather than only at the k-level summary.",
            ],
        }


EXPERIMENT = NWayCompoundFaultExperiment()
