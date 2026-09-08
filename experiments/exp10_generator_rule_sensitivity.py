"""
E10 -- GENERATOR-stage rule sensitivity: any-unsupported-claim vs. a
fraction-based threshold, scored against both E1's synthetic ground truth
and E8's live ground truth.

The question this answers
--------------------------
E8 (exp08_live_diagnostic_accuracy) found live diagnostic accuracy far below
E1's 95.6% synthetic figure (22.5% overall, 64% false-positive rate on
healthy questions). Digging into individual rows found the dominant cause:
``PipelineStateAnalyzer``'s GENERATOR stage fails on *any* single unsupported
claim (``some_unsupported = len(unsupported_verifications) > 0``), with no
tolerance for the claim-level noise that is unavoidable at realistic claim
counts (E3's own data: mean 5.87 claims per substantive answer) -- citation
claims like "these three modes are set out in Section 45" are true but
structurally unverifiable against body-text NLI, and get counted the same as
a genuine hallucination.

The obvious fix -- fail only when *more than a fraction* of claims are
unsupported -- cannot simply be adopted, because E1's own synthetic
DILUTE_CONTEXT fault is constructed with as few as 1 unsupported claim added
to a base of up to 8 supported claims (unsupported_fraction as low as ~0.11),
and a fraction threshold set high enough to absorb the live noise could
silently blind E1's own detection of a real generator fault.

This experiment makes that tradeoff visible rather than guessing at one
setting: sweep the fraction threshold and report BOTH accuracies at each
point --

    E1 single-fault accuracy (102 synthetic cases, re-diagnosed under each
    setting -- same cases E1 already validated, at seed 20260801)

    E8 live accuracy (the same 40 real rows E8 scored, re-diagnosed under
    each setting by reconstructing the real PipelineStateAnalyzer's inputs
    from already-collected artifacts -- no new pipeline or LLM calls)

so the current setting (None -- any unsupported claim) can be read as one
point on a real curve, not defended or attacked in isolation.

Reconstruction, not re-simulation
-----------------------------------
The live side does not re-run generation, claim decomposition, or NLI. It
reloads the already-saved ``RAGTrace`` (``artifacts/rag_traces/``, which
carries the real retrieval scores and corpus-distance signal CORPUS/
RETRIEVER/CHUNKING need) and reconstructs a ``VerificationSummary`` from the
already-saved diagnostic report's ``evidence_analysis`` (which carries the
real, live ``verification_status`` per claim -- the ground truth signal that
matters). Only the GENERATOR-stage NLI numeric scores (entailment/
contradiction/neutral) are not preserved in the saved report and are
defaulted to 0.0; they are not read by any PipelineStateAnalyzer stage logic
(confirmed by inspection -- only ``verification_status``, ``claim_id``, and
``best_chunk_id`` are used), so this default has no effect on any stage's
verdict. The real ``PipelineStateAnalyzer`` and ``RootCauseReasoner`` classes
are called unmodified; nothing about their logic is duplicated here.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional

from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    accuracy,
    load_eval_dataset,
    wilson_interval,
)
from experiments.exp05_diagnostic_agreement import load_real_rows
from experiments.exp08_live_diagnostic_accuracy import _ground_truth_cause
from experiments.faults import ALL_FAULTS, FaultType, build_case, diagnose
from src.claim_verifier import VerificationResult, VerificationStatus, VerificationSummary
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.rag_trace import RAGTrace
from src.root_cause_reasoner import FailureType, RootCauseReasoner

logger = logging.getLogger(__name__)

#: The sweep. None reproduces the exact current (unmodified) behavior --
#: always included first so it is the reference point every other setting is
#: compared against, not just another entry in a list.
FRACTION_SETTINGS: List[Optional[float]] = [None, 0.15, 0.25, 0.34, 0.5, 0.75]

#: Same base-scenario count as E1's single-fault arm, same ids, same seed --
#: this experiment re-diagnoses E1's own cases, not a fresh sample.
BASES_PER_FAULT = 17
NEAR_THRESHOLD_EVERY = 4


def _setting_label(fraction: Optional[float]) -> str:
    return "any_unsupported" if fraction is None else f"frac_gt_{fraction:.2f}"


def _load_trace(trace_id: str) -> Optional[RAGTrace]:
    matches = glob.glob(f"artifacts/rag_traces/**/trace_{trace_id}.json", recursive=True)
    if not matches:
        return None
    with open(matches[0], encoding="utf-8") as f:
        return RAGTrace.from_json(f.read())


def _reconstruct_verification(trace_id: str) -> Optional[VerificationSummary]:
    """Rebuild a VerificationSummary from the saved diagnostic report's
    evidence_analysis. Only verification_status, claim_id, and best_chunk_id
    are read by PipelineStateAnalyzer -- everything else is a harmless
    placeholder, documented in the module docstring above."""
    report_path = os.path.join("artifacts", "reports", f"{trace_id}.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    results = []
    for i, claim in enumerate(report.get("evidence_analysis", [])):
        results.append(VerificationResult(
            verification_id=f"reconstructed-{trace_id}-{i}",
            trace_id=trace_id,
            claim_id=claim.get("claim_id", f"claim-{i}"),
            claim_text=claim.get("claim_text", ""),
            verification_status=VerificationStatus(claim["verification_status"]),
            verification_reason="reconstructed from saved report for E10",
            confidence=0.0,
            best_chunk_id=claim.get("supporting_chunk_id"),
            best_chunk_rank=claim.get("supporting_chunk_rank"),
            best_chunk_score=None,
            best_sentence_id=None,
            evidence_text=claim.get("supporting_evidence"),
            entailment_score=0.0,
            contradiction_score=0.0,
            neutral_score=0.0,
        ))

    total = len(results)
    return VerificationSummary(
        trace_id=trace_id,
        total_claims=total,
        supported_claims=sum(1 for r in results if r.verification_status == VerificationStatus.SUPPORTED),
        partially_supported_claims=sum(1 for r in results if r.verification_status == VerificationStatus.PARTIALLY_SUPPORTED),
        contradicted_claims=sum(1 for r in results if r.verification_status == VerificationStatus.CONTRADICTED),
        unsupported_claims=sum(1 for r in results if r.verification_status == VerificationStatus.UNSUPPORTED),
        not_verifiable_claims=sum(1 for r in results if r.verification_status == VerificationStatus.NOT_VERIFIABLE),
        average_entailment_score=0.0,
        total_verification_latency_ms=0.0,
        results=results,
    )


class GeneratorRuleSensitivityExperiment(Experiment):
    key = "exp10_generator_rule_sensitivity"
    number = 10
    title = "GENERATOR-stage rule sensitivity: any-unsupported vs. fraction threshold"
    claim = (
        "The current any-unsupported-claim GENERATOR rule is one point on a real "
        "accuracy tradeoff between E1's synthetic detection and E8's live false-positive "
        "rate; no single fraction threshold dominates on both."
    )
    supported_modes = ("offline", "live")  # performs no new pipeline/LLM calls either way

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        specs: List[ExampleSpec] = []
        for fraction in FRACTION_SETTINGS:
            label = _setting_label(fraction)
            for fault in ALL_FAULTS:
                for base in range(BASES_PER_FAULT):
                    specs.append(ExampleSpec(
                        example_id=f"{label}/e1/{fault.value}/base{base:02d}",
                        payload={
                            "source": "e1",
                            "fraction": fraction,
                            "fault": fault.value,
                            "base": base,
                            "near_threshold": base % NEAR_THRESHOLD_EVERY == 0,
                        },
                    ))
            for row in load_real_rows():
                eid = row.get("eval_id")
                specs.append(ExampleSpec(
                    example_id=f"{label}/live/{eid}",
                    payload={"source": "live", "fraction": fraction, "row": row},
                ))
        return specs

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload
        fraction = payload["fraction"]
        analyzer = PipelineStateAnalyzer(generator_fail_min_unsupported_fraction=fraction)

        if payload["source"] == "e1":
            # Same id shape and seed salt as E1 itself, so this is the exact
            # same case E1 already validated -- not a fresh random draw.
            base_example_id = f"single/{payload['fault']}/base{payload['base']:02d}"
            rng = ctx.rng_for(f"e1:{base_example_id}")
            case = build_case(
                base_id=base_example_id.replace("/", "-"),
                faults=[FaultType(payload["fault"])],
                rng=rng,
                near_threshold=payload["near_threshold"],
                preserve_upstream=True,
            )
            _, rca = diagnose(case, analyzer=analyzer)
            predicted = rca.primary_cause.value
            expected = case.expected_primary_cause
            return {
                "setting": _setting_label(fraction),
                "fraction": fraction,
                "source": "e1",
                "expected_cause": expected,
                "predicted_cause": predicted,
                "correct": predicted == expected,
            }

        row = payload["row"]
        trace_id = row.get("trace_id")
        expected = _ground_truth_cause(row)
        trace = _load_trace(trace_id) if trace_id else None
        verification = _reconstruct_verification(trace_id) if trace_id else None

        if trace is None or verification is None:
            # Missing artifact for this row (should not happen given the
            # coverage check run before this experiment was written, but
            # fail loud rather than silently mis-scoring on absence).
            return {
                "setting": _setting_label(fraction),
                "fraction": fraction,
                "source": "live",
                "eval_id": row.get("eval_id"),
                "expected_cause": expected,
                "predicted_cause": None,
                "correct": False,
                "reconstruction_failed": True,
            }

        psm = analyzer.analyze(trace, None, verification)
        rca = RootCauseReasoner().analyze(psm)
        predicted = rca.primary_cause.value
        return {
            "setting": _setting_label(fraction),
            "fraction": fraction,
            "source": "live",
            "eval_id": row.get("eval_id"),
            "expected_cause": expected,
            "predicted_cause": predicted,
            "correct": predicted == expected,
            "reconstruction_failed": False,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        by_setting: Dict[str, Any] = {}
        for fraction in FRACTION_SETTINGS:
            label = _setting_label(fraction)
            e1_rows = [r for r in records if r["setting"] == label and r["source"] == "e1"]
            live_rows = [r for r in records if r["setting"] == label and r["source"] == "live"]
            live_failures = [r for r in live_rows if r["expected_cause"] != FailureType.UNKNOWN.value]
            live_healthy = [r for r in live_rows if r["expected_cause"] == FailureType.UNKNOWN.value]

            e1_correct = sum(1 for r in e1_rows if r["correct"])
            live_correct = sum(1 for r in live_rows if r["correct"])
            live_fp = sum(1 for r in live_healthy if not r["correct"])

            by_setting[label] = {
                "fraction": fraction,
                "e1_single_fault_accuracy": e1_correct / len(e1_rows) if e1_rows else None,
                "e1_n": len(e1_rows),
                "live_overall_accuracy": live_correct / len(live_rows) if live_rows else None,
                "live_attribution_accuracy_on_labeled_failures": (
                    accuracy([r["expected_cause"] for r in live_failures], [r["predicted_cause"] for r in live_failures])
                    if live_failures else None
                ),
                "live_false_positive_rate_on_healthy": live_fp / len(live_healthy) if live_healthy else None,
                "live_n": len(live_rows),
                "reconstruction_failures": sum(1 for r in live_rows if r.get("reconstruction_failed")),
            }

        reference = by_setting[_setting_label(None)]
        best_live = max(
            (s for s in by_setting.values() if s["live_overall_accuracy"] is not None),
            key=lambda s: s["live_overall_accuracy"],
        )
        e1_preserved = [
            label for label, s in by_setting.items()
            if s["e1_single_fault_accuracy"] is not None and s["e1_single_fault_accuracy"] >= reference["e1_single_fault_accuracy"] - 1e-9
        ]

        return {
            "headline": {
                "reference_setting": _setting_label(None),
                "reference_e1_accuracy": reference["e1_single_fault_accuracy"],
                "reference_live_accuracy": reference["live_overall_accuracy"],
                "best_live_accuracy_setting": _setting_label(best_live["fraction"]),
                "best_live_accuracy": best_live["live_overall_accuracy"],
                "settings_that_do_not_regress_e1": e1_preserved,
            },
            "by_setting": by_setting,
            "interpretation_notes": [
                "e1_single_fault_accuracy is scored against the exact 102 cases E1's own "
                "single-fault arm validates (same seed, same ids) -- a setting that drops "
                "this number is regressing an already-published result, not a free variable.",
                "live_* numbers are reconstructed from already-collected artifacts (no new "
                "pipeline or LLM calls); see the module docstring for exactly what is "
                "reconstructed vs. defaulted, and why the defaults cannot affect the verdict.",
                "A setting that improves live accuracy without lowering e1_single_fault_accuracy "
                "is a candidate for adoption; a setting that improves live accuracy only by also "
                "lowering e1_single_fault_accuracy is evidence the rule is fundamentally a "
                "precision/recall tradeoff at this claim-count scale, not a bug with one correct fix.",
            ],
        }


EXPERIMENT = GeneratorRuleSensitivityExperiment()
