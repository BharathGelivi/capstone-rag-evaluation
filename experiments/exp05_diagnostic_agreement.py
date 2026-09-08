"""
E5 -- Diagnostic agreement: high score correlation, low cause-level kappa.

The critique
------------
Cross-framework validation in the RAG evaluation literature is almost always
reported as a correlation between scalar scores. Two frameworks correlate at
r = 0.8 on faithfulness, and that is offered as evidence they measure the same
thing. It is not. Correlation on *how bad* a run was says nothing about
agreement on *what went wrong*, and "what went wrong" is the only output an
engineer can act on.

This experiment reports both numbers side by side over the same examples:

    Pearson / Spearman r on the scalar scores   -- expected: high
    Cohen's kappa on the attributed cause       -- expected: low

If that pattern holds, the scalar agreement literature is measuring
redundancy, not validity, and a framework can look "validated" while
disagreeing with every other framework about every actionable conclusion.

Method
------
Two diagnosers over the same traces:

*X-RAG* -- this framework's ``RootCauseReasoner`` verdict.

*Baseline-rule* -- what a practitioner actually does with RAGAS/RAGChecker
numbers: threshold them into a stage attribution. Low context precision means
retrieval; high hallucination with good context means generation; and so on.
The rule is written out explicitly in ``BASELINE_ATTRIBUTION_RULE`` rather than
left implicit, so it can be argued with. This is a *steelman* of the baselines,
not a strawman: it grants them the localisation their scalar outputs cannot
express on their own.

Two record sources, kept separate and labeled in the output:

``real_baseline``
    Rows from ``artifacts/benchmark_comparison/results.json`` -- genuine
    RAGAS / RAGChecker / ARES scores against genuine X-RAG verdicts.

``injected``
    E1's fault-injection cases, where the *true* cause is known. Baseline-style
    faithfulness and hallucination are recomputed from the same verification
    summary using each metric's own published definition (supported / total,
    contradicted / total) -- a reimplementation of the definition, not a
    fabricated score, and flagged as such. These rows add what the real rows
    cannot: a ground truth to score both diagnosers against, so the experiment
    reports not only that the two disagree but which one is right.

Scalar correlation is computed only over ``real_baseline`` rows, because
correlating X-RAG against a metric recomputed from X-RAG's own verification
output would be circular. The kappa is computed over both, reported separately.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    accuracy,
    binary_cohens_kappa,
    confusion_matrix,
    multiclass_cohens_kappa,
    pearson_r,
    spearman_rho,
)
from experiments.faults import FaultType, build_case, diagnose
from src.root_cause_reasoner import FailureType

logger = logging.getLogger(__name__)

RESULTS_JSON = os.path.join("artifacts", "benchmark_comparison", "results.json")

#: Thresholds for the baseline attribution rule. Chosen to match the defaults
#: already used in scripts/analyze_agreement.py so the two analyses cannot
#: diverge on threshold choice alone.
FAITHFULNESS_FAIL_BELOW = 0.7
HALLUCINATION_FAIL_ABOVE = 0.5
CONTEXT_FAIL_BELOW = 0.5

BASELINE_ATTRIBUTION_RULE = """
Given RAGAS/RAGChecker scalars, attribute a stage the way a practitioner would:

  context_precision or context_recall < 0.5           -> RETRIEVAL_MISS
  else hallucination > 0.5                            -> GROUNDING_FAILURE
  else faithfulness < 0.7                             -> UNSUPPORTED_GENERATION
  else                                                -> UNKNOWN (no failure)

Retrieval is checked first for the same reason X-RAG walks its stages in causal
order: a retrieval failure makes the generation numbers meaningless, so reading
them first would attribute the symptom.
""".strip()


def baseline_attribution(row: Dict[str, Any]) -> Optional[str]:
    """
    Turn scalar baseline metrics into a stage attribution. Returns None when
    the row carries no usable baseline scores at all -- an absent judgement is
    not agreement, and must not be counted as one.
    """
    context = _first_not_none(
        row.get("ragchecker_context_precision"),
        row.get("ragas_context_precision"),
        row.get("ragas_context_recall"),
        row.get("ares_context_relevance"),
    )
    hallucination = row.get("ragchecker_hallucination")
    faithfulness = _first_not_none(
        row.get("ragas_faithfulness"),
        row.get("ragchecker_faithfulness"),
        row.get("ares_answer_faithfulness"),
    )

    if context is None and hallucination is None and faithfulness is None:
        return None

    if context is not None and context < CONTEXT_FAIL_BELOW:
        return FailureType.RETRIEVAL_MISS.value
    if hallucination is not None and hallucination > HALLUCINATION_FAIL_ABOVE:
        return FailureType.GROUNDING_FAILURE.value
    if faithfulness is not None and faithfulness < FAITHFULNESS_FAIL_BELOW:
        return FailureType.UNSUPPORTED_GENERATION.value
    return FailureType.UNKNOWN.value


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _flags_failure(cause: Optional[str]) -> Optional[bool]:
    if cause is None:
        return None
    return cause not in ("", FailureType.UNKNOWN.value)


def load_real_rows(path: str = RESULTS_JSON) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else []


class DiagnosticAgreementExperiment(Experiment):
    key = "exp05_diagnostic_agreement"
    number = 5
    title = "Diagnostic agreement: high score correlation, low cause-level kappa"
    claim = (
        "Frameworks that correlate strongly on scalar faithfulness scores agree barely "
        "better than chance on which pipeline stage failed."
    )

    #: Injected cases added so the study has ground truth and clears the sample
    #: floor even when the real benchmark run is partial.
    N_INJECTED = 36

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        specs = [
            ExampleSpec(
                example_id=f"real/{row.get('eval_id', i)}",
                payload={"source": "real_baseline", "row": row},
            )
            for i, row in enumerate(load_real_rows())
        ]

        faults = [
            FaultType.NONE,
            FaultType.REMOVE_GOLD_CHUNK,
            FaultType.FORCE_BAD_RANK,
            FaultType.TRUNCATE_AT_BOUNDARY,
            FaultType.DILUTE_CONTEXT,
            FaultType.CONTRADICT_EVIDENCE,
        ]
        for i in range(self.N_INJECTED):
            fault = faults[i % len(faults)]
            specs.append(ExampleSpec(
                example_id=f"injected/{fault.value}/{i:03d}",
                payload={"source": "injected", "fault": fault.value, "index": i},
            ))
        return specs

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload

        if payload["source"] == "real_baseline":
            row = payload["row"]
            xrag_cause = row.get("xrag_primary_cause") or FailureType.UNKNOWN.value
            return {
                "source": "real_baseline",
                "eval_id": row.get("eval_id"),
                "true_cause": None,  # no ground truth for real rows
                "xrag_cause": xrag_cause,
                "baseline_cause": baseline_attribution(row),
                "xrag_score": row.get("xrag_avg_entailment_score"),
                "ragas_faithfulness": row.get("ragas_faithfulness"),
                "ragchecker_faithfulness": row.get("ragchecker_faithfulness"),
                "ragchecker_precision": row.get("ragchecker_precision"),
                "ragchecker_hallucination": row.get("ragchecker_hallucination"),
                "expected_failure_type": row.get("expected_failure_type") or None,
            }

        rng = ctx.rng_for(f"e5:{spec.example_id}")
        case = build_case(
            base_id=spec.example_id.replace("/", "-"),
            faults=[FaultType(payload["fault"])],
            rng=rng,
            near_threshold=payload["index"] % 4 == 0,
        )
        _, rca = diagnose(case)

        derived = self._derive_baseline_scalars(case)
        return {
            "source": "injected",
            "fault": payload["fault"],
            "true_cause": case.expected_primary_cause,
            "xrag_cause": rca.primary_cause.value,
            "baseline_cause": baseline_attribution(derived),
            "xrag_score": case.verification.average_entailment_score,
            **derived,
            "baseline_scalars_are_recomputed": True,
        }

    @staticmethod
    def _derive_baseline_scalars(case) -> Dict[str, Any]:
        """
        Recompute RAGAS-style faithfulness and RAGChecker-style hallucination
        and context precision from the injected case's verification summary,
        each by its own published definition.

        Stated plainly for the paper: these are reimplementations of published
        formulas applied to synthetic verification output, not scores returned
        by RAGAS or RAGChecker. They exist so the injected rows can be scored
        by the same baseline *rule* as the real rows. Every conclusion drawn
        from them is reported under the "injected" label, never pooled with the
        real-baseline correlation.
        """
        summary = case.verification
        total = max(1, summary.total_claims)
        supported = summary.supported_claims + summary.partially_supported_claims
        refs = case.trace.retrieved_chunk_references

        used_chunks = {
            r.best_chunk_id for r in summary.results
            if r.best_chunk_id and r.verification_status.value in ("SUPPORTED", "PARTIALLY_SUPPORTED")
        }
        context_precision = len(used_chunks) / len(refs) if refs else 0.0

        return {
            "ragas_faithfulness": supported / total,
            "ragchecker_faithfulness": supported / total,
            "ragchecker_hallucination": summary.contradicted_claims / total,
            "ragchecker_precision": supported / total,
            "ragchecker_context_precision": context_precision,
            "ragas_context_precision": context_precision,
        }

    # -- aggregation -----------------------------------------------------

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        real = [r for r in records if r["source"] == "real_baseline"]
        injected = [r for r in records if r["source"] == "injected"]

        correlations = self._correlations(real)
        cause_agreement = {
            "real_baseline": self._cause_agreement(real),
            "injected": self._cause_agreement(injected),
            "pooled": self._cause_agreement(records),
        }

        ground_truth = self._ground_truth_scoring(injected)

        headline = {
            "score_correlation_real_rows": correlations.get("best_absolute_r"),
            "cause_level_kappa_real_rows": cause_agreement["real_baseline"].get("cause_kappa"),
            "binary_failure_kappa_real_rows": cause_agreement["real_baseline"].get("binary_kappa"),
            "cause_level_kappa_pooled": cause_agreement["pooled"].get("cause_kappa"),
            "correlation_minus_cause_kappa": _gap(
                correlations.get("best_absolute_r"),
                cause_agreement["real_baseline"].get("cause_kappa"),
            ),
            "who_is_right_on_injected_faults": ground_truth,
        }

        return {
            "headline": headline,
            "scalar_correlations_real_rows_only": correlations,
            "cause_level_agreement": cause_agreement,
            "baseline_attribution_rule": BASELINE_ATTRIBUTION_RULE,
            "n_real_rows": len(real),
            "n_injected_rows": len(injected),
            "interpretation_notes": [
                "The quotable pattern is a high score correlation next to a low cause "
                "kappa. Correlation says the two frameworks rank runs similarly by "
                "severity; kappa says they disagree about what to fix. Only the second "
                "changes what an engineer does on Monday.",
                "Binary failure kappa is reported alongside cause kappa on purpose. Binary "
                "agreement is usually the higher of the two, and the gap between them is "
                "the precise thing scalar-correlation validation hides: agreeing that a run "
                "failed is not agreeing on why.",
                "Injected rows carry recomputed, not measured, baseline scalars (see "
                "_derive_baseline_scalars). They are excluded from the correlation and "
                "included in the kappa and ground-truth scoring, and are always reported "
                "under their own label.",
                "A low kappa is not by itself proof that X-RAG is the correct one. That is "
                "what who_is_right_on_injected_faults answers, and it is the only part of "
                "this experiment with an oracle.",
            ],
        }

    @staticmethod
    def _correlations(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(rows) < 2:
            return {"note": "fewer than 2 real-baseline rows; correlation not computed"}

        xrag = [r.get("xrag_score") for r in rows]
        pairs = {
            "xrag_vs_ragas_faithfulness": [r.get("ragas_faithfulness") for r in rows],
            "xrag_vs_ragchecker_faithfulness": [r.get("ragchecker_faithfulness") for r in rows],
            "xrag_vs_ragchecker_precision": [r.get("ragchecker_precision") for r in rows],
        }
        out: Dict[str, Any] = {}
        for name, ys in pairs.items():
            out[name] = {
                "pearson_r": pearson_r(xrag, ys),
                "spearman_rho": spearman_rho(xrag, ys),
                "n_paired": sum(
                    1 for x, y in zip(xrag, ys) if x is not None and y is not None
                ),
            }
        rs = [v["pearson_r"] for v in out.values() if isinstance(v, dict) and v.get("pearson_r") is not None]
        out["best_absolute_r"] = max((abs(r) for r in rs), default=None)
        return out

    @staticmethod
    def _cause_agreement(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        paired = [
            (r["xrag_cause"], r["baseline_cause"])
            for r in rows
            if r.get("baseline_cause") is not None and r.get("xrag_cause") is not None
        ]
        if not paired:
            return {"n_paired": 0, "note": "no rows with both a X-RAG and a baseline verdict"}

        xrag_causes = [a for a, _ in paired]
        baseline_causes = [b for _, b in paired]

        return {
            "n_paired": len(paired),
            "raw_cause_agreement": accuracy(xrag_causes, baseline_causes),
            "cause_kappa": multiclass_cohens_kappa(xrag_causes, baseline_causes),
            "binary_kappa": binary_cohens_kappa(
                [_flags_failure(c) for c in xrag_causes],
                [_flags_failure(c) for c in baseline_causes],
            ),
            "raw_binary_agreement": accuracy(
                [str(_flags_failure(c)) for c in xrag_causes],
                [str(_flags_failure(c)) for c in baseline_causes],
            ),
            "confusion_xrag_rows_vs_baseline_columns": confusion_matrix(
                xrag_causes, baseline_causes
            ),
        }

    @staticmethod
    def _ground_truth_scoring(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The part with an oracle: on injected faults, which diagnoser is right?"""
        scored = [r for r in rows if r.get("true_cause")]
        if not scored:
            return {"n": 0}
        truth = [r["true_cause"] for r in scored]
        return {
            "n": len(scored),
            "xrag_accuracy": accuracy(truth, [r["xrag_cause"] for r in scored]),
            "baseline_rule_accuracy": accuracy(
                truth, [r.get("baseline_cause") or FailureType.UNKNOWN.value for r in scored]
            ),
        }


def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


EXPERIMENT = DiagnosticAgreementExperiment()
