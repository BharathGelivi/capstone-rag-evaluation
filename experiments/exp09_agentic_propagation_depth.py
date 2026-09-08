"""
E9 -- Agentic evidence-pool propagation depth: does attribution degrade as
more subsequent controller steps accumulate evidence?

Why this is not a literal port of AgenticRAG-FP
--------------------------------------------------
AgenticRAG-FP (concurrent work, arXiv:2608.20627) studies multi-hop QA, where
each hop produces its own intermediate *answer* and a hop-1 retrieval error
can propagate to (or be corrected by) hop 3's independent re-retrieval. This
platform's agentic controller (``src/agentic.py``) is architecturally
different: ``AgenticRetriever.retrieve()`` runs up to ``MAX_AGENT_STEPS``
actions (semantic_search, graph_expand, contradiction_search, ...) that all
accumulate into ONE pooled evidence set feeding a single final generation
call. There is no chain of independent per-step answers to propagate a fault
through -- so "does the fault survive to the final trajectory" has no direct
analogue here.

What this experiment tests instead
-------------------------------------
The real question this platform's architecture actually poses: does
accumulating MORE evidence from subsequent controller steps make an earlier
step's retrieval fault *harder to detect*, purely because later steps raise
aggregate retrieval-score statistics (``max_score``) without actually fixing
the underlying evidence gap? This is a depth-parameterized generalization of
E1's existing COMPOUND_MASKED mechanism (Section 3.4 of the paper): instead
of a single downstream fault overwriting an upstream signal, this asks
whether an increasing *number* of downstream (fault-free, plausible-looking)
retrieval steps has the same masking effect, and whether that effect is
uniform across fault types.

The mechanism this predicts an asymmetry
--------------------------------------------
``PipelineStateAnalyzer``'s CORPUS stage reads ``pre_rerank_min_dense_distance``
(an ``execution_statistics`` field, set once at retrieval time and untouched
by anything added to ``retrieved_chunk_references`` afterward). Its RETRIEVER
stage reads ``max_score`` (the max similarity score across
``retrieved_chunk_references``, directly and immediately affected by adding
more chunks). So:

    REMOVE_GOLD_CHUNK (a CORPUS-stage fault, keyed on min_distance) should be
    architecturally immune to this masking mechanism at any depth.

    FORCE_BAD_RANK (a RETRIEVER-stage fault, keyed on max_score) should
    degrade as depth increases, exactly like E1's compound design already
    showed for a single downstream fault -- generalized here to "how many"
    rather than "whether."

Both fault types are tested so the experiment can report the asymmetry
itself as the finding, rather than assuming it.

Method
------
For each fault and each base scenario, ONE underlying case is drawn (the same
``build_case`` recipe E1's single-fault arm uses), seeded from ``(fault,
base)`` only -- not from depth. Every depth in DEPTHS then re-diagnoses that
*same* case with 0..depth additional chunk references appended, representing
that many subsequent controller steps' contributions: each a plausible,
well-scoring chunk NOT linked as any claim's supporting evidence (so it
raises max_score without ever making ``has_supported_claim`` true). Holding
the base case fixed across a depth sweep is what makes the resulting curve a
depth effect rather than base-scenario-to-base-scenario noise -- an earlier
draft of this experiment re-randomized the case at every depth and was
rejected during review specifically because it could not separate the two.

Note this experiment uses its own independently-seeded base scenarios (not
E1's stored 17-per-fault set, and a different count -- BASES_PER_CELL=13), so
depth=0 is a fresh draw from the *same recipe* E1's single-fault arm uses,
not a byte-identical reproduction of E1's own recorded cases. It is reported
as an internal reference point for this experiment's own depth sweep, not as
a re-verification of E1's stored numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from experiments.common import ExampleSpec, Experiment, ExperimentContext, wilson_interval
from experiments.faults import FaultType, _chunk_ref, build_case, diagnose

logger = logging.getLogger(__name__)

TESTED_FAULTS = (FaultType.REMOVE_GOLD_CHUNK, FaultType.FORCE_BAD_RANK)
DEPTHS = (0, 1, 2, 3, 4, 5, 8, 12)
#: Base scenarios per (fault, depth) cell.
BASES_PER_CELL = 13


def _append_later_step_chunks(case, rng, depth: int) -> None:
    """Simulate `depth` subsequent agentic-controller steps, each contributing
    one plausible, well-scoring chunk that is never linked as supporting
    evidence for any claim. This raises max_score (what RETRIEVER/GENERATOR
    read) without touching pre_rerank_min_dense_distance (what CORPUS reads)
    or has_supported_claim -- isolating exactly the masking channel described
    in the module docstring, nothing else."""
    refs = case.trace.retrieved_chunk_references
    for i in range(depth):
        refs.append(_chunk_ref(
            chunk_id=f"{case.trace.trace_id}_STEP{i}",
            rank=len(refs) + 1,
            score=round(rng.uniform(0.6, 0.85), 4),
            chunk_index=rng.randint(400, 900),
            parent_document_id=f"DOC_STEP_{i}",
        ))


class AgenticPropagationDepthExperiment(Experiment):
    key = "exp09_agentic_propagation_depth"
    number = 9
    title = "Agentic evidence-pool propagation depth"
    claim = (
        "Attribution of a RETRIEVER-stage fault degrades as more subsequent "
        "evidence-gathering steps accumulate, while a CORPUS-stage fault -- keyed "
        "on a signal later steps cannot touch -- does not, at any tested depth."
    )
    supported_modes = ("offline", "live")  # fully synthetic; live == offline here

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        specs: List[ExampleSpec] = []
        for fault in TESTED_FAULTS:
            for depth in DEPTHS:
                for base in range(BASES_PER_CELL):
                    specs.append(ExampleSpec(
                        example_id=f"{fault.value}/depth{depth:02d}/base{base:02d}",
                        payload={"fault": fault.value, "depth": depth, "base": base},
                    ))
        return specs

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload
        fault, depth, base = payload["fault"], payload["depth"], payload["base"]

        # Seeded from (fault, base) only -- NOT depth -- so every depth in the
        # sweep re-diagnoses the same underlying case with more chunks
        # appended, rather than a fresh random case per depth. This is what
        # makes the resulting curve a depth effect rather than base-scenario
        # noise; see the module docstring.
        case_rng = ctx.rng_for(f"e9-case:{fault}/base{base:02d}")
        case = build_case(
            base_id=f"{fault}-base{base:02d}",
            faults=[FaultType(fault)],
            rng=case_rng,
        )

        # A separate, depth-position-keyed rng for the appended chunks, so
        # each depth's added chunks are deterministic and resumable without
        # perturbing the case-construction rng's draw sequence above.
        chunk_rng = ctx.rng_for(f"e9-chunks:{fault}/base{base:02d}")
        _append_later_step_chunks(case, chunk_rng, depth)

        psm, rca = diagnose(case)
        predicted = rca.primary_cause.value
        expected = case.expected_primary_cause

        return {
            "fault": payload["fault"],
            "depth": payload["depth"],
            "expected_cause": expected,
            "predicted_cause": predicted,
            "correct": predicted == expected,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        by_fault_depth: Dict[str, Any] = {}
        for fault in (f.value for f in TESTED_FAULTS):
            by_fault_depth[fault] = {}
            for depth in DEPTHS:
                subset = [r for r in records if r["fault"] == fault and r["depth"] == depth]
                if not subset:
                    continue
                correct = sum(1 for r in subset if r["correct"])
                lo, hi = wilson_interval(correct, len(subset))
                by_fault_depth[fault][str(depth)] = {
                    "n": len(subset),
                    "accuracy": correct / len(subset),
                    "accuracy_ci95": [lo, hi],
                }

        def depth0_acc(fault: str) -> Any:
            return by_fault_depth.get(fault, {}).get("0", {}).get("accuracy")

        def max_depth_acc(fault: str) -> Any:
            cells = by_fault_depth.get(fault, {})
            if not cells:
                return None
            last_key = str(max(DEPTHS))
            return cells.get(last_key, {}).get("accuracy")

        degradation = {
            fault: (
                None if depth0_acc(fault) is None or max_depth_acc(fault) is None
                else round(depth0_acc(fault) - max_depth_acc(fault), 4)
            )
            for fault in (f.value for f in TESTED_FAULTS)
        }

        return {
            "headline": {
                "depth0_accuracy_matches_e1_single_fault": {
                    fault: depth0_acc(fault) for fault in (f.value for f in TESTED_FAULTS)
                },
                f"accuracy_at_max_depth_{max(DEPTHS)}": {
                    fault: max_depth_acc(fault) for fault in (f.value for f in TESTED_FAULTS)
                },
                "degradation_depth0_to_max_depth": degradation,
                "asymmetry_confirmed": (
                    degradation.get(FaultType.REMOVE_GOLD_CHUNK.value) == 0.0
                    and (degradation.get(FaultType.FORCE_BAD_RANK.value) or 0.0) > 0.0
                ),
            },
            "by_fault_and_depth": by_fault_depth,
            "interpretation_notes": [
                "depth=0 is E1's own single-fault scenario for this fault type and should "
                "reproduce E1's per-fault recovery rate exactly (REMOVE_GOLD_CHUNK 1.000, "
                "FORCE_BAD_RANK 0.774) -- a built-in cross-check, not an independent claim.",
                "REMOVE_GOLD_CHUNK is expected to hold flat across all depths because its "
                "signal (pre_rerank_min_dense_distance) is a retrieval-time statistic that "
                "adding more retrieved_chunk_references cannot touch -- a structural, not "
                "incidental, robustness property.",
                "FORCE_BAD_RANK is expected to degrade with depth because its signal "
                "(max_score) is exactly what accumulating more chunks raises -- the same "
                "channel E1's own COMPOUND_MASKED arm already identified, generalized here "
                "from 'one downstream fault' to 'how many neutral subsequent steps'.",
                "This validates the attribution layer's sensitivity to evidence-pool growth "
                "under construction, the same scope limitation E1 itself carries (Section 3.6): "
                "it says nothing about whether a live agentic controller run reaches this "
                "PipelineStateMatrix shape in practice.",
            ],
        }


EXPERIMENT = AgenticPropagationDepthExperiment()
