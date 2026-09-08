"""
Causal fault injection for stage-attributed failure diagnosis (E1, reused by E5).

Why synthetic ground truth
--------------------------
The usual way to validate a failure-attribution system is to correlate its
verdicts with human labels. That is weak evidence: annotators disagree about
*which* stage caused a failure precisely because the causal question is hard,
so a correlation study measures agreement with a noisy oracle rather than
correctness. Fault injection inverts the problem. We *cause* a specific
pipeline fault, so the ground-truth cause is known by construction, and the
question becomes checkable: does the diagnosis recover the fault we injected?

What this validates -- and what it does not
-------------------------------------------
The injectors mutate the *observable stage signals* that a real run emits and
that the diagnosis consumes: pre-rerank dense distance, retrieval scores,
chunk adjacency and provenance, and per-claim verification verdicts. So this
validates the attribution layer -- ``PipelineStateAnalyzer`` plus
``RootCauseReasoner``, including the causal-precedence rule -- given faithful
stage signals.

It does **not** validate the layers that produce those signals (claim
decomposition, NLI verification, retrieval itself). A fault injected here
reaches the analyzer intact by construction; in a live run it must first
survive decomposition and NLI. E1 reports attribution accuracy; the upstream
signal fidelity is a separate question, and the paper should not conflate them.

Non-triviality
--------------
An injector that simply hard-set the answer would prove nothing. Three design
choices keep the task genuinely hard:

* **Nuisance variation.** Claim counts, score magnitudes, distractor chunks
  and chunk provenance vary per base scenario, so no fixed input shape maps to
  a fixed verdict.
* **Near-threshold cases.** A configurable fraction of scenarios place the
  decisive score within a hair of its threshold, exposing brittleness that
  comfortably-separated cases would hide. Accuracy is reported split by this
  flag.
* **Compound faults.** Upstream and downstream faults are injected together
  and the ground truth is the *upstream* one. These cases fail unless the
  reasoner's causal-precedence rule actually works -- a confidence-ranking
  reasoner picks the downstream symptom, because downstream stages accumulate
  more evidence and score higher.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from configs.thresholds import (
    CORPUS_MAX_RELEVANT_DISTANCE,
    RETRIEVAL_SCORE_THRESHOLD,
)
from src.claim_decomposer import CandidateClaim, CandidateClaimSet
from src.claim_verifier import (
    VerificationResult,
    VerificationStatus,
    VerificationSummary,
)
from src.rag_trace import RAGTrace
from src.root_cause_reasoner import FailureType

# ---------------------------------------------------------------------------
# Fault taxonomy
# ---------------------------------------------------------------------------


class FaultType(str, Enum):
    """Injectable faults, one per diagnosable pipeline stage (plus a control)."""

    NONE = "NONE"
    REMOVE_GOLD_CHUNK = "REMOVE_GOLD_CHUNK"
    FORCE_BAD_RANK = "FORCE_BAD_RANK"
    TRUNCATE_AT_BOUNDARY = "TRUNCATE_AT_BOUNDARY"
    DILUTE_CONTEXT = "DILUTE_CONTEXT"
    CONTRADICT_EVIDENCE = "CONTRADICT_EVIDENCE"


ALL_FAULTS: Tuple[str, ...] = (
    FaultType.NONE,
    FaultType.REMOVE_GOLD_CHUNK,
    FaultType.FORCE_BAD_RANK,
    FaultType.TRUNCATE_AT_BOUNDARY,
    FaultType.DILUTE_CONTEXT,
    FaultType.CONTRADICT_EVIDENCE,
)

#: The cause each single-fault injection is *supposed* to be diagnosed as.
#: This is the ground truth E1 scores against.
FAULT_TO_EXPECTED_CAUSE: Dict[str, str] = {
    FaultType.NONE: FailureType.UNKNOWN.value,           # healthy: no failure to name
    FaultType.REMOVE_GOLD_CHUNK: FailureType.MISSING_CORPUS.value,
    FaultType.FORCE_BAD_RANK: FailureType.RETRIEVAL_MISS.value,
    FaultType.TRUNCATE_AT_BOUNDARY: FailureType.CHUNK_BOUNDARY.value,
    FaultType.DILUTE_CONTEXT: FailureType.UNSUPPORTED_GENERATION.value,
    FaultType.CONTRADICT_EVIDENCE: FailureType.GROUNDING_FAILURE.value,
}

#: Causal depth, used to resolve the ground truth of a compound injection: the
#: expected cause is the *shallowest* (most upstream) fault applied.
FAULT_CAUSAL_DEPTH: Dict[str, int] = {
    FaultType.REMOVE_GOLD_CHUNK: 0,     # CORPUS
    FaultType.FORCE_BAD_RANK: 1,        # RETRIEVER
    FaultType.TRUNCATE_AT_BOUNDARY: 2,  # CHUNKING
    FaultType.DILUTE_CONTEXT: 3,        # GENERATOR
    FaultType.CONTRADICT_EVIDENCE: 4,   # GROUNDING
    FaultType.NONE: 99,
}

#: Compound pairs (upstream, downstream). Each exists to break a reasoner that
#: ranks by confidence instead of walking the causal chain: the downstream
#: symptom is the louder signal, and the correct answer is the quiet upstream
#: one.
COMPOUND_PAIRS: Tuple[Tuple[str, str], ...] = (
    (FaultType.REMOVE_GOLD_CHUNK, FaultType.CONTRADICT_EVIDENCE),
    (FaultType.REMOVE_GOLD_CHUNK, FaultType.DILUTE_CONTEXT),
    (FaultType.FORCE_BAD_RANK, FaultType.DILUTE_CONTEXT),
    (FaultType.TRUNCATE_AT_BOUNDARY, FaultType.DILUTE_CONTEXT),
)


class Arm(str, Enum):
    """
    The three question types E1 asks. Accuracy is reported per arm, because
    they test different things and a pooled number would hide which.

    SINGLE
        One fault, clean signals. Tests whether each stage's detection rule
        fires on its own signature -- and, for the CORPUS/RETRIEVER pair,
        whether the diagnosis separates two faults with identical downstream
        symptoms.

    COMPOUND_PRESERVED
        Two faults, upstream signature intact. Tests the causal-precedence
        rule in isolation: the reasoner must name the upstream fault even
        though the downstream stage carries higher-confidence evidence.

    COMPOUND_MASKED
        Two faults where the downstream one *overwrites* an observable the
        upstream detection rule depends on -- the realistic case, since a
        generator fault genuinely does change retrieval-score statistics.
        Ground truth is still the upstream fault. This arm is expected to be
        the hard one, and its score is reported rather than pooled away: a low
        number here is a real limitation of signal-threshold attribution, not
        a bug in the harness.
    """

    SINGLE = "SINGLE"
    COMPOUND_PRESERVED = "COMPOUND_PRESERVED"
    COMPOUND_MASKED = "COMPOUND_MASKED"


# ---------------------------------------------------------------------------
# Injected case
# ---------------------------------------------------------------------------


@dataclass
class InjectedCase:
    """A synthetic pipeline observation with a known injected cause."""

    case_id: str
    faults_applied: List[str]
    expected_primary_cause: str
    trace: RAGTrace
    claim_set: CandidateClaimSet
    verification: VerificationSummary
    arm: str = Arm.SINGLE.value
    near_threshold: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_compound(self) -> bool:
        return len([f for f in self.faults_applied if f != FaultType.NONE]) > 1

    @property
    def is_healthy(self) -> bool:
        return self.expected_primary_cause == FailureType.UNKNOWN.value


# ---------------------------------------------------------------------------
# Baseline construction
# ---------------------------------------------------------------------------

_DOC_IDS = ("BNS.pdf", "BNSS.pdf", "BSA.pdf")


def _chunk_ref(
    chunk_id: str,
    rank: int,
    score: float,
    chunk_index: int,
    parent_document_id: str,
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "rank": rank,
        "similarity_score": score,
        "rrf_score": round(1.0 / (60 + rank), 6),
        "reranker_score": score,
        "dense_score": round(1.0 - score, 4),
        "sparse_score": round(score * 8.0, 4),
        "dense_rank": rank,
        "sparse_rank": rank,
        "page_number": str(10 + chunk_index),
        "source_file": parent_document_id,
        "chunk_index": chunk_index,
        "parent_document_id": parent_document_id,
    }


def _verification(
    trace_id: str,
    claim_id: str,
    status: str,
    best_chunk_id: Optional[str],
    entailment: float,
    contradiction: float,
    neutral: float,
) -> VerificationResult:
    return VerificationResult(
        verification_id=f"V_{claim_id}",
        trace_id=trace_id,
        claim_id=claim_id,
        claim_text=f"Synthetic claim {claim_id}.",
        verification_status=VerificationStatus(status),
        verification_reason="synthetic",
        confidence=max(entailment, contradiction),
        best_chunk_id=best_chunk_id,
        best_chunk_rank=1,
        best_chunk_score=entailment,
        best_sentence_id=f"{best_chunk_id}_s0" if best_chunk_id else None,
        evidence_text="synthetic evidence" if best_chunk_id else None,
        entailment_score=entailment,
        contradiction_score=contradiction,
        neutral_score=neutral,
    )


def _summarize(trace_id: str, results: List[VerificationResult]) -> VerificationSummary:
    def count(status: VerificationStatus) -> int:
        return sum(1 for r in results if r.verification_status == status)

    total = len(results)
    return VerificationSummary(
        trace_id=trace_id,
        total_claims=total,
        supported_claims=count(VerificationStatus.SUPPORTED),
        partially_supported_claims=count(VerificationStatus.PARTIALLY_SUPPORTED),
        contradicted_claims=count(VerificationStatus.CONTRADICTED),
        unsupported_claims=count(VerificationStatus.UNSUPPORTED),
        not_verifiable_claims=count(VerificationStatus.NOT_VERIFIABLE),
        average_entailment_score=(
            sum(r.entailment_score for r in results) / total if total else 0.0
        ),
        total_verification_latency_ms=0.0,
        results=results,
    )


def build_healthy_case(base_id: str, rng: random.Random, near_threshold: bool = False) -> InjectedCase:
    """
    A clean run: relevant corpus, strong retrieval, every claim supported.

    This is both the control arm (a diagnosis that invents a failure here is
    a false positive) and the substrate every fault is injected into.

    ``near_threshold`` moves the decisive signals to within a hair of their
    configured thresholds -- still on the passing side, but close enough that a
    brittle rule flips. Accuracy on these cases is reported separately.
    """
    trace_id = f"FI-{base_id}"
    doc = _DOC_IDS[rng.randrange(len(_DOC_IDS))]

    n_chunks = rng.randint(4, 7)
    # Chunk indices spaced >= 3 apart so no accidental adjacency triggers the
    # CHUNKING boundary rule -- boundary adjacency is injected explicitly, by
    # TRUNCATE_AT_BOUNDARY, and must not appear by chance in other arms.
    base_index = rng.randint(20, 200)
    chunk_refs = []
    for i in range(n_chunks):
        if near_threshold:
            score = round(RETRIEVAL_SCORE_THRESHOLD + rng.uniform(0.005, 0.03), 4)
        else:
            score = round(rng.uniform(0.68, 0.96), 4)
        chunk_refs.append(
            _chunk_ref(
                chunk_id=f"{trace_id}_C{i}",
                rank=i + 1,
                score=score,
                chunk_index=base_index + i * rng.randint(3, 6),
                parent_document_id=doc,
            )
        )
    chunk_refs.sort(key=lambda c: c["similarity_score"], reverse=True)
    for rank, ref in enumerate(chunk_refs, start=1):
        ref["rank"] = rank

    if near_threshold:
        min_distance = round(CORPUS_MAX_RELEVANT_DISTANCE - rng.uniform(0.005, 0.03), 4)
    else:
        min_distance = round(rng.uniform(0.12, 0.55), 4)

    n_claims = rng.randint(3, 8)
    claims = []
    results = []
    claim_set = CandidateClaimSet(trace_id=trace_id)
    for i in range(n_claims):
        claim_id = f"{trace_id}_K{i}"
        claim_set.add_claim(
            CandidateClaim(
                candidate_id=claim_id,
                trace_id=trace_id,
                claim_text=f"Synthetic claim {claim_id}.",
                sentence_id=f"S{i}",
                claim_index=i,
                character_start=i * 40,
                character_end=i * 40 + 39,
            )
        )
        entail = round(rng.uniform(0.78, 0.97), 4)
        results.append(
            _verification(
                trace_id,
                claim_id,
                VerificationStatus.SUPPORTED.value,
                chunk_refs[i % len(chunk_refs)]["chunk_id"],
                entail,
                round(rng.uniform(0.0, 0.04), 4),
                round(1.0 - entail - 0.02, 4),
            )
        )
        claims.append(claim_id)

    trace = RAGTrace(
        trace_id=trace_id,
        trace_version="1.0",
        pipeline_version="1.0",
        framework_version="1.0",
        timestamp="2026-08-01T00:00:00Z",
        question=f"Synthetic question for base {base_id}?",
        generated_answer="Synthetic answer.",
        prompt_snapshot="synthetic prompt",
        prompt_length=1234,
        retrieved_chunk_references=chunk_refs,
        configuration_snapshot={"synthetic": True},
        execution_statistics={
            "retrieval_time": 0.4,
            "generation_time": 1.2,
            "total_pipeline_time": 1.6,
            "pre_rerank_candidate_pool_size": rng.randint(28, 48),
            "pre_rerank_min_dense_distance": min_distance,
        },
        pipeline_stage_status={},
    )

    return InjectedCase(
        case_id=base_id,
        faults_applied=[FaultType.NONE],
        expected_primary_cause=FailureType.UNKNOWN.value,
        trace=trace,
        claim_set=claim_set,
        verification=_summarize(trace_id, results),
        near_threshold=near_threshold,
        provenance={
            "n_chunks": n_chunks,
            "n_claims": n_claims,
            "document": doc,
            "min_dense_distance": min_distance,
        },
    )


# ---------------------------------------------------------------------------
# Injectors
# ---------------------------------------------------------------------------


def _next_claim_id(case: InjectedCase, tag: str, i: int) -> str:
    return f"{case.trace.trace_id}_{tag}{i}"


def _add_claim(case: InjectedCase, claim_id: str) -> None:
    idx = case.claim_set.total_candidates
    case.claim_set.add_claim(
        CandidateClaim(
            candidate_id=claim_id,
            trace_id=case.trace.trace_id,
            claim_text=f"Synthetic claim {claim_id}.",
            sentence_id=f"S{idx}",
            claim_index=idx,
            character_start=idx * 40,
            character_end=idx * 40 + 39,
        )
    )


def inject_remove_gold_chunk(case: InjectedCase, rng: random.Random) -> None:
    """
    CORPUS fault: the answer-bearing passage is not in the corpus at all.

    Signature: even the closest pre-rerank candidate sits beyond the relevance
    threshold, and nothing retrieved supports any claim. Distinguishing this
    from FORCE_BAD_RANK is the whole point -- both produce identical downstream
    symptoms (no supported claims), and only the pre-rerank distance separates
    "the corpus does not contain it" from "retrieval failed to surface it".
    """
    stats = case.trace.execution_statistics
    if case.near_threshold:
        stats["pre_rerank_min_dense_distance"] = round(
            CORPUS_MAX_RELEVANT_DISTANCE + rng.uniform(0.005, 0.03), 4
        )
    else:
        stats["pre_rerank_min_dense_distance"] = round(rng.uniform(0.82, 0.98), 4)

    _starve_retrieval(case, rng)
    case.provenance["removed_gold_chunk"] = True


def inject_force_bad_rank(case: InjectedCase, rng: random.Random) -> None:
    """
    RETRIEVER fault: the passage exists in the corpus but ranks below the
    window, so the generator never sees it.

    Signature: pre-rerank distance stays *inside* the relevance band (the
    corpus is fine) while nothing retrieved supports any claim.

    Must NOT touch pre_rerank_min_dense_distance if REMOVE_GOLD_CHUNK has
    already run on this case: build_case applies faults in causal order, so
    REMOVE_GOLD_CHUNK (depth 0) always runs before this (depth 1) when both
    are present, and this fault unconditionally overwriting that field used
    to silently erase REMOVE_GOLD_CHUNK's own signal -- a real bug caught by
    E11 (exp11_nway_compound_faults.py), which stacks combinations E1's own
    curated COMPOUND_PAIRS list never tests together. The corpus-distance
    signal belongs to whichever fault is causally upstream, not to whichever
    injector happens to run second.
    """
    stats = case.trace.execution_statistics
    if not case.provenance.get("removed_gold_chunk"):
        if case.near_threshold:
            stats["pre_rerank_min_dense_distance"] = round(
                CORPUS_MAX_RELEVANT_DISTANCE - rng.uniform(0.005, 0.03), 4
            )
        else:
            stats["pre_rerank_min_dense_distance"] = round(rng.uniform(0.18, 0.6), 4)

    _starve_retrieval(case, rng)
    case.provenance["displaced_gold_rank"] = rng.randint(12, 40)


def _starve_retrieval(case: InjectedCase, rng: random.Random) -> None:
    """Shared downstream symptom of the two retrieval-side faults: every
    retrieved chunk scores below the retrieval threshold and every claim comes
    back unsupported."""
    for ref in case.trace.retrieved_chunk_references:
        if case.near_threshold:
            ref["similarity_score"] = round(
                RETRIEVAL_SCORE_THRESHOLD - rng.uniform(0.005, 0.03), 4
            )
        else:
            ref["similarity_score"] = round(rng.uniform(0.05, 0.38), 4)
        ref["reranker_score"] = ref["similarity_score"]

    for result in case.verification.results:
        result.verification_status = VerificationStatus.UNSUPPORTED
        result.entailment_score = round(rng.uniform(0.01, 0.18), 4)
        result.contradiction_score = round(rng.uniform(0.0, 0.06), 4)
        result.neutral_score = round(1.0 - result.entailment_score - result.contradiction_score, 4)
    case.verification = _summarize(case.trace.trace_id, case.verification.results)


def inject_truncate_at_boundary(case: InjectedCase, rng: random.Random) -> None:
    """
    CHUNKING fault: a fact is split across a chunk boundary, so the best
    evidence only partially supports the claim while its continuation sits in
    the adjacent chunk.

    Signature: PARTIALLY_SUPPORTED claims whose best chunk is adjacent (by
    chunk_index, same parent document) to another retrieved chunk. Retrieval
    and corpus stay healthy -- the content was found, it was merely cut.
    """
    refs = case.trace.retrieved_chunk_references
    anchor = refs[0]
    neighbour_index = anchor["chunk_index"] + 1
    neighbour_id = f"{case.trace.trace_id}_ADJ"
    refs.append(
        _chunk_ref(
            chunk_id=neighbour_id,
            rank=len(refs) + 1,
            score=round(anchor["similarity_score"] - rng.uniform(0.01, 0.08), 4),
            chunk_index=neighbour_index,
            parent_document_id=anchor["parent_document_id"],
        )
    )

    n_split = rng.randint(1, 2)
    for i in range(n_split):
        claim_id = _next_claim_id(case, "SPLIT", i)
        _add_claim(case, claim_id)
        entail = round(rng.uniform(0.42, 0.66), 4)
        case.verification.results.append(
            _verification(
                case.trace.trace_id,
                claim_id,
                VerificationStatus.PARTIALLY_SUPPORTED.value,
                anchor["chunk_id"],
                entail,
                round(rng.uniform(0.0, 0.05), 4),
                round(1.0 - entail - 0.05, 4),
            )
        )
    case.verification = _summarize(case.trace.trace_id, case.verification.results)
    case.provenance["boundary_split_claims"] = n_split
    case.provenance["adjacent_chunk_id"] = neighbour_id


def inject_dilute_context(case: InjectedCase, rng: random.Random, preserve_upstream: bool = False) -> None:
    """
    GENERATOR fault: retrieval succeeded, but the context window is padded
    with distractors and the model asserts things the evidence does not carry.

    Signature: unsupported claims *despite* retrieval scores clearing the
    threshold -- the discriminator that separates a generator fault from a
    retrieval fault.

    ``preserve_upstream=True`` holds every observable an upstream detection
    rule reads. Concretely: when retrieval has already been starved, the
    injected distractors are scored *below* the retrieval threshold so
    ``max_score`` still reports the starvation. Compound-preserved cases use
    this so they test causal precedence alone.

    ``preserve_upstream=False`` on top of an upstream fault is the
    COMPOUND_MASKED arm: high-scoring distractors legitimately overwrite
    ``max_score``, erasing the retrieval-side evidence. That is what happens in
    a real pipeline, and E1 reports it as its own number rather than folding it
    into the headline.
    """
    refs = case.trace.retrieved_chunk_references
    current_max = max((r["similarity_score"] for r in refs), default=0.0)
    retrieval_already_starved = current_max < RETRIEVAL_SCORE_THRESHOLD

    n_distractors = rng.randint(2, 4)
    for i in range(n_distractors):
        if preserve_upstream and retrieval_already_starved:
            score = round(rng.uniform(0.05, RETRIEVAL_SCORE_THRESHOLD - 0.05), 4)
        else:
            # Distractors are off-topic, not low-scoring: a reranker that
            # promoted them scored them highly. Their damage is dilution,
            # which is why they never become supporting evidence below.
            score = round(rng.uniform(0.55, 0.8), 4)
        refs.append(
            _chunk_ref(
                chunk_id=f"{case.trace.trace_id}_DIST{i}",
                rank=len(refs) + 1,
                score=score,
                chunk_index=rng.randint(400, 900),
                parent_document_id=_DOC_IDS[rng.randrange(len(_DOC_IDS))],
            )
        )

    if not preserve_upstream:
        # A standalone generator fault needs retrieval to look healthy --
        # unsupported claims *despite* good retrieval is the whole signature.
        for ref in refs:
            if ref["similarity_score"] < RETRIEVAL_SCORE_THRESHOLD:
                ref["similarity_score"] = round(
                    RETRIEVAL_SCORE_THRESHOLD + rng.uniform(0.06, 0.3), 4
                )
                ref["reranker_score"] = ref["similarity_score"]

    n_unsupported = rng.randint(1, 3)
    for i in range(n_unsupported):
        claim_id = _next_claim_id(case, "HALL", i)
        _add_claim(case, claim_id)
        entail = round(rng.uniform(0.02, 0.2), 4)
        contradiction = round(rng.uniform(0.0, 0.08), 4)
        case.verification.results.append(
            _verification(
                case.trace.trace_id,
                claim_id,
                VerificationStatus.UNSUPPORTED.value,
                refs[-1]["chunk_id"],
                entail,
                contradiction,
                round(1.0 - entail - contradiction, 4),
            )
        )
    case.verification = _summarize(case.trace.trace_id, case.verification.results)
    case.provenance["distractor_chunks"] = n_distractors
    case.provenance["unsupported_claims_added"] = n_unsupported


def inject_contradict_evidence(case: InjectedCase, rng: random.Random, preserve_upstream: bool = False) -> None:
    """
    GROUNDING fault: the answer asserts something the retrieved evidence
    actively refutes -- misinformation, not merely unsupported speculation.

    Signature: CONTRADICTED claims with everything upstream healthy. Kept
    distinct from the generator fault because the corrective action differs:
    an unsupported claim wants tighter grounding instructions, a contradicted
    one wants the answer suppressed.
    """
    n_contradicted = rng.randint(1, 3)
    best_chunk = case.trace.retrieved_chunk_references[0]["chunk_id"]
    for i in range(n_contradicted):
        claim_id = _next_claim_id(case, "CONTRA", i)
        _add_claim(case, claim_id)
        contradiction = round(rng.uniform(0.74, 0.97), 4)
        case.verification.results.append(
            _verification(
                case.trace.trace_id,
                claim_id,
                VerificationStatus.CONTRADICTED.value,
                best_chunk,
                round(rng.uniform(0.0, 0.08), 4),
                contradiction,
                round(1.0 - contradiction - 0.04, 4),
            )
        )
    case.verification = _summarize(case.trace.trace_id, case.verification.results)
    case.provenance["contradicted_claims_added"] = n_contradicted


_INJECTORS = {
    FaultType.REMOVE_GOLD_CHUNK: inject_remove_gold_chunk,
    FaultType.FORCE_BAD_RANK: inject_force_bad_rank,
    FaultType.TRUNCATE_AT_BOUNDARY: inject_truncate_at_boundary,
    FaultType.DILUTE_CONTEXT: inject_dilute_context,
    FaultType.CONTRADICT_EVIDENCE: inject_contradict_evidence,
}

#: Injectors that accept ``preserve_upstream``.
_MASKING_AWARE = {FaultType.DILUTE_CONTEXT, FaultType.CONTRADICT_EVIDENCE}


def build_case(
    base_id: str,
    faults: List[str],
    rng: random.Random,
    near_threshold: bool = False,
    preserve_upstream: bool = True,
) -> InjectedCase:
    """
    Build one injected case: a healthy baseline with ``faults`` applied in
    causal order (upstream first).

    The expected primary cause is the most upstream applied fault -- exactly
    the claim ``RootCauseReasoner`` makes, and therefore exactly what E1
    scores.

    ``preserve_upstream`` only matters for compound injections and selects the
    arm: ``True`` gives COMPOUND_PRESERVED (causal precedence in isolation),
    ``False`` gives COMPOUND_MASKED (the downstream fault is allowed to
    overwrite upstream observables, as it would in a live pipeline).
    """
    case = build_healthy_case(base_id, rng, near_threshold=near_threshold)

    real_faults = [f for f in faults if f != FaultType.NONE]
    ordered = sorted(real_faults, key=lambda f: FAULT_CAUSAL_DEPTH[f])

    for position, fault in enumerate(ordered):
        injector = _INJECTORS[fault]
        if position > 0 and fault in _MASKING_AWARE:
            injector(case, rng, preserve_upstream=preserve_upstream)
        else:
            injector(case, rng)

    case.faults_applied = [f.value for f in ordered] or [FaultType.NONE.value]
    if ordered:
        case.expected_primary_cause = FAULT_TO_EXPECTED_CAUSE[ordered[0]]
    else:
        case.expected_primary_cause = FailureType.UNKNOWN.value

    if len(ordered) > 1:
        case.arm = (
            Arm.COMPOUND_PRESERVED.value if preserve_upstream else Arm.COMPOUND_MASKED.value
        )
    else:
        case.arm = Arm.SINGLE.value

    case.case_id = base_id
    return case


def diagnose(case: InjectedCase, analyzer=None):
    """
    Run the real diagnosis stack over an injected case.

    ``analyzer``, when given, overrides the default ``PipelineStateAnalyzer()``
    -- used by E10 (experiments/exp10_generator_rule_sensitivity.py) to
    re-score the same 158 E1 cases under alternative GENERATOR-stage
    aggregation rules without duplicating the fault-construction logic.
    Defaults to ``None`` so every existing call site (E1, E5) is unaffected.

    Returns ``(PipelineStateMatrix, RootCauseAnalysis)``. Imported lazily so
    the fault library itself stays importable without the diagnostic modules'
    dependencies.
    """
    from src.pipeline_state_analyzer import PipelineStateAnalyzer
    from src.root_cause_reasoner import RootCauseReasoner

    psm = (analyzer or PipelineStateAnalyzer()).analyze(case.trace, case.claim_set, case.verification)
    rca = RootCauseReasoner().analyze(psm)
    return psm, rca
