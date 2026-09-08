"""
Evaluation for the retrieval-strategy comparison.

Retrieval and generation are scored **separately**, on purpose. A system can
retrieve the right evidence and then write a bad answer, or write a fluent
answer over the wrong evidence; a single end-to-end score cannot tell those
apart, and telling them apart is the entire point of this study.

Three families:

**Retrieval** (against the benchmark's gold evidence)
    ``recall@k``, ``MRR``, ``nDCG@k`` at chunk level, plus document-level
    ``evidence_recall`` and, for multi-hop questions, ``multi_hop_evidence_recall``
    -- the fraction of questions where *every* gold document was retrieved. That
    last one is the metric multi-hop methods live or die on: getting one of the
    three hops is not answering the question.

**Generation** (against the retrieved evidence and, where available, gold)
    Claim support rate, unsupported rate and contradiction rate come from the
    existing ``ClaimVerifier``; citation precision/recall/correctness come from
    ``src.citation_check``. Nothing here re-implements claim decomposition.

**System**
    Latency, LLM calls, retrieval calls, agent steps, graph hops, and answer
    length as a token proxy. Cost is a first-class result: a technique that buys
    two points of recall for four extra LLM calls is a different recommendation
    from one that buys it for free.

``compute_metric_panel`` assembles the thirteen scores the UI shows, reusing
``RagasEvaluator``, ``ClaimVerifier`` and ``AnswerCorrectnessEvaluator``. The
RAGAS-, RAGChecker- and ARES-style names are computed *by this project's own
judge* against its own definitions -- they are comparable across arms here, and
are not claimed to be the numbers those packages would return.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> Optional[float]:
    """Fraction of gold items appearing in the top ``k`` retrieved items."""
    gold_set = set(gold)
    if not gold_set:
        return None
    return round(len(gold_set & set(retrieved[:k])) / len(gold_set), 4)


def hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> Optional[float]:
    """1.0 if any gold item is in the top ``k``. The "did it find anything" floor."""
    gold_set = set(gold)
    if not gold_set:
        return None
    return 1.0 if gold_set & set(retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> Optional[float]:
    """Reciprocal rank of the first gold item; 0.0 if none was retrieved."""
    gold_set = set(gold)
    if not gold_set:
        return None
    for rank, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> Optional[float]:
    """Binary-relevance nDCG@k.

    Binary because the benchmark's gold labels are binary -- a chunk either is
    or is not evidence for the question. Graded relevance would need an
    annotation pass this corpus does not have, and inventing grades would make
    the metric look more precise than the labels underneath it.
    """
    gold_set = set(gold)
    if not gold_set:
        return None
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, item in enumerate(retrieved[:k], start=1) if item in gold_set)
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(gold_set), k) + 1))
    return round(dcg / ideal, 4) if ideal else None


@dataclass
class RetrievalMetrics:
    """Retrieval scores for one benchmark question."""
    question_id: str
    question_type: str
    hops: int
    n_retrieved: int
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    recall_at_20: Optional[float] = None
    hit_at_10: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    evidence_recall: Optional[float] = None
    document_recall: Optional[float] = None
    all_gold_documents_found: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_retrieval(
    question: Dict[str, Any],
    retrieved_chunk_ids: Sequence[str],
    registry=None,
) -> RetrievalMetrics:
    """Score one retrieval against the benchmark row's gold evidence.

    Document-level recall is computed alongside chunk-level recall because the
    two answer different questions. Chunk recall asks whether the *exact*
    passage was found; document recall asks whether the right judgment was
    found at all. A chunking change moves the first without necessarily moving
    the second, so reporting only one would confound chunking with retrieval.
    """
    gold_chunks = question.get("gold_chunk_ids") or []
    gold_documents = set(question.get("gold_document_ids") or [])

    retrieved_documents: List[str] = []
    if registry is not None:
        for chunk_id in retrieved_chunk_ids:
            record = registry.get_chunk(chunk_id)
            if record is not None:
                document_id = record.metadata.get("document_id")
                if document_id and document_id not in retrieved_documents:
                    retrieved_documents.append(document_id)

    found_documents = gold_documents & set(retrieved_documents)

    return RetrievalMetrics(
        question_id=question["id"],
        question_type=question["type"],
        hops=int(question.get("hops", 1)),
        n_retrieved=len(retrieved_chunk_ids),
        recall_at_5=recall_at_k(retrieved_chunk_ids, gold_chunks, 5),
        recall_at_10=recall_at_k(retrieved_chunk_ids, gold_chunks, 10),
        recall_at_20=recall_at_k(retrieved_chunk_ids, gold_chunks, 20),
        hit_at_10=hit_at_k(retrieved_chunk_ids, gold_chunks, 10),
        mrr=mrr(retrieved_chunk_ids, gold_chunks),
        ndcg_at_10=ndcg_at_k(retrieved_chunk_ids, gold_chunks, 10),
        evidence_recall=recall_at_k(retrieved_chunk_ids, gold_chunks, len(retrieved_chunk_ids) or 1),
        document_recall=(round(len(found_documents) / len(gold_documents), 4)
                         if gold_documents else None),
        # The multi-hop question: were *all* the hops found, not just one.
        all_gold_documents_found=(1.0 if gold_documents and found_documents == gold_documents
                                  else (0.0 if gold_documents else None)),
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationMetrics:
    """Claim- and citation-level scores for one answer."""
    total_claims: int = 0
    supported_claims: int = 0
    partially_supported_claims: int = 0
    unsupported_claims: int = 0
    contradicted_claims: int = 0
    not_verifiable_claims: int = 0
    claim_support_rate: Optional[float] = None
    unsupported_claim_rate: Optional[float] = None
    contradiction_rate: Optional[float] = None
    citation_precision: Optional[float] = None
    citation_recall: Optional[float] = None
    citation_correctness: Optional[float] = None
    fabricated_citations: int = 0
    total_citations: int = 0
    answer_chars: int = 0
    refused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Surface forms of a refusal, shared with experiments/exp03. A refusal must be
#: counted as one rather than scored as a perfectly faithful answer -- see E3.
def is_refusal(answer: str) -> bool:
    from experiments.exp03_refusal_calibration import is_refusal as _is_refusal

    return _is_refusal(answer)


def evaluate_generation(verification, citation_report, answer: str) -> GenerationMetrics:
    """Assemble generation metrics from artifacts the pipeline already produced."""
    metrics = GenerationMetrics(answer_chars=len(answer or ""), refused=is_refusal(answer or ""))

    if verification is not None:
        total = verification.total_claims
        metrics.total_claims = total
        metrics.supported_claims = verification.supported_claims
        metrics.partially_supported_claims = verification.partially_supported_claims
        metrics.unsupported_claims = verification.unsupported_claims
        metrics.contradicted_claims = verification.contradicted_claims
        metrics.not_verifiable_claims = verification.not_verifiable_claims
        if total:
            metrics.claim_support_rate = round(verification.supported_claims / total, 4)
            metrics.unsupported_claim_rate = round(verification.unsupported_claims / total, 4)
            metrics.contradiction_rate = round(verification.contradicted_claims / total, 4)

    if citation_report is not None:
        metrics.citation_precision = citation_report.citation_precision
        metrics.citation_recall = citation_report.citation_recall
        metrics.citation_correctness = citation_report.citation_correctness
        metrics.fabricated_citations = citation_report.unverifiable
        metrics.total_citations = citation_report.total_citations

    return metrics


# ---------------------------------------------------------------------------
# System cost
# ---------------------------------------------------------------------------

@dataclass
class SystemMetrics:
    """What the answer cost. Reported next to quality, never instead of it."""
    total_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    verification_latency_s: float = 0.0
    llm_calls: int = 0
    retrieval_calls: int = 0
    agent_steps: int = 0
    graph_hops: int = 0
    answer_tokens_estimate: int = 0
    context_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_system(retrieval_result, generation_result, verification=None,
                    extra_llm_calls: int = 0) -> SystemMetrics:
    """Read cost off the trace metadata the strategies already record."""
    meta = retrieval_result.retrieval_metadata if retrieval_result else {}
    answer = getattr(generation_result, "generated_answer", "") or ""

    llm_calls = int(meta.get("ircot_llm_calls", 0)) + int(meta.get("agent_llm_calls", 0))
    llm_calls += 1 if generation_result is not None else 0      # the answer call
    llm_calls += extra_llm_calls

    retrieval_calls = int(meta.get("ircot_retrieval_calls", 0)) or int(
        meta.get("agent_retrieval_calls", 0)) or 1

    context_chars = sum(len(c.chunk_text) for c in
                        getattr(retrieval_result, "retrieved_chunks", []) or [])

    return SystemMetrics(
        total_latency_s=round(
            getattr(retrieval_result, "retrieval_time", 0.0)
            + getattr(generation_result, "generation_time", 0.0), 3),
        retrieval_latency_s=round(getattr(retrieval_result, "retrieval_time", 0.0), 3),
        generation_latency_s=round(getattr(generation_result, "generation_time", 0.0), 3),
        verification_latency_s=round(
            getattr(verification, "total_verification_latency_ms", 0.0) / 1000.0, 3),
        llm_calls=llm_calls,
        retrieval_calls=retrieval_calls,
        agent_steps=int(meta.get("agent_steps", 0)),
        graph_hops=int(meta.get("agent_graph_hops", 0)),
        # ~4 characters per token: a proxy, labelled as one. The endpoint does
        # not return usage, so a measured token count is not available.
        answer_tokens_estimate=len(answer) // 4,
        context_chars=context_chars,
    )


# ---------------------------------------------------------------------------
# The UI metric panel
# ---------------------------------------------------------------------------

#: Provenance for every panel score: which family the name comes from, and what
#: this project actually computes for it. Shown in the UI so a reader is never
#: misled into thinking these are the upstream packages' own outputs.
METRIC_PROVENANCE: Dict[str, Dict[str, str]] = {
    "faithfulness": {"family": "RAGAS", "definition": "supported claims / total claims (strict)",
                     "judge": "local NLI verifier"},
    "answer_relevancy": {"family": "RAGAS", "definition": "mean cosine similarity between the question and N reverse-generated questions",
                         "judge": "LLM + embeddings"},
    "context_precision": {"family": "RAGAS", "definition": "rank-weighted precision of retrieved chunks",
                          "judge": "LLM"},
    "context_recall": {"family": "RAGAS", "definition": "reference sentences entailed by the context",
                       "judge": "local NLI verifier", "requires": "reference answer"},
    "answer_correctness": {"family": "RAGAS", "definition": "claim F1 vs reference, blended with similarity",
                           "judge": "local NLI + embeddings", "requires": "reference answer"},
    "precision": {"family": "RAGChecker", "definition": "supported answer claims / all answer claims",
                  "judge": "local NLI verifier"},
    "recall": {"family": "RAGChecker", "definition": "gold claims recalled by the answer",
               "judge": "local NLI verifier", "requires": "reference answer"},
    "f1": {"family": "RAGChecker", "definition": "harmonic mean of precision and recall",
           "judge": "derived"},
    "hallucination": {"family": "RAGChecker", "definition": "(unsupported + contradicted) claims / total claims",
                      "judge": "local NLI verifier"},
    "context_relevance": {"family": "ARES", "definition": "fraction of retrieved chunks judged relevant to the question",
                          "judge": "LLM"},
    "answer_relevance": {"family": "ARES", "definition": "same computation as RAGAS answer relevancy",
                         "judge": "LLM + embeddings"},
    "answer_faithfulness": {"family": "ARES", "definition": "same computation as RAGAS faithfulness",
                            "judge": "local NLI verifier"},
    "citation_correctness": {"family": "this project", "definition": "1 - (unverifiable citations / total citations)",
                             "judge": "corpus manifest lookup"},
}


def compute_metric_panel(
    question: str,
    answer: str,
    retrieved_chunks: Sequence[Any],
    verification=None,
    reference: Optional[str] = None,
    ragas_evaluator=None,
    answer_correctness=None,
    citation_report=None,
) -> Dict[str, Optional[float]]:
    """Compute the thirteen-metric panel from already-loaded components.

    Every value may be ``None``, which means *not computed* -- because no
    reference answer was supplied, or the judge was unavailable. ``None`` is
    never silently replaced with 0.0 or 1.0: a missing measurement and a bad
    measurement are different things, and conflating them is how evaluation
    harnesses end up rewarding silence.
    """
    panel: Dict[str, Optional[float]] = {name: None for name in METRIC_PROVENANCE}

    if verification is not None and verification.total_claims:
        from src.ragas_metrics import compute_faithfulness_strict

        total = verification.total_claims
        supported = verification.supported_claims
        faithfulness = compute_faithfulness_strict(verification)
        panel["faithfulness"] = faithfulness
        panel["answer_faithfulness"] = faithfulness
        panel["precision"] = round(supported / total, 4)
        panel["hallucination"] = round(
            (verification.unsupported_claims + verification.contradicted_claims) / total, 4)

    if answer_correctness is not None:
        panel["recall"] = round(float(answer_correctness.claim_recall), 4)

    if panel["precision"] is not None and panel["recall"] is not None:
        precision, recall = panel["precision"], panel["recall"]
        panel["f1"] = round(2 * precision * recall / (precision + recall), 4) if (
            precision + recall) else 0.0

    if ragas_evaluator is not None:
        try:
            relevancy = ragas_evaluator.compute_answer_relevancy(question, answer)
            panel["answer_relevancy"] = relevancy
            panel["answer_relevance"] = relevancy
            panel["context_precision"] = ragas_evaluator.compute_context_precision(
                question, answer, list(retrieved_chunks))
            panel["context_relevance"] = ragas_evaluator.compute_context_relevancy(
                question, list(retrieved_chunks))
            if reference:
                panel["context_recall"] = ragas_evaluator.compute_context_recall(
                    reference, list(retrieved_chunks))
                panel["answer_correctness"] = ragas_evaluator.compute_answer_correctness(
                    answer, reference)
        except Exception as exc:
            logger.warning("metric panel: RAGAS-family computation failed: %s", exc)

    if citation_report is not None:
        panel["citation_correctness"] = citation_report.citation_correctness

    return panel


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


def aggregate(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Optional[float]]:
    return {key: mean([row.get(key) for row in rows]) for key in keys}


def aggregate_by(rows: Sequence[Dict[str, Any]], group_key: str,
                 keys: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key)), []).append(row)
    return {group: {**aggregate(items, keys), "n": len(items)}
            for group, items in sorted(groups.items())}


def demo() -> None:
    """Self-check for the metric maths. No models, no corpus."""
    retrieved = ["a", "b", "c", "d", "e", "f"]
    gold = ["c", "z"]

    assert recall_at_k(retrieved, gold, 5) == 0.5
    assert recall_at_k(retrieved, gold, 2) == 0.0
    assert hit_at_k(retrieved, gold, 5) == 1.0
    assert hit_at_k(retrieved, gold, 2) == 0.0
    assert mrr(retrieved, gold) == round(1 / 3, 4)
    assert mrr(["x", "y"], gold) == 0.0
    assert recall_at_k(retrieved, [], 5) is None, "empty gold must be undefined, not zero"

    # nDCG: one gold hit at rank 3 out of two possible, against an ideal of two
    # hits at ranks 1 and 2.
    expected = (1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(retrieved, gold, 10) == round(expected, 4)
    assert ndcg_at_k(["c", "z"], gold, 10) == 1.0, "perfect ranking must score 1.0"

    class FakeRecord:
        def __init__(self, document_id):
            self.metadata = {"document_id": document_id}

    class FakeRegistry:
        def __init__(self, mapping):
            self.mapping = {k: FakeRecord(v) for k, v in mapping.items()}

        def get_chunk(self, chunk_id):
            return self.mapping.get(chunk_id)

    registry = FakeRegistry({"a": "doc1", "b": "doc1", "c": "doc2", "d": "doc3"})
    question = {"id": "q1", "type": "citation_chain", "hops": 3,
                "gold_chunk_ids": ["c", "z"], "gold_document_ids": ["doc2", "doc9"]}

    metrics = evaluate_retrieval(question, retrieved, registry)
    assert metrics.document_recall == 0.5, metrics.to_dict()
    # Partial multi-hop coverage is a failure on the metric that matters.
    assert metrics.all_gold_documents_found == 0.0, metrics.to_dict()

    complete = evaluate_retrieval(
        {"id": "q2", "type": "case_to_case", "hops": 2,
         "gold_chunk_ids": ["a"], "gold_document_ids": ["doc1"]},
        retrieved, registry)
    assert complete.all_gold_documents_found == 1.0
    assert complete.mrr == 1.0

    class FakeVerification:
        total_claims = 10
        supported_claims = 6
        partially_supported_claims = 1
        unsupported_claims = 2
        contradicted_claims = 1
        not_verifiable_claims = 0
        total_verification_latency_ms = 2500.0

    class FakeCitationReport:
        citation_precision = 0.8
        citation_recall = 0.5
        citation_correctness = 0.9
        unverifiable = 1
        total_citations = 10

    generation = evaluate_generation(FakeVerification(), FakeCitationReport(),
                                     "Some answer with content.")
    assert generation.claim_support_rate == 0.6
    assert generation.contradiction_rate == 0.1
    assert generation.unsupported_claim_rate == 0.2
    assert generation.refused is False

    refused = evaluate_generation(None, None, "I do not have enough information to answer this.")
    assert refused.refused is True, "a refusal must be flagged, not scored as a normal answer"

    class FakeAnswerCorrectness:
        claim_recall = 0.5

    panel = compute_metric_panel(
        "q?", "a.", [], verification=FakeVerification(),
        answer_correctness=FakeAnswerCorrectness(), citation_report=FakeCitationReport())

    assert panel["faithfulness"] == 0.6
    assert panel["answer_faithfulness"] == panel["faithfulness"], "ARES/RAGAS aliases must agree"
    assert panel["precision"] == 0.6 and panel["recall"] == 0.5
    assert panel["f1"] == round(2 * 0.6 * 0.5 / 1.1, 4), panel["f1"]
    assert panel["hallucination"] == 0.3
    assert panel["citation_correctness"] == 0.9
    # Reference-based metrics stay None without a reference, never 0.0.
    assert panel["context_recall"] is None and panel["answer_correctness"] is None
    assert set(panel) == set(METRIC_PROVENANCE), "panel and provenance table must not drift"

    class FakeRetrievalResult:
        retrieval_time = 1.5
        retrieved_chunks: list = []
        retrieval_metadata = {"ircot_llm_calls": 2, "ircot_retrieval_calls": 3}

    class FakeGenerationResult:
        generation_time = 4.0
        generated_answer = "x" * 400

    system = evaluate_system(FakeRetrievalResult(), FakeGenerationResult(), FakeVerification())
    assert system.llm_calls == 3, system.to_dict()      # 2 planner + 1 answer
    assert system.retrieval_calls == 3
    assert system.total_latency_s == 5.5
    assert system.answer_tokens_estimate == 100

    rows = [{"recall_at_10": 0.5, "type": "a"}, {"recall_at_10": None, "type": "a"},
            {"recall_at_10": 1.0, "type": "b"}]
    assert aggregate(rows, ["recall_at_10"])["recall_at_10"] == 0.75
    grouped = aggregate_by(rows, "type", ["recall_at_10"])
    assert grouped["a"]["n"] == 2 and grouped["a"]["recall_at_10"] == 0.5

    print("rag_eval demo OK")


if __name__ == "__main__":
    demo()
