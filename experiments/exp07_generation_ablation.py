"""
E7 -- Generation ablation: does better retrieval produce a better *answer*?

E6 measures retrieval. It cannot answer the question a user actually has, which
is whether the answer is grounded, correctly cited, and honest about conflict.
An arm can retrieve more evidence and write a worse answer -- more context is
also more opportunity to assert something the evidence does not carry.

Scope, and why it is smaller than E6
------------------------------------
Four arms on a question subset, not eleven arms on everything. Each example here
costs a generation call, a claim-decomposition call and one NLI pass per claim;
running the full grid would take hours and buy little, because E6 already
established which arms differ in *retrieval*. The four arms chosen are the ones
whose retrieval behaviour differs most:

    C_hybrid_rerank   the shipped pipeline (baseline)
    D_ircot           multi-hop retrieval
    F_graphrag        citation-graph expansion
    I_full            IRCoT + graph + contradiction search

Metrics
-------
Claim-level support, unsupported and contradiction rates come from the existing
``ClaimDecomposer`` -> ``ClaimVerifier`` path -- no second claim framework.
Citation precision/correctness come from ``src.citation_check``. The LLM-judged
RAGAS-family metrics are deliberately *not* computed per example here: they add
roughly ten judge calls per row and would dominate the runtime of an experiment
whose question is about grounding, not relevance. The UI computes the full
thirteen-metric panel on demand for a single query instead.

Refusal is recorded, never scored as success. An arm that answers "I do not have
enough information" to everything would post a perfect support rate; E3
established why that must be reported rather than rewarded.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from experiments.common import ExampleSpec, Experiment, ExperimentContext
from experiments.exp06_strategy_ablation import ARMS, execute_arm, load_benchmark

logger = logging.getLogger(__name__)

#: Arms carried forward from E6. Fewer arms, deeper measurement.
GENERATION_ARMS = ("C_hybrid_rerank", "D_ircot", "F_graphrag", "I_full")

#: Question types that most stress grounding and citation behaviour.
PRIORITY_TYPES = ("citation_chain", "case_to_case", "contradictory", "temporal",
                  "exact_citation", "statute_to_cases")

DEFAULT_QUESTIONS_PER_ARM = 14

GENERATION_KEYS = [
    "claim_support_rate", "unsupported_claim_rate", "contradiction_rate",
    "citation_precision", "citation_correctness", "citation_recall",
    "total_claims", "total_citations", "fabricated_citations",
    "answer_chars", "latency_s", "llm_calls",
]

#: Citation-first instructions. The corpus is judgments, so the answer must
#: name authorities -- and must not name any it was not shown.
SYSTEM_INSTRUCTIONS = (
    "You are a legal research assistant answering from retrieved Indian Supreme Court "
    "judgments.\n"
    "\n"
    "CITATION RULES (these are strict)\n"
    "1. Cite an authority ONLY if its citation appears in the retrieved context. Never "
    "recall a case from memory.\n"
    "2. Quote citations exactly as they appear in the context, e.g. [2024] 10 S.C.R. 1.\n"
    "3. Attach the [Chunk-ID] to every proposition you take from the context.\n"
    "4. If the context does not support a point, say so instead of supplying an authority.\n"
    "\n"
    "CONFLICTING AUTHORITY\n"
    "If the retrieved passages point in different directions, say the evidence is "
    "conflicting and set out both positions. Do not resolve a conflict the material does "
    "not resolve.\n"
    "\n"
    "FORMAT\n"
    "Lead with a direct answer, then the supporting authority. Be concise."
)


class GenerationAblationExperiment(Experiment):
    key = "exp07_generation_ablation"
    number = 7
    title = "Generation ablation: grounding and citation correctness by retrieval strategy"
    claim = (
        "Retrieval gains do not automatically become answer quality: grounding and "
        "citation correctness must be measured on the generated answer itself."
    )
    supported_modes = ("live",)

    def __init__(self) -> None:
        self._benchmark: Dict[str, Any] = {}
        self._registry = None
        self._retriever = None
        self._graph = None
        self._generator = None
        self._decomposer = None
        self._verifier = None
        self._corpus_index: Dict[str, Any] = {}

    # -- planning --------------------------------------------------------

    def _questions(self, ctx: ExperimentContext) -> List[Dict[str, Any]]:
        benchmark = self._benchmark or load_benchmark(
            ctx.extra.get("benchmark", os.path.join("eval", "legal_benchmark.json")))
        self._benchmark = benchmark
        per_arm = int(ctx.extra.get("questions_per_arm", DEFAULT_QUESTIONS_PER_ARM))

        # Priority types first, then whatever else fills the budget -- so a small
        # subset still spans the failure modes the arms are supposed to address.
        ordered = [q for t in PRIORITY_TYPES for q in benchmark["questions"] if q["type"] == t]
        ordered += [q for q in benchmark["questions"] if q["type"] not in PRIORITY_TYPES]
        seen: set = set()
        unique = []
        for question in ordered:
            if question["id"] not in seen:
                seen.add(question["id"])
                unique.append(question)
        return unique[:per_arm]

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        questions = self._questions(ctx)
        arms = ctx.extra.get("arms") or GENERATION_ARMS
        if isinstance(arms, str):
            arms = arms.split(",")
        return [
            ExampleSpec(example_id=f"{arm}/{q['id']}",
                        payload={"arm": arm, "question_id": q["id"]})
            for arm in arms for q in questions
        ]

    # -- resources -------------------------------------------------------

    def setup(self, ctx: ExperimentContext) -> None:
        from src.chunk_registry import ChunkRegistry
        from src.citation_check import load_corpus_index
        from src.claim_decomposer import ClaimDecomposer
        from src.claim_verifier import ClaimVerifier
        from src.generator import Generator
        from src.legal_graph import GRAPH_PATH, load_graph
        from src.retriever import Retriever
        from src.vector_store import ChromaVectorStore
        from scripts.build_legal_corpus import COLLECTIONS, registry_path

        self._registry = ChunkRegistry.load_from_json(registry_path("legal"))
        store = ChromaVectorStore(collection_name=COLLECTIONS["legal"])
        store.initialize_collection()
        self._retriever = Retriever(store, self._registry)
        self._graph = load_graph(GRAPH_PATH) if os.path.exists(GRAPH_PATH) else None
        self._generator = Generator(system_instructions=SYSTEM_INSTRUCTIONS)
        # Answers come from the 70B model; retrieval planning from the 8B one.
        from configs.models import NVIDIA_PLANNER_MODEL

        self._planner_llm = Generator(model_name=NVIDIA_PLANNER_MODEL).llm
        self._decomposer = ClaimDecomposer()
        self._verifier = ClaimVerifier()
        self._corpus_index = load_corpus_index()

    # -- execution -------------------------------------------------------

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        from src.citation_check import validate_answer_citations
        from src.rag_eval import evaluate_generation, evaluate_retrieval
        from src.rag_trace import RAGTraceBuilder
        from src.retriever import RetrievalResult

        arm = spec.payload["arm"]
        question = next(q for q in self._benchmark["questions"]
                        if q["id"] == spec.payload["question_id"])

        start = time.time()
        chunks, meta, graph_added = execute_arm(
            arm, question["question"], self._registry, self._retriever,
            self._graph, self._planner_llm,
        )
        retrieval_s = time.time() - start

        retrieval_result = RetrievalResult(
            question=question["question"],
            question_embedding_dimension=int(meta.get("question_embedding_dimension", 0)),
            retrieved_chunks=chunks,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            similarity_scores=[c.similarity_score for c in chunks],
            retrieval_time=retrieval_s, top_k=len(chunks),
            retrieval_metadata={**meta, "arm": arm, "graph_added": graph_added},
        )

        generation = self._generator.generate(retrieval_result)
        if not generation.ok:
            return {"arm": arm, "question_id": question["id"], "type": question["type"],
                    "hops": question["hops"], "error": generation.error,
                    "latency_s": round(time.time() - start, 3)}

        # The trace is the existing RAGTrace, with the strategy's own step log
        # in its diagnostics block -- no competing trace format.
        trace = RAGTraceBuilder.build(retrieval_result, generation,
                                      retrieval_s + generation.generation_time)
        trace.diagnostics = {"arm": arm, "retrieval_metadata": meta,
                             "graph_added": graph_added}
        trace_path = RAGTraceBuilder.save_to_json(trace)

        claims = self._decomposer.decompose(trace)
        verification = self._verifier.verify_all(claims, trace.trace_id, chunks)
        self._verifier.save_artifacts(verification)

        citation_report = validate_answer_citations(
            generation.generated_answer, chunks, self._registry,
            self._corpus_index, gold_chunk_ids=question.get("gold_chunk_ids"),
        )

        generation_metrics = evaluate_generation(
            verification, citation_report, generation.generated_answer)
        retrieval_metrics = evaluate_retrieval(
            question, retrieval_result.retrieved_chunk_ids, self._registry)

        return {
            "arm": arm,
            "question_id": question["id"],
            "type": question["type"],
            "hops": question["hops"],
            "trace_id": trace.trace_id,
            "trace_path": trace_path,
            "latency_s": round(time.time() - start, 3),
            "llm_calls": int(meta.get("ircot_llm_calls", 0))
                         + int(meta.get("agent_llm_calls", 0)) + 2,  # answer + decomposition
            "graph_added": graph_added,
            "document_recall": retrieval_metrics.document_recall,
            "all_gold_documents_found": retrieval_metrics.all_gold_documents_found,
            **generation_metrics.to_dict(),
            "citation_verdicts": [v.to_dict() for v in citation_report.verdicts],
        }

    # -- aggregation -----------------------------------------------------

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        from src.rag_eval import aggregate, aggregate_by

        usable = [r for r in records if not r.get("error")]
        errors = len(records) - len(usable)

        by_arm: Dict[str, Any] = {}
        for arm in dict.fromkeys(r["arm"] for r in usable):
            rows = [r for r in usable if r["arm"] == arm]
            refusals = sum(1 for r in rows if r.get("refused"))
            by_arm[arm] = {
                "n": len(rows),
                **aggregate(rows, GENERATION_KEYS),
                "document_recall": aggregate(rows, ["document_recall"])["document_recall"],
                "refusal_rate": round(refusals / len(rows), 4) if rows else None,
                "answers_with_zero_citations": sum(1 for r in rows if not r.get("total_citations")),
                "by_type": aggregate_by(rows, "type",
                                        ["claim_support_rate", "citation_correctness",
                                         "contradiction_rate"]),
            }

        baseline = by_arm.get("C_hybrid_rerank", {})

        def delta(arm: str, key: str) -> Optional[float]:
            a, b = by_arm.get(arm, {}).get(key), baseline.get(key)
            return None if a is None or b is None else round(a - b, 4)

        return {
            "n_errors": errors,
            "headline": {
                "baseline_arm": "C_hybrid_rerank",
                "baseline_claim_support_rate": baseline.get("claim_support_rate"),
                "baseline_citation_correctness": baseline.get("citation_correctness"),
                "best_claim_support_rate": max(
                    ((arm, d.get("claim_support_rate")) for arm, d in by_arm.items()),
                    key=lambda kv: kv[1] or 0.0, default=(None, None)),
                "best_citation_correctness": max(
                    ((arm, d.get("citation_correctness")) for arm, d in by_arm.items()),
                    key=lambda kv: kv[1] or 0.0, default=(None, None)),
            },
            "research_questions": {
                "does_contradiction_search_reduce_unsupported_claims": {
                    "comparison": "I_full vs C_hybrid_rerank",
                    "delta_unsupported_claim_rate": delta("I_full", "unsupported_claim_rate"),
                    "delta_contradiction_rate": delta("I_full", "contradiction_rate"),
                    "delta_claim_support_rate": delta("I_full", "claim_support_rate"),
                },
                "does_better_retrieval_improve_grounding": {
                    arm: {"delta_claim_support_rate": delta(arm, "claim_support_rate"),
                          "delta_citation_correctness": delta(arm, "citation_correctness"),
                          "delta_document_recall": delta(arm, "document_recall"),
                          "extra_latency_s": delta(arm, "latency_s")}
                    for arm in by_arm if arm != "C_hybrid_rerank"
                },
            },
            "by_arm": by_arm,
            "interpretation_notes": [
                "Refusal rate is reported next to support rate on purpose: a refusal "
                "produces no claims and would otherwise inflate every grounding metric.",
                "citation_precision and citation_correctness are undefined for an answer "
                "that cites nothing; those rows are excluded from the mean rather than "
                "scored 1.0, and answers_with_zero_citations reports how many there were.",
                "UNVERIFIABLE citations mean 'not in this 400-judgment corpus', which is "
                "not the same as 'does not exist'. The label is deliberately weaker than "
                "'hallucinated'.",
                "The LLM-judged RAGAS-family metrics are not computed per example here; "
                "they are available for a single query in the UI panel.",
            ],
        }


EXPERIMENT = GenerationAblationExperiment()
