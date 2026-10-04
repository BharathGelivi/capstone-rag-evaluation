"""
E6 -- Retrieval-strategy ablation: which technique fixes which retrieval failure.

Eleven arms over the same benchmark, the same corpus and the same gold
evidence. Retrieval only: no generation, no NLI verification. That separation is
deliberate and is what makes the experiment affordable enough to run every arm
on every question -- and what stops a generation difference from being read as a
retrieval difference. Generation is scored separately in E7.

Arms
----
    A_vector                dense bi-encoder only
    A_bm25                  lexical only
    B_hybrid                dense + BM25, RRF fused, no reranker
    C_hybrid_rerank         + cross-encoder reranker      (the shipped pipeline)
    C_fixed_chunking        identical to C, over fixed-size chunks
    D_ircot                 interleaved retrieval/reasoning over C
    E_agentic               bounded controller over C, no graph
    F_graphrag              C, then citation-graph expansion
    G_ircot_graph           D, then citation-graph expansion
    H_agentic_graph         controller with graph actions enabled
    I_full                  IRCoT + graph expansion + contradiction search

Every arm but ``C_fixed_chunking`` runs over the legal-chunked corpus, so
chunking is isolated to exactly one comparison (C vs C_fixed_chunking) rather
than confounded with everything else.

What the design controls for
----------------------------
* **Same gold labels across arms.** Chunk-level gold ids belong to the legal
  registry, so ``C_fixed_chunking`` can only be compared on *document* recall.
  The summary reports that arm separately and says why.
* **Same question set, same order, same seed.** Arms differ in one dimension.
* **Cost recorded per arm.** A technique that buys recall with four extra LLM
  calls is a different recommendation from one that buys it free.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from experiments.common import ExampleSpec, Experiment, ExperimentContext, mean

logger = logging.getLogger(__name__)

BENCHMARK_PATH = os.path.join("eval", "legal_benchmark.json")

#: ``arm -> configuration``. One dict per arm keeps the whole design readable in
#: one screen and makes "what exactly differs between C and D" answerable
#: without reading the executor.
ARMS: Dict[str, Dict[str, Any]] = {
    "A_vector":         {"chunking": "legal", "mode": "vector", "rerank": False, "strategy": "plain"},
    "A_bm25":           {"chunking": "legal", "mode": "bm25",   "rerank": False, "strategy": "plain"},
    "B_hybrid":         {"chunking": "legal", "mode": "hybrid", "rerank": False, "strategy": "plain"},
    "C_hybrid_rerank":  {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "plain"},
    "C_fixed_chunking": {"chunking": "fixed", "mode": "hybrid", "rerank": True,  "strategy": "plain"},
    "D_ircot":          {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "ircot"},
    "E_agentic":        {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "agentic",
                         "graph": False},
    "F_graphrag":       {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "plain",
                         "graph_expand": True},
    "G_ircot_graph":    {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "ircot",
                         "graph_expand": True},
    "H_agentic_graph":  {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "agentic",
                         "graph": True},
    "I_full":           {"chunking": "legal", "mode": "hybrid", "rerank": True,  "strategy": "ircot",
                         "graph_expand": True, "contradiction": True},
}

#: Arms whose chunk ids come from a different registry than the gold labels.
#: Scored on document recall only; chunk-level numbers would be meaningless.
DOCUMENT_LEVEL_ONLY = {"C_fixed_chunking"}

RETRIEVAL_KEYS = [
    "recall_at_5", "recall_at_10", "recall_at_20", "hit_at_10", "mrr",
    "ndcg_at_10", "evidence_recall", "document_recall", "all_gold_documents_found",
]
COST_KEYS = ["latency_s", "llm_calls", "retrieval_calls", "n_retrieved", "graph_added"]

#: Chunks every arm's first retrieval returns. Held constant across arms so a
#: recall difference is a ranking difference, not a budget difference. Wider
#: than the shipped RERANKER_TOP_N because this is a retrieval study, and
#: recall@10 cannot exceed what was retrieved.
BASE_TOP_N = 10

GRAPH_EXPAND_CHUNKS = 6
CONTRADICTION_CHUNKS = 4


# ---------------------------------------------------------------------------
# Arm execution -- shared with E7 so the two experiments cannot drift apart
# ---------------------------------------------------------------------------

def graph_expand(graph, chunk_ids, registry, relations=None, limit: int = GRAPH_EXPAND_CHUNKS):
    if graph is None:
        return []
    from src.legal_graph import expand

    return expand(graph, list(chunk_ids), registry, max_hops=2,
                  max_chunks=limit, relations=relations)


def materialise(chunk_ids, registry):
    """Turn registry records into RetrievedChunks so graph results are the same
    type as retrieval results and need no special-casing downstream."""
    from src.retriever import RetrievedChunk

    out = []
    for chunk_id in chunk_ids:
        record = registry.get_chunk(chunk_id)
        if record is None:
            continue
        out.append(RetrievedChunk(
            chunk_id=chunk_id, similarity_score=0.0, rank=0,
            page_number=str(record.metadata.get("page_number", "")),
            source_file=record.source_file, chunk_index=record.chunk_index,
            chunk_text=record.text, parent_document_id=record.parent_document_id,
        ))
    return out


def execute_arm(arm: str, question_text: str, registry, retriever, graph, llm):
    """Run one arm's retrieval. Returns ``(chunks, retrieval_metadata, graph_added)``.

    Single definition of what each arm *is*, imported by both E6 (retrieval
    scoring) and E7 (generation scoring). Two copies of this logic would let the
    experiments quietly measure different things under the same arm names.
    """
    from src.agentic import ADVERSE_RELATIONS, CONTRADICTION_TERMS, AgenticRetriever
    from src.ircot import IRCoTRetriever

    config = ARMS[arm]
    graph_added = 0

    if config["strategy"] == "plain":
        result = retriever.retrieve(question_text, mode=config["mode"],
                                    rerank=config["rerank"], top_n=BASE_TOP_N)
    elif config["strategy"] == "ircot":
        result = IRCoTRetriever(retriever, llm, retrieval_mode=config["mode"],
                                rerank=config["rerank"],
                                first_hop_top_n=BASE_TOP_N).retrieve(question_text)
    elif config["strategy"] == "agentic":
        result = AgenticRetriever(
            retriever, llm, graph=graph if config.get("graph") else None,
            registry=registry, retrieval_mode=config["mode"], rerank=config["rerank"],
        ).retrieve(question_text)
    else:
        raise ValueError(f"unknown strategy in arm {arm}")

    chunks = list(result.retrieved_chunks)

    if config.get("graph_expand"):
        hits = graph_expand(graph, [c.chunk_id for c in chunks[:6]], registry)
        known = {c.chunk_id for c in chunks}
        new = [h.chunk_id for h in hits if h.chunk_id not in known]
        chunks += materialise(new, registry)
        graph_added += len(new)

    if config.get("contradiction"):
        adverse = retriever.retrieve(f"{question_text} {CONTRADICTION_TERMS}",
                                     mode="bm25", rerank=False, top_n=CONTRADICTION_CHUNKS)
        known = {c.chunk_id for c in chunks}
        chunks += [c for c in adverse.retrieved_chunks if c.chunk_id not in known]
        hits = graph_expand(graph, [c.chunk_id for c in chunks[:6]], registry,
                            relations=ADVERSE_RELATIONS, limit=CONTRADICTION_CHUNKS)
        known = {c.chunk_id for c in chunks}
        new = [h.chunk_id for h in hits if h.chunk_id not in known]
        chunks += materialise(new, registry)
        graph_added += len(new)

    for rank, chunk in enumerate(chunks, start=1):
        chunk.rank = rank
    metadata = dict(result.retrieval_metadata)
    metadata["retrieval_calls"] = (int(metadata.get("ircot_retrieval_calls", 0))
                                   or int(metadata.get("agent_retrieval_calls", 0)) or 1) + int(bool(config.get("contradiction")))
    metadata["initial_retrieval_budget"] = BASE_TOP_N if config["strategy"] != "agentic" else 6
    metadata["n_retrieved"] = len(chunks)
    return chunks, metadata, graph_added


def load_benchmark(path: str = BENCHMARK_PATH) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} not found -- run `python -m scripts.build_benchmark` first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class StrategyAblationExperiment(Experiment):
    key = "exp06_strategy_ablation"
    number = 6
    title = "Retrieval-strategy ablation over a multi-hop legal benchmark"
    claim = (
        "Retrieval techniques do not improve RAG uniformly: each fixes a specific "
        "failure mode, and the ablation localises which."
    )
    # Always runs against the ingested corpus; there is no simulated variant,
    # so the suite's mode flag is irrelevant here.
    supported_modes = ("live",)

    def __init__(self) -> None:
        self._benchmark: Dict[str, Any] = {}
        self._registries: Dict[str, Any] = {}
        self._retrievers: Dict[str, Any] = {}
        self._graph = None
        self._llm = None

    # -- planning --------------------------------------------------------

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        benchmark = load_benchmark(ctx.extra.get("benchmark", BENCHMARK_PATH))
        self._benchmark = benchmark
        arms = ctx.extra.get("arms") or list(ARMS)
        if isinstance(arms, str):
            arms = arms.split(",")

        specs = []
        for arm in arms:
            for question in benchmark["questions"]:
                specs.append(ExampleSpec(
                    example_id=f"{arm}/{question['id']}",
                    payload={"arm": arm, "question_id": question["id"]},
                ))
        return specs

    # -- resources -------------------------------------------------------

    def setup(self, ctx: ExperimentContext) -> None:
        from src.chunk_registry import ChunkRegistry
        from src.generator import Generator
        from src.legal_graph import load_graph, GRAPH_PATH
        from src.retriever import Retriever
        from src.vector_store import ChromaVectorStore
        from scripts.build_legal_corpus import COLLECTIONS, registry_path

        if not self._benchmark:
            self._benchmark = load_benchmark(ctx.extra.get("benchmark", BENCHMARK_PATH))

        needed = {config["chunking"] for arm, config in ARMS.items()
                  if arm in {s.example_id.split("/")[0] for s in self.plan(ctx)}}

        for chunking in sorted(needed):
            logger.info("loading registry + store for chunking=%s", chunking)
            registry = ChunkRegistry.load_from_json(registry_path(chunking))
            store = ChromaVectorStore(collection_name=COLLECTIONS[chunking])
            store.initialize_collection()
            self._registries[chunking] = registry
            # One Retriever per chunking, reused across arms: the retrieval mode
            # and reranker are per-call arguments, so the heavy models load once.
            self._retrievers[chunking] = Retriever(store, registry)

        if os.path.exists(GRAPH_PATH):
            self._graph = load_graph(GRAPH_PATH)
            logger.info("graph loaded: %d nodes", self._graph.number_of_nodes())
        else:
            logger.warning("no knowledge graph at %s -- graph arms will degrade", GRAPH_PATH)

        # The planner/controller LLM is deliberately the small model: these calls
        # choose the next query, they do not write the answer. E7 keeps the 70B
        # model for generation.
        from configs.models import NVIDIA_PLANNER_MODEL

        self._llm = Generator(model_name=NVIDIA_PLANNER_MODEL).llm

    # -- retrieval per arm ------------------------------------------------

    def _graph_expand(self, chunk_ids, registry, relations=None, limit=GRAPH_EXPAND_CHUNKS):
        return graph_expand(self._graph, chunk_ids, registry, relations, limit)

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        from src.rag_eval import evaluate_retrieval

        arm = spec.payload["arm"]
        config = ARMS[arm]
        question = next(q for q in self._benchmark["questions"]
                        if q["id"] == spec.payload["question_id"])
        registry = self._registries[config["chunking"]]

        start = time.time()
        chunks, meta, graph_added = execute_arm(
            arm, question["question"], registry,
            self._retrievers[config["chunking"]], self._graph, self._llm,
        )

        retrieved_ids = [c.chunk_id for c in chunks]
        metrics = evaluate_retrieval(question, retrieved_ids, registry)

        record: Dict[str, Any] = {
            "arm": arm,
            "question_id": question["id"],
            "type": question["type"],
            "hops": question["hops"],
            "chunking": config["chunking"],
            "document_level_only": arm in DOCUMENT_LEVEL_ONLY,
            "latency_s": round(time.time() - start, 3),
            "llm_calls": int(meta.get("ircot_llm_calls", 0)) + int(meta.get("agent_llm_calls", 0)),
            "retrieval_calls": int(meta["retrieval_calls"]),
            "n_retrieved": len(retrieved_ids),
            "graph_added": graph_added,
            "termination_reason": meta.get("ircot_termination_reason")
                                  or meta.get("agent_termination_reason"),
            "agent_actions": meta.get("agent_actions"),
            **metrics.to_dict(),
        }
        # Chunk ids of the gold-hit evidence, so a disputed score can be checked
        # against the actual passage rather than re-run.
        record["gold_hits"] = sorted(set(retrieved_ids) & set(question.get("gold_chunk_ids") or []))
        return record

    # -- aggregation -----------------------------------------------------

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        from src.rag_eval import aggregate, aggregate_by

        by_arm: Dict[str, Any] = {}
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm]
            if not rows:
                continue
            by_arm[arm] = {
                "n": len(rows),
                **aggregate(rows, RETRIEVAL_KEYS),
                **aggregate(rows, COST_KEYS),
                "chunking": rows[0]["chunking"],
                "document_level_only": rows[0]["document_level_only"],
                "by_type": aggregate_by(rows, "type",
                                        ["recall_at_10", "document_recall",
                                         "all_gold_documents_found", "mrr"]),
                "by_hops": aggregate_by(rows, "hops",
                                        ["document_recall", "all_gold_documents_found"]),
            }

        baseline = by_arm.get("C_hybrid_rerank", {})

        def delta(arm: str, key: str) -> Optional[float]:
            a, b = by_arm.get(arm, {}).get(key), baseline.get(key)
            return None if a is None or b is None else round(a - b, 4)

        # Each research question, answered by one controlled comparison.
        answers = {
            "does_bm25_reduce_lexical_misses": {
                # Document recall, not chunk recall: the question is whether the
                # right judgment was found from a verbatim identifier, and the
                # gold chunk for such a question is a headnote that seldom wins
                # a top-10 slot even when the document is correctly retrieved.
                "comparison": "A_bm25 vs A_vector, document recall",
                "overall": {
                    "bm25": by_arm.get("A_bm25", {}).get("document_recall"),
                    "vector": by_arm.get("A_vector", {}).get("document_recall"),
                    "hybrid": by_arm.get("B_hybrid", {}).get("document_recall"),
                },
                "exact_citation": {
                    "bm25": _type_score(by_arm, "A_bm25", "exact_citation", "document_recall"),
                    "vector": _type_score(by_arm, "A_vector", "exact_citation", "document_recall"),
                    "hybrid": _type_score(by_arm, "B_hybrid", "exact_citation", "document_recall"),
                },
                "entity_resolution": {
                    "bm25": _type_score(by_arm, "A_bm25", "entity_resolution", "document_recall"),
                    "vector": _type_score(by_arm, "A_vector", "entity_resolution", "document_recall"),
                    "hybrid": _type_score(by_arm, "B_hybrid", "entity_resolution", "document_recall"),
                },
                "lexical_mismatch": {
                    "bm25": _type_score(by_arm, "A_bm25", "lexical_mismatch", "document_recall"),
                    "vector": _type_score(by_arm, "A_vector", "lexical_mismatch", "document_recall"),
                    "hybrid": _type_score(by_arm, "B_hybrid", "lexical_mismatch", "document_recall"),
                },
            },
            "does_reranking_improve_precision": {
                "comparison": "C_hybrid_rerank vs B_hybrid",
                "delta_mrr": _delta_between(by_arm, "C_hybrid_rerank", "B_hybrid", "mrr"),
                "delta_ndcg_at_10": _delta_between(by_arm, "C_hybrid_rerank", "B_hybrid", "ndcg_at_10"),
                "delta_recall_at_10": _delta_between(by_arm, "C_hybrid_rerank", "B_hybrid", "recall_at_10"),
                "delta_document_recall": _delta_between(
                    by_arm, "C_hybrid_rerank", "B_hybrid", "document_recall"),
            },
            "does_ircot_improve_multi_hop": {
                "comparison": "D_ircot vs C_hybrid_rerank on multi-hop questions",
                "delta_all_gold_documents_found": _multihop_delta(by_arm, "D_ircot", "C_hybrid_rerank"),
                "delta_document_recall": delta("D_ircot", "document_recall"),
                "extra_llm_calls": delta("D_ircot", "llm_calls"),
                "extra_latency_s": delta("D_ircot", "latency_s"),
            },
            "does_graphrag_improve_citation_chains": {
                "comparison": "F_graphrag vs C_hybrid_rerank on case_to_case / citation_chain / temporal",
                "delta_citation_chain_document_recall": _type_delta(
                    by_arm, "F_graphrag", "C_hybrid_rerank", "citation_chain", "document_recall"),
                "delta_case_to_case_document_recall": _type_delta(
                    by_arm, "F_graphrag", "C_hybrid_rerank", "case_to_case", "document_recall"),
                "delta_temporal_document_recall": _type_delta(
                    by_arm, "F_graphrag", "C_hybrid_rerank", "temporal", "document_recall"),
            },
            "does_agentic_improve_evidence_completeness": {
                "comparison": "E_agentic / H_agentic_graph vs C_hybrid_rerank",
                "delta_all_gold_documents_found_no_graph": delta("E_agentic", "all_gold_documents_found"),
                "delta_all_gold_documents_found_with_graph": delta("H_agentic_graph", "all_gold_documents_found"),
                "extra_llm_calls": delta("H_agentic_graph", "llm_calls"),
            },
            "does_legal_chunking_reduce_missing_evidence": {
                "comparison": "C_hybrid_rerank vs C_fixed_chunking, document recall only",
                "legal_document_recall": by_arm.get("C_hybrid_rerank", {}).get("document_recall"),
                "fixed_document_recall": by_arm.get("C_fixed_chunking", {}).get("document_recall"),
                "delta_document_recall": _delta_between(
                    by_arm, "C_hybrid_rerank", "C_fixed_chunking", "document_recall"),
                "note": "chunk-level metrics are not comparable across chunkings; "
                        "document recall is the only sound comparison here",
            },
            "which_combinations_only_add_cost": {
                arm: {"delta_document_recall": delta(arm, "document_recall"),
                      "delta_all_gold_documents_found": delta(arm, "all_gold_documents_found"),
                      "extra_latency_s": delta(arm, "latency_s"),
                      "extra_llm_calls": delta(arm, "llm_calls")}
                for arm in ("D_ircot", "E_agentic", "F_graphrag", "G_ircot_graph",
                            "H_agentic_graph", "I_full")
                if arm in by_arm
            },
        }

        best_overall = max(by_arm.items(),
                           key=lambda kv: (kv[1].get("document_recall") or 0.0))
        best_multihop = max(
            ((arm, _multihop_score(data)) for arm, data in by_arm.items()),
            key=lambda kv: kv[1] or 0.0, default=(None, None))

        return {
            "benchmark_version": self._benchmark.get("benchmark_version"),
            "n_questions": len(self._benchmark.get("questions", [])),
            "headline": {
                "best_arm_by_document_recall": {
                    "arm": best_overall[0],
                    "document_recall": best_overall[1].get("document_recall"),
                },
                "best_arm_on_multi_hop": {"arm": best_multihop[0], "score": best_multihop[1]},
                "baseline_arm": "C_hybrid_rerank",
                "baseline_document_recall": baseline.get("document_recall"),
            },
            "research_questions": answers,
            "by_arm": by_arm,
            "interpretation_notes": [
                "Retrieval only. An arm that retrieves better may still answer worse; "
                "generation is measured separately in E7 and the two must not be pooled.",
                "Gold chunk ids belong to the legal-chunked registry. C_fixed_chunking is "
                "therefore comparable on document recall only, and its chunk-level scores "
                "are reported but must not be read as a chunking result.",
                "Gold evidence is derived from corpus structure (citation edges, reporter "
                "tables), which is weak supervision. It is applied identically to every "
                "arm, so it cannot manufacture a difference between arms, but it is not an "
                "absolute recall figure.",
                "all_gold_documents_found is the multi-hop metric that matters: partial "
                "coverage of a citation chain does not answer the question.",
            ],
        }


def _multihop_score(data: Dict[str, Any]) -> Optional[float]:
    by_hops = data.get("by_hops") or {}
    scores = [v.get("all_gold_documents_found") for k, v in by_hops.items()
              if k.isdigit() and int(k) >= 2 and v.get("all_gold_documents_found") is not None]
    return mean(scores) if scores else None


def _multihop_delta(by_arm: Dict[str, Any], arm: str, baseline: str) -> Optional[float]:
    a, b = _multihop_score(by_arm.get(arm, {})), _multihop_score(by_arm.get(baseline, {}))
    return None if a is None or b is None else round(a - b, 4)


def _type_score(by_arm: Dict[str, Any], arm: str, qtype: str, key: str) -> Optional[float]:
    return (by_arm.get(arm, {}).get("by_type", {}).get(qtype, {}) or {}).get(key)


def _type_delta(by_arm: Dict[str, Any], arm: str, baseline: str,
                qtype: str, key: str) -> Optional[float]:
    a, b = _type_score(by_arm, arm, qtype, key), _type_score(by_arm, baseline, qtype, key)
    return None if a is None or b is None else round(a - b, 4)


def _delta_between(by_arm: Dict[str, Any], arm: str, other: str, key: str) -> Optional[float]:
    a, b = by_arm.get(arm, {}).get(key), by_arm.get(other, {}).get(key)
    return None if a is None or b is None else round(a - b, 4)


EXPERIMENT = StrategyAblationExperiment()
