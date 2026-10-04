"""
Chat pipeline as a plain function, decoupled from Streamlit.

This is `ui/app.py`'s `run_rag_pipeline()` (:235-475) with every
`st.session_state`/`st.cache_resource` reference replaced by explicit
arguments and `functools.lru_cache`, so the same orchestration can be called
from `src/api_ui.py` for the React UI as well as from Streamlit. The 7 steps
(memory retrieval, query condensing, execute_arm retrieval, generation,
RAGTrace build, claim decomposition, NLI verification) are unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import copy
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Generator as TypingGenerator, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.memory_manager import MemoryManager
from src.cache_utils import serialized_cache

logger = logging.getLogger(__name__)

#: Serializes access to GPU-resident models (reranker, NLI verifier) across
#: concurrent request threads -- a single-worker uvicorn process still runs
#: each sync route in its own threadpool thread, and CUDA contexts are not
#: safe to hit from two threads at once.
GPU_LOCK = threading.Lock()

CORPORA = {
    "statutes": (
        os.path.join("artifacts", "chunk_registry.json"),
        "rag_benchmark_collection",
        "Statutes (BNS / BNSS / BSA)",
    ),
    "judgments": (
        os.path.join("artifacts", "legal", "chunk_registry_legal.json"),
        "legal_corpus_legal",
        "Supreme Court judgments (legal corpus)",
    ),
}

#: "both" isn't a real collection -- it retrieves from the two corpora above
#: independently and merges the results (see the ``corpus == "both"`` branch
#: in ``_run_chat_turn``). Exposed as a corpus option in the UI alongside the
#: real ones so a question can be answered from statutes and case law at once.
CORPUS_OPTIONS = list(CORPORA.keys()) + ["both"]
CORPORA["judgments_fixed"] = ("artifacts/legal/chunk_registry_fixed.json", "legal_corpus_fixed", "Judgments (fixed chunks)")

#: Arms whose retrieval strategy is itself multi-step (interleaved
#: retrieval/reasoning, agentic planning, or graph expansion) get a prompted
#: think-aloud generation pass so the UI can show a real reasoning trace.
#: Single-shot arms (A/B/C) rely on the stage/condensation events instead.
REASONING_ARMS = frozenset({
    "D_ircot", "E_agentic", "F_graphrag", "G_ircot_graph", "H_agentic_graph", "I_full",
})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def _registry_signature(corpus):
    path = os.path.join(PROJECT_ROOT, CORPORA[corpus][0])
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


def load_pipeline(corpus: str = "statutes"):
    return _load_pipeline(corpus, _registry_signature(corpus))


@serialized_cache(maxsize=3)
def _load_pipeline(corpus, signature):
    from src.chunk_registry import ChunkRegistry
    from src.vector_store import ChromaVectorStore
    from src.retriever import Retriever
    from src.generator import Generator

    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("Set NVIDIA_API_KEY in the server environment before starting chat")

    relative_registry, collection, _label = CORPORA[corpus]
    registry_path = os.path.join(PROJECT_ROOT, relative_registry)
    if not os.path.exists(registry_path):
        hint = ("Run run_pipeline.py first." if corpus == "statutes"
                else "Run `python -m scripts.build_legal_corpus` first.")
        return None, None, None, f"Corpus '{corpus}' is not ingested. {hint}"

    registry = ChunkRegistry.load_from_json(registry_path)
    vector_store = ChromaVectorStore(collection_name=collection)
    vector_store.initialize_collection()
    retriever = Retriever(vector_store, registry)
    generator = Generator()
    return retriever, generator, registry, None


def load_registry(corpus: str = "statutes"):
    return _load_registry(corpus, _registry_signature(corpus))


@serialized_cache(maxsize=3)
def _load_registry(corpus, signature):
    """Just the chunk registry -- a JSON file load, no vector store or GPU
    reranker/generator involved. For read-only chunk-text lookups (e.g.
    backfilling a historical trace's chunk text, see /ui/trace in api_ui.py)
    that have no business paying full pipeline construction cost. Calling
    load_pipeline() there instead was the bug: fetching a session's traces on
    load fires one lookup per message, and each one loaded a *fresh*
    Cross-Encoder Reranker onto the GPU (lru_cache doesn't dedupe concurrent
    cache misses), piling up until 8.6GB of VRAM was exhausted and the
    process died.
    """
    from src.chunk_registry import ChunkRegistry

    relative_registry, _collection, _label = CORPORA[corpus]
    registry_path = os.path.join(PROJECT_ROOT, relative_registry)
    if not os.path.exists(registry_path):
        return None
    return ChunkRegistry.load_from_json(registry_path)


@serialized_cache(maxsize=1)
def load_knowledge_graph():
    from src.legal_graph import GRAPH_PATH, load_graph

    path = os.path.join(PROJECT_ROOT, GRAPH_PATH)
    if not os.path.exists(path):
        return None
    return load_graph(path)


@serialized_cache(maxsize=1)
def load_verifier():
    from src.claim_verifier import ClaimVerifier
    return ClaimVerifier()


@serialized_cache(maxsize=1)
def load_decomposer():
    from src.claim_decomposer import ClaimDecomposer
    return ClaimDecomposer()


@serialized_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    mm = MemoryManager()
    mm.initialize()
    return mm


def run_chat_turn(
    question: str,
    session_id: str,
    arm: str = "C_hybrid_rerank",
    corpus: str = "statutes",
    chat_history: Optional[List[Dict[str, str]]] = None,
    memory_enabled: bool = True,
    deep_analysis: bool = False,
) -> TypingGenerator[Dict[str, Any], None, None]:
    """Run one chat turn, yielding SSE-ready event dicts.

    Event sequence: ``meta`` -> ``chunks`` -> ``token`` (many) -> ``strategy``
    -> ``done`` (or ``error`` at any point). An uncaught exception anywhere in
    the turn is converted to a terminal ``error`` event rather than silently
    truncating the SSE stream -- the frontend has no other way to tell "the
    connection just ended" apart from "the turn finished cleanly".
    """
    try:
        yield from _run_chat_turn(
            question, session_id, arm, corpus, chat_history, memory_enabled, deep_analysis)
    except Exception as exc:
        reference = uuid.uuid4().hex
        logger.exception("Chat failed [ref=%s]", reference)
        yield {"event": "error", "message": "Chat could not complete. Please retry.", "reference_id": reference}


def _run_chat_turn(
    question: str,
    session_id: str,
    arm: str,
    corpus: str,
    chat_history: Optional[List[Dict[str, str]]],
    memory_enabled: bool,
    deep_analysis: bool,
) -> TypingGenerator[Dict[str, Any], None, None]:
    mm = get_memory_manager()
    pipeline_corpus = "judgments_fixed" if arm == "C_fixed_chunking" else corpus
    if corpus == "both":
        # Combined mode has no single retriever/registry -- each corpus's
        # retriever is pulled from the (lru_cache'd) per-corpus pipeline in
        # the retrieval step below and the results are merged. Only the
        # baseline hybrid-rerank strategy applies (graph/IRCoT arms are built
        # around the judgments-only citation graph and don't have a "both"
        # analogue). The generator is corpus-agnostic, so any one pipeline's
        # is fine for query condensation ahead of retrieval.
        retriever, generator, registry, error = load_pipeline("judgments")
        if error:
            yield {"event": "error", "message": error}
            return
        registry = None
        arm = "C_hybrid_rerank"
    else:
        retriever, generator, registry, error = load_pipeline(pipeline_corpus)
        if error:
            yield {"event": "error", "message": error}
            return

    generator = copy.copy(generator)
    t0 = time.time()
    warnings = []

    yield {"event": "stage", "stage": "memory", "label": "Searching memory…"}

    # Step 1: memory retrieval
    memory_context = ""
    recalled_memories: list = []
    t_mem = time.time()
    if memory_enabled:
        try:
            memory_results = mm.search_memory(question, top_k=3, session_id=session_id)
            memory_context = mm.format_memory_context(memory_results, session_id=session_id)
            recalled_memories = [
                {
                    "question": r.memory.question,
                    "answer": r.memory.answer[:300],
                    "session_id": r.memory.session_id,
                    "timestamp": r.memory.timestamp,
                    "semantic_score": round(r.semantic_score, 4),
                    "recency_score": round(r.recency_score, 4),
                    "frequency_score": round(r.frequency_score, 4),
                    "importance_score": round(r.importance_score, 4),
                    "final_score": round(r.final_score, 4),
                }
                for r in memory_results
            ]
        except Exception:
            logger.exception("Memory recall failed")
            warnings.append("Memory recall was unavailable")
            memory_context = ""
    memory_time = time.time() - t_mem
    yield {
        "event": "memory",
        "recalled": recalled_memories,
        "memory_time": round(memory_time, 3),
    }

    yield {"event": "stage", "stage": "condense", "label": "Condensing query…"}

    # Step 1b: condense follow-ups into a standalone retrieval query
    history = (chat_history or [])[-8:]
    search_query = question
    if history:
        try:
            search_query = generator.condense_query(question, history)
        except Exception:
            search_query = question

    yield {
        "event": "meta",
        "search_query": search_query,
        "query_was_condensed": search_query != question,
        "arm": arm,
    }

    yield {"event": "stage", "stage": "retrieval", "label": "Retrieving documents…"}

    # Step 2: retrieval via execute_arm -- identical to the ablation experiment
    t1 = time.time()
    from experiments.exp06_strategy_ablation import ARMS, execute_arm
    from src.retriever import RetrievalResult

    # Every branch here does GPU cross-encoder reranking (directly, or inside
    # execute_arm), so the lock must cover the whole step -- restricting it to
    # just the execute_arm branch left C_hybrid_rerank's direct retrieve()
    # call free to race a concurrent request's rerank pass on the same GPU
    # context, which surfaced as silently truncated SSE streams under load.
    with GPU_LOCK:
        if corpus == "both":
            r_stat, _, _, err_stat = load_pipeline("statutes")
            r_judg, _, _, err_judg = load_pipeline("judgments")
            if err_stat or err_judg:
                yield {"event": "error", "message": err_stat or err_judg}
                return
            res_stat = r_stat.retrieve(search_query)
            res_judg = r_judg.retrieve(search_query)
            # Each corpus already ran its own hybrid+rerank pass, so the merge
            # is just re-sorting the two reranked lists together and cutting
            # to one corpus's top_k -- no re-ranking across corpora needed.
            merged = sorted(
                res_stat.retrieved_chunks + res_judg.retrieved_chunks,
                key=lambda c: c.reranker_score, reverse=True,
            )[: r_judg.top_k]
            retrieval_result = RetrievalResult(
                question=search_query,
                question_embedding_dimension=res_judg.question_embedding_dimension,
                retrieved_chunks=merged,
                retrieved_chunk_ids=[c.chunk_id for c in merged],
                similarity_scores=[c.similarity_score for c in merged],
                retrieval_time=time.time() - t1,
                top_k=len(merged),
                retrieval_metadata={
                    "arm": arm,
                    "combined_corpora": ["statutes", "judgments"],
                },
            )
        elif arm == "C_hybrid_rerank":
            retrieval_result = retriever.retrieve(search_query)
        else:
            needs_graph = ARMS[arm].get("graph") or ARMS[arm].get("graph_expand")
            graph = load_knowledge_graph() if needs_graph else None
            if needs_graph and graph is None:
                yield {"event": "error", "message": "This strategy requires an ingested legal knowledge graph."}
                return
            chunks, strategy_meta, graph_added = execute_arm(
                arm, search_query, registry, retriever, graph, generator.llm)
            retrieval_result = RetrievalResult(
                question=search_query,
                question_embedding_dimension=int(strategy_meta.get("question_embedding_dimension", 0)),
                retrieved_chunks=chunks,
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
                similarity_scores=[c.similarity_score for c in chunks],
                retrieval_time=time.time() - t1, top_k=len(chunks),
                retrieval_metadata={**strategy_meta, "arm": arm, "graph_added": graph_added},
            )

    from experiments.exp06_strategy_ablation import ARMS
    config = ARMS[arm]
    retrieval_result.retrieval_metadata.update({"corpus": corpus, "arm": arm,
        "registry_path": CORPORA[pipeline_corpus][0] if pipeline_corpus in CORPORA else None,
        "chunking_strategy": config["chunking"], "retrieval_mode": config["mode"],
        "bm25_enabled": config["mode"] in ("hybrid", "bm25"), "reranker_enabled": config["rerank"]})
    retrieval_time = time.time() - t1
    yield {
        "event": "chunks",
        "retrieval_time": round(retrieval_time, 3),
        "retrieval_metadata": retrieval_result.retrieval_metadata or {},
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "source_file": os.path.basename(c.source_file or ""),
                "page_number": c.page_number,
                "similarity_score": round(c.similarity_score, 4),
                "text": c.chunk_text[:400],
            }
            for c in retrieval_result.retrieved_chunks
        ],
    }

    # Step 3: streaming generation
    from configs.prompts import LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS

    system_instructions = LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS
    if memory_context:
        system_instructions += (
            "\n\nRELEVANT PAST CONVERSATIONS (recalled from long-term memory; "
            "cite as [Memory] if you use them):\n" + memory_context
        )

    yield {"event": "stage", "stage": "generating", "label": "Generating answer…"}

    # Multi-step arms (interleaved retrieval, agentic planning, graph
    # expansion) are where a visible reasoning trace is actually informative;
    # single-shot retrieval arms get the pipeline's own stage/condensation
    # trace instead (already sent above) rather than paying the extra tokens
    # and latency for an elicited chain-of-thought that wouldn't say much.
    want_reasoning = arm in REASONING_ARMS

    t2 = time.time()
    for item in generator.generate_stream(
        retrieval_result,
        system_instructions=system_instructions,
        chat_history=history,
        question_override=question,
        reasoning=want_reasoning,
    ):
        if want_reasoning:
            kind, delta = item
            yield {"event": "reasoning" if kind == "reasoning" else "token", "text": delta}
        else:
            yield {"event": "token", "text": item}

    generation_result = generator.last_stream_result
    generation_time = time.time() - t2

    if generation_result.error:
        logger.error("Generation failed: %s", generation_result.error)
        yield {"event": "error", "message": "Generation failed. Please retry."}
        return

    # Step 4: RAGTrace
    from src.rag_trace import RAGTraceBuilder
    trace = RAGTraceBuilder.build(
        retrieval_result, generation_result,
        retrieval_result.retrieval_time + generation_result.generation_time,
    )
    trace_path = RAGTraceBuilder.save_to_json(trace)

    strategy_event = {
        "event": "strategy",
        "arm": arm,
        "retrieval_metadata": retrieval_result.retrieval_metadata,
        "trace_id": trace.trace_id,
    }

    claim_count = 0
    if deep_analysis:
        # Steps 5-6: claim decomposition + NLI verification, opt-in -- same
        # 30-115s/claim cost on CPU as the Streamlit path, so gated identically.
        try:
            decomposer = load_decomposer()
            claim_set = decomposer.decompose(trace)
            trace.diagnostics["decomposition_success"] = claim_set.metadata.get("diagnostics", {}).get("success", True)
            claim_count = claim_set.total_candidates
            if not claim_set.candidate_claims:
                strategy_event["verification"] = {"claim_count": 0, "results": []}
            else:
                with GPU_LOCK:
                    verifier = load_verifier()
                    verification = verifier.verify_all(
                        claim_set, trace.trace_id, retrieval_result.retrieved_chunks)
                strategy_event["verification"] = {
                    "claim_count": claim_count,
                    "results": [
                        {
                            "claim_id": vr.claim_id,
                            "claim_text": vr.claim_text,
                            "status": getattr(vr.verification_status, "value", str(vr.verification_status)),
                            "reason": vr.verification_reason,
                            "confidence": vr.confidence,
                            "entailment_score": vr.entailment_score,
                            "contradiction_score": vr.contradiction_score,
                            "neutral_score": vr.neutral_score,
                            "best_chunk_id": vr.best_chunk_id,
                            "best_chunk_rank": vr.best_chunk_rank,
                            "evidence_text": vr.evidence_text,
                            "verified_by": vr.verified_by,
                        }
                        for vr in verification.results
                    ],
                }
        except Exception as exc:
            logger.exception("Claim analysis failed")
            strategy_event["claim_error"] = "Claim analysis was unavailable"
            warnings.append("Claim analysis was unavailable")

        # The trace file was already saved above, before claim decomposition
        # ran -- without this, verification results only ever existed on the
        # live SSE `strategy` event and vanished the moment the page was
        # reloaded or the session revisited (same class of bug the chunk-text
        # backfill in /ui/trace fixed for retrieval, just for claims instead).
        if "verification" in strategy_event or "claim_error" in strategy_event:
            try:
                with open(trace_path, "r", encoding="utf-8") as f:
                    trace_json = json.load(f)
                if "verification" in strategy_event:
                    trace_json.setdefault("diagnostics", {})["claim_verification"] = strategy_event["verification"]
                if "claim_error" in strategy_event:
                    trace_json.setdefault("diagnostics", {})["claim_error"] = strategy_event["claim_error"]
                trace_json["diagnostics"]["decomposition_success"] = trace.diagnostics.get("decomposition_success", False)
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(trace_json, f, indent=2)
            except Exception:
                logger.exception("Diagnostic persistence failed")
                warnings.append("Diagnostic persistence failed")

    yield strategy_event

    # Step 5: Compute Standard Evaluation Metrics (Groundedness, Faithfulness, Relevancy, Precision)
    import re
    import numpy as np

    def cosine_sim(a, b):
        a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        na, nb = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
        return float(np.dot(a_arr, b_arr) / (na * nb)) if (na > 0 and nb > 0) else 0.0

    # 1. Answer Relevancy
    answer_relevancy = None
    try:
        if getattr(retriever, "embed_model", None):
            with GPU_LOCK:
                q_emb = retriever.embed_model.get_text_embedding(question)
                a_emb = retriever.embed_model.get_text_embedding(generation_result.generated_answer[:600])
            answer_relevancy = round(max(0.0, min(1.0, cosine_sim(q_emb, a_emb))), 4)
    except Exception:
        pass

    # 2. Context Precision & Context Relevancy
    scores = [c.similarity_score for c in retrieval_result.retrieved_chunks]
    context_precision = None
    context_relevancy = None

    # 3. Citation Analysis
    chunk_pattern = re.compile(r'\[(?:Chunk-ID:\s*|Chunk:\s*)([a-f0-9\-]+)\]', re.IGNORECASE)
    cited_ids = set(chunk_pattern.findall(generation_result.generated_answer))
    retrieved_ids = set(retrieval_result.retrieved_chunk_ids)
    if cited_ids:
        grounded_citations = cited_ids.intersection(retrieved_ids)
        citation_precision = round(len(grounded_citations) / len(cited_ids), 4)
    else:
        citation_precision = None

    # 4. Groundedness & Faithfulness
    ans_text = generation_result.generated_answer.lower()
    combined_context = " ".join(c.chunk_text.lower() for c in retrieval_result.retrieved_chunks)
    sentences = [s.strip() for s in re.split(r'[.!?\n]+', ans_text) if len(s.strip()) > 15]
    grounded_count = 0
    stopwords = {'that', 'this', 'with', 'from', 'which', 'under', 'shall', 'have', 'been', 'about', 'there', 'their'}
    for s in sentences:
        words = [w for w in re.findall(r'\b\w{4,}\b', s) if w not in stopwords]
        if words:
            overlap = sum(1 for w in words if w in combined_context) / len(words)
            if overlap >= 0.45:
                grounded_count += 1
    groundedness = round(grounded_count / len(sentences), 4) if sentences else None
    faithfulness = None
    if deep_analysis and strategy_event.get("verification", {}).get("results"):
        results = strategy_event["verification"]["results"]
        faithfulness = round(sum(r["status"] == "SUPPORTED" for r in results) / len(results), 4)

    from src.device import describe_device
    device_name = describe_device()

    yield {
        "event": "evaluation",
        "groundedness": groundedness,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_relevancy": context_relevancy,
        "citation_precision": citation_precision,
        "citations_found": len(cited_ids),
        "device": device_name,
        "metric_methods": {"groundedness": "lexical_overlap_heuristic", "answer_relevancy": "embedding_cosine",
                           "faithfulness": "verified_supported_claim_fraction", "context_precision": "not_computed",
                           "context_relevancy": "not_computed"},
    }

    # Step 7: save to memory -- unconditional, same as the fast path in
    # ui/app.py, so a turn is remembered whether or not deep analysis ran.
    claim_ids = [r["claim_id"] for r in strategy_event.get("verification", {}).get("results", [])]
    if memory_enabled:
        try:
            mm.save_interaction(
                question=question,
                answer=generation_result.generated_answer,
                session_id=session_id,
                trace_id=trace.trace_id,
                retrieved_chunk_ids=retrieval_result.retrieved_chunk_ids,
                claim_ids=claim_ids,
            )
        except Exception:
            logger.exception("Interaction persistence failed")
            warnings.append("The interaction could not be saved to memory")

    yield {
        "event": "done",
        "trace_id": trace.trace_id,
        "answer": generation_result.generated_answer,
        "total_time": round(time.time() - t0, 3),
        "generation_time": round(generation_time, 3),
        "retrieval_time": round(retrieval_time, 3),
        "memory_time": round(memory_time, 3),
        "status": "partial" if warnings or generation_result.generation_metadata.get("finish_reason") == "length" else "completed",
        "warnings": warnings,
    }
