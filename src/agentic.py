"""
Agentic RAG -- a small, bounded retrieval controller.

This is a *controller*, not an agent framework. It has a fixed action set, a
hard step budget, no tool registry, no planner/executor split, no memory of its
own, and no ability to call itself. Every loop it can run terminates in at most
``MAX_AGENT_STEPS`` decisions and ``MAX_RETRIEVAL_ROUNDS`` retrievals, which is
what makes its cost bounded and its behaviour reportable.

Why have it at all, given IRCoT
-------------------------------
IRCoT always does the same thing: reformulate, retrieve, repeat. It cannot
decide that a question needs an *exact citation lookup* rather than a semantic
one, or that the missing evidence is structural (who later cited this) rather
than textual, or that the honest answer requires looking for authority that
*contradicts* what has been found. The controller's contribution is action
selection; if the ablation shows it never beats IRCoT, that is a result worth
publishing, and the step log is what makes the claim checkable either way.

Actions
-------
``semantic_search``      dense/hybrid retrieval over a reformulated query
``lexical_search``       BM25-only, for exact citations, section numbers, names
``graph_expand``         one to ``MAX_GRAPH_HOPS`` hops along citation edges
``find_citing_cases``    reverse citation lookup for a specific authority
``contradiction_search`` deliberately hunt contrary / limiting authority
``stop``                 declare the evidence sufficient

Only structured decisions are recorded -- action, argument, why (one short
public sentence), what came back. No free-form chain-of-thought is stored.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

from src.ircot import structured_call, _format_evidence
from src.retriever import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)

MAX_AGENT_STEPS = 6
MAX_RETRIEVAL_ROUNDS = 5
MAX_GRAPH_HOPS = 2
CHUNKS_PER_ACTION = 3

ACTIONS = (
    "semantic_search",
    "lexical_search",
    "graph_expand",
    "find_citing_cases",
    "contradiction_search",
    "stop",
)

#: Treatment relations that indicate an authority was limited rather than
#: applied. These are what a contradiction search is actually looking for.
ADVERSE_RELATIONS = (
    "CASE_OVERRULES_CASE",
    "CASE_DISTINGUISHES_CASE",
    "CASE_DOUBTS_CASE",
    "CASE_DOES_NOT_FOLLOW_CASE",
)

#: Query terms that surface passages where a court declined to follow something.
#: Lexical rather than semantic on purpose: "distinguished" and "overruled" are
#: terms of art, and an embedding of the claim retrieves passages that *agree*
#: with it -- which is precisely the bias contradiction search exists to correct.
CONTRADICTION_TERMS = "overruled distinguished not applicable does not apply contrary view per incuriam"

_DECIDE_PROMPT = """You are directing evidence retrieval for a legal research question.

Question: {question}

Evidence gathered so far ({n_chunks} passages):
{evidence}

Actions already taken: {history}
Remaining retrieval budget: {budget}

Choose ONE next action:
- "semantic_search": meaning-based search. Use for concepts and doctrines. arg = search text.
- "lexical_search": exact-term search. Use for citations, section numbers, case names. arg = exact terms.
- "graph_expand": follow citation links from the cases already retrieved. arg = "".
- "find_citing_cases": find judgments that cite a specific authority. arg = the citation, e.g. "(2013) 5 SCC 762".
- "contradiction_search": look for authority that limits, distinguishes or overrules what has been found. arg = the proposition to challenge.
- "stop": the evidence is sufficient, or further search will not help.

Choose "stop" if the evidence already answers the question. Choose
"contradiction_search" at least once before stopping if the answer states a
legal proposition that later authority might have limited.

Return ONLY this JSON object, no other text:
{{"action": "<one of the actions above>",
  "arg": "<argument, or empty string>",
  "why": "<one short sentence, no reasoning chain>"}}"""


@dataclass
class AgentStep:
    """One controller decision and its outcome."""
    step: int
    action: str
    arg: str
    why: str = ""
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    new_evidence_chunk_ids: List[str] = field(default_factory=list)
    n_new_evidence: int = 0
    graph_hops: int = 0
    relation_paths: List[List[str]] = field(default_factory=list)
    contradiction_candidates: List[str] = field(default_factory=list)
    latency_s: float = 0.0
    llm_calls: int = 0
    retrieval_calls: int = 0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgenticRetriever:
    """Bounded controller over an existing retriever and (optionally) the graph."""

    def __init__(
        self,
        retriever,
        llm,
        graph=None,
        registry=None,
        max_steps: int = MAX_AGENT_STEPS,
        max_retrieval_rounds: int = MAX_RETRIEVAL_ROUNDS,
        max_graph_hops: int = MAX_GRAPH_HOPS,
        retrieval_mode: Optional[str] = None,
        rerank: Optional[bool] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.graph = graph
        self.registry = registry
        self.max_steps = max_steps
        self.max_retrieval_rounds = max_retrieval_rounds
        self.max_graph_hops = max_graph_hops
        self.retrieval_mode = retrieval_mode
        self.rerank = rerank

    # -- actions ---------------------------------------------------------

    def _search(self, query: str, mode: Optional[str], top_n: int) -> List[RetrievedChunk]:
        result = self.retriever.retrieve(
            query, mode=mode or self.retrieval_mode, rerank=self.rerank, top_n=top_n
        )
        return list(result.retrieved_chunks)

    def _graph_expand(self, seed_ids: Sequence[str], relations=None):
        if self.graph is None or self.registry is None:
            return []
        from src.legal_graph import expand

        return expand(self.graph, list(seed_ids), self.registry,
                      max_hops=self.max_graph_hops, max_chunks=CHUNKS_PER_ACTION * 2,
                      relations=relations)

    def _chunks_from_ids(self, chunk_ids: Sequence[str], score: float = 0.0) -> List[RetrievedChunk]:
        """Materialise registry records as RetrievedChunks so graph results are
        the same type as retrieval results and need no special-casing later."""
        out = []
        for chunk_id in chunk_ids:
            record = self.registry.get_chunk(chunk_id) if self.registry else None
            if record is None:
                continue
            out.append(RetrievedChunk(
                chunk_id=chunk_id, similarity_score=score, rank=0,
                page_number=str(record.metadata.get("page_number", "")),
                source_file=record.source_file, chunk_index=record.chunk_index,
                chunk_text=record.text,
                parent_document_id=record.parent_document_id,
            ))
        return out

    def _find_citing(self, citation_arg: str) -> List[str]:
        if self.graph is None:
            return []
        from src.legal_corpus import CITATION_PATTERNS, normalize_case_citation
        from src.legal_graph import find_citing_cases

        key = None
        for reporter, pattern in CITATION_PATTERNS.items():
            match = pattern.search(citation_arg)
            if match:
                key = normalize_case_citation(reporter, match.groups())
                break
        if key is None:
            return []
        chunk_ids: List[str] = []
        for row in find_citing_cases(self.graph, key):
            document_id = row.get("document_id")
            if not document_id or self.registry is None:
                continue
            for record in self.registry._records.values():
                if record.metadata.get("document_id") == document_id:
                    chunk_ids.append(record.chunk_id)
                    break
        return chunk_ids[: CHUNKS_PER_ACTION * 2]

    # -- loop ------------------------------------------------------------

    def retrieve(self, question: str) -> RetrievalResult:
        start = time.time()
        steps: List[AgentStep] = []
        evidence: List[RetrievedChunk] = []
        seen: set = set()
        contradiction_ids: set = set()
        llm_calls = 0
        retrieval_calls = 0
        history: List[str] = []

        # Always ground the loop in one ordinary retrieval; letting the model
        # choose the first action wastes a call to rediscover the obvious.
        first_result = self.retriever.retrieve(
            question, mode=self.retrieval_mode, rerank=self.rerank
        )
        retrieval_calls += 1
        for chunk in first_result.retrieved_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                evidence.append(chunk)
        steps.append(AgentStep(
            step=0, action="semantic_search", arg=question,
            why="ground the loop in a direct retrieval",
            retrieved_chunk_ids=[c.chunk_id for c in first_result.retrieved_chunks],
            new_evidence_chunk_ids=[c.chunk_id for c in first_result.retrieved_chunks],
            n_new_evidence=len(first_result.retrieved_chunks),
            latency_s=round(time.time() - start, 3), retrieval_calls=1,
        ))
        history.append("semantic_search")

        termination = "max_steps"

        for step_number in range(1, self.max_steps + 1):
            if retrieval_calls >= self.max_retrieval_rounds:
                termination = "retrieval_budget_exhausted"
                break

            step_start = time.time()
            decision = structured_call(
                self.llm,
                _DECIDE_PROMPT.format(
                    question=question, n_chunks=len(evidence),
                    evidence=_format_evidence(evidence),
                    history=", ".join(history[-6:]) or "none",
                    budget=self.max_retrieval_rounds - retrieval_calls,
                ),
                default={"action": "stop", "arg": "", "why": ""},
            )
            llm_calls += 1

            if decision.get("_error"):
                steps.append(AgentStep(
                    step=step_number, action="stop", arg="",
                    why="controller unavailable", note=str(decision["_error"])[:160],
                    latency_s=round(time.time() - step_start, 3), llm_calls=1,
                ))
                termination = "controller_error"
                break

            action = str(decision.get("action", "stop")).strip()
            arg = str(decision.get("arg", "") or "").strip()
            why = str(decision.get("why", ""))[:200]

            if action not in ACTIONS:
                steps.append(AgentStep(
                    step=step_number, action="stop", arg=arg, why=why,
                    note=f"invalid action {action!r}",
                    latency_s=round(time.time() - step_start, 3), llm_calls=1,
                ))
                termination = "invalid_action"
                break

            if action == "stop":
                steps.append(AgentStep(
                    step=step_number, action="stop", arg="", why=why,
                    latency_s=round(time.time() - step_start, 3), llm_calls=1,
                ))
                termination = "agent_stop"
                break

            step = AgentStep(step=step_number, action=action, arg=arg, why=why, llm_calls=1)
            hits: List[RetrievedChunk] = []

            if action == "semantic_search":
                hits = self._search(arg or question, "hybrid", CHUNKS_PER_ACTION)
                step.retrieval_calls = 1
                retrieval_calls += 1
            elif action == "lexical_search":
                hits = self._search(arg or question, "bm25", CHUNKS_PER_ACTION)
                step.retrieval_calls = 1
                retrieval_calls += 1
            elif action == "contradiction_search":
                # Two complementary probes: lexical terms of art, and the
                # graph's own adverse-treatment edges.
                probe = f"{arg} {CONTRADICTION_TERMS}".strip()
                hits = self._search(probe, "bm25", CHUNKS_PER_ACTION)
                step.retrieval_calls = 1
                retrieval_calls += 1
                graph_hits = self._graph_expand([c.chunk_id for c in evidence[:5]],
                                                relations=ADVERSE_RELATIONS)
                hits += self._chunks_from_ids([h.chunk_id for h in graph_hits])
                step.graph_hops = max([h.hops for h in graph_hits], default=0)
                step.relation_paths = [h.relation_path for h in graph_hits]
                for chunk in hits:
                    contradiction_ids.add(chunk.chunk_id)
                step.contradiction_candidates = [c.chunk_id for c in hits]
            elif action == "graph_expand":
                graph_hits = self._graph_expand([c.chunk_id for c in evidence[:5]])
                hits = self._chunks_from_ids([h.chunk_id for h in graph_hits])
                step.graph_hops = max([h.hops for h in graph_hits], default=0)
                step.relation_paths = [h.relation_path for h in graph_hits]
                if not hits:
                    step.note = "graph unavailable or no linked documents"
            elif action == "find_citing_cases":
                hits = self._chunks_from_ids(self._find_citing(arg))
                if not hits:
                    step.note = "no citing cases in corpus for that authority"

            new_ids = []
            for chunk in hits:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    evidence.append(chunk)
                    new_ids.append(chunk.chunk_id)

            step.retrieved_chunk_ids = [c.chunk_id for c in hits]
            step.new_evidence_chunk_ids = new_ids
            step.n_new_evidence = len(new_ids)
            step.latency_s = round(time.time() - step_start, 3)
            steps.append(step)
            history.append(action)

        evidence.sort(key=lambda c: c.similarity_score, reverse=True)
        for rank, chunk in enumerate(evidence, start=1):
            chunk.rank = rank

        actions_taken = [s.action for s in steps]
        return RetrievalResult(
            question=question,
            question_embedding_dimension=first_result.question_embedding_dimension,
            retrieved_chunks=evidence,
            retrieved_chunk_ids=[c.chunk_id for c in evidence],
            similarity_scores=[c.similarity_score for c in evidence],
            retrieval_time=time.time() - start,
            top_k=len(evidence),
            retrieval_metadata={
                **first_result.retrieval_metadata,
                "strategy": "agentic",
                "agent_steps": len(steps),
                "agent_actions": actions_taken,
                "agent_termination_reason": termination,
                "agent_llm_calls": llm_calls,
                "agent_retrieval_calls": retrieval_calls,
                "agent_graph_hops": max([s.graph_hops for s in steps], default=0),
                "agent_contradiction_searched": "contradiction_search" in actions_taken,
                "agent_contradiction_chunk_ids": sorted(contradiction_ids),
                "agent_total_evidence": len(evidence),
                "agent_step_log": [s.to_dict() for s in steps],
            },
        )


def demo() -> None:
    """Self-check with a scripted controller, a fake corpus and a real graph."""
    import networkx as nx

    def chunk(chunk_id, text, score=0.5):
        return RetrievedChunk(chunk_id=chunk_id, similarity_score=score, rank=1,
                              page_number="1", source_file="x.pdf", chunk_index=0,
                              chunk_text=text)

    corpus = {
        "principle of promissory estoppel": [chunk("c1", "The doctrine of promissory estoppel binds the State.", 0.9)],
        "(2013) 5 SCC 762": [chunk("c2", "Exact citation match passage.", 0.8)],
        "promissory estoppel " + CONTRADICTION_TERMS: [chunk("c3", "This court distinguished the earlier view.", 0.6)],
    }

    class FakeRetriever:
        def __init__(self):
            self.modes = []

        def retrieve(self, query, mode=None, rerank=None, top_n=None):
            self.modes.append((query, mode))
            hits = corpus.get(query.strip(), [])
            return RetrievalResult(
                question=query, question_embedding_dimension=768,
                retrieved_chunks=list(hits),
                retrieved_chunk_ids=[c.chunk_id for c in hits],
                similarity_scores=[c.similarity_score for c in hits],
                retrieval_time=0.01, top_k=len(hits),
                retrieval_metadata={"retrieval_mode": mode or "hybrid"},
            )

    class ScriptedLLM:
        def __init__(self, script):
            self.script = list(script)

        def chat(self, messages):
            payload = self.script.pop(0) if self.script else {"action": "stop", "arg": "", "why": "done"}

            class R:
                class message:
                    content = json.dumps(payload)
            return R()

    class FakeRecord:
        def __init__(self, chunk_id, document_id, text):
            self.chunk_id = chunk_id
            self.text = text
            self.source_file = "x.pdf"
            self.chunk_index = 0
            self.parent_document_id = document_id
            self.metadata = {"document_id": document_id, "page_number": "1"}

    class FakeRegistry:
        def __init__(self, records):
            self._records = {r.chunk_id: r for r in records}

        def get_chunk(self, chunk_id):
            return self._records.get(chunk_id)

    registry = FakeRegistry([
        FakeRecord("c1", "docA", "The doctrine of promissory estoppel binds the State."),
        FakeRecord("g1", "docB", "A later judgment limiting the doctrine."),
    ])

    graph = nx.MultiDiGraph()
    graph.add_node("case:A", type="case", document_id="docA", chunk_ids=["c1"], in_corpus=True)
    graph.add_node("case:B", type="case", document_id="docB", chunk_ids=["g1"], in_corpus=True)
    graph.add_edge("case:A", "case:B", relation="CASE_DISTINGUISHES_CASE",
                   document_id="docA", chunk_id="c1", source_text="distinguished",
                   source_url="https://example.invalid/a.pdf")

    retriever = FakeRetriever()
    agent = AgenticRetriever(
        retriever,
        ScriptedLLM([
            {"action": "lexical_search", "arg": "(2013) 5 SCC 762", "why": "exact citation lookup"},
            {"action": "contradiction_search", "arg": "promissory estoppel", "why": "check for limiting authority"},
            {"action": "stop", "arg": "", "why": "evidence sufficient"},
        ]),
        graph=graph, registry=registry,
    )
    result = agent.retrieve("principle of promissory estoppel")
    meta = result.retrieval_metadata

    assert meta["strategy"] == "agentic"
    assert meta["agent_termination_reason"] == "agent_stop", meta
    assert meta["agent_actions"] == [
        "semantic_search", "lexical_search", "contradiction_search", "stop"], meta["agent_actions"]
    # lexical_search must actually route to the BM25 arm, not silently to hybrid.
    assert ("(2013) 5 SCC 762", "bm25") in retriever.modes, retriever.modes
    assert meta["agent_contradiction_searched"] is True
    # Contradiction search reached the adverse-treatment neighbour through the graph.
    assert "g1" in meta["agent_contradiction_chunk_ids"], meta["agent_contradiction_chunk_ids"]
    assert "g1" in result.retrieved_chunk_ids
    assert meta["agent_graph_hops"] >= 1

    for entry in meta["agent_step_log"]:
        assert set(entry) == set(AgentStep(0, "", "").to_dict())
        assert len(entry["why"]) <= 200          # decisions, not chain-of-thought

    # Budgets are hard: a controller that never stops still terminates.
    class NeverStops:
        def chat(self, messages):
            class R:
                class message:
                    content = json.dumps({"action": "semantic_search", "arg": "x", "why": "more"})
            return R()

    bounded = AgenticRetriever(FakeRetriever(), NeverStops(), max_steps=10,
                               max_retrieval_rounds=3).retrieve("principle of promissory estoppel")
    assert bounded.retrieval_metadata["agent_retrieval_calls"] <= 3
    assert bounded.retrieval_metadata["agent_termination_reason"] == "retrieval_budget_exhausted"

    # An invalid action is refused rather than executed.
    class Rogue:
        def chat(self, messages):
            class R:
                class message:
                    content = json.dumps({"action": "rm -rf /", "arg": "", "why": "no"})
            return R()

    rogue = AgenticRetriever(FakeRetriever(), Rogue()).retrieve("principle of promissory estoppel")
    assert rogue.retrieval_metadata["agent_termination_reason"] == "invalid_action"

    print(f"agentic demo OK  (actions={meta['agent_actions']} "
          f"evidence={len(result.retrieved_chunks)})")


if __name__ == "__main__":
    demo()
