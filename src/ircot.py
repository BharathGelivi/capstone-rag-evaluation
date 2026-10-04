"""
IRCoT -- Interleaving Retrieval with Chain-of-Thought Reasoning.

Implements the central mechanism of Trivedi et al., *Interleaving Retrieval with
Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions*
(arXiv:2212.10509): retrieval and reasoning alternate, and each reasoning step
supplies the query for the next retrieval, so evidence for hop *n+1* can be
found using what hop *n* just established.

    question -> retrieve -> reason -> retrieve -> ... -> answer

Why this is not just "retrieve more"
------------------------------------
A one-shot retriever embeds the *question*. For "which later cases limited the
principle in X", the documents that answer it do not resemble the question --
they resemble a fact the question does not contain (the name of X, the section
it construed). One-shot retrieval cannot reach them at any top-k, because
ranking further down a wrong ranking does not help. IRCoT rewrites the query
between hops, which is a different operation from widening one.

On chain-of-thought
-------------------
The model's private reasoning is deliberately **not** persisted. Each step
records only what is needed to audit and reproduce retrieval: the query issued,
the chunks it returned, their scores, which evidence was new, why the loop
stopped, and how long it took. The model is asked for a short public
``reasoning_summary`` -- a statement of what it is now looking for -- and that
is what gets stored. Nothing here reconstructs a hidden CoT trace.

Integration
-----------
The loop returns an ordinary :class:`~src.retriever.RetrievalResult`, so the
generator, ``RAGTraceBuilder``, claim decomposition and verification are all
unchanged. Per-step metadata rides in ``RetrievalResult.retrieval_metadata``
and is written to ``RAGTrace.diagnostics`` by the runner -- the existing trace
is the trace. There is no second tracing system.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.llms import ChatMessage, MessageRole

from src.retriever import RetrievalResult, RetrievedChunk
from src import rate_limiter

logger = logging.getLogger(__name__)

#: Hard bound on interleaving rounds. IRCoT's own ablations show returns
#: flattening after a handful of steps, and an unbounded loop is a cost bug
#: waiting to happen.
MAX_ITERATIONS = 4

#: Chunks pulled per follow-up hop. Smaller than the first hop on purpose: later
#: hops are targeted lookups, and padding them dilutes the context the generator
#: finally sees.
CHUNKS_PER_STEP = 3

#: Stop if a hop adds fewer than this many previously-unseen chunks. Repeated
#: retrieval of the same evidence is the loop's natural fixed point.
MIN_NEW_CHUNKS = 1

_STEP_PROMPT = """You are planning evidence retrieval for a legal research question.

Question: {question}

Evidence gathered so far ({n_chunks} passages):
{evidence}

Decide what to retrieve NEXT. Follow these rules:
- If the evidence already answers the question completely, set "have_enough" to true.
- Otherwise write ONE specific search query for the single most important missing fact.
- The query must be a standalone search string, not a question about the evidence.
- Prefer naming concrete entities the evidence revealed: case citations, statute
  sections, doctrines, parties. That is the point of searching again.
- Do not repeat a query already issued: {previous_queries}

Return ONLY this JSON object, no other text:
{{"have_enough": <true|false>,
  "next_query": "<search string, or empty if have_enough>",
  "reasoning_summary": "<one short sentence naming what you are looking for and why>"}}"""


@dataclass
class IRCoTStep:
    """One interleaved round. This is the auditable record, not the CoT."""
    iteration: int
    query: str
    is_initial: bool
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    new_evidence_chunk_ids: List[str] = field(default_factory=list)
    n_new_evidence: int = 0
    reasoning_summary: str = ""
    termination_reason: Optional[str] = None
    latency_s: float = 0.0
    llm_calls: int = 0
    retrieval_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def structured_call(llm, prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """One LLM call that must return a JSON object.

    Falls back to ``default`` on any failure. A control loop that crashes
    because a model emitted prose is a worse failure than a loop that stops
    early, so every parse problem degrades to termination rather than an
    exception. The fallback is recorded by the caller as a termination reason.
    """
    try:
        response = rate_limiter.call(llm.chat, [ChatMessage(role=MessageRole.USER, content=prompt)])
        text = str(response.message.content).strip()
    except Exception as exc:
        logger.warning("structured_call: LLM error %s", exc)
        return {**default, "_error": f"{type(exc).__name__}: {exc}"}

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {**default, "_error": "no JSON object in response"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {**default, "_error": f"unparseable JSON: {exc}"}

    if not isinstance(parsed, dict):
        return {**default, "_error": "JSON was not an object"}
    return {**default, **parsed}


def _format_evidence(chunks: Sequence[RetrievedChunk], budget_chars: int = 2600) -> str:
    """Compact evidence view for the planner.

    Truncated hard: the planner needs to know what is *already known*, not to
    re-read the corpus, and a long prompt here is paid on every iteration.
    """
    if not chunks:
        return "(nothing retrieved yet)"
    parts = []
    used = 0
    for chunk in chunks:
        snippet = " ".join(chunk.chunk_text.split())[:400]
        line = f"[{chunk.chunk_id[:8]}] {snippet}"
        if used + len(line) > budget_chars:
            parts.append(f"... and {len(chunks) - len(parts)} more passages")
            break
        parts.append(line)
        used += len(line)
    return "\n".join(parts)


class IRCoTRetriever:
    """Interleaved retrieval/reasoning over an existing :class:`Retriever`.

    Wraps rather than replaces: the underlying retriever's mode, fusion and
    reranker are whatever the ablation configured, so IRCoT can be measured on
    top of any retrieval arm.
    """

    def __init__(
        self,
        retriever,
        llm,
        max_iterations: int = MAX_ITERATIONS,
        chunks_per_step: int = CHUNKS_PER_STEP,
        retrieval_mode: Optional[str] = None,
        rerank: Optional[bool] = None,
        first_hop_top_n: Optional[int] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_iterations = max_iterations
        self.chunks_per_step = chunks_per_step
        self.retrieval_mode = retrieval_mode
        self.rerank = rerank
        self.first_hop_top_n = first_hop_top_n

    def _retrieve(self, query: str, top_n: Optional[int] = None) -> RetrievalResult:
        return self.retriever.retrieve(
            query, mode=self.retrieval_mode, rerank=self.rerank, top_n=top_n
        )

    def retrieve(self, question: str) -> RetrievalResult:
        """Run the interleaved loop and return one merged RetrievalResult."""
        start = time.time()
        steps: List[IRCoTStep] = []
        evidence: List[RetrievedChunk] = []
        seen: set = set()
        queries: List[str] = [question]
        llm_calls = 0
        retrieval_calls = 0

        # --- Hop 0: the question itself ------------------------------------
        step_start = time.time()
        first = self._retrieve(question, top_n=self.first_hop_top_n)
        retrieval_calls += 1
        for chunk in first.retrieved_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                evidence.append(chunk)
        steps.append(IRCoTStep(
            iteration=0, query=question, is_initial=True,
            retrieved_chunk_ids=[c.chunk_id for c in first.retrieved_chunks],
            scores=[float(c.similarity_score) for c in first.retrieved_chunks],
            new_evidence_chunk_ids=[c.chunk_id for c in first.retrieved_chunks],
            n_new_evidence=len(first.retrieved_chunks),
            reasoning_summary="initial retrieval from the question as asked",
            latency_s=round(time.time() - step_start, 3),
            retrieval_calls=1,
        ))

        termination = "max_iterations"

        # --- Hops 1..N: reason, then retrieve ------------------------------
        for iteration in range(1, self.max_iterations + 1):
            step_start = time.time()
            plan = structured_call(
                self.llm,
                _STEP_PROMPT.format(
                    question=question,
                    n_chunks=len(evidence),
                    evidence=_format_evidence(evidence),
                    previous_queries="; ".join(queries[-4:]),
                ),
                default={"have_enough": True, "next_query": "", "reasoning_summary": ""},
            )
            llm_calls += 1

            if plan.get("_error"):
                steps.append(IRCoTStep(
                    iteration=iteration, query="", is_initial=False,
                    reasoning_summary="planner unavailable",
                    termination_reason=f"planner_error: {plan['_error'][:120]}",
                    latency_s=round(time.time() - step_start, 3), llm_calls=1,
                ))
                termination = "planner_error"
                break

            next_query = (plan.get("next_query") or "").strip()
            if plan.get("have_enough") or not next_query:
                steps.append(IRCoTStep(
                    iteration=iteration, query="", is_initial=False,
                    reasoning_summary=str(plan.get("reasoning_summary", ""))[:300],
                    termination_reason="sufficient_evidence",
                    latency_s=round(time.time() - step_start, 3), llm_calls=1,
                ))
                termination = "sufficient_evidence"
                break

            if next_query.lower() in {q.lower() for q in queries}:
                steps.append(IRCoTStep(
                    iteration=iteration, query=next_query, is_initial=False,
                    reasoning_summary=str(plan.get("reasoning_summary", ""))[:300],
                    termination_reason="repeated_query",
                    latency_s=round(time.time() - step_start, 3), llm_calls=1,
                ))
                termination = "repeated_query"
                break

            queries.append(next_query)
            hop = self._retrieve(next_query, top_n=self.chunks_per_step)
            retrieval_calls += 1

            new_ids = []
            for chunk in hop.retrieved_chunks:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    evidence.append(chunk)
                    new_ids.append(chunk.chunk_id)

            step = IRCoTStep(
                iteration=iteration, query=next_query, is_initial=False,
                retrieved_chunk_ids=[c.chunk_id for c in hop.retrieved_chunks],
                scores=[float(c.similarity_score) for c in hop.retrieved_chunks],
                new_evidence_chunk_ids=new_ids, n_new_evidence=len(new_ids),
                reasoning_summary=str(plan.get("reasoning_summary", ""))[:300],
                latency_s=round(time.time() - step_start, 3),
                llm_calls=1, retrieval_calls=1,
            )

            if len(new_ids) < MIN_NEW_CHUNKS:
                step.termination_reason = "no_new_evidence"
                steps.append(step)
                termination = "no_new_evidence"
                break
            steps.append(step)
        else:
            termination = "max_iterations"

        # Rank the merged evidence so the generator still sees the strongest
        # passage last (the position the prompt builder relies on).
        evidence.sort(key=lambda c: c.similarity_score, reverse=True)
        for rank, chunk in enumerate(evidence, start=1):
            chunk.rank = rank

        total = time.time() - start
        return RetrievalResult(
            question=question,
            question_embedding_dimension=first.question_embedding_dimension,
            retrieved_chunks=evidence,
            retrieved_chunk_ids=[c.chunk_id for c in evidence],
            similarity_scores=[c.similarity_score for c in evidence],
            retrieval_time=total,
            top_k=len(evidence),
            retrieval_metadata={
                **first.retrieval_metadata,
                "strategy": "ircot",
                "ircot_iterations": len(steps),
                "ircot_termination_reason": termination,
                "ircot_queries": queries,
                "ircot_llm_calls": llm_calls,
                "ircot_retrieval_calls": retrieval_calls,
                "ircot_total_evidence": len(evidence),
                "ircot_steps": [s.to_dict() for s in steps],
            },
        )


def demo() -> None:
    """Self-check with a scripted planner and a fake corpus. No API, no models."""

    def chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id, similarity_score=score, rank=1, page_number="1",
            source_file="x.pdf", chunk_index=0, chunk_text=text,
        )

    corpus = {
        "what did kesavananda decide": [chunk("c1", "Kesavananda established the basic structure doctrine.", 0.9)],
        "basic structure doctrine later cases": [chunk("c2", "Indira Nehru Gandhi applied basic structure.", 0.8),
                                                  chunk("c1", "Kesavananda established the basic structure doctrine.", 0.7)],
        "indira nehru gandhi holding": [chunk("c3", "The election law amendment was struck down.", 0.75)],
    }

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, mode=None, rerank=None, top_n=None):
            self.calls.append(query)
            chunks = corpus.get(query.lower(), [])
            return RetrievalResult(
                question=query, question_embedding_dimension=768,
                retrieved_chunks=list(chunks),
                retrieved_chunk_ids=[c.chunk_id for c in chunks],
                similarity_scores=[c.similarity_score for c in chunks],
                retrieval_time=0.01, top_k=len(chunks),
                retrieval_metadata={"retrieval_mode": mode or "hybrid"},
            )

    class ScriptedLLM:
        """Returns planned queries, then declares sufficiency."""
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            payload = self.script.pop(0) if self.script else {
                "have_enough": True, "next_query": "", "reasoning_summary": "done"}

            class R:
                class message:
                    content = json.dumps(payload)
            return R()

    retriever = FakeRetriever()
    llm = ScriptedLLM([
        {"have_enough": False, "next_query": "basic structure doctrine later cases",
         "reasoning_summary": "need the cases that applied the doctrine"},
        {"have_enough": False, "next_query": "indira nehru gandhi holding",
         "reasoning_summary": "need what that case actually held"},
        {"have_enough": True, "next_query": "", "reasoning_summary": "chain complete"},
    ])

    result = IRCoTRetriever(retriever, llm, max_iterations=4).retrieve(
        "what did kesavananda decide")

    meta = result.retrieval_metadata
    # The multi-hop property: evidence the first query could not reach.
    assert set(result.retrieved_chunk_ids) == {"c1", "c2", "c3"}, result.retrieved_chunk_ids
    assert meta["strategy"] == "ircot"
    assert meta["ircot_termination_reason"] == "sufficient_evidence", meta
    assert meta["ircot_retrieval_calls"] == 3, meta
    assert len(meta["ircot_queries"]) == 3, meta

    steps = meta["ircot_steps"]
    assert steps[0]["is_initial"] and steps[0]["query"] == "what did kesavananda decide"
    assert steps[1]["n_new_evidence"] == 1, steps[1]
    # Structured metadata only -- no field carries free-form model reasoning.
    for step in steps:
        assert set(step) == set(IRCoTStep(0, "", True).to_dict()), set(step)
        assert len(step["reasoning_summary"]) <= 300

    # Ranks are re-assigned over the merged set, densest evidence first.
    assert [c.rank for c in result.retrieved_chunks] == [1, 2, 3]
    assert result.retrieved_chunks[0].similarity_score >= result.retrieved_chunks[-1].similarity_score

    # Termination: a planner that keeps asking for the same thing must stop.
    repeat_llm = ScriptedLLM([
        {"have_enough": False, "next_query": "basic structure doctrine later cases",
         "reasoning_summary": "one"},
        {"have_enough": False, "next_query": "basic structure doctrine later cases",
         "reasoning_summary": "again"},
    ])
    repeated = IRCoTRetriever(FakeRetriever(), repeat_llm).retrieve("what did kesavananda decide")
    assert repeated.retrieval_metadata["ircot_termination_reason"] == "repeated_query"

    # Termination: a hop that adds nothing new is a fixed point.
    stale_llm = ScriptedLLM([
        {"have_enough": False, "next_query": "basic structure doctrine later cases", "reasoning_summary": "x"},
        {"have_enough": False, "next_query": "unknown query with no hits", "reasoning_summary": "y"},
    ])
    stale = IRCoTRetriever(FakeRetriever(), stale_llm).retrieve("what did kesavananda decide")
    assert stale.retrieval_metadata["ircot_termination_reason"] == "no_new_evidence"

    # A broken planner degrades to termination, never to an exception.
    class BrokenLLM:
        def chat(self, messages):
            raise RuntimeError("endpoint down")

    broken = IRCoTRetriever(FakeRetriever(), BrokenLLM()).retrieve("what did kesavananda decide")
    assert broken.retrieval_metadata["ircot_termination_reason"] == "planner_error"
    assert broken.retrieved_chunk_ids == ["c1"], broken.retrieved_chunk_ids

    print(f"ircot demo OK  (evidence={len(result.retrieved_chunks)} "
          f"hops={meta['ircot_retrieval_calls']} termination={meta['ircot_termination_reason']})")


if __name__ == "__main__":
    demo()
