"""
Shared, dependency-free intermediate representation for baseline adapters.

`ragchecker` and `ares-ai` each need their own separate Python 3.10 venv (see
requirements-eval-ragchecker.txt / requirements-eval-ares.txt) that does NOT
have this project's own heavy dependencies (llama-index, chromadb, ...)
installed. So resolving a RAGTrace's chunk references into actual text via
ChunkRegistry must happen once, in the main project venv (which has both),
producing a plain, JSON-serializable representation that the other venvs can
consume without ever importing `src.rag_trace` / `src.chunk_registry`.

ragas runs fine in the main venv alongside `src/`, so its adapter can (and
does) work directly off RAGTrace + ChunkRegistry -- this module exists for the
two adapters that can't.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class ResolvedExample:
    """One eval example with all chunk text already resolved to plain strings."""
    trace_id: str
    question: str
    answer: str
    contexts: List[str] = field(default_factory=list)
    gold_answer: Optional[str] = None


def resolve_examples(traces, registry, gold_answers: Optional[Dict[str, str]] = None) -> List[ResolvedExample]:
    """
    Builds ResolvedExamples from X-RAG RAGTraces + a ChunkRegistry. Must be
    called from the main project venv (needs src.rag_trace / src.chunk_registry).
    """
    gold_answers = gold_answers or {}
    examples = []
    for trace in traces:
        contexts = []
        for ref in trace.retrieved_chunk_references:
            record = registry.get_chunk(ref["chunk_id"])
            if record is not None:
                contexts.append(record.text)
        examples.append(ResolvedExample(
            trace_id=trace.trace_id,
            question=trace.question,
            answer=trace.generated_answer,
            contexts=contexts,
            gold_answer=gold_answers.get(trace.trace_id),
        ))
    return examples


def save_resolved_examples(examples: List[ResolvedExample], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in examples], f, indent=2)


def load_resolved_examples(path: str) -> List[ResolvedExample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ResolvedExample(**row) for row in data]
