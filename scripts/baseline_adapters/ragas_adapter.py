"""
Converts X-RAG's own artifacts (RAGTrace + ChunkRegistry) plus the labeled eval
set into the exact input format the installed `ragas` library expects.

Schema note (verified against the actually-installed package, not assumed):
this targets ragas==0.2.15 (pinned; see requirements-eval.txt -- newer ragas
releases hard-import a langchain-community submodule that no longer exists on
PyPI; see docs/RESEARCH_LOG.md for the full investigation). ragas 0.2.15 uses
the "SingleTurnSample" schema (user_input/response/retrieved_contexts/reference),
NOT the older question/answer/contexts/ground_truth column naming shown in some
outdated ragas tutorials.
"""

from typing import Dict, List, Optional

from src.rag_trace import RAGTrace
from src.chunk_registry import ChunkRegistry


def resolve_contexts(trace: RAGTrace, registry: ChunkRegistry) -> List[str]:
    """
    Resolves the actual chunk text for every chunk a trace references.
    RAGTrace.retrieved_chunk_references intentionally omits full chunk text
    (see src/rag_trace.py) -- the canonical text lives in the ChunkRegistry,
    the same lookup src/claim_verifier.py's verify() method already performs.
    """
    contexts = []
    for ref in trace.retrieved_chunk_references:
        record = registry.get_chunk(ref["chunk_id"])
        if record is not None:
            contexts.append(record.text)
    return contexts


def to_ragas_dataset(
    traces: List[RAGTrace],
    registry: ChunkRegistry,
    gold_answers: Optional[Dict[str, str]] = None,
):
    """
    Builds a ragas EvaluationDataset from X-RAG RAGTraces.

    gold_answers: optional dict mapping trace_id -> reference (gold) answer.
    Traces without a matching gold answer get reference=None; ragas skips
    reference-based metrics (context_recall, answer_correctness) for those rows.
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    gold_answers = gold_answers or {}
    samples = [
        SingleTurnSample(
            user_input=trace.question,
            response=trace.generated_answer,
            retrieved_contexts=resolve_contexts(trace, registry),
            reference=gold_answers.get(trace.trace_id),
        )
        for trace in traces
    ]
    return EvaluationDataset(samples=samples)


def to_ragas_dataset_from_resolved(examples):
    """
    Same as to_ragas_dataset, but takes already-resolved ResolvedExamples
    (see common.py) instead of raw RAGTrace + ChunkRegistry. Useful for
    consistency with the ragchecker/ARES adapters, which must work off this
    dependency-free intermediate format since they run in a separate venv.
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=example.question,
            response=example.answer,
            retrieved_contexts=example.contexts,
            reference=example.gold_answer,
        )
        for example in examples
    ]
    return EvaluationDataset(samples=samples)


def build_ragas_llm(model: Optional[str] = None):
    """Wrap the NVIDIA judge LLM for use as ragas's ``llm=`` argument."""
    from ragas.llms import LangchainLLMWrapper
    from configs.models import NVIDIA_RAGAS_JUDGE_MODEL
    from scripts.baseline_adapters.nvidia_judge import build_nvidia_chat_openai

    return LangchainLLMWrapper(build_nvidia_chat_openai(model or NVIDIA_RAGAS_JUDGE_MODEL))


def build_ragas_embeddings(embed_model):
    """
    Wraps an already-loaded llama_index embedding model (e.g. Retriever.embed_model)
    for use as ragas's `embeddings=` argument -- avoids loading a second embedding model.
    """
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper

    return LlamaIndexEmbeddingsWrapper(embed_model)
