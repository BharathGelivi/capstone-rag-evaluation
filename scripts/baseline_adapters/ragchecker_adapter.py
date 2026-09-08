"""
Converts the shared ResolvedExample intermediate format (see common.py) into
the RAGResults format the installed `ragchecker` package expects.

Schema note (verified against the installed package's container classes, not
assumed): `ragchecker.container.RAGResult` requires `query_id`/`query`/
`gt_answer`/`response`, and `gt_answer` has no default -- unlike ragas,
RAGChecker cannot score an example at all without a gold answer, so examples
with no gold answer are skipped here rather than included with a null
reference.

Install note: `ragchecker` needs its own Python 3.10 venv (see
requirements-eval-ragchecker.txt) -- this module is meant to be run under
that venv's interpreter, not this project's main venv, and therefore works
off ResolvedExample (plain strings) rather than RAGTrace/ChunkRegistry
directly (those need llama-index, which isn't installed in that venv).
"""

import logging
import os
from typing import List

from scripts.baseline_adapters.common import ResolvedExample

logger = logging.getLogger(__name__)


def to_ragchecker_results(examples: List[ResolvedExample]):
    """
    Builds a ragchecker RAGResults object from ResolvedExamples. Examples
    without a gold answer are skipped (logged), since RAGChecker requires one
    for every example.
    """
    from ragchecker import RAGResults
    from ragchecker.container import RAGResult, RetrievedDoc

    results = []
    for example in examples:
        if example.gold_answer is None:
            logger.warning(f"Skipping trace {example.trace_id} for RAGChecker: no gold answer available.")
            continue

        results.append(RAGResult(
            query_id=example.trace_id,
            query=example.question,
            gt_answer=example.gold_answer,
            response=example.answer,
            retrieved_context=[RetrievedDoc(doc_id=f"{example.trace_id}_c{i}", text=text) for i, text in enumerate(example.contexts)],
        ))

    return RAGResults(results=results)


def build_ragchecker(model: str = None, batch_size: int = 1):
    """Build a RAGChecker instance whose LLM calls route through the NVIDIA NIM endpoint.

    Uses litellm's ``openai/<model>`` custom-endpoint convention so RAGChecker's
    internal litellm calls land on NVIDIA's OpenAI-compatible API.

    batch_size=1, not RAGChecker's default of 4: this project's own
    generation/decomposition/verification code shares this same NVIDIA
    account's 40-requests/minute free-tier budget (src/rate_limiter.py), which
    RAGChecker's separate-venv litellm calls cannot see or throttle against.
    Measured directly: a realistic example (6 real chunks, a real gold answer)
    at batch_size=4 was still running past 24 minutes with repeated internal
    429 retries before being killed; serializing RAGChecker's own concurrent
    calls removes the self-inflicted burst that was causing most of that
    contention, at the cost of losing RAGChecker's own internal parallelism.

    Note: this module runs under venv_eval_ragchecker's own Python 3.10
    interpreter (see module docstring), which does NOT have configs/ installed
    — the API key and base URL are read directly from environment variables.
    """
    from ragchecker import RAGChecker

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY must be set to use the NVIDIA-hosted judge LLM "
            "for RAGChecker. Set it in your .env file."
        )

    api_base = "https://integrate.api.nvidia.com/v1"
    # meta/llama-3.1-8b-instruct reached end-of-life on NVIDIA's catalog on
    # 2026-08-26 (confirmed live: the endpoint now returns 410 Gone for it),
    # which was silently turning every RAGChecker call into a hard failure
    # rather than a rate-limit-style transient one. nemotron-3-super-120b-a12b
    # is the one model this project has confirmed actually serves completions
    # on this account (see configs/models.py) -- use the same one here so
    # RAGChecker isn't relying on a model nothing else in the project trusts.
    resolved_model = model or "nvidia/nemotron-3-super-120b-a12b"
    litellm_model = f"openai/{resolved_model}"

    return RAGChecker(
        extractor_name=litellm_model,
        checker_name=litellm_model,
        extractor_api_base=api_base,
        checker_api_base=api_base,
        openai_api_key=api_key,
        batch_size_extractor=batch_size,
        batch_size_checker=batch_size,
    )
