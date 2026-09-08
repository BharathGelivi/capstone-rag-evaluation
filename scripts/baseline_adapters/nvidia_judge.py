"""
Shared helper for pointing baseline evaluation frameworks' LLM judges at the
NVIDIA NIM OpenAI-compatible inference endpoint.

Used by ragas_adapter.py, ares_worker.py, and any other baseline adapter that
needs a LangChain-compatible LLM client for NVIDIA models.
"""

import os

from configs.models import NVIDIA_BASE_URL


def build_nvidia_chat_openai(model: str, temperature: float = 0.0, max_tokens: int = 2048):
    """Build a LangChain ChatOpenAI client pointed at the NVIDIA NIM endpoint.

    ``max_tokens`` is set explicitly (rather than left at LangChain's default)
    because RAGAS and RAGChecker both require a complete, immediately-parseable
    JSON/score response in a single completion — a low default risks the same
    truncation that reasoning models hit (RAGAS: LLMDidNotFinishException;
    RAGChecker: silently degenerated to all-zero scores).

    Args:
        model:       NVIDIA NIM model name (e.g. ``meta/llama-3.1-70b-instruct``).
        temperature: Sampling temperature. 0.0 for deterministic judge outputs.
        max_tokens:  Maximum tokens to generate.
    """
    from langchain_openai import ChatOpenAI

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY must be set to use the NVIDIA-hosted judge LLM "
            "for baseline comparisons."
        )
    return ChatOpenAI(
        model=model,
        base_url=NVIDIA_BASE_URL,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
