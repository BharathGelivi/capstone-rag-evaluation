"""
Shared helper for pointing baseline evaluation frameworks' LLM judges at
Groq's free-tier OpenAI-compatible endpoint, reusing this project's own
GROQ_API_KEY -- the free alternative to hf_judge.py's HF-router path while HF
Inference Providers credits are exhausted (see docs/RESEARCH_LOG.md).
"""

import os

from configs.models import GROQ_BASE_URL


def build_groq_chat_openai(model: str, temperature: float = 0.0, max_tokens: int = 2048):
    """Builds a LangChain ChatOpenAI client pointed at Groq's OpenAI-compatible endpoint.

    max_tokens is set generously (rather than left at LangChain's None/model
    default) since RAGAS/RAGChecker need a complete, parseable response in one
    shot -- a low default risks the same truncation that reasoning models hit.
    """
    from langchain_openai import ChatOpenAI

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY must be set to use the Groq-hosted judge LLM for baseline comparisons.")
    return ChatOpenAI(model=model, base_url=GROQ_BASE_URL, api_key=key, temperature=temperature, max_tokens=max_tokens)


def build_nvidia_chat_openai(model: str, temperature: float = 0.0, max_tokens: int = 2048):
    """Builds a LangChain ChatOpenAI client pointed at NVIDIA's OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI
    from configs.models import NVIDIA_BASE_URL

    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY must be set to use the NVIDIA-hosted judge LLM for baseline comparisons.")
    return ChatOpenAI(model=model, base_url=NVIDIA_BASE_URL, api_key=key, temperature=temperature, max_tokens=max_tokens)
