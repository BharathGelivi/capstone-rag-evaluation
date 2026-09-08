"""
Device selection for local models.

Every local model in the pipeline — the bi-encoder, the cross-encoder reranker,
and the NLI verifier — went to CPU by default, which is what made a single
verified turn cost minutes. This module centralises the choice so there is one
place to reason about placement and VRAM.

Resolution order:
    1. ``XRAG_DEVICE`` env var, if set (``cuda``, ``cuda:0``, ``cpu``, ``mps``).
    2. CUDA, if a GPU is actually usable.
    3. Apple MPS, if present.
    4. CPU.

The CUDA check is deliberately stricter than ``torch.cuda.is_available()``: a
CPU-only torch wheel reports ``False`` there, but a *driver* mismatch can let
that call succeed and then fail on first allocation. Allocating a probe tensor
catches both, so callers can trust the returned device.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_device(prefer: Optional[str] = None) -> str:
    """Return the torch device string that local models should load onto.

    Cached: the probe allocates on the GPU, and there is no reason to repeat it
    for every model constructed during a run.
    """
    override = prefer or os.environ.get("XRAG_DEVICE", "").strip()

    try:
        import torch
    except ImportError:
        logger.warning("torch is not installed; falling back to CPU.")
        return "cpu"

    if override:
        if override.startswith("cuda") and not _cuda_usable(torch):
            logger.warning(
                "XRAG_DEVICE=%s requested but CUDA is not usable; using CPU.",
                override,
            )
            return "cpu"
        logger.info("Using device from override: %s", override)
        return override

    if _cuda_usable(torch):
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("CUDA available: %s (%.1f GB) — loading models on GPU.", name, total)
        return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        logger.info("Apple MPS available — loading models on MPS.")
        return "mps"

    logger.info(
        "No GPU detected — loading models on CPU. Reranking and NLI "
        "verification will be slow; see docs/gpu_setup.md."
    )
    return "cpu"


def _cuda_usable(torch) -> bool:
    """True only if CUDA is present *and* an allocation actually succeeds."""
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return False
        torch.zeros(1).cuda()
        return True
    except Exception as exc:
        logger.warning("CUDA reported available but unusable (%s); using CPU.", exc)
        return False


def get_hf_device_index() -> int:
    """Device argument for ``transformers.pipeline``, which wants an int.

    ``0`` selects the first GPU; ``-1`` means CPU.
    """
    device = get_device()
    if device.startswith("cuda"):
        _, _, index = device.partition(":")
        return int(index) if index else 0
    return -1


def describe_device() -> str:
    """Human-readable device summary for the UI and logs."""
    device = get_device()
    if not device.startswith("cuda"):
        return device.upper()
    try:
        import torch

        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        return f"{name} ({free / 1e9:.1f} / {total / 1e9:.1f} GB free)"
    except Exception:
        return device
