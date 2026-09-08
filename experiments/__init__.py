"""
X-RAG paper experiments (E1-E5).

Each experiment is a self-contained, resumable study that writes its per-example
records and an aggregate summary under ``artifacts/experiments/<key>/``.

    E1  exp01_fault_injection      Stage-attributed failure taxonomy via causal
                                   fault injection (the paper's spine).
    E2  exp02_reranker_window      The reranker-window pathology: when
                                   rerank_input approximate top_n, the reranker is a no-op.
    E3  exp03_refusal_calibration  Refusal calibration -- faithfulness metrics
                                   reward silence.
    E4  exp04_corpus_quality       Corpus quality as an upstream determinant
                                   (TOC chunks outranking real provisions).
    E5  exp05_diagnostic_agreement Diagnostic agreement: high score
                                   correlation, low cause-level kappa.

Run them all (resumable) with::

    python -m experiments.run_all
"""

from experiments.common import (  # noqa: F401
    Experiment,
    ExampleSpec,
    ExperimentContext,
    Checkpoint,
    EXPERIMENTS_DIR,
)

__all__ = [
    "Experiment",
    "ExampleSpec",
    "ExperimentContext",
    "Checkpoint",
    "EXPERIMENTS_DIR",
]
