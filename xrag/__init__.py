"""Public entry points for external RAG trace evaluation."""
from .evaluator import TraceEvaluator, evaluate_trace

__all__ = ["TraceEvaluator", "evaluate_trace"]
