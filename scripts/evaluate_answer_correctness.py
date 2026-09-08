"""
Standalone entry point for AnswerCorrectnessEvaluator -- the ONLY supported
way to invoke it right now (not from runner.py, not from the API; see
src/answer_correctness_evaluator.py's module docstring for why).

Usage:
    python -m scripts.evaluate_answer_correctness <trace_id_or_path> <gold_answer_text_or_file>

Examples:
    python -m scripts.evaluate_answer_correctness 84c9b989-e765-4497-b0ee-8797c4782798 "The right commences..."
    python -m scripts.evaluate_answer_correctness artifacts/rag_traces/2026-07-19/trace_X.json gold_answer.txt
"""

import argparse
import glob
import json
import os

from src.answer_correctness_evaluator import AnswerCorrectnessEvaluator

TRACES_DIR = "artifacts/rag_traces"


def load_trace_generated_answer(trace_id_or_path: str) -> tuple:
    """Returns (trace_id, generated_answer). Accepts either a bare trace_id
    (looked up under artifacts/rag_traces/**/trace_<id>.json) or a direct
    path to a trace JSON file."""
    if os.path.exists(trace_id_or_path):
        path = trace_id_or_path
    else:
        matches = glob.glob(os.path.join(TRACES_DIR, "**", f"trace_{trace_id_or_path}.json"), recursive=True)
        if not matches:
            raise FileNotFoundError(f"No trace found for '{trace_id_or_path}' under {TRACES_DIR}")
        path = matches[0]

    with open(path, encoding="utf-8") as f:
        trace_data = json.load(f)
    return trace_data["trace_id"], trace_data["generated_answer"]


def load_gold_answer(gold_answer_text_or_file: str) -> str:
    if os.path.exists(gold_answer_text_or_file):
        with open(gold_answer_text_or_file, encoding="utf-8") as f:
            return f.read().strip()
    return gold_answer_text_or_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate answer correctness (claim recall) standalone.")
    parser.add_argument("trace_id_or_path", help="trace_id or path to a RAGTrace JSON file.")
    parser.add_argument("gold_answer_text_or_file", help="Gold answer text, or a path to a file containing it.")
    args = parser.parse_args()

    trace_id, generated_answer = load_trace_generated_answer(args.trace_id_or_path)
    gold_answer = load_gold_answer(args.gold_answer_text_or_file)

    evaluator = AnswerCorrectnessEvaluator()
    summary = evaluator.evaluate(generated_answer, gold_answer, trace_id)
    path = summary.save()

    recalled = sum(1 for r in summary.results if r.verification_status in ("SUPPORTED", "PARTIALLY_SUPPORTED"))
    print(f"claim_recall: {summary.claim_recall:.3f} ({recalled}/{summary.total_gold_claims} gold claims recalled)")
    for r in summary.results:
        print(f"  [{r.verification_status}] {r.claim_text}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
