"""
Worker script meant to run under venv_eval_ragchecker's interpreter (invoked
as a subprocess by scripts/run_baseline_comparison.py, which runs in the main
project venv where ragchecker itself isn't installed).

Usage:
    venv_eval_ragchecker\\Scripts\\python.exe -m scripts.baseline_adapters.ragchecker_worker <input_examples.json> <output_scores.json> [--model MODEL]

Reads a list of ResolvedExample dicts (see common.py) from <input_examples.json>,
scores them with RAGChecker, and writes {trace_id: {metric_name: score, ...}} to
<output_scores.json>.
"""

import argparse
import json
import sys

from scripts.baseline_adapters.common import load_resolved_examples
from scripts.baseline_adapters.ragchecker_adapter import to_ragchecker_results, build_ragchecker


def main():
    parser = argparse.ArgumentParser(description="Score ResolvedExamples with RAGChecker.")
    parser.add_argument("input_path", help="Path to a JSON file of ResolvedExamples.")
    parser.add_argument("output_path", help="Path to write per-example scores JSON to.")
    parser.add_argument("--model", default=None, help="Defaults to a provider-appropriate model (see build_ragchecker) if not given.")
    args = parser.parse_args()

    examples = load_resolved_examples(args.input_path)
    results = to_ragchecker_results(examples)

    if len(results.results) == 0:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        return

    checker = build_ragchecker(model=args.model)
    checker.evaluate(results, metrics="all_metrics")

    scores = {result.query_id: result.metrics for result in results.results}
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ragchecker_worker failed: {e}", file=sys.stderr)
        sys.exit(1)
