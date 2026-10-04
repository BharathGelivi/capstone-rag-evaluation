"""Installed CLI commands."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG trace JSON with local NLI")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--claim-mode", choices=["sentences", "llm"], default="sentences")
    args = parser.parse_args()
    from .evaluator import evaluate_trace
    try:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        result = evaluate_trace(trace, claim_mode=args.claim_mode)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


def serve():
    parser = argparse.ArgumentParser(description="Start the standalone X-RAG trace API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run("xrag.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
