"""
One-off utility: re-run a single eval example using a specified generator
model, overwrite its entry in results.json / manifest.json, and regenerate
the HTML diagnostic report.

Usage (from the project root):
    python -m scripts.rerun_example_with_model --eval-id 10 --model llama-3.1-8b-instant

This is useful when the default generation model (llama-3.3-70b-versatile) has
exhausted its daily token quota and you need to regenerate an example with a
different Groq model.  The old trace and report are preserved in their dated
subdirectories; only the manifest/results pointers are updated to the new run.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time

logger = logging.getLogger(__name__)

RESULTS_DIR = "artifacts/benchmark_comparison"
MANIFEST_PATH = os.path.join(RESULTS_DIR, "manifest.json")
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "results.json")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")
DIAGNOSTIC_REPORTS_DIR = "artifacts/diagnostic_reports"

RESULT_COLUMNS = [
    "eval_id", "question", "expected_failure_type", "trace_id",
    "xrag_primary_cause", "xrag_avg_entailment_score",
    "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision",
    "ragas_context_recall", "ragas_answer_correctness",
    "ragchecker_precision", "ragchecker_recall", "ragchecker_f1",
    "ragchecker_claim_recall", "ragchecker_context_precision",
    "ragchecker_faithfulness", "ragchecker_hallucination",
    "ares_context_relevance", "ares_answer_relevance", "ares_answer_faithfulness",
    "ragas_latency_ms", "ragchecker_latency_ms", "ares_latency_ms",
    "generator_model",  # extra column to document which model was used
]


def load_eval_row(eval_id: str, path: str = "eval/eval_dataset.csv"):
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["id"] == eval_id:
                return row
    raise ValueError(f"eval_id={eval_id} not found in {path}")


def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_eval_ids": [], "eval_id_to_trace_id": {}}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_results():
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(rows):
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    # Rebuild the CSV from scratch to keep it in sync
    if rows:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = [c for c in RESULT_COLUMNS if c in all_keys] + sorted(all_keys - set(RESULT_COLUMNS))
        with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})


def generate_html_report(eval_id: str, trace_id: str):
    from scripts.generate_diagnostic_report import generate
    try:
        report_path = generate(trace_id, output_dir=DIAGNOSTIC_REPORTS_DIR)
        named_path = os.path.join(DIAGNOSTIC_REPORTS_DIR, f"{eval_id}.html")
        shutil.move(report_path, named_path)
        logger.info(f"Diagnostic report written to {named_path}")
        return named_path
    except Exception as e:
        logger.warning(f"Could not generate diagnostic report: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Re-run one eval example with a specified generator model."
    )
    parser.add_argument("--eval-id", required=True, help="Eval example ID (e.g. 10)")
    parser.add_argument(
        "--model",
        default="llama-3.1-8b-instant",
        help="Groq model name to use for generation (default: llama-3.1-8b-instant)",
    )
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from dotenv import load_dotenv
    load_dotenv()
    from src.env_check import ensure_llm_credentials
    ensure_llm_credentials()

    from src.chunk_registry import ChunkRegistry
    from src.vector_store import ChromaVectorStore
    from src.retriever import get_retriever
    from src.generator import Generator
    from src.runner import PipelineRunner
    from src.rag_trace import RAGTraceBuilder

    eval_row = load_eval_row(args.eval_id, args.eval_dataset)
    question = eval_row["question"]
    gold_answer = eval_row.get("gold_answer") or None
    expected_failure_type = eval_row.get("expected_failure_type") or ""

    logger.info(f"Re-running eval {args.eval_id} with model={args.model}")
    logger.info(f"Question: {question[:120]}")

    chunk_registry = ChunkRegistry.load_from_json("artifacts/chunk_registry.json")
    vector_store = ChromaVectorStore()
    vector_store.initialize_collection()
    retriever = get_retriever(vector_store, chunk_registry)

    # Use the specified model for generation
    generator = Generator(model_name=args.model)
    runner = PipelineRunner()

    retrieval_result = retriever.retrieve(question)
    generation_result = generator.generate(retrieval_result)
    total_time = retrieval_result.retrieval_time + generation_result.generation_time
    trace = RAGTraceBuilder.build(retrieval_result, generation_result, total_time)
    RAGTraceBuilder.save_to_json(trace)

    report = runner.run(trace, gold_answer=gold_answer)
    report.save()

    new_row = {
        "eval_id": args.eval_id,
        "question": question,
        "expected_failure_type": expected_failure_type,
        "trace_id": trace.trace_id,
        "xrag_primary_cause": report.root_cause_analysis.primary_cause,
        "xrag_avg_entailment_score": report.evaluation_metrics.average_entailment,
        "generator_model": args.model,
    }
    logger.info(f"X-RAG primary cause: {new_row['xrag_primary_cause']}")
    logger.info(f"X-RAG avg entailment: {new_row['xrag_avg_entailment_score']:.4f}")

    # Update manifest + results (remove old entry for this eval_id, insert new one)
    manifest = load_manifest()
    all_rows = load_results()
    all_rows = [r for r in all_rows if str(r.get("eval_id")) != str(args.eval_id)]
    all_rows.append(new_row)

    manifest["eval_id_to_trace_id"][args.eval_id] = trace.trace_id
    if args.eval_id not in manifest["completed_eval_ids"]:
        manifest["completed_eval_ids"].append(args.eval_id)

    save_results(all_rows)
    save_manifest(manifest)
    logger.info("Results and manifest updated.")

    report_path = generate_html_report(args.eval_id, trace.trace_id)
    if report_path:
        logger.info(f"HTML report: {report_path}")

    print(f"\n✓  eval {args.eval_id} complete")
    print(f"   Model:       {args.model}")
    print(f"   Trace ID:    {trace.trace_id}")
    print(f"   Primary cause: {new_row['xrag_primary_cause']}")
    print(f"   Avg entailment: {new_row['xrag_avg_entailment_score']:.4f}")
    print(f"   HTML report: {report_path or 'not generated'}")


if __name__ == "__main__":
    main()
