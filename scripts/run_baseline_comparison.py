"""
Benchmark runner: compares X-RAG's own diagnosis against RAGAS, RAGChecker,
and ARES (ues_idp mode) over the labeled eval set (eval/eval_dataset.csv).

Usage:
    python -m scripts.run_baseline_comparison [--dry-run] [--limit N] [--skip-ragchecker] [--skip-ares]

Design notes:
- ragchecker and ares-ai each need their own separate Python 3.10 venv (see
  requirements-eval-ragchecker.txt / requirements-eval-ares.txt and
  docs/RESEARCH_LOG.md) -- this script runs in the MAIN project venv (which
  has X-RAG + ragas) and shells out to the other two venvs' interpreters via
  subprocess, passing data through the dependency-free ResolvedExample JSON
  format (scripts/baseline_adapters/common.py) and reading scores back.
- Resumable: writes artifacts/benchmark_comparison/results.csv (+ .json)
  incrementally, one row at a time, and a manifest.json tracking which eval
  ids are done. Re-running the same command skips already-completed rows.
- Resilient: a failure in any one baseline for any one example is caught,
  logged to artifacts/benchmark_comparison/failures.log, and recorded as NaN
  for that baseline's columns -- it does not abort the whole run.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RESULTS_DIR = "artifacts/benchmark_comparison"
MANIFEST_PATH = os.path.join(RESULTS_DIR, "manifest.json")
RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "results.json")
FAILURES_LOG_PATH = os.path.join(RESULTS_DIR, "failures.log")
DIAGNOSTIC_REPORTS_DIR = "artifacts/diagnostic_reports"

RAGCHECKER_PYTHON = os.path.join("venv_eval_ragchecker", "Scripts", "python.exe")
ARES_PYTHON = os.path.join("venv_eval_ares", "Scripts", "python.exe")

RESULT_COLUMNS = [
    "eval_id", "question", "expected_failure_type", "trace_id",
    "xrag_primary_cause", "xrag_avg_entailment_score",
    "ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision",
    "ragas_context_recall", "ragas_answer_correctness",
    "ragchecker_precision", "ragchecker_recall", "ragchecker_f1",
    "ragchecker_claim_recall", "ragchecker_context_precision",
    "ragchecker_faithfulness", "ragchecker_hallucination",
    "ares_context_relevance", "ares_answer_relevance", "ares_answer_faithfulness",
    # Research Improvement #5: real wall-clock cost per baseline, for an
    # honest cost/latency comparison instead of an approximate call-count table.
    "ragas_latency_ms", "ragchecker_latency_ms", "ares_latency_ms",
]


def load_eval_dataset(path: str = "eval/eval_dataset.csv") -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest() -> Dict[str, Any]:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_eval_ids": [], "eval_id_to_trace_id": {}}


def save_manifest(manifest: Dict[str, Any]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def log_failure(eval_id: str, baseline: str, error: Exception) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(FAILURES_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] eval_id={eval_id} baseline={baseline}: {error}\n")
    logger.warning(f"{baseline} failed for {eval_id}: {error}")


def append_result_row(row: Dict[str, Any]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(RESULTS_CSV_PATH)
    with open(RESULTS_CSV_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})

    all_rows = []
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, encoding="utf-8") as f:
            all_rows = json.load(f)
    all_rows.append(row)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)


class GenerationFailedError(RuntimeError):
    """Raised when the generator itself failed (timeout, malformed response,
    etc.) rather than producing a real answer. Generator.generate() catches
    the underlying exception and returns a placeholder "did not respond in
    time" string with GenerationResult.error set, so a caller that does not
    check .error will happily run that placeholder through claim
    decomposition and NLI verification as if it were the model's real answer
    -- guaranteed to score as unsupported/hallucinated, corrupting the row
    with a diagnosis that has nothing to do with the pipeline's actual
    behavior on that question. Caught by main()'s per-example try/except,
    exactly like any other per-baseline failure: the row is skipped (not
    appended, not marked complete in the manifest) and retried on the next
    run, mirroring how E3's trace harvest excludes rather than imputes
    failed generations."""


def build_xrag_trace_and_report(question: str, retriever, generator, runner, gold_answer: Optional[str] = None):
    """Runs the full X-RAG pipeline for one question: retrieve -> generate -> diagnose.

    gold_answer, when provided, also computes answer correctness (claim recall)
    as part of the same report -- see src/answer_correctness_evaluator.py.
    """
    from src.rag_trace import RAGTraceBuilder

    retrieval_result = retriever.retrieve(question)
    generation_result = generator.generate(retrieval_result)
    if generation_result.error:
        raise GenerationFailedError(
            f"Generation failed for {question[:80]!r}: {generation_result.error}"
        )
    total_time = retrieval_result.retrieval_time + generation_result.generation_time
    trace = RAGTraceBuilder.build(retrieval_result, generation_result, total_time)
    RAGTraceBuilder.save_to_json(trace)

    report = runner.run(trace, gold_answer=gold_answer)
    report.save()
    return trace, report


def compute_ragas_scores(question: str, answer: str, contexts: List[str], gold_answer: Optional[str], embed_model) -> Dict[str, Optional[float]]:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
    from scripts.baseline_adapters.ragas_adapter import to_ragas_dataset_from_resolved, build_ragas_llm, build_ragas_embeddings
    from scripts.baseline_adapters.common import ResolvedExample

    example = ResolvedExample(trace_id="_", question=question, answer=answer, contexts=contexts, gold_answer=gold_answer)
    dataset = to_ragas_dataset_from_resolved([example])

    metrics = [faithfulness, answer_relevancy, context_precision]
    if gold_answer:
        metrics += [context_recall, answer_correctness]

    result = evaluate(dataset, metrics=metrics, llm=build_ragas_llm(), embeddings=build_ragas_embeddings(embed_model))
    df = result.to_pandas()
    row = df.iloc[0].to_dict()
    return {
        "ragas_faithfulness": row.get("faithfulness"),
        "ragas_answer_relevancy": row.get("answer_relevancy"),
        "ragas_context_precision": row.get("context_precision"),
        "ragas_context_recall": row.get("context_recall"),
        "ragas_answer_correctness": row.get("answer_correctness"),
    }


def run_subprocess_worker(python_exe: str, module: str, example, timeout: int = 900) -> Dict[str, Any]:
    """
    Writes a single ResolvedExample to a temp JSON, invokes `python_exe -m module
    <in> <out>` as a subprocess, and reads the resulting per-trace_id scores back.
    Raises on failure (non-zero exit or timeout) -- caller is responsible for
    catching and recording NaN.
    """
    from scripts.baseline_adapters.common import save_resolved_examples

    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.json")
        out_path = os.path.join(tmpdir, "out.json")
        save_resolved_examples([example], in_path)

        result = subprocess.run(
            [python_exe, "-m", module, in_path, out_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{module} exited {result.returncode}: {result.stderr[-2000:]}")

        with open(out_path, encoding="utf-8") as f:
            scores = json.load(f)
        return scores.get(example.trace_id, {})


def compute_ragchecker_scores(example) -> Dict[str, Optional[float]]:
    """
    Reads the per-result metrics dict ragchecker_worker.py writes out. Verified
    against the installed package's source (ragchecker/evaluator.py,
    ragchecker/computation.py, ragchecker/metrics.py): RAGResult.metrics is a
    FLAT dict keyed by the metric's own name string (e.g. "precision",
    "recall", "f1", "claim_recall", "context_precision", "faithfulness",
    "hallucination", ...) -- not grouped/nested like the separate
    RAGResults.metrics aggregate (which groups by "overall_metrics" /
    "retriever_metrics" / "generator_metrics").
    """
    if example.gold_answer is None:
        return {}
    # 2700s (45 min), not the 900s default: measured directly against a
    # realistic example (6 real chunks, real gold answer) that was still
    # running past 24 minutes under 429 contention before batch_size was
    # dropped to 1 (see ragchecker_adapter.build_ragchecker). Generous rather
    # than tight, since a killed-too-early call wastes the work already done
    # and forces a full re-run on resume, whereas a slow-but-finishing call
    # only costs wall-clock time.
    raw = run_subprocess_worker(
        RAGCHECKER_PYTHON, "scripts.baseline_adapters.ragchecker_worker", example, timeout=2700,
    )
    return {
        "ragchecker_precision": raw.get("precision"),
        "ragchecker_recall": raw.get("recall"),
        "ragchecker_f1": raw.get("f1"),
        "ragchecker_claim_recall": raw.get("claim_recall"),
        "ragchecker_context_precision": raw.get("context_precision"),
        "ragchecker_faithfulness": raw.get("faithfulness"),
        "ragchecker_hallucination": raw.get("hallucination"),
    }


def compute_ares_scores(example) -> Dict[str, Optional[float]]:
    raw = run_subprocess_worker(ARES_PYTHON, "scripts.baseline_adapters.ares_worker", example)
    return {
        "ares_context_relevance": raw.get("Context Relevance Scores"),
        "ares_answer_relevance": raw.get("Answer Relevance Scores"),
        "ares_answer_faithfulness": raw.get("Answer Faithfulness Scores"),
    }


def process_one_example(eval_row, retriever, generator, runner, chunk_registry, skip_ragchecker: bool, skip_ares: bool) -> Dict[str, Any]:
    from scripts.baseline_adapters.common import resolve_examples

    eval_id = eval_row["id"]
    question = eval_row["question"]
    gold_answer = eval_row["gold_answer"] or None
    expected_failure_type = eval_row.get("expected_failure_type") or ""

    trace, report = build_xrag_trace_and_report(question, retriever, generator, runner, gold_answer=gold_answer)
    example = resolve_examples([trace], chunk_registry, gold_answers={trace.trace_id: gold_answer} if gold_answer else {})[0]
    example.trace_id = trace.trace_id  # keep IDs consistent across all baselines for this row

    row: Dict[str, Any] = {
        "eval_id": eval_id,
        "question": question,
        "expected_failure_type": expected_failure_type,
        "trace_id": trace.trace_id,
        "xrag_primary_cause": report.root_cause_analysis.primary_cause,
        "xrag_avg_entailment_score": report.evaluation_metrics.average_entailment,
    }

    try:
        t0 = time.time()
        row.update(compute_ragas_scores(question, trace.generated_answer, example.contexts, gold_answer, retriever.embed_model))
        row["ragas_latency_ms"] = (time.time() - t0) * 1000
    except Exception as e:
        log_failure(eval_id, "ragas", e)

    if not skip_ragchecker:
        try:
            t0 = time.time()
            row.update(compute_ragchecker_scores(example))
            row["ragchecker_latency_ms"] = (time.time() - t0) * 1000
        except Exception as e:
            log_failure(eval_id, "ragchecker", e)

    if not skip_ares:
        try:
            t0 = time.time()
            row.update(compute_ares_scores(example))
            row["ares_latency_ms"] = (time.time() - t0) * 1000
        except Exception as e:
            log_failure(eval_id, "ares", e)

    return row


def generate_and_name_report(eval_id: str, trace_id: str) -> None:
    """
    Generates the detailed per-trace HTML diagnostic report (see
    scripts/generate_diagnostic_report.py) and names it by eval_id
    (1.html, 2.html, ... 40.html) rather than trace_id, for easy reference
    against eval/eval_dataset.csv. Failures here are logged, not raised --
    a report-generation glitch should never take down the whole eval run.
    """
    from scripts.generate_diagnostic_report import generate

    try:
        report_path = generate(trace_id, output_dir=DIAGNOSTIC_REPORTS_DIR)
        named_path = os.path.join(DIAGNOSTIC_REPORTS_DIR, f"{eval_id}.html")
        shutil.move(report_path, named_path)
        logger.info(f"Diagnostic report for eval {eval_id} written to {named_path}")
    except Exception as e:
        logger.warning(f"Could not generate diagnostic report for eval {eval_id} (trace {trace_id}): {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare X-RAG against RAGAS/RAGChecker/ARES over the labeled eval set.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned examples without model calls or writes.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N examples.")
    parser.add_argument("--skip-ragchecker", action="store_true")
    parser.add_argument("--skip-ares", action="store_true")
    parser.add_argument("--eval-dataset", default="eval/eval_dataset.csv")
    args = parser.parse_args()
    if args.dry_run:
        rows = load_eval_dataset(args.eval_dataset)
        if args.limit is not None:
            rows = rows[:args.limit]
        print(json.dumps({"dry_run": True, "examples": [r["id"] for r in rows],
                          "skip_ragchecker": args.skip_ragchecker, "skip_ares": args.skip_ares}, indent=2))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from src.env_check import ensure_llm_credentials
    from dotenv import load_dotenv
    load_dotenv()
    ensure_llm_credentials()

    from src.chunk_registry import ChunkRegistry
    from src.vector_store import ChromaVectorStore
    from src.retriever import get_retriever
    from src.generator import Generator
    from src.runner import PipelineRunner

    eval_rows = load_eval_dataset(args.eval_dataset)
    if args.limit is not None:
        eval_rows = eval_rows[:args.limit]

    manifest = load_manifest()
    remaining = [row for row in eval_rows if row["id"] not in manifest["completed_eval_ids"]]
    logger.info(f"{len(eval_rows) - len(remaining)} of {len(eval_rows)} examples already done; {len(remaining)} remaining.")

    if not remaining:
        logger.info("Nothing to do.")
        return

    registry_path = "artifacts/chunk_registry.json"
    if not os.path.exists(registry_path):
        logger.error(f"{registry_path} not found. Run run_pipeline.py first.")
        sys.exit(1)
    chunk_registry = ChunkRegistry.load_from_json(registry_path)

    vector_store = ChromaVectorStore()
    vector_store.initialize_collection()
    retriever = get_retriever(vector_store, chunk_registry)
    generator = Generator()
    runner = PipelineRunner()

    for eval_row in remaining:
        eval_id = eval_row["id"]
        logger.info(f"Processing eval example {eval_id}: {eval_row['question'][:80]}")
        try:
            row = process_one_example(eval_row, retriever, generator, runner, chunk_registry, args.skip_ragchecker, args.skip_ares)
        except Exception as e:
            log_failure(eval_id, "xrag_pipeline", e)
            continue

        append_result_row(row)
        manifest["completed_eval_ids"].append(eval_id)
        manifest["eval_id_to_trace_id"][eval_id] = row["trace_id"]
        save_manifest(manifest)
        generate_and_name_report(eval_id, row["trace_id"])

    logger.info(f"Done. Results written to {RESULTS_CSV_PATH} / {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
