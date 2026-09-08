"""
Run the paper's experiment suite (E1-E5), resumably.

    python -m experiments.run_all                 # run everything not yet done
    python -m experiments.run_all --status        # what is done, what is not
    python -m experiments.run_all --from 3        # start at experiment 3
    python -m experiments.run_all --only 2 4      # just these
    python -m experiments.run_all --mode live     # use the real corpus / LLM
    python -m experiments.run_all --force         # discard checkpoints, rerun

Resumption works at two levels, and both matter:

*Between experiments.* A completed experiment is recorded in
``artifacts/experiments/manifest.json`` and skipped on the next invocation. Stop
after experiment 5 and the next run continues at 5 -- it does not restart at 1.

*Within an experiment.* Each example is appended to ``records.jsonl`` as it
finishes, so a run killed partway through experiment 3 resumes at the example
it was on. Nothing already computed is recomputed.

The safety property behind both: example ids are deterministic functions of the
plan, and every example's randomness is seeded from its own id. A resumed run
therefore produces the same records as an uninterrupted one. If the plan does
change -- a different seed, a different mode, an edited grid -- the checkpoint's
stored fingerprint no longer matches and the run refuses to continue rather
than silently blending results from two different studies. ``--force`` is the
way to say you meant it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from experiments.common import (
    EXPERIMENTS_DIR,
    Checkpoint,
    ConfigDriftError,
    DEFAULT_SEED,
    Experiment,
    ExperimentContext,
    utc_now,
)

logger = logging.getLogger("experiments")

MANIFEST_NAME = "manifest.json"
REPORT_NAME = "SUITE_REPORT.md"


def load_experiments() -> List[Experiment]:
    """Import order defines run order; each module exposes a single EXPERIMENT."""
    from experiments.exp01_fault_injection import EXPERIMENT as e1
    from experiments.exp02_reranker_window import EXPERIMENT as e2
    from experiments.exp03_refusal_calibration import EXPERIMENT as e3
    from experiments.exp04_corpus_quality import EXPERIMENT as e4
    from experiments.exp05_diagnostic_agreement import EXPERIMENT as e5
    from experiments.exp06_strategy_ablation import EXPERIMENT as e6
    from experiments.exp07_generation_ablation import EXPERIMENT as e7
    from experiments.exp08_live_diagnostic_accuracy import EXPERIMENT as e8
    from experiments.exp09_agentic_propagation_depth import EXPERIMENT as e9
    from experiments.exp10_generator_rule_sensitivity import EXPERIMENT as e10
    from experiments.exp11_nway_compound_faults import EXPERIMENT as e11
    from experiments.exp12_diagnostic_cost_comparison import EXPERIMENT as e12
    from experiments.exp13_external_fever_validation import EXPERIMENT as e13

    return sorted([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13], key=lambda e: e.number)


# ---------------------------------------------------------------------------
# Suite manifest
# ---------------------------------------------------------------------------


def manifest_path(base_dir: str) -> str:
    return os.path.join(base_dir, MANIFEST_NAME)


def load_manifest(base_dir: str) -> Dict[str, Any]:
    path = manifest_path(base_dir)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Unreadable manifest at %s; starting a fresh one.", path)
    return {"experiments": {}, "created_at": utc_now()}


def save_manifest(base_dir: str, manifest: Dict[str, Any]) -> None:
    os.makedirs(base_dir, exist_ok=True)
    manifest["updated_at"] = utc_now()
    with open(manifest_path(base_dir), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def experiment_status(experiment: Experiment, base_dir: str) -> Dict[str, Any]:
    checkpoint = Checkpoint(experiment.key, base_dir=base_dir)
    state = checkpoint.load_state()
    return {
        "number": experiment.number,
        "key": experiment.key,
        "title": experiment.title,
        "status": state.get("status", "not_started"),
        "n_recorded": len(checkpoint.completed_ids()),
        "n_planned": state.get("n_planned"),
        "mode": state.get("mode"),
        "seed": state.get("seed"),
        "completed_at": state.get("completed_at"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_suite_report(base_dir: str, experiments: List[Experiment]) -> str:
    """A single readable page of every experiment's headline, for the paper."""
    lines = [
        "# X-RAG Experiment Suite -- Results",
        "",
        f"Generated {utc_now()}.",
        "",
        "Each section is the headline block of that experiment's `summary.json`.",
        "Full per-example records are in `artifacts/experiments/<key>/records.jsonl`.",
        "",
    ]

    for experiment in experiments:
        checkpoint = Checkpoint(experiment.key, base_dir=base_dir)
        summary = checkpoint.load_summary()
        lines.append(f"## E{experiment.number} -- {experiment.title}")
        lines.append("")
        lines.append(f"**Claim.** {experiment.claim}")
        lines.append("")
        if not summary:
            lines += ["_Not yet run._", ""]
            continue
        lines.append(
            f"`mode={summary.get('mode')}` `seed={summary.get('seed')}` "
            f"`n={summary.get('n_examples')}`"
        )
        lines += ["", "```json", json.dumps(summary.get("headline", {}), indent=2, default=str), "```", ""]
        for note in summary.get("interpretation_notes", []):
            lines.append(f"> {note}")
            lines.append("")

    path = os.path.join(base_dir, REPORT_NAME)
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def print_status(experiments: List[Experiment], base_dir: str) -> None:
    print(f"\nExperiment suite status  ({base_dir})\n")
    print(f"{'#':<3} {'status':<12} {'records':<10} {'mode':<9} key")
    print("-" * 72)
    for experiment in experiments:
        status = experiment_status(experiment, base_dir)
        planned = status["n_planned"]
        recorded = f"{status['n_recorded']}/{planned}" if planned else str(status["n_recorded"])
        print(
            f"{status['number']:<3} {status['status']:<12} {recorded:<10} "
            f"{str(status['mode'] or '-'):<9} {status['key']}"
        )
    print()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def select(experiments: List[Experiment], args) -> List[Experiment]:
    if args.only:
        wanted = set(args.only)
        return [e for e in experiments if e.number in wanted]
    return [e for e in experiments if e.number >= args.start_from]


def run_suite(experiments: List[Experiment], ctx: ExperimentContext, rerun_complete: bool) -> int:
    manifest = load_manifest(ctx.base_dir)
    failures = 0

    for experiment in experiments:
        checkpoint = Checkpoint(experiment.key, base_dir=ctx.base_dir)

        if checkpoint.is_complete() and not rerun_complete and not ctx.force:
            logger.info(
                "E%d %s: already complete (%d records) -- skipping. "
                "Use --force to recompute.",
                experiment.number, experiment.key, len(checkpoint.completed_ids()),
            )
            manifest["experiments"][experiment.key] = experiment_status(experiment, ctx.base_dir)
            save_manifest(ctx.base_dir, manifest)
            continue

        if ctx.is_live and "live" not in experiment.supported_modes:
            logger.info("E%d %s: no live mode; running offline.", experiment.number, experiment.key)
            experiment_ctx = ExperimentContext(**{**ctx.__dict__, "mode": "offline"})
        else:
            experiment_ctx = ctx

        logger.info("=" * 70)
        logger.info("E%d -- %s [%s]", experiment.number, experiment.title, experiment_ctx.mode)
        logger.info("=" * 70)

        t0 = time.time()
        try:
            summary = experiment.run(experiment_ctx)
        except ConfigDriftError as e:
            logger.error("E%d %s: %s", experiment.number, experiment.key, e)
            failures += 1
            manifest["experiments"][experiment.key] = {
                **experiment_status(experiment, ctx.base_dir),
                "last_error": str(e),
            }
            save_manifest(ctx.base_dir, manifest)
            continue
        except Exception as e:
            logger.error(
                "E%d %s failed: %s\n%s",
                experiment.number, experiment.key, e, traceback.format_exc(),
            )
            failures += 1
            # Partial records survive on disk; the next run resumes from them
            # rather than starting over.
            manifest["experiments"][experiment.key] = {
                **experiment_status(experiment, ctx.base_dir),
                "last_error": str(e),
            }
            save_manifest(ctx.base_dir, manifest)
            continue

        logger.info(
            "E%d complete in %.1fs -- %d examples. Summary: %s",
            experiment.number, time.time() - t0, summary.get("n_examples"),
            checkpoint.summary_path,
        )
        manifest["experiments"][experiment.key] = experiment_status(experiment, ctx.base_dir)
        save_manifest(ctx.base_dir, manifest)

    report = write_suite_report(ctx.base_dir, load_experiments())
    logger.info("Suite report written to %s", report)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the X-RAG paper experiment suite (E1-E5), resumably.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--from", dest="start_from", type=int, default=1, metavar="N",
        help="Start at experiment N (default 1). Earlier experiments are left as they are.",
    )
    parser.add_argument(
        "--only", type=int, nargs="+", metavar="N",
        help="Run only these experiment numbers.",
    )
    parser.add_argument(
        "--mode", choices=("offline", "live"), default="offline",
        help="offline (default): deterministic, no corpus or API needed. "
             "live: real retriever/generator over the ingested corpus.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap examples per experiment (smoke runs; implies --allow-small-sample).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Discard existing checkpoints and recompute from scratch.",
    )
    parser.add_argument(
        "--rerun-complete", action="store_true",
        help="Re-run experiments already marked complete, keeping their records "
             "(resumes rather than discarding; use --force to discard).",
    )
    parser.add_argument("--status", action="store_true", help="Print suite status and exit.")
    parser.add_argument("--base-dir", default=EXPERIMENTS_DIR)
    parser.add_argument(
        "--allow-small-sample", action="store_true",
        help="Bypass the 50-example floor. For smoke runs only -- results below "
             "the floor must not be reported.",
    )
    parser.add_argument(
        "--extra", nargs="*", default=[], metavar="KEY=VALUE",
        help="Per-experiment knobs, e.g. --extra window_max=20 verifier=nli",
    )
    return parser


def parse_extra(pairs: List[str]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--extra expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = value
    return extra


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # The heavy libraries are chatty enough to bury the suite's own progress.
    for noisy in ("httpx", "sentence_transformers", "transformers", "chromadb", "src.retriever"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    experiments = load_experiments()

    if args.status:
        print_status(experiments, args.base_dir)
        return 0

    selected = select(experiments, args)
    if not selected:
        print("Nothing selected. Check --from / --only.", file=sys.stderr)
        return 1

    ctx = ExperimentContext(
        mode=args.mode,
        seed=args.seed,
        limit=args.limit,
        force=args.force,
        base_dir=args.base_dir,
        allow_small_sample=args.allow_small_sample or args.limit is not None,
        extra=parse_extra(args.extra),
    )

    logger.info(
        "Running E%s in %s mode (seed=%d).",
        ",".join(str(e.number) for e in selected), ctx.mode, ctx.seed,
    )
    failures = run_suite(selected, ctx, rerun_complete=args.rerun_complete)

    print_status(experiments, args.base_dir)
    if failures:
        logger.error(
            "%d experiment(s) failed. Partial records were kept -- re-running the same "
            "command resumes from where each one stopped.", failures,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
