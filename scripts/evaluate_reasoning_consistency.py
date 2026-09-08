"""
Research Improvement #4: Reasoning Chain Validation.

Diagnostic accuracy (#1) checks whether the final *label* (xrag_primary_cause)
matches ground truth. It says nothing about whether the *reasoning* is sound
-- a report could have the right label for the wrong reason, or a
self-contradictory narrative that happens to name the right stage. This is
a genuinely different failure mode and needs genuinely different evidence:
real, independent human judgment on whether each `reasoning_chain` actually
makes sense, which no script can substitute for.

This script does two separate things, and they must not be conflated:

1. An AUTOMATED LOGICAL-CONSISTENCY CHECK (not a substitute for human
   judgment): verifies the reasoning_chain narrative doesn't contradict the
   underlying PipelineStateMatrix/RootCauseAnalysis data it was built from --
   e.g. does the stage named as the primary cause actually have status=FAIL,
   is a "no failures detected" narrative accompanied by zero FAIL stages, is
   the STAGE_TO_FAILURE_MAP direction correct. This catches "the explanation
   contradicts its own conclusion" bugs. It does NOT catch "the explanation
   is coherent but the underlying diagnosis is still wrong" -- that class of
   error needs a human reader.

2. A HUMAN-RATING WORKSHEET (artifacts/benchmark_comparison/
   reasoning_chain_worksheet.csv): one row per trace with the question,
   answer, primary cause, and full reasoning chain, plus empty
   `sound_yn`/`notes` columns for an actual person to fill in. Real human
   validation is NOT performed by this script -- filling in that worksheet
   is a separate, manual step, and the report must say so honestly rather
   than imply this script did it.

Usage:
    python -m scripts.evaluate_reasoning_consistency
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List

from src.root_cause_reasoner import RootCauseReasoner

RESULTS_DIR = "artifacts/benchmark_comparison"
STAGE_TO_FAILURE = {stage.value: failure.value for stage, failure in RootCauseReasoner.STAGE_TO_FAILURE_MAP.items()}
FAILURE_TO_STAGE = {v: k for k, v in STAGE_TO_FAILURE.items()}


def load_results_full(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_report(trace_id: str, reports_dir: str = "artifacts/reports") -> Dict[str, Any]:
    path = os.path.join(reports_dir, f"{trace_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_consistency(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict of individual boolean checks plus an overall 'consistent'
    flag (True only if every check passes). Each check is independently
    inspectable so a disagreement can be traced to exactly which assumption
    broke, rather than a single opaque pass/fail.
    """
    rca = report["root_cause_analysis"]
    stages = {s["stage"]: s for s in report["pipeline_overview"]["pipeline_stages"]}
    primary_cause = rca["primary_cause"]
    reasoning_chain = rca.get("reasoning_chain", [])
    chain_text = " ".join(reasoning_chain)

    checks = {}

    if primary_cause == "UNKNOWN":
        # Healthy narrative: no stage should be FAIL, and the chain should say so.
        checks["no_fail_stages_when_healthy"] = not any(s["status"] == "FAIL" for s in stages.values())
        checks["chain_states_no_failure"] = "no failures detected" in chain_text.lower()
        checks["primary_cause_stage_is_fail"] = None  # not applicable
    else:
        expected_stage = FAILURE_TO_STAGE.get(primary_cause)
        stage_state = stages.get(expected_stage) if expected_stage else None
        checks["primary_cause_stage_is_fail"] = bool(stage_state and stage_state["status"] == "FAIL")
        checks["chain_names_primary_cause"] = primary_cause in chain_text
        checks["no_fail_stages_when_healthy"] = None  # not applicable

    # The chain should mention every stage that's actually FAIL (as either
    # the primary cause or a logged secondary effect) -- a FAIL stage
    # silently missing from the narrative would be a real inconsistency.
    fail_stages = [stage for stage, s in stages.items() if s["status"] == "FAIL"]
    mapped_failures = [STAGE_TO_FAILURE.get(stage) for stage in fail_stages]
    checks["all_fail_stages_named_in_chain"] = all(
        (failure_type in chain_text) for failure_type in mapped_failures if failure_type
    )

    applicable = [v for v in checks.values() if v is not None]
    checks["consistent"] = all(applicable) if applicable else True
    return checks


def build_worksheet_row(eval_id: str, report: Dict[str, Any]) -> Dict[str, str]:
    rca = report["root_cause_analysis"]
    return {
        "eval_id": eval_id,
        "question": report["executive_summary"]["question"],
        "generated_answer": report["executive_summary"]["generated_answer"],
        "primary_cause": rca["primary_cause"],
        "reasoning_chain": " | ".join(rca.get("reasoning_chain", [])),
        "sound_yn": "",
        "notes": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Automated reasoning-chain consistency check + human-rating worksheet generator.")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "results.json"))
    args = parser.parse_args()

    rows = load_results_full(args.results)

    consistency_by_eval: Dict[str, Any] = {}
    worksheet_rows = []
    for row in rows:
        eval_id = row["eval_id"]
        trace_id = row["trace_id"]
        report = load_report(trace_id)

        consistency_by_eval[eval_id] = check_consistency(report)
        worksheet_rows.append(build_worksheet_row(eval_id, report))

    n_consistent = sum(1 for c in consistency_by_eval.values() if c["consistent"])
    output = {
        "n_examples": len(consistency_by_eval),
        "n_consistent": n_consistent,
        "consistency_rate": (n_consistent / len(consistency_by_eval)) if consistency_by_eval else None,
        "per_example": consistency_by_eval,
        "note": (
            "This is an AUTOMATED self-consistency check (does the narrative match its own "
            "underlying data), NOT a substitute for human judgment on whether the reasoning is "
            "actually sound. See reasoning_chain_worksheet.csv for the human-rating step, which "
            "has not yet been completed."
        ),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "reasoning_consistency.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    worksheet_path = os.path.join(RESULTS_DIR, "reasoning_chain_worksheet.csv")
    with open(worksheet_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(worksheet_rows[0].keys()) if worksheet_rows else [])
        writer.writeheader()
        writer.writerows(worksheet_rows)

    print(json.dumps(output, indent=2))
    print(f"\nSaved consistency results to {json_path}")
    print(f"Saved human-rating worksheet (not yet filled in) to {worksheet_path}")


if __name__ == "__main__":
    main()
