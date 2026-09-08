"""
E13 -- External validation: the real NLI verifier (unmodified model, unmodified
thresholds) scored against FEVER, a public, human-annotated claim-verification
benchmark, not this project's own corpus or synthetic constructions.

The gap this closes
----------------------
Every other experiment in this suite (E1-E12) validates the diagnostic layer
against either synthetic ground truth (E1, E9, E10, E11) or this project's own
40-question labeled legal-QA set (E8, E12) -- both are single-corpus, and the
NLI thresholds themselves (ENTAILMENT_THRESHOLD=0.7, PARTIAL_SUPPORT=0.4,
CONTRADICTION=0.7, NEUTRAL_UNSUPPORTED=0.8 -- configs/thresholds.py) have never
been checked against an external, independently-labeled dataset built for
exactly this task. This experiment is that check: FEVER (Thorne et al., 2018)
pairs a short claim with retrieved Wikipedia evidence and a human verdict
(SUPPORTS / REFUTES / NOT ENOUGH INFO) -- structurally the same three-way
judgment ClaimVerifier makes (SUPPORTED / CONTRADICTED / everything-else), over
a domain (Wikipedia, general knowledge) that shares zero vocabulary or style
with this project's Indian legal corpus. A verifier that holds up here is
holding up out-of-domain, not merely on data it was implicitly tuned against.

What is and is not reused
-----------------------------
``ClaimVerifier.run_nli()`` and ``ClaimVerifier._determine_status_and_reason()``
are called directly, unmodified, with the real ``cross-encoder/nli-deberta-v3-large``
model this project ships and the real threshold constants from
configs/thresholds.py -- nothing about the classification logic is
reimplemented or approximated. What is NOT exercised: claim decomposition (FEVER
claims are already atomic, single-sentence assertions -- there is nothing to
decompose) and the LLM-judge escalation path (``enable_llm_judge=False`` here,
so this measures the NLI model alone, the same component E8/E10 identified as
the live-accuracy bottleneck).

Field mapping (dataset quirk, not a project decision)
----------------------------------------------------------
``pietrolesci/nli_fever``'s own column names are the reverse of what they
contain: its ``premise`` column holds the short FEVER claim, and its
``hypothesis`` column holds the (often multi-sentence) Wikipedia evidence
passage. This experiment passes them to ``run_nli`` in the ORIENTATION
ClaimVerifier expects (evidence as premise, claim as hypothesis), not in the
dataset's own column order -- i.e. dataset['hypothesis'] -> our premise,
dataset['premise'] -> our hypothesis. Verified directly against three printed
examples before writing this file, not assumed from column names.

Label correspondence (many-to-one, disclosed rather than hidden)
-----------------------------------------------------------------------
FEVER's three-way judgment is coarser than VerificationStatus's five-way one.
SUPPORTS and REFUTES map to exact single statuses (SUPPORTED, CONTRADICTED);
NOT ENOUGH INFO is scored correct against ANY of {PARTIALLY_SUPPORTED,
UNSUPPORTED, NOT_VERIFIABLE} -- the three statuses that all mean "the NLI
model did not find clear support or clear contradiction," which is exactly
what NEI asserts. The full confusion matrix (all 5 VerificationStatus values
per FEVER gold label) is reported alongside the collapsed accuracy, so this
mapping choice is checkable rather than hidden inside one aggregate number.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

from experiments.common import ExampleSpec, Experiment, ExperimentContext, accuracy, confusion_matrix

logger = logging.getLogger(__name__)

DATASET_NAME = "pietrolesci/nli_fever"
DATASET_SPLIT = "dev"
#: Stratified sample: this many examples per FEVER gold label (3 labels),
#: seeded for reproducibility. 200/label = 600 total, comfortably above the
#: suite's MIN_EXAMPLES floor and large enough for a stable per-label estimate.
PER_LABEL_SAMPLE = 200

#: FEVER's three-way verdict -> the VerificationStatus values that count as a
#: correct prediction. SUPPORTS/REFUTES require an exact match; NOT ENOUGH INFO
#: accepts any of the three "neither clear support nor clear contradiction"
#: statuses -- see module docstring for why this is the honest mapping, not a
#: lenient one chosen to inflate the number.
LABEL_TO_CORRECT_STATUSES: Dict[str, tuple] = {
    "SUPPORTS": ("SUPPORTED",),
    "REFUTES": ("CONTRADICTED",),
    "NOT ENOUGH INFO": ("PARTIALLY_SUPPORTED", "UNSUPPORTED", "NOT_VERIFIABLE"),
}


class ExternalFeverValidationExperiment(Experiment):
    key = "exp13_external_fever_validation"
    number = 13
    title = "External validation: NLI verifier scored against FEVER"
    claim = (
        "The unmodified NLI verifier and its unmodified production thresholds, "
        "scored against a public, human-annotated, out-of-domain claim-verification "
        "benchmark (FEVER) rather than this project's own corpus or synthetic data."
    )
    supported_modes = ("offline", "live")  # local model inference only, no API calls

    def __init__(self):
        self._verifier = None

    def setup(self, ctx: ExperimentContext) -> None:
        from src.claim_verifier import ClaimVerifier
        # enable_llm_judge=False: this experiment measures the NLI model alone
        # (see module docstring) and requires no NVIDIA_API_KEY to run.
        self._verifier = ClaimVerifier(enable_llm_judge=False)

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        from datasets import load_dataset

        ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        rng = ctx.rng_for("e13:sample")

        by_label: Dict[str, list] = {label: [] for label in LABEL_TO_CORRECT_STATUSES}
        for i, row in enumerate(ds):
            label = row.get("fever_gold_label")
            if label in by_label:
                by_label[label].append(i)

        specs: List[ExampleSpec] = []
        for label, indices in by_label.items():
            chosen = indices if len(indices) <= PER_LABEL_SAMPLE else rng.sample(indices, PER_LABEL_SAMPLE)
            for idx in sorted(chosen):
                row = ds[idx]
                # See module docstring: this dataset's column names are
                # reversed relative to their content.
                claim_text = row["premise"]
                evidence_text = row["hypothesis"]
                specs.append(ExampleSpec(
                    example_id=f"{label.replace(' ', '_')}/{idx}",
                    payload={"claim": claim_text, "evidence": evidence_text, "gold_label": label},
                ))
        return specs

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload
        scores = self._verifier.run_nli(premise=payload["evidence"], hypothesis=payload["claim"])
        status, reason = self._verifier._determine_status_and_reason(
            scores["entailment"], scores["contradiction"], scores["neutral"]
        )
        gold_label = payload["gold_label"]
        correct = status.value in LABEL_TO_CORRECT_STATUSES[gold_label]
        return {
            "gold_label": gold_label,
            "predicted_status": status.value,
            "entailment": scores["entailment"],
            "contradiction": scores["contradiction"],
            "neutral": scores["neutral"],
            "correct": correct,
        }

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        overall_correct = sum(1 for r in records if r["correct"])

        by_label: Dict[str, Any] = {}
        for label in LABEL_TO_CORRECT_STATUSES:
            subset = [r for r in records if r["gold_label"] == label]
            if not subset:
                continue
            correct = sum(1 for r in subset if r["correct"])
            by_label[label] = {
                "n": len(subset),
                "accuracy": correct / len(subset),
                "predicted_status_distribution": dict(Counter(r["predicted_status"] for r in subset)),
            }

        gold = [r["gold_label"] for r in records]
        predicted = [r["predicted_status"] for r in records]

        return {
            "headline": {
                "dataset": DATASET_NAME,
                "split": DATASET_SPLIT,
                "n_total": len(records),
                "overall_collapsed_accuracy": overall_correct / len(records) if records else None,
                "supports_accuracy": by_label.get("SUPPORTS", {}).get("accuracy"),
                "refutes_accuracy": by_label.get("REFUTES", {}).get("accuracy"),
                "not_enough_info_accuracy": by_label.get("NOT ENOUGH INFO", {}).get("accuracy"),
            },
            "by_gold_label": by_label,
            "confusion_matrix_gold_rows_vs_predicted_status_columns": confusion_matrix(gold, predicted),
            "interpretation_notes": [
                "This is the first out-of-domain, externally-labeled check of the exact NLI "
                "model and thresholds the live pipeline uses -- FEVER is Wikipedia-domain "
                "general-knowledge claims, sharing no vocabulary or style with this project's "
                "Indian legal corpus, and was built independently for exactly this "
                "claim-verification task rather than adapted from a different one.",
                "REFUTES accuracy is the number most relevant to GROUNDING_FAILURE detection "
                "(Section 4.4/6.1's CONTRADICTED status): a verifier that cannot reliably "
                "recognize contradiction on a clean, human-labeled external set is unlikely to "
                "do better on live, noisier pipeline output.",
                "The confusion matrix reports all 5 VerificationStatus values per FEVER gold "
                "label, not only the collapsed pass/fail used in the headline accuracy -- e.g. "
                "whether NOT ENOUGH INFO errors lean toward false SUPPORTED or false "
                "CONTRADICTED is visible there and is not the same failure mode either way.",
                "This experiment does not exercise claim decomposition or the LLM-judge "
                "escalation path (enable_llm_judge=False) -- it isolates the NLI model itself, "
                "the component E8/E10 identified as the live-accuracy bottleneck, from the "
                "rest of the pipeline.",
            ],
        }


EXPERIMENT = ExternalFeverValidationExperiment()
