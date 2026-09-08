"""
E3 -- Refusal calibration: faithfulness metrics reward silence.

The pathology
-------------
Faithfulness-style metrics -- RAGAS faithfulness, RAGChecker's precision, and
this framework's own average entailment -- are all fractions whose numerator
and denominator are counts of *claims made by the answer*:

    faithfulness = supported_claims / total_claims

An answer that says "I do not have enough information to answer this" makes no
verifiable claims. The fraction is 0/0. Every implementation has to pick a
convention for that case, and the convention that makes the metric
well-behaved elsewhere -- treat a vacuous answer as perfectly faithful -- makes
refusal the score-maximising strategy. A system that refuses every question
scores 1.0 on faithfulness. It is also useless.

This is not a hypothetical: it is why faithfulness must never be reported
without a coverage term alongside it, and why an evaluation harness that ranks
systems on faithfulness alone will select for silence.

What is measured
----------------
**Arm A (offline, always available).** Over answers this pipeline actually
produced -- harvested from ``artifacts/rag_traces/``, not written for the
experiment -- compare refusals against substantive answers on faithfulness
(under both conventions), on claim count, and on recall against the gold answer
where the question is in the labeled eval set. The predicted result is that
refusals win on faithfulness and score zero on recall.

**Arm B (live).** The prompt A/B. The same questions are answered under the
superseded refusal-first instruction (``STRICT_REFUSAL_SYSTEM_INSTRUCTIONS_GEN1``)
and the current refusal-as-last-resort instruction, over a mix of answerable
eval questions and deliberately out-of-corpus probes with known
``should_refuse`` labels. That yields genuine refusal calibration: over-refusal
rate, under-refusal rate, F1 -- and shows what the faithfulness metric would
have said about each prompt, which is the point.

Verification backend
--------------------
Claim support is decided by the project's NLI verifier in live mode and by a
lexical-overlap proxy offline (the proxy keeps the default run fast and free of
a 1.6 GB model download). The backend used is recorded in every record and in
the summary. The headline finding does not depend on it: a refusal's claim
count is zero under any verifier, and zero is what drives the pathology.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    iter_saved_traces,
    load_eval_dataset,
    mean,
    wilson_interval,
)

logger = logging.getLogger(__name__)

PROBES_PATH = os.path.join("experiments", "data", "refusal_probes.csv")

#: Surface forms of a refusal. Deliberately broad: an under-detected refusal
#: would be counted as a substantive answer and would weaken the finding, so
#: the detector errs toward recall.
_REFUSAL_PATTERNS = [
    r"do(?:es)? not have enough information",
    r"don'?t have enough information",
    r"not enough information",
    r"insufficient information",
    r"unable to (?:answer|determine|find)",
    r"cannot (?:answer|determine|be answered)",
    r"can'?t (?:answer|determine)",
    r"no (?:relevant )?information (?:is )?(?:available|provided|found)",
    r"the (?:retrieved )?context does not (?:contain|provide|mention)",
    r"i (?:do not|don'?t) know",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

#: Sentences that assert nothing checkable and must not be counted as claims --
#: counting them would inflate a refusal's claim count and hide the 0/0 case.
_NON_ASSERTIVE_RE = re.compile(
    r"^\s*(?:"
    r"i (?:do not|don'?t|cannot|can'?t)\b"
    r"|(?:the )?(?:retrieved )?context (?:does not|doesn'?t)\b"
    r"|(?:however|unfortunately|please|note that|in summary|let me know)\b"
    r"|there (?:is|are) no\b"
    r")",
    re.IGNORECASE,
)

#: A persisted transport/API failure, not an answer. Older traces recorded the
#: error string in ``generated_answer`` (the bug ``GenerationResult.error``
#: exists to prevent), so they have to be excluded here or they would be scored
#: as if the model had said them.
_GENERATION_ERROR_RE = re.compile(
    r"^\s*(?:error generating answer|error:|\{'message':)", re.IGNORECASE
)

#: Context blocks as PromptBuilder writes them into the prompt snapshot.
_CONTEXT_BLOCK_RE = re.compile(
    r"--- Context chunk \d+ \[Chunk-ID: [^\]]*\] ---\n(.*?)(?=\n--- Context chunk |\n\nQuestion: |\Z)",
    re.DOTALL,
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "by", "as", "it", "its", "this", "that",
    "with", "from", "which", "who", "whom", "shall", "may", "any", "such", "not",
    "if", "then", "under", "means", "person", "he", "she", "they",
}

#: Minimum content-word overlap for the lexical proxy to call a claim supported.
LEXICAL_SUPPORT_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Answer analysis primitives (pure -- unit-testable without any model)
# ---------------------------------------------------------------------------


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer or ""))


def is_generation_error(answer: str) -> bool:
    return bool(_GENERATION_ERROR_RE.match(answer or ""))


def contexts_from_prompt_snapshot(prompt: str) -> List[str]:
    """
    Recover the retrieved context from a trace's prompt snapshot.

    Preferred over resolving the trace's chunk ids against the current
    ``chunk_registry.json``: the registry is rebuilt on every re-ingest and its
    ids do not survive that, so most historical traces resolve to nothing. The
    prompt snapshot is immutable and is literally the text the generator saw,
    which is the correct premise for a faithfulness measurement anyway.
    """
    return [block.strip() for block in _CONTEXT_BLOCK_RE.findall(prompt or "") if block.strip()]


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def extract_claims(answer: str) -> List[str]:
    """
    Rule-based atomic-claim extraction: assertive sentences, with markdown
    scaffolding and non-assertive framing removed.

    A rule is used rather than the LLM decomposer so that Arm A costs nothing
    to reproduce and cannot drift with an API model. The measurement that
    matters here is a *count*, and specifically whether it is zero.
    """
    claims: List[str] = []
    for sentence in split_sentences(answer):
        cleaned = re.sub(r"[*_`#>]|^\s*[-•]\s*|^\s*\d+\.\s*", " ", sentence).strip()
        cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned).strip()  # drop [Chunk-ID] citations
        if len(_content_words(cleaned)) < 3:
            continue
        if _NON_ASSERTIVE_RE.match(cleaned):
            continue
        claims.append(cleaned)
    return claims


def _content_words(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS]


def lexical_support(claim: str, contexts: Sequence[str]) -> float:
    """Fraction of the claim's content words that appear anywhere in the
    context. A crude proxy for entailment, and labeled as such wherever it is
    reported."""
    words = set(_content_words(claim))
    if not words:
        return 0.0
    haystack = set(_content_words(" ".join(contexts)))
    return len(words & haystack) / len(words)


def faithfulness_scores(
    n_claims: int, n_supported: int, empty_answer_convention: float = 1.0
) -> Dict[str, Optional[float]]:
    """
    The two conventions, computed side by side.

    ``conventional`` implements what evaluation harnesses actually do with a
    claimless answer: score it 1.0, because there is nothing unsupported in it.
    ``undefined_on_empty`` returns None instead, which is the honest reading --
    faithfulness of an answer that asserts nothing is not defined.

    ``silence_adjusted`` multiplies faithfulness by the answer's coverage of
    the question, so that an answer cannot buy a high score by asserting
    nothing. The coverage term is supplied by the caller (claim recall against
    a gold answer where one exists), because faithfulness alone genuinely
    cannot distinguish a careful refusal from a lazy one.
    """
    if n_claims == 0:
        return {
            "conventional": empty_answer_convention,
            "undefined_on_empty": None,
        }
    return {
        "conventional": n_supported / n_claims,
        "undefined_on_empty": n_supported / n_claims,
    }


def load_probes(path: str = PROBES_PATH) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


class RefusalCalibrationExperiment(Experiment):
    key = "exp03_refusal_calibration"
    number = 3
    title = "Refusal calibration -- faithfulness metrics reward silence"
    claim = (
        "Faithfulness-style metrics score a refusal at or near 1.0 while its answer "
        "recall is 0, so ranking systems on faithfulness selects for silence."
    )

    def __init__(self) -> None:
        self._registry = None
        self._verifier = None
        self._backend = "lexical"
        self._gold_by_question: Dict[str, str] = {}
        self._retriever = None
        self._generator = None

    # -- planning --------------------------------------------------------

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        if ctx.is_live:
            rows = load_eval_dataset()
            probes = load_probes()
            specs: List[ExampleSpec] = []
            for generation in ("gen1_strict", "gen2_current"):
                for row in rows:
                    specs.append(ExampleSpec(
                        example_id=f"{generation}/eval/{row['id']}",
                        payload={
                            "prompt_generation": generation,
                            "question": row["question"],
                            "gold_answer": row.get("gold_answer") or "",
                            "should_refuse": False,
                            "source": "eval",
                        },
                    ))
                for probe in probes:
                    specs.append(ExampleSpec(
                        example_id=f"{generation}/probe/{probe['probe_id']}",
                        payload={
                            "prompt_generation": generation,
                            "question": probe["question"],
                            "gold_answer": "",
                            "should_refuse": probe["should_refuse"] == "1",
                            "source": "probe",
                        },
                    ))
            return specs

        # Offline: every distinct (question, answer) the pipeline has produced.
        seen = set()
        specs = []
        for trace in iter_saved_traces():
            question = (trace.get("question") or "").strip()
            answer = (trace.get("generated_answer") or "").strip()
            if not question or not answer or is_generation_error(answer):
                continue
            fingerprint = (question, answer)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            specs.append(ExampleSpec(
                example_id=f"trace/{trace['trace_id']}",
                payload={
                    "question": question,
                    "answer": answer,
                    "prompt_snapshot": trace.get("prompt_snapshot") or "",
                    "chunk_ids": [
                        ref.get("chunk_id")
                        for ref in trace.get("retrieved_chunk_references", [])
                    ],
                },
            ))

        if not specs:
            raise RuntimeError(
                "offline mode harvests answers from artifacts/rag_traces/, which is empty. "
                "Run the pipeline (or scripts.run_baseline_comparison) first, or use --mode live."
            )
        # Sorted so the plan is order-stable regardless of filesystem listing.
        return sorted(specs, key=lambda s: s.example_id)

    # -- resources -------------------------------------------------------

    def setup(self, ctx: ExperimentContext) -> None:
        from src.chunk_registry import ChunkRegistry

        registry_path = "artifacts/chunk_registry.json"
        if os.path.exists(registry_path):
            self._registry = ChunkRegistry.load_from_json(registry_path)

        self._gold_by_question = {
            (row["question"] or "").strip(): (row.get("gold_answer") or "")
            for row in load_eval_dataset()
        }

        backend = ctx.extra.get("verifier", "nli" if ctx.is_live else "lexical")
        if backend == "nli":
            try:
                from src.claim_verifier import ClaimVerifier

                self._verifier = ClaimVerifier()
                self._backend = "nli"
            except Exception as e:  # model download / device failure
                logger.warning("NLI verifier unavailable (%s); falling back to lexical proxy.", e)
                self._backend = "lexical"
        else:
            self._backend = "lexical"

        if ctx.is_live:
            from src.generator import Generator
            from src.retriever import get_retriever
            from src.vector_store import ChromaVectorStore

            if self._registry is None:
                raise RuntimeError(
                    "live mode needs artifacts/chunk_registry.json. Run `python run_pipeline.py`."
                )
            vector_store = ChromaVectorStore()
            vector_store.initialize_collection()
            self._retriever = get_retriever(vector_store, self._registry)
            self._generator = Generator()

    def _contexts_for(self, payload: Dict[str, Any]) -> Tuple[List[str], str]:
        """Resolve the context the generator saw. Returns ``(texts, source)``."""
        texts = contexts_from_prompt_snapshot(payload.get("prompt_snapshot", ""))
        if texts:
            return texts, "prompt_snapshot"

        if self._registry:
            resolved = []
            for chunk_id in payload.get("chunk_ids") or []:
                record = self._registry.get_chunk(chunk_id)
                if record:
                    resolved.append(record.text)
            if resolved:
                return resolved, "chunk_registry"
        return [], "unavailable"

    # -- scoring ---------------------------------------------------------

    def _support_fractions(self, claims: Sequence[str], contexts: Sequence[str]) -> Tuple[int, List[float]]:
        if not claims or not contexts:
            return 0, []

        if self._backend == "nli" and self._verifier is not None:
            sources = [(f"ctx{i}", i + 1, text) for i, text in enumerate(contexts)]
            supported = 0
            scores = []
            for claim in claims:
                outcome = self._verifier._verify_against_sources(claim, sources)
                scores.append(outcome["entailment"])
                if outcome["status"].value in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                    supported += 1
            return supported, scores

        scores = [lexical_support(claim, contexts) for claim in claims]
        return sum(1 for s in scores if s >= LEXICAL_SUPPORT_THRESHOLD), scores

    def _claim_recall(self, answer: str, gold_answer: str) -> Optional[float]:
        """
        Coverage: what fraction of the gold answer's claims the answer carries.
        This is the term faithfulness is missing, and the one on which a refusal
        scores zero.
        """
        if not gold_answer:
            return None
        gold_claims = extract_claims(gold_answer) or split_sentences(gold_answer)
        if not gold_claims:
            return None
        supported, _ = self._support_fractions(gold_claims, [answer])
        return supported / len(gold_claims)

    def _score_answer(
        self, question: str, answer: str, contexts: Sequence[str], gold_answer: str
    ) -> Dict[str, Any]:
        claims = extract_claims(answer)
        n_supported, support_scores = self._support_fractions(claims, contexts)
        faith = faithfulness_scores(len(claims), n_supported)
        recall = self._claim_recall(answer, gold_answer)

        silence_adjusted = None
        if faith["conventional"] is not None and recall is not None:
            silence_adjusted = faith["conventional"] * recall

        return {
            "refused": is_refusal(answer),
            "answer_chars": len(answer),
            "n_claims": len(claims),
            "n_supported_claims": n_supported,
            "faithfulness_conventional": faith["conventional"],
            "faithfulness_undefined_on_empty": faith["undefined_on_empty"],
            "claim_recall_vs_gold": recall,
            "silence_adjusted_faithfulness": silence_adjusted,
            "mean_support_score": mean(support_scores),
            "verifier_backend": self._backend,
            "has_gold": bool(gold_answer),
        }

    # -- execution -------------------------------------------------------

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        payload = spec.payload

        if ctx.is_live:
            from configs.prompts import (
                LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS,
                STRICT_REFUSAL_SYSTEM_INSTRUCTIONS_GEN1,
            )

            instructions = (
                STRICT_REFUSAL_SYSTEM_INSTRUCTIONS_GEN1
                if payload["prompt_generation"] == "gen1_strict"
                else LEGAL_DOMAIN_SYSTEM_INSTRUCTIONS
            )
            question = payload["question"]
            retrieval = self._retriever.retrieve(question)
            generation = self._generator.generate(retrieval, system_instructions=instructions)
            if not generation.ok:
                return {
                    "error": generation.error,
                    "prompt_generation": payload["prompt_generation"],
                    "source": payload["source"],
                    "should_refuse": payload["should_refuse"],
                }
            contexts = [c.chunk_text for c in retrieval.retrieved_chunks]
            record = self._score_answer(
                question, generation.generated_answer, contexts, payload["gold_answer"]
            )
            record.update({
                "prompt_generation": payload["prompt_generation"],
                "source": payload["source"],
                "should_refuse": payload["should_refuse"],
                "question": question,
                "answer": generation.generated_answer,
            })
            return record

        question = payload["question"]
        answer = payload["answer"]
        contexts, context_source = self._contexts_for(payload)
        gold = self._gold_by_question.get(question, "")
        record = self._score_answer(question, answer, contexts, gold)
        record.update({
            "prompt_generation": "as_run",
            # Harvested traces come from answerable eval questions where a gold
            # answer exists; anything else is left unlabeled rather than guessed.
            "should_refuse": False if gold else None,
            "source": "harvested_trace",
            "question": question,
            "n_contexts": len(contexts),
            "context_source": context_source,
        })
        return record

    # -- aggregation -----------------------------------------------------

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        scored = [r for r in records if not r.get("error")]
        if not scored:
            return {"error": "no usable records", "n_errors": len(records)}

        # A record whose context could not be recovered would score every claim
        # unsupported and drag substantive answers toward 0 -- manufacturing the
        # very gap the experiment claims to find. Excluded from the faithfulness
        # comparison, and the exclusion is reported.
        usable = [r for r in scored if r.get("n_contexts", 1) > 0]
        if not usable:
            return {
                "error": "no records with recoverable context",
                "n_scored": len(scored),
            }

        refusals = [r for r in usable if r["refused"]]
        answers = [r for r in usable if not r["refused"]]

        silence_reward = {
            "n_refusals": len(refusals),
            "n_substantive_answers": len(answers),
            "refusal_rate": len(refusals) / len(usable),
            "mean_faithfulness_conventional_refusals": mean(
                [r["faithfulness_conventional"] for r in refusals]
            ),
            "mean_faithfulness_conventional_answers": mean(
                [r["faithfulness_conventional"] for r in answers]
            ),
            "mean_claim_recall_refusals": mean([r["claim_recall_vs_gold"] for r in refusals]),
            "mean_claim_recall_answers": mean([r["claim_recall_vs_gold"] for r in answers]),
            "mean_claims_refusals": mean([r["n_claims"] for r in refusals]),
            "mean_claims_answers": mean([r["n_claims"] for r in answers]),
            "n_refusals_with_zero_claims": sum(1 for r in refusals if r["n_claims"] == 0),
            "mean_silence_adjusted_refusals": mean(
                [r["silence_adjusted_faithfulness"] for r in refusals]
            ),
            "mean_silence_adjusted_answers": mean(
                [r["silence_adjusted_faithfulness"] for r in answers]
            ),
        }

        faith_gap = None
        if (
            silence_reward["mean_faithfulness_conventional_refusals"] is not None
            and silence_reward["mean_faithfulness_conventional_answers"] is not None
        ):
            faith_gap = (
                silence_reward["mean_faithfulness_conventional_refusals"]
                - silence_reward["mean_faithfulness_conventional_answers"]
            )

        summary: Dict[str, Any] = {
            "headline": {
                "faithfulness_advantage_of_refusing": faith_gap,
                "recall_cost_of_refusing": _delta(
                    silence_reward["mean_claim_recall_refusals"],
                    silence_reward["mean_claim_recall_answers"],
                ),
                "silence_adjusted_advantage_of_refusing": _delta(
                    silence_reward["mean_silence_adjusted_refusals"],
                    silence_reward["mean_silence_adjusted_answers"],
                ),
                "verifier_backend": usable[0].get("verifier_backend"),
            },
            "silence_reward_analysis": silence_reward,
            "n_generation_errors": len(records) - len(scored),
            "n_excluded_no_recoverable_context": len(scored) - len(usable),
            "n_scored_for_faithfulness": len(usable),
            "context_sources": _count_by(usable, "context_source"),
            "interpretation_notes": [
                "faithfulness_advantage_of_refusing > 0 with recall_cost_of_refusing < 0 is "
                "the finding: refusing improves the metric and destroys the answer. "
                "silence_adjusted_advantage_of_refusing should be strongly negative -- that "
                "is the check that the proposed correction actually removes the incentive.",
                "The conventional 0/0 -> 1.0 rule is the mechanism. Reported alongside the "
                "undefined-on-empty variant so the choice is visible rather than buried in "
                "a metric implementation.",
                "Offline records are harvested from real pipeline runs and were not written "
                "for this experiment; their refusal rate is whatever the shipped prompt "
                "produced, not a designed condition.",
            ],
        }

        labeled = [r for r in usable if r.get("should_refuse") is not None]
        if labeled:
            summary["refusal_calibration"] = self._calibration(labeled)

        generations = sorted({r["prompt_generation"] for r in usable})
        if len(generations) > 1:
            summary["prompt_generation_ab"] = {
                generation: self._arm_summary([r for r in usable if r["prompt_generation"] == generation])
                for generation in generations
            }
            summary["headline"]["over_refusal_delta_gen1_minus_gen2"] = _delta(
                summary["prompt_generation_ab"].get("gen1_strict", {}).get("over_refusal_rate"),
                summary["prompt_generation_ab"].get("gen2_current", {}).get("over_refusal_rate"),
            )

        return summary

    @staticmethod
    def _calibration(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Treating "refuse" as the positive class against the ``should_refuse``
        label: does the system refuse when, and only when, it should?
        """
        tp = sum(1 for r in records if r["should_refuse"] and r["refused"])
        fp = sum(1 for r in records if not r["should_refuse"] and r["refused"])
        fn = sum(1 for r in records if r["should_refuse"] and not r["refused"])
        tn = sum(1 for r in records if not r["should_refuse"] and not r["refused"])

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        answerable = tp + fp + tn + fn - (tp + fn)
        over_lo, over_hi = wilson_interval(fp, fp + tn) if (fp + tn) else (None, None)

        return {
            "n_labeled": len(records),
            "confusion": {
                "correct_refusal": tp,
                "over_refusal": fp,
                "under_refusal": fn,
                "correct_answer": tn,
            },
            "refusal_precision": precision,
            "refusal_recall": recall,
            "refusal_f1": f1,
            # The two errors are not symmetric: an over-refusal wastes an
            # answerable question, an under-refusal emits an ungrounded answer.
            "over_refusal_rate": fp / (fp + tn) if (fp + tn) else None,
            "over_refusal_rate_ci95": [over_lo, over_hi],
            "under_refusal_rate": fn / (fn + tp) if (fn + tp) else None,
            "n_answerable": answerable,
        }

    def _arm_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        labeled = [r for r in records if r.get("should_refuse") is not None]
        arm = {
            "n": len(records),
            "refusal_rate": (
                sum(1 for r in records if r["refused"]) / len(records) if records else None
            ),
            "mean_faithfulness_conventional": mean(
                [r["faithfulness_conventional"] for r in records]
            ),
            "mean_claim_recall_vs_gold": mean([r["claim_recall_vs_gold"] for r in records]),
            "mean_claims_per_answer": mean([r["n_claims"] for r in records]),
        }
        if labeled:
            arm.update(self._calibration(labeled))
        return arm


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else a - b


def _count_by(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        key = str(record.get(field))
        counts[key] = counts.get(key, 0) + 1
    return counts


EXPERIMENT = RefusalCalibrationExperiment()
