"""
Answer Correctness Evaluator (Claim Recall).

Measures whether a generated answer captures the facts asserted by a gold
answer: the recall-side mirror of the existing faithfulness/precision check
(which asks "is every generated claim supported by evidence?"). This asks
the opposite direction: "is every gold claim reflected in the generated
answer?"

Reuses the existing ClaimDecomposer (to decompose the GOLD answer into
atomic claims, exactly as it decomposes a generated answer) and
ClaimVerifier's NLI sentence-scoring core (to check each gold claim against
the generated answer's own sentences as the evidence pool) -- no new model
or dependency is introduced.

Wired into PipelineRunner.run() (src/runner.py) as an optional step: pass
gold_answer= and it's computed automatically, persisted to
artifacts/answer_correctness/TRACE_<id>.json, and surfaced in the report as
DiagnosticEvaluationReport.answer_correctness (src/report.py). It is purely
additive -- it never affects overall_health_score, primary_issue, or any
existing pipeline stage. Can still be run standalone via
scripts/evaluate_answer_correctness.py when no gold answer was available at
pipeline-run time.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional

from src.rag_trace import RAGTrace
from src.claim_decomposer import ClaimDecomposer, CandidateClaimSet
from src.claim_verifier import ClaimVerifier, VerificationStatus

RECALLED_STATUSES = (VerificationStatus.SUPPORTED.value, VerificationStatus.PARTIALLY_SUPPORTED.value)


@dataclass
class GoldClaimResult:
    claim_id: str
    claim_text: str
    verification_status: str
    best_matching_sentence: Optional[str]
    confidence: float


@dataclass
class AnswerCorrectnessSummary:
    """Structurally parallel to VerificationSummary (see claim_verifier.py),
    but measuring recall of gold claims in the generated answer rather than
    support of generated claims in the retrieved context."""
    trace_id: str
    gold_answer: str
    artifact_version: str = "1.0"
    total_gold_claims: int = 0
    claim_recall: float = 0.0
    results: List[GoldClaimResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> "AnswerCorrectnessSummary":
        data = json.loads(data_str)
        data["results"] = [GoldClaimResult(**r) for r in data.get("results", [])]
        return cls(**data)

    def save(self, base_dir: str = "artifacts/answer_correctness") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "AnswerCorrectnessSummary":
        with open(filepath, encoding="utf-8") as f:
            return cls.from_json(f.read())


class AnswerCorrectnessEvaluator:
    def __init__(self, decomposer: Optional[ClaimDecomposer] = None, verifier: Optional[ClaimVerifier] = None):
        self.decomposer = decomposer or ClaimDecomposer()
        self.verifier = verifier or ClaimVerifier()

    def _decompose_gold_answer(self, gold_answer: str, trace_id: str) -> CandidateClaimSet:
        """
        Reuses ClaimDecomposer.decompose() as-is (same prompt, same
        JSON-recovery routine, same CLAIM_DECOMPOSER_PROMPT_VERSION) by
        feeding it a synthetic RAGTrace whose generated_answer is the gold
        answer. decompose() only ever reads trace.generated_answer and
        trace.trace_id, so this requires no fork of its logic.
        """
        synthetic_trace = RAGTrace(
            trace_id=trace_id,
            trace_version="1.0",
            pipeline_version="1.0",
            framework_version="1.0",
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            question="",
            generated_answer=gold_answer,
            prompt_snapshot="",
            prompt_length=0,
            retrieved_chunk_references=[],
            configuration_snapshot={},
            execution_statistics={},
            pipeline_stage_status={},
        )
        return self.decomposer.decompose(synthetic_trace)

    def evaluate(self, generated_answer: str, gold_answer: str, trace_id: str) -> AnswerCorrectnessSummary:
        gold_claim_set = self._decompose_gold_answer(gold_answer, trace_id)

        # Single evidence source: the generated answer's own sentences.
        sources = [("generated_answer", 0, generated_answer)]

        results: List[GoldClaimResult] = []
        for claim in gold_claim_set.candidate_claims:
            outcome = self.verifier._verify_against_sources(claim.claim_text, sources)
            best = outcome["best_sentence"]
            results.append(GoldClaimResult(
                claim_id=claim.candidate_id,
                claim_text=claim.claim_text,
                verification_status=outcome["status"].value,
                best_matching_sentence=best.text if best else None,
                confidence=outcome["confidence"],
            ))

        total = len(results)
        recalled = sum(1 for r in results if r.verification_status in RECALLED_STATUSES)
        claim_recall = (recalled / total) if total else 0.0

        return AnswerCorrectnessSummary(
            trace_id=trace_id,
            gold_answer=gold_answer,
            total_gold_claims=total,
            claim_recall=claim_recall,
            results=results,
        )
