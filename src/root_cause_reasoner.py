"""
Root Cause Reasoner Module.

Performs causal root cause analysis over a PipelineStateMatrix.

Design:
    Primary cause selection uses causal propagation, not confidence ranking.
    Pipeline stages have a strict causal order:
        CORPUS → RETRIEVER → CHUNKING → GENERATOR → GROUNDING
    An upstream failure (e.g. CORPUS) causes downstream failures (e.g. RETRIEVAL_MISS).
    Selecting the downstream failure with the highest confidence as "primary" would
    mis-attribute the symptom as the cause. Instead, we select the earliest failing
    stage in causal order — the upstream root — and classify everything downstream
    as a secondary effect.
"""

import os
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineStage, PipelineStatus


class FailureType(str, Enum):
    MISSING_CORPUS             = "MISSING_CORPUS"
    RETRIEVAL_MISS             = "RETRIEVAL_MISS"
    CHUNK_BOUNDARY             = "CHUNK_BOUNDARY"
    UNSUPPORTED_GENERATION     = "UNSUPPORTED_GENERATION"
    CONTRADICTORY_GENERATION   = "CONTRADICTORY_GENERATION"
    GROUNDING_FAILURE          = "GROUNDING_FAILURE"
    UNKNOWN                    = "UNKNOWN"


@dataclass
class RootCauseAnalysis:
    trace_id: str
    artifact_version: str = "1.0"
    primary_cause: FailureType = FailureType.UNKNOWN
    secondary_effects: List[FailureType] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    recommendations_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = asdict(self)
        data["primary_cause"] = data["primary_cause"].value
        data["secondary_effects"] = [e.value for e in data["secondary_effects"]]
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> "RootCauseAnalysis":
        data = json.loads(data_str)
        data["primary_cause"] = FailureType(data["primary_cause"])
        data["secondary_effects"] = [FailureType(e) for e in data.get("secondary_effects", [])]
        return cls(**data)

    def save(self, base_dir: str = "artifacts/root_cause_analysis") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "RootCauseAnalysis":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())


class RootCauseReasoner:
    """Selects the earliest upstream failing stage as the primary root cause.

    Rationale: pipeline stages form a causal chain.
        CORPUS failure → causes → RETRIEVAL_MISS → causes → GROUNDING_FAILURE

    Selecting the highest-confidence failure as "primary" (as the previous
    implementation did) often picks the downstream symptom because downstream
    stages accumulate more evidence and therefore have higher confidence scores.
    The correct approach is to walk the chain and stop at the first (earliest)
    failing stage — that is the root, not the symptom.
    """

    # Strict causal traversal order: upstream stages come first.
    CAUSAL_ORDER = [
        PipelineStage.CORPUS,
        PipelineStage.RETRIEVER,
        PipelineStage.CHUNKING,
        PipelineStage.GENERATOR,
        PipelineStage.GROUNDING,
    ]

    STAGE_TO_FAILURE_MAP = {
        PipelineStage.CORPUS:     FailureType.MISSING_CORPUS,
        PipelineStage.RETRIEVER:  FailureType.RETRIEVAL_MISS,
        PipelineStage.CHUNKING:   FailureType.CHUNK_BOUNDARY,
        PipelineStage.GENERATOR:  FailureType.UNSUPPORTED_GENERATION,
        PipelineStage.GROUNDING:  FailureType.GROUNDING_FAILURE,
    }

    def analyze(self, psm: PipelineStateMatrix) -> RootCauseAnalysis:
        reasoning_chain: List[str] = []
        confidence_scores: List[float] = []
        fail_stages = []  # List[(PipelineStage, FailureType, PipelineState)] in causal order

        reasoning_chain.append(f"Starting root cause analysis for trace {psm.trace_id}.")

        for stage in self.CAUSAL_ORDER:
            state = psm.get(stage)
            if not state:
                reasoning_chain.append(
                    f"Skipping {stage.value}: no observable state in matrix."
                )
                continue

            if state.status == PipelineStatus.UNKNOWN:
                reasoning_chain.append(
                    f"Skipping {stage.value}: status is UNKNOWN (insufficient evidence)."
                )
                continue

            if state.status == PipelineStatus.FAIL:
                failure_type = self.STAGE_TO_FAILURE_MAP.get(stage, FailureType.UNKNOWN)
                fail_stages.append((stage, failure_type, state))
                confidence_scores.append(state.confidence)
                reasoning_chain.append(
                    f"Failure at {stage.value}: {failure_type.value} "
                    f"(confidence={state.confidence:.2f}). Observation: {state.observation}"
                )
            elif state.status == PipelineStatus.PASS:
                confidence_scores.append(state.confidence)
                reasoning_chain.append(
                    f"{stage.value} passed (confidence={state.confidence:.2f})."
                )

        if fail_stages:
            # Primary cause = first failure in causal order (upstream root).
            # All subsequent failures in the chain are secondary effects (propagated).
            primary_stage, primary_failure, primary_state = fail_stages[0]
            secondary_effects = [ft for (_, ft, _) in fail_stages[1:]]

            reasoning_chain.append(
                f"Primary root cause: {primary_failure.value} at {primary_stage.value} "
                f"(earliest upstream failure, confidence={primary_state.confidence:.2f})."
            )
            for stage, failure_type, state in fail_stages[1:]:
                reasoning_chain.append(
                    f"Secondary effect (propagated from {primary_stage.value}): "
                    f"{failure_type.value} at {stage.value} "
                    f"(confidence={state.confidence:.2f})."
                )
            reasoning_chain.append(
                f"Analysis complete. Primary cause: {primary_failure.value}."
            )
            needs_rec = True
        else:
            primary_failure = FailureType.UNKNOWN
            secondary_effects = []
            reasoning_chain.append(
                "No failures detected. Pipeline is healthy or all evidence is UNKNOWN."
            )
            needs_rec = False

        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores else 0.0
        )

        return RootCauseAnalysis(
            trace_id=psm.trace_id,
            primary_cause=primary_failure,
            secondary_effects=secondary_effects,
            reasoning_chain=reasoning_chain,
            confidence=round(avg_confidence, 2),
            recommendations_needed=needs_rec,
        )
