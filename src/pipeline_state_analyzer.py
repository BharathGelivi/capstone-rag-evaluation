import os
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.rag_trace import RAGTrace
from src.claims import ClaimSet
from src.claim_verifier import VerificationSummary, VerificationStatus
from configs.thresholds import CORPUS_MAX_RELEVANT_DISTANCE, RETRIEVAL_SCORE_THRESHOLD

class PipelineStage(str, Enum):
    CORPUS = "CORPUS"
    RETRIEVER = "RETRIEVER"
    CHUNKING = "CHUNKING"
    GENERATOR = "GENERATOR"
    GROUNDING = "GROUNDING"

class PipelineStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"

@dataclass
class PipelineState:
    stage: PipelineStage
    status: PipelineStatus
    observation: str
    confidence: float
    supporting_claim_ids: List[str] = field(default_factory=list)
    supporting_chunk_ids: List[str] = field(default_factory=list)
    supporting_verification_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineStateMatrix:
    trace_id: str
    artifact_version: str = "1.0"
    pipeline_states: List[PipelineState] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def get(self, stage: PipelineStage) -> Optional[PipelineState]:
        for state in self.pipeline_states:
            if state.stage == stage:
                return state
        return None

    def summary(self) -> Dict[str, str]:
        return {state.stage.value: state.status.value for state in self.pipeline_states}

    def to_json(self) -> str:
        data = asdict(self)
        for state in data["pipeline_states"]:
            state["stage"] = state["stage"].value
            state["status"] = state["status"].value
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'PipelineStateMatrix':
        data = json.loads(data_str)
        states = []
        for state_data in data.get("pipeline_states", []):
            state_data["stage"] = PipelineStage(state_data["stage"])
            state_data["status"] = PipelineStatus(state_data["status"])
            states.append(PipelineState(**state_data))
        data["pipeline_states"] = states
        return cls(**data)

    def save(self, base_dir: str = "artifacts/pipeline_state_matrix") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'PipelineStateMatrix':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

class PipelineStateAnalyzer:
    def __init__(
        self,
        retrieval_score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
        generator_fail_min_unsupported_fraction: Optional[float] = None,
    ):
        """
        generator_fail_min_unsupported_fraction: when None (default), GENERATOR
        fails on *any* unsupported claim -- the original, unchanged behavior,
        so every existing caller (E1, E5, production) is bit-for-bit
        unaffected. When set, GENERATOR instead requires more than this
        fraction of claims to be unsupported. Exists so E10
        (experiments/exp10_generator_rule_sensitivity.py) can sweep the rule
        against both E1's synthetic ground truth and E8's live ground truth
        without a second, duplicated analyzer implementation.
        """
        self.retrieval_score_threshold = retrieval_score_threshold
        self.generator_fail_min_unsupported_fraction = generator_fail_min_unsupported_fraction

    def analyze(self, trace: RAGTrace, claim_set: ClaimSet, verification: VerificationSummary) -> PipelineStateMatrix:
        states = []
        
        supported_verifications = [
            v for v in verification.results 
            if v.verification_status in (VerificationStatus.SUPPORTED, VerificationStatus.PARTIALLY_SUPPORTED)
        ]
        unsupported_verifications = [
            v for v in verification.results 
            if v.verification_status == VerificationStatus.UNSUPPORTED
        ]
        
        has_supported_claim = len(supported_verifications) > 0
        all_unsupported = len(unsupported_verifications) == len(verification.results) and len(verification.results) > 0
        some_unsupported = len(unsupported_verifications) > 0
        
        max_score = 0.0
        if trace.retrieved_chunk_references:
            max_score = max((chunk.get("similarity_score", 0.0) for chunk in trace.retrieved_chunk_references), default=0.0)
            
        supporting_claim_ids = [v.claim_id for v in supported_verifications]
        supporting_chunk_ids = [v.best_chunk_id for v in supported_verifications if v.best_chunk_id]
        supporting_verification_ids = [v.verification_id for v in supported_verifications]
        
        # 1. Corpus
        min_distance = trace.execution_statistics.get("pre_rerank_min_dense_distance")
        pool_size = trace.execution_statistics.get("pre_rerank_candidate_pool_size")

        if min_distance is None:
            corpus_status = PipelineStatus.UNKNOWN
            corpus_obs = "Pre-rerank candidate scores are unavailable for this trace."
            corpus_conf = 1.0
        elif min_distance > CORPUS_MAX_RELEVANT_DISTANCE:
            corpus_status = PipelineStatus.FAIL
            corpus_obs = (
                f"Even the closest of {pool_size} pre-rerank candidates (distance={min_distance:.3f}) "
                f"falls outside the relevance threshold, suggesting the corpus lacks this content."
            )
            corpus_conf = 0.85
        elif has_supported_claim:
            corpus_status = PipelineStatus.PASS
            corpus_obs = "Corpus contains sufficient relevant content, evidenced by at least one supported claim."
            corpus_conf = 0.9
        else:
            corpus_status = PipelineStatus.UNKNOWN
            corpus_obs = "Corpus relevance is inconclusive."
            corpus_conf = 0.5

        states.append(PipelineState(
            stage=PipelineStage.CORPUS,
            status=corpus_status,
            observation=corpus_obs,
            confidence=corpus_conf,
            supporting_claim_ids=[],
            supporting_chunk_ids=[],
            supporting_verification_ids=[],
            metadata={"pre_rerank_min_dense_distance": min_distance, "pool_size": pool_size}
        ))
        
        # 2. Retriever
        if has_supported_claim:
            retriever_status = PipelineStatus.PASS
            retriever_obs = "At least one retrieved chunk provides supporting evidence for a verified claim."
            retriever_conf = 0.95
        elif all_unsupported and max_score < self.retrieval_score_threshold:
            retriever_status = PipelineStatus.FAIL
            retriever_obs = "No retrieved chunks provide support, and retrieval scores strongly indicate a lack of useful evidence."
            retriever_conf = 0.85
        else:
            retriever_status = PipelineStatus.UNKNOWN
            retriever_obs = "Retrieval evidence does not definitively confirm or refute pipeline success."
            retriever_conf = 0.5

        # Chunk utilization: how much of the retrieved evidence was actually
        # used (informational only -- does not affect retriever_status/_conf
        # above, which are already finalized by this point).
        retrieved_chunk_ids = {ref["chunk_id"] for ref in trace.retrieved_chunk_references}
        used_chunk_ids = set(supporting_chunk_ids)
        chunk_utilization_rate = (len(used_chunk_ids) / len(retrieved_chunk_ids)) if retrieved_chunk_ids else None

        states.append(PipelineState(
            stage=PipelineStage.RETRIEVER,
            status=retriever_status,
            observation=retriever_obs,
            confidence=retriever_conf,
            supporting_claim_ids=supporting_claim_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            supporting_verification_ids=supporting_verification_ids,
            metadata={
                "max_score": max_score,
                "threshold": self.retrieval_score_threshold,
                "chunk_utilization_rate": chunk_utilization_rate,
                "chunks_used": len(used_chunk_ids),
                "chunks_retrieved": len(retrieved_chunk_ids),
            }
        ))
        
        # 3. Chunking: detect claims whose best evidence chunk sits directly
        # adjacent (by chunk_index, same parent document) to another retrieved
        # chunk -- a signal that a fact was likely split across a chunk boundary.
        partially_supported = [
            v for v in verification.results
            if v.verification_status == VerificationStatus.PARTIALLY_SUPPORTED and v.best_chunk_id
        ]
        chunk_ref_by_id = {ref["chunk_id"]: ref for ref in trace.retrieved_chunk_references}
        boundary_claim_ids = []
        for v in partially_supported:
            ref = chunk_ref_by_id.get(v.best_chunk_id)
            if not ref or ref.get("chunk_index") is None or not ref.get("parent_document_id"):
                continue
            for other in trace.retrieved_chunk_references:
                if (other["chunk_id"] != v.best_chunk_id
                        and other.get("parent_document_id") == ref["parent_document_id"]
                        and other.get("chunk_index") is not None
                        and abs(other["chunk_index"] - ref["chunk_index"]) == 1):
                    boundary_claim_ids.append(v.claim_id)
                    break

        if boundary_claim_ids:
            chunking_status = PipelineStatus.FAIL
            chunking_obs = (
                f"{len(boundary_claim_ids)} partially supported claim(s) have best evidence adjacent "
                f"to another retrieved chunk from the same document."
            )
            chunking_conf = 0.8
            
            states.append(PipelineState(
                stage=PipelineStage.CHUNKING,
                status=chunking_status,
                observation=chunking_obs,
                confidence=chunking_conf,
                supporting_claim_ids=boundary_claim_ids,
                supporting_chunk_ids=[],
                supporting_verification_ids=[],
                metadata={"boundary_claim_ids": boundary_claim_ids}
            ))
        
        # 4. Generator
        unsupported_fraction = (
            len(unsupported_verifications) / len(verification.results)
            if verification.results else 0.0
        )
        if self.generator_fail_min_unsupported_fraction is None:
            generator_fail_signal = some_unsupported
        else:
            generator_fail_signal = unsupported_fraction > self.generator_fail_min_unsupported_fraction

        if not verification.results or all(v.verification_status == VerificationStatus.NOT_VERIFIABLE for v in verification.results):
            gen_status = PipelineStatus.UNKNOWN
            gen_obs = "No assessable claims were available; generation quality is unknown."
            gen_conf = 0.5
        elif generator_fail_signal and max_score >= self.retrieval_score_threshold:
            gen_status = PipelineStatus.FAIL
            gen_obs = f"{len(unsupported_verifications)} generated claims remain unsupported despite high retrieval scores."
            gen_conf = 0.85
        elif some_unsupported:
            gen_status = PipelineStatus.UNKNOWN
            gen_obs = f"{len(unsupported_verifications)} generated claims are unsupported, but retrieval quality is ambiguous."
            gen_conf = 0.5
        else:
            gen_status = PipelineStatus.PASS
            gen_obs = "No generated claims were classified as unsupported; see grounding for contradictions."
            gen_conf = 0.95
            
        states.append(PipelineState(
            stage=PipelineStage.GENERATOR,
            status=gen_status,
            observation=gen_obs,
            confidence=gen_conf,
            supporting_claim_ids=supporting_claim_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            supporting_verification_ids=supporting_verification_ids,
            metadata={
                "unsupported_count": len(unsupported_verifications),
                "max_score": max_score,
                "unsupported_fraction": unsupported_fraction,
                "generator_fail_min_unsupported_fraction": self.generator_fail_min_unsupported_fraction,
            }
        ))
        
        # 5. Grounding: GROUNDING_FAILURE means active misinformation (claims
        # directly contradicted by evidence), which is a qualitatively different
        # failure from GENERATOR's unsupported-but-not-contradicted claims.
        contradicted_verifications = [
            v for v in verification.results
            if v.verification_status == VerificationStatus.CONTRADICTED
        ]
        if contradicted_verifications:
            grounding_status = PipelineStatus.FAIL
            grounding_obs = f"{len(contradicted_verifications)} claim(s) are actively contradicted by retrieved evidence."
            grounding_conf = 0.95
        elif len(verification.results) == 0:
            grounding_status = PipelineStatus.UNKNOWN
            grounding_obs = "No claims were available to assess grounding."
            grounding_conf = 0.5
        else:
            grounding_status = PipelineStatus.PASS
            grounding_obs = "No claims are contradicted by retrieved evidence."
            grounding_conf = 0.9

        states.append(PipelineState(
            stage=PipelineStage.GROUNDING,
            status=grounding_status,
            observation=grounding_obs,
            confidence=grounding_conf,
            supporting_claim_ids=supporting_claim_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            supporting_verification_ids=supporting_verification_ids,
            metadata={"contradicted_count": len(contradicted_verifications)}
        ))

        return PipelineStateMatrix(
            trace_id=trace.trace_id,
            pipeline_states=states
        )
