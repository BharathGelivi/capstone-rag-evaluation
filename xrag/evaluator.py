"""Evaluate external traces with the existing NLI and diagnostic components."""
import json
import os
import re
import threading
from typing import Literal

from .schema import TraceInput


class TraceEvaluator:
    """Reuse one lazily loaded NLI model without persisting artifacts."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self._verifier = None
        self._decomposer = None
        self._lock = threading.Lock()

    def evaluate(self, trace, *, claim_mode: Literal["sentences", "llm"] = "sentences") -> dict:
        if claim_mode not in ("sentences", "llm"):
            raise ValueError("claim_mode must be 'sentences' or 'llm'")
        if isinstance(trace, str):
            trace = json.loads(trace)
        if not isinstance(trace, TraceInput):
            if hasattr(trace, "to_json"):
                trace = json.loads(trace.to_json())
            trace = TraceInput.model_validate(trace)
        else:
            trace = trace.model_copy(deep=True)
        if trace.claims is None and claim_mode == "llm" and not os.environ.get("NVIDIA_API_KEY"):
            raise ValueError("claim_mode='llm' requires NVIDIA_API_KEY; supply claims or use sentences")
        with self._lock:
            return self._evaluate(trace, claim_mode)

    def _evaluate(self, trace, claim_mode):
        from src.rag_trace import RAGTrace
        from src.claim_decomposer import CandidateClaim, CandidateClaimSet, ClaimDecomposer
        from src.claim_verifier import ClaimVerifier, VerificationStatus
        from src.pipeline_state_analyzer import PipelineStateAnalyzer, PipelineStage, PipelineStatus
        from src.root_cause_reasoner import RootCauseReasoner
        from src.corrective_action_engine import CorrectiveActionEngine
        from src.report_builder import ReportBuilder

        internal = RAGTrace(
            trace_id=trace.trace_id, trace_version="1.0", pipeline_version="1.0",
            framework_version="1.0", timestamp=trace.timestamp, question=trace.question,
            generated_answer=trace.answer, prompt_snapshot=trace.prompt_snapshot,
            prompt_length=len(trace.prompt_snapshot),
            retrieved_chunk_references=[c.model_dump(exclude_none=True) for c in trace.retrieved_chunks],
            configuration_snapshot=dict(trace.configuration_snapshot),
            execution_statistics=dict(trace.execution_statistics),
            pipeline_stage_status=dict(trace.pipeline_stage_status), diagnostics={})
        claim_source = "supplied"
        if trace.claims is None and claim_mode == "llm":
            if self._decomposer is None:
                self._decomposer = ClaimDecomposer(debug=False)
            claims = self._decomposer.decompose(internal)
            if not claims.metadata.get("diagnostics", {}).get("success", True) or not claims.total_candidates:
                raise RuntimeError("Atomic claim decomposition failed or returned no claims")
            claim_source = "llm"
        else:
            texts = trace.claims
            if texts is None:
                texts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", trace.answer) if s.strip()]
                claim_source = "sentences"
            claims = CandidateClaimSet(trace_id=trace.trace_id)
            for i, text in enumerate(dict.fromkeys(texts)):
                offset = trace.answer.find(text)
                claims.add_claim(CandidateClaim(
                    candidate_id=f"{trace.trace_id}_C{i + 1:03d}", trace_id=trace.trace_id,
                    claim_text=text, sentence_id=f"S{i + 1}", claim_index=i,
                    character_start=offset, character_end=offset + len(text) if offset >= 0 else -1))
        if self._verifier is None:
            kwargs = {"enable_llm_judge": False}
            if self.model_name:
                kwargs["model_name"] = self.model_name
            self._verifier = ClaimVerifier(**kwargs)
        # Supplied evidence is authoritative; never resolve through a local registry.
        internal.prompt_snapshot = ""
        chunks = self._verifier.build_retrieved_chunks_from_trace(internal, None)
        verification = self._verifier.verify_all(claims, trace.trace_id, chunks)
        if not chunks:
            # Absence of evidence cannot establish an unsupported factual claim.
            for result in verification.results:
                result.verification_status = VerificationStatus.NOT_VERIFIABLE
                result.verification_reason = "No evidence supplied; the claim cannot be assessed."
            verification.unsupported_claims = 0
            verification.not_verifiable_claims = verification.total_claims
        internal.prompt_snapshot = trace.prompt_snapshot
        internal.diagnostics["decomposition_success"] = True
        psm = PipelineStateAnalyzer().analyze(internal, claims, verification)
        if not any(c.similarity_score is not None for c in trace.retrieved_chunks):
            for state in psm.pipeline_states:
                if state.stage == PipelineStage.RETRIEVER and state.status == PipelineStatus.FAIL:
                    state.status = PipelineStatus.UNKNOWN
                    state.observation = "Claims lack support, but retrieval scores were not supplied."
                    state.confidence = 0.5
        rca = RootCauseReasoner().analyze(psm)
        cap = CorrectiveActionEngine().generate(rca, psm=psm, config_snapshot=internal.configuration_snapshot)
        report = ReportBuilder().build(trace=internal, psm=psm, rca=rca, cap=cap, verification=verification)
        if not chunks or not trace.pipeline_stage_status or not trace.configuration_snapshot:
            report.analysis_status = "PARTIAL_ANALYSIS"
            report.executive_summary.summary = "Trace verification completed; pipeline metadata or evidence is incomplete."
        total = verification.total_claims
        return {
            "trace_id": trace.trace_id,
            "verification": json.loads(verification.to_json()),
            "diagnostic_report": json.loads(report.to_json()),
            "metrics": {"claim_support_rate": verification.supported_claims / total if total else None,
                        "contradiction_rate": verification.contradicted_claims / total if total else None},
            "metadata": {"claim_source": claim_source, "nli_model": self._verifier.model_name,
                         "llm_judge_enabled": False, "evidence_source": "trace",
                         "warnings": (["Sentence segmentation does not guarantee atomic claims."] if claim_source == "sentences" else [])
                         + (["No evidence supplied; claims cannot be verified."] if not chunks else [])
                         + (["Pipeline metadata is incomplete; stage diagnoses may be unavailable."]
                            if not trace.pipeline_stage_status or not trace.configuration_snapshot else [])}}


_default_evaluator = TraceEvaluator()


def evaluate_trace(trace, *, claim_mode="sentences") -> dict:
    """Evaluate a dictionary, JSON string, or canonical RAGTrace without file writes."""
    return _default_evaluator.evaluate(trace, claim_mode=claim_mode)
