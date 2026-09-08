import os
import logging
from typing import Optional

from src.rag_trace import RAGTrace
from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineStatus
from src.root_cause_reasoner import RootCauseAnalysis, FailureType
from src.corrective_action_engine import CorrectiveActionPlan
from src.claim_verifier import VerificationSummary, VerificationStatus
from src.ragas_metrics import RagasMetrics
from src.answer_correctness_evaluator import AnswerCorrectnessSummary
from src.report import (
    DiagnosticEvaluationReport,
    FrameworkMetadata,
    ExecutiveSummary,
    PipelineStageResult,
    PipelineOverview,
    EvaluationMetrics,
    EvidenceAnalysis,
    RootCauseSection,
    CorrectiveActionSection,
    OverallAssessment,
    RagasMetricsSection,
    AnswerCorrectnessSection
)

logger = logging.getLogger(__name__)

class ReportBuilder:
    """
    Assembles a DiagnosticEvaluationReport from existing pipeline artifacts.
    Performs NO reasoning or calculations, only data mapping.
    """
    def build(
        self,
        trace: Optional[RAGTrace] = None,
        psm: Optional[PipelineStateMatrix] = None,
        rca: Optional[RootCauseAnalysis] = None,
        cap: Optional[CorrectiveActionPlan] = None,
        verification: Optional[VerificationSummary] = None,
        ragas_metrics: Optional[RagasMetrics] = None,
        answer_correctness: Optional[AnswerCorrectnessSummary] = None
    ) -> DiagnosticEvaluationReport:
        
        # 1. Trace ID validation
        trace_ids = set()
        for artifact in [trace, psm, rca, cap, verification]:
            if artifact is not None and hasattr(artifact, "trace_id"):
                trace_ids.add(artifact.trace_id)
                
        if len(trace_ids) > 1:
            raise ValueError(f"Trace ID mismatch detected across artifacts: {trace_ids}")
            
        trace_id = trace_ids.pop() if trace_ids else "UNKNOWN"
        
        # 2. Status determination
        is_partial = any(a is None for a in [trace, psm, rca, cap, verification])
        analysis_status = "PARTIAL_ANALYSIS" if is_partial else "COMPLETED"
        if trace_id == "UNKNOWN":
            analysis_status = "FAILED"
            
        metadata = {"missing_artifacts": [name for name, val in {
            "RAGTrace": trace,
            "PipelineStateMatrix": psm,
            "RootCauseAnalysis": rca,
            "CorrectiveActionPlan": cap,
            "VerificationSummary": verification
        }.items() if val is None]}

        # 3. Framework Metadata
        framework_metadata = FrameworkMetadata(
            trace_id=trace_id,
            metadata={"analysis_status_reason": "Some artifacts were unavailable."} if is_partial else {}
        )

        # 4. Pipeline Overview
        pipeline_stages = []
        if psm:
            for state in psm.pipeline_states:
                pipeline_stages.append(PipelineStageResult(
                    stage=state.stage.value,
                    status=state.status.value,
                    observation=state.observation,
                    confidence=str(state.confidence),
                    metadata=state.metadata
                ))
        pipeline_overview = PipelineOverview(pipeline_stages=pipeline_stages)

        # 5. Evidence Analysis & Evaluation Metrics
        evidence_list = []
        metrics = EvaluationMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        if verification and trace:
            for res in verification.results:
                evidence_list.append(EvidenceAnalysis(
                    claim_id=res.claim_id,
                    claim_text=res.claim_text,
                    verification_status=res.verification_status.value,
                    supporting_chunk_id=res.best_chunk_id,
                    supporting_chunk_rank=res.best_chunk_rank,
                    supporting_evidence=res.evidence_text or ""
                ))
                
            metrics.retrieved_chunks = len(trace.retrieved_chunk_references)
            metrics.verified_claims = len(verification.results)
            metrics.supported_claims = sum(1 for v in verification.results if v.verification_status == VerificationStatus.SUPPORTED)
            metrics.partially_supported_claims = sum(1 for v in verification.results if v.verification_status == VerificationStatus.PARTIALLY_SUPPORTED)
            metrics.unsupported_claims = sum(1 for v in verification.results if v.verification_status == VerificationStatus.UNSUPPORTED)
            metrics.contradicted_claims = sum(1 for v in verification.results if v.verification_status == VerificationStatus.CONTRADICTED)
            
            if metrics.verified_claims > 0:
                metrics.grounding_score = (metrics.supported_claims + 0.5 * metrics.partially_supported_claims) / metrics.verified_claims
                metrics.average_entailment = sum(v.entailment_score for v in verification.results) / metrics.verified_claims
                metrics.evidence_coverage = len(set(v.best_chunk_id for v in verification.results if v.best_chunk_id)) / max(1, metrics.retrieved_chunks)
            
            metrics.retrieval_latency_ms = trace.execution_statistics.get("retrieval_time", 0.0) * 1000
            metrics.generation_latency_ms = trace.execution_statistics.get("generation_time", 0.0) * 1000
            metrics.verification_latency_ms = getattr(verification, "total_verification_latency_ms", 0.0)

        # 6. Root Cause Analysis
        root_cause_section = RootCauseSection(primary_cause="UNKNOWN")
        if rca:
            root_cause_section.primary_cause = rca.primary_cause.value
            root_cause_section.secondary_effects = [e.value for e in rca.secondary_effects]
            root_cause_section.reasoning_chain = rca.reasoning_chain
            root_cause_section.diagnosis_confidence = rca.confidence

        # 7. Corrective Actions
        corrective_actions = []
        if cap:
            all_actions = cap.immediate_actions + cap.short_term_actions + cap.experimental_actions + cap.informational_actions
            for action in all_actions:
                corrective_actions.append(CorrectiveActionSection(
                    priority=action.priority,
                    title=action.title,
                    description=action.description,
                    observed_evidence=action.observed_evidence,
                    expected_improvement=action.expected_improvement,
                    success_metric=action.success_metric,
                    tradeoff=action.tradeoff
                ))

        # 8. Composite overall health score
        # Weighted composite avoids over-indexing on a single metric:
        #   grounding (faithfulness) carries the most weight as it directly
        #   measures factual grounding; the remaining metrics fill in retrieval
        #   and relevancy quality. Weights normalise over available metrics.
        WEIGHTS = [
            ("grounding",          0.50, metrics.grounding_score if verification else None),
            ("context_precision",  0.20, ragas_metrics.context_precision if ragas_metrics else None),
            ("answer_relevancy",   0.20, ragas_metrics.answer_relevancy if ragas_metrics else None),
            ("context_recall",     0.10, ragas_metrics.context_recall if ragas_metrics else None),
        ]
        weighted_sum = sum(w * v for _, w, v in WEIGHTS if v is not None)
        total_weight = sum(w for _, w, v in WEIGHTS if v is not None)
        overall_health = round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

        primary_issue = rca.primary_cause.value if rca else "UNKNOWN"
        summary_text = "Analysis completed." if not is_partial else "Analysis completed with missing artifacts."

        executive_summary = ExecutiveSummary(
            question=trace.question if trace else "UNKNOWN",
            generated_answer=trace.generated_answer if trace else "UNKNOWN",
            overall_health_score=overall_health,
            primary_issue=primary_issue,
            supported_claims=metrics.supported_claims,
            total_claims=metrics.verified_claims,
            number_of_corrective_actions=len(corrective_actions),
            summary=summary_text
        )

        # 9. Overall Assessment
        # major_strength is derived from the highest-confidence PASS stage in the
        # PSM rather than being hardcoded. Falls back to "N/A" only when no stage
        # passed (i.e. all stages failed or are UNKNOWN).
        major_weakness = primary_issue
        next_priority = corrective_actions[0].title if corrective_actions else "Monitor pipeline"

        if psm:
            passing_states = [
                s for s in psm.pipeline_states if s.status == PipelineStatus.PASS
            ]
            if passing_states:
                best_pass = max(passing_states, key=lambda s: s.confidence)
                major_strength = (
                    f"{best_pass.stage.value} is healthy "
                    f"(confidence={best_pass.confidence:.0%}): {best_pass.observation}"
                )
            else:
                major_strength = "N/A"
        else:
            major_strength = "N/A"

        if overall_health > 0.8:
            overall_rec = "Pipeline is healthy. Monitor for distribution shift."
        elif overall_health > 0.5:
            overall_rec = "Pipeline requires targeted improvements to the primary failure stage."
        else:
            overall_rec = "Pipeline requires architectural adjustments across multiple stages."

        overall_assessment = OverallAssessment(
            major_strength=major_strength,
            major_weakness=major_weakness,
            next_priority=next_priority,
            overall_recommendation=overall_rec
        )

        # 10. RAGAS-style Metrics
        ragas_section = None
        if ragas_metrics is not None:
            ragas_section = RagasMetricsSection(
                faithfulness=ragas_metrics.faithfulness,
                answer_relevancy=ragas_metrics.answer_relevancy,
                context_precision=ragas_metrics.context_precision,
                context_relevancy=ragas_metrics.context_relevancy,
                context_recall=ragas_metrics.context_recall,
                answer_similarity=ragas_metrics.answer_similarity,
                answer_correctness=ragas_metrics.answer_correctness
            )

        # 11. Answer Correctness (Claim Recall) -- purely additive, does not
        # affect overall_health_score/primary_issue computed above.
        answer_correctness_section = None
        if answer_correctness is not None:
            recalled = sum(
                1 for r in answer_correctness.results
                if r.verification_status in ("SUPPORTED", "PARTIALLY_SUPPORTED")
            )
            answer_correctness_section = AnswerCorrectnessSection(
                claim_recall=answer_correctness.claim_recall,
                total_gold_claims=answer_correctness.total_gold_claims,
                recalled_gold_claims=recalled
            )

        return DiagnosticEvaluationReport(
            framework_metadata=framework_metadata,
            executive_summary=executive_summary,
            pipeline_overview=pipeline_overview,
            evaluation_metrics=metrics,
            root_cause_analysis=root_cause_section,
            overall_assessment=overall_assessment,
            evidence_analysis=evidence_list,
            corrective_actions=corrective_actions,
            ragas_metrics=ragas_section,
            answer_correctness=answer_correctness_section,
            analysis_status=analysis_status,
            metadata=metadata
        )
