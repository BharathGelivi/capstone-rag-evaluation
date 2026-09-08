import unittest
import os
import tempfile
from src.report import (
    FrameworkMetadata,
    ExecutiveSummary,
    PipelineStageResult,
    PipelineOverview,
    EvaluationMetrics,
    EvidenceAnalysis,
    RootCauseSection,
    CorrectiveActionSection,
    OverallAssessment,
    AnswerCorrectnessSection,
    DiagnosticEvaluationReport
)

class TestReport(unittest.TestCase):
    def setUp(self):
        self.metadata = FrameworkMetadata(trace_id="TRACE_TEST_123")
        
        self.executive_summary = ExecutiveSummary(
            question="What is the punishment for murder?",
            generated_answer="The punishment is life imprisonment or death.",
            overall_health_score=0.85,
            primary_issue="None",
            supported_claims=2,
            total_claims=2,
            number_of_corrective_actions=0,
            summary="All claims are perfectly grounded in the provided context."
        )
        
        self.pipeline_overview = PipelineOverview(
            pipeline_stages=[
                PipelineStageResult(stage="CORPUS", status="PASS", observation="Docs found", confidence="HIGH")
            ]
        )
        
        self.evaluation_metrics = EvaluationMetrics(
            retrieved_chunks=5,
            verified_claims=2,
            supported_claims=2,
            partially_supported_claims=0,
            unsupported_claims=0,
            contradicted_claims=0,
            grounding_score=1.0,
            evidence_coverage=1.0,
            average_entailment=0.98,
            retrieval_latency_ms=150.5,
            generation_latency_ms=800.0,
            verification_latency_ms=450.2
        )
        
        self.evidence_analysis = [
            EvidenceAnalysis(
                claim_id="claim-1",
                claim_text="Punishment is life imprisonment.",
                verification_status="SUPPORTED",
                supporting_chunk_id="chunk-1",
                supporting_chunk_rank=1,
                supporting_evidence="Section 302: Whoever commits murder shall be punished with death, or imprisonment for life."
            )
        ]
        
        self.root_cause = RootCauseSection(
            primary_cause="UNKNOWN",
            secondary_effects=[],
            reasoning_chain=["No failures detected."]
        )
        
        self.corrective_actions = [
            CorrectiveActionSection(
                priority="low",
                title="None",
                description="Pipeline healthy.",
                observed_evidence="100% grounding.",
                expected_improvement="N/A",
                success_metric="N/A",
                tradeoff="N/A"
            )
        ]
        
        self.overall_assessment = OverallAssessment(
            major_strength="Perfect grounding score.",
            major_weakness="None identified.",
            next_priority="Continue monitoring.",
            overall_recommendation="No action needed."
        )
        
        self.report = DiagnosticEvaluationReport(
            framework_metadata=self.metadata,
            executive_summary=self.executive_summary,
            pipeline_overview=self.pipeline_overview,
            evaluation_metrics=self.evaluation_metrics,
            root_cause_analysis=self.root_cause,
            overall_assessment=self.overall_assessment,
            evidence_analysis=self.evidence_analysis,
            corrective_actions=self.corrective_actions
        )

    def test_pipeline_stage_result_metadata_defaults_and_roundtrips(self):
        stage = PipelineStageResult(stage="RETRIEVER", status="PASS", observation="OK", confidence="0.9")
        self.assertEqual(stage.metadata, {})

        stage_with_meta = PipelineStageResult(
            stage="RETRIEVER", status="PASS", observation="OK", confidence="0.9",
            metadata={"chunk_utilization_rate": 0.25, "chunks_used": 1, "chunks_retrieved": 4}
        )
        report = DiagnosticEvaluationReport(
            framework_metadata=self.metadata,
            executive_summary=self.executive_summary,
            pipeline_overview=PipelineOverview(pipeline_stages=[stage_with_meta]),
            evaluation_metrics=self.evaluation_metrics,
            root_cause_analysis=self.root_cause,
            overall_assessment=self.overall_assessment,
        )
        reloaded = DiagnosticEvaluationReport.from_json(report.to_json())
        self.assertEqual(reloaded.pipeline_overview.pipeline_stages[0].metadata["chunk_utilization_rate"], 0.25)

    def test_answer_correctness_defaults_to_none_and_roundtrips(self):
        self.assertIsNone(self.report.answer_correctness)

        self.report.answer_correctness = AnswerCorrectnessSection(claim_recall=0.75, total_gold_claims=4, recalled_gold_claims=3)
        reloaded = DiagnosticEvaluationReport.from_json(self.report.to_json())
        self.assertEqual(reloaded.answer_correctness.claim_recall, 0.75)
        self.assertEqual(reloaded.answer_correctness.recalled_gold_claims, 3)

    def test_serialization(self):
        json_str = self.report.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("TRACE_TEST_123", json_str)
        
        reloaded = DiagnosticEvaluationReport.from_json(json_str)
        self.assertEqual(reloaded.framework_metadata.trace_id, self.report.framework_metadata.trace_id)
        self.assertEqual(reloaded.artifact_version, self.report.artifact_version)
        self.assertEqual(reloaded.framework_metadata.framework_version, self.report.framework_metadata.framework_version)
        
        # Test nested dataclass equality
        self.assertEqual(reloaded.executive_summary.question, self.report.executive_summary.question)
        self.assertEqual(len(reloaded.pipeline_overview.pipeline_stages), 1)
        self.assertEqual(reloaded.pipeline_overview.pipeline_stages[0].stage, "CORPUS")
        self.assertEqual(reloaded.evaluation_metrics.grounding_score, 1.0)
        self.assertEqual(len(reloaded.evidence_analysis), 1)
        self.assertEqual(reloaded.evidence_analysis[0].claim_text, "Punishment is life imprisonment.")
        self.assertEqual(reloaded.root_cause_analysis.primary_cause, "UNKNOWN")
        self.assertEqual(reloaded.overall_assessment.major_strength, "Perfect grounding score.")
        self.assertEqual(len(reloaded.corrective_actions), 1)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = self.report.save(tmpdir)
            self.assertTrue(os.path.exists(filepath))
            
            reloaded = DiagnosticEvaluationReport.load(filepath)
            self.assertEqual(reloaded.framework_metadata.trace_id, "TRACE_TEST_123")

if __name__ == '__main__':
    unittest.main()
