import unittest
from src.report_presenter import DiagnosticReportPresenter
from src.report import (
    DiagnosticEvaluationReport,
    FrameworkMetadata,
    ExecutiveSummary,
    PipelineOverview,
    PipelineStageResult,
    EvaluationMetrics,
    EvidenceAnalysis,
    RootCauseSection,
    CorrectiveActionSection,
    OverallAssessment,
    RagasMetricsSection
)

class TestDiagnosticReportPresenter(unittest.TestCase):
    def setUp(self):
        self.report = DiagnosticEvaluationReport(
            framework_metadata=FrameworkMetadata(trace_id="TRACE_123"),
            executive_summary=ExecutiveSummary(
                question="Q", generated_answer="A", overall_health_score=0.9,
                primary_issue="None", supported_claims=1, total_claims=1,
                number_of_corrective_actions=0, summary="Good"
            ),
            pipeline_overview=PipelineOverview(
                pipeline_stages=[
                    PipelineStageResult(stage="RETRIEVER", status="PASS", observation="OK", confidence="0.9")
                ]
            ),
            evaluation_metrics=EvaluationMetrics(
                retrieved_chunks=1, verified_claims=1, supported_claims=1,
                partially_supported_claims=0, unsupported_claims=0, contradicted_claims=0,
                grounding_score=1.0, evidence_coverage=1.0, average_entailment=0.9,
                retrieval_latency_ms=10.0, generation_latency_ms=10.0, verification_latency_ms=10.0
            ),
            root_cause_analysis=RootCauseSection(primary_cause="UNKNOWN"),
            overall_assessment=OverallAssessment("Good", "None", "None", "None"),
            evidence_analysis=[
                EvidenceAnalysis("C1", "Claim", "SUPPORTED", "Chunk1", 1, "Evidence")
            ],
            corrective_actions=[
                CorrectiveActionSection("low", "Monitor", "Desc", "Ev", "Imp", "Metric", "Tradeoff")
            ]
        )
        self.presenter = DiagnosticReportPresenter(self.report)

    def test_render_console(self):
        output = self.presenter.render_console()
        self.assertIn("X-RAG DIAGNOSTIC EVALUATION REPORT", output)
        self.assertIn("TRACE_123", output)
        self.assertIn("APPENDIX", output)

    def test_render_markdown(self):
        output = self.presenter.render_markdown()
        self.assertIn("# Diagnostic Evaluation Report", output)
        self.assertIn("## 9. Evidence Traceability", output)
        self.assertIn("APPENDIX", output)

    def test_render_markdown_ragas_not_computed(self):
        output = self.presenter.render_markdown()
        self.assertIn("## 5. RAGAS Metrics", output)
        self.assertIn("Not computed for this trace.", output)

    def test_render_console_and_markdown_with_ragas_metrics(self):
        self.report.ragas_metrics = RagasMetricsSection(
            faithfulness=0.8, answer_relevancy=0.75, context_precision=0.6,
            context_relevancy=0.7, context_recall=0.5, answer_similarity=0.9, answer_correctness=0.65
        )
        console_output = self.presenter.render_console()
        markdown_output = self.presenter.render_markdown()

        self.assertIn("0.8000", console_output)
        self.assertIn("Faithfulness", console_output)
        self.assertIn("0.8000", markdown_output)
        self.assertIn("Answer Correctness", markdown_output)

    def test_render_html(self):
        output = self.presenter.render_html()
        self.assertIn("<!DOCTYPE html>", output)
        self.assertIn("TRACE_123", output)

    def test_render_pdf(self):
        pdf_bytes = self.presenter.render_pdf()
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 0)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_retriever_efficiency_line_rendered_when_metadata_present(self):
        self.report.pipeline_overview.pipeline_stages = [
            PipelineStageResult(
                stage="RETRIEVER", status="PASS", observation="OK", confidence="0.9",
                metadata={"chunk_utilization_rate": 0.25, "chunks_used": 1, "chunks_retrieved": 4}
            )
        ]
        console_output = self.presenter.render_console()
        markdown_output = self.presenter.render_markdown()

        self.assertIn("Efficiency: 1/4 chunks used (25%)", console_output)
        self.assertIn("Efficiency: 1/4 chunks used (25%)", markdown_output)

    def test_no_efficiency_line_when_metadata_absent(self):
        # Default fixture stage has no metadata (older-report backward compatibility).
        console_output = self.presenter.render_console()
        self.assertNotIn("Efficiency:", console_output)

    def test_efficiency_line_only_for_retriever_stage(self):
        self.report.pipeline_overview.pipeline_stages = [
            PipelineStageResult(
                stage="GENERATOR", status="PASS", observation="OK", confidence="0.9",
                metadata={"chunk_utilization_rate": 0.25, "chunks_used": 1, "chunks_retrieved": 4}
            )
        ]
        output = self.presenter.render_console()
        self.assertNotIn("Efficiency:", output)

    def test_answer_correctness_rendered_when_present(self):
        from src.report import AnswerCorrectnessSection

        self.report.answer_correctness = AnswerCorrectnessSection(claim_recall=0.5, total_gold_claims=2, recalled_gold_claims=1)
        console_output = self.presenter.render_console()
        markdown_output = self.presenter.render_markdown()

        self.assertIn("ANSWER CORRECTNESS", console_output)
        self.assertIn("0.5000", console_output)
        self.assertIn("Answer Correctness (Claim Recall)", markdown_output)
        self.assertIn("1 / 2 recalled", markdown_output)

    def test_answer_correctness_not_computed_placeholder(self):
        console_output = self.presenter.render_console()
        self.assertIn("ANSWER CORRECTNESS", console_output)
        self.assertIn("Not computed for this trace.", console_output)

    def test_traceability(self):
        tr = self.presenter._get_traceability_data()
        self.assertEqual(tr["primary_cause"], "UNKNOWN")
        self.assertIn("C1", tr["supporting_claims"])
        self.assertIn("Chunk1", tr["supporting_chunks"])

if __name__ == '__main__':
    unittest.main()
