import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.rag_trace import RAGTrace
from src.claims import ClaimSet, Claim
from src.claim_verifier import VerificationSummary, VerificationResult, VerificationStatus
from src.pipeline_state_analyzer import PipelineStateAnalyzer
from src.root_cause_reasoner import RootCauseReasoner
from src.corrective_action_engine import CorrectiveActionEngine
from src.report_builder import ReportBuilder
from src.report_presenter import DiagnosticReportPresenter
from src.runner import PipelineRunner


def make_trace(trace_id="TRACE_E2E", max_score=0.9):
    return RAGTrace(
        trace_id=trace_id,
        trace_version="1.0",
        pipeline_version="1.0",
        framework_version="1.0",
        timestamp="2026-07-17T12:00:00Z",
        question="What is the punishment for murder?",
        generated_answer="The punishment is death or life imprisonment.",
        prompt_snapshot="Prompt",
        prompt_length=100,
        retrieved_chunk_references=[{"chunk_id": "chunk1", "similarity_score": max_score}],
        configuration_snapshot={},
        execution_statistics={"retrieval_time": 0.2, "generation_time": 0.3},
        pipeline_stage_status={},
    )


def make_claim_set(trace_id, num_claims=1):
    claim_set = ClaimSet(trace_id=trace_id)
    for i in range(num_claims):
        claim_set.add_claim(Claim(
            claim_id=f"{trace_id}_C{i:03d}",
            trace_id=trace_id,
            claim_text=f"Claim {i}",
            sentence_id=f"S{i}",
            claim_index=i,
            character_start=0,
            character_end=10,
        ))
    return claim_set


def make_verification(trace_id, statuses):
    results = [
        VerificationResult(
            verification_id=f"V{i}",
            trace_id=trace_id,
            claim_id=f"{trace_id}_C{i:03d}",
            claim_text=f"Claim {i}",
            verification_status=status,
            verification_reason="Reason",
            confidence=0.9,
            best_chunk_id="chunk1",
            best_chunk_rank=1,
            best_chunk_score=0.9,
            best_sentence_id="S1",
            evidence_text="Evidence",
            entailment_score=0.9 if status == VerificationStatus.SUPPORTED else 0.1,
            contradiction_score=0.1,
            neutral_score=0.1,
        )
        for i, status in enumerate(statuses)
    ]
    return VerificationSummary(
        trace_id=trace_id,
        total_claims=len(statuses),
        supported_claims=statuses.count(VerificationStatus.SUPPORTED),
        partially_supported_claims=statuses.count(VerificationStatus.PARTIALLY_SUPPORTED),
        contradicted_claims=statuses.count(VerificationStatus.CONTRADICTED),
        unsupported_claims=statuses.count(VerificationStatus.UNSUPPORTED),
        not_verifiable_claims=statuses.count(VerificationStatus.NOT_VERIFIABLE),
        average_entailment_score=0.5,
        total_verification_latency_ms=100.0,
        results=results,
    )


class TestEndToEndFramework(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        original = os.getcwd()
        os.chdir(directory.name)
        self.addCleanup(os.chdir, original)
        for target in ("src.runner.Generator", "src.embedding_engine.get_shared_embed_model"):
            patcher = patch(target)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch("src.runner.RagasEvaluator", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
    def test_full_diagnostic_chain_produces_report(self):
        trace = make_trace()
        claim_set = make_claim_set(trace.trace_id)
        verification = make_verification(trace.trace_id, [VerificationStatus.SUPPORTED])

        psm = PipelineStateAnalyzer().analyze(trace, claim_set, verification)
        rca = RootCauseReasoner().analyze(psm)
        cap = CorrectiveActionEngine().generate(rca)
        report = ReportBuilder().build(trace=trace, psm=psm, rca=rca, cap=cap, verification=verification)

        self.assertEqual(report.framework_metadata.trace_id, trace.trace_id)
        self.assertEqual(report.analysis_status, "COMPLETED")
        self.assertEqual(report.executive_summary.supported_claims, 1)

    def test_report_presenter_renders_without_error(self):
        trace = make_trace()
        claim_set = make_claim_set(trace.trace_id)
        verification = make_verification(trace.trace_id, [VerificationStatus.UNSUPPORTED])

        psm = PipelineStateAnalyzer().analyze(trace, claim_set, verification)
        rca = RootCauseReasoner().analyze(psm)
        cap = CorrectiveActionEngine().generate(rca)
        report = ReportBuilder().build(trace=trace, psm=psm, rca=rca, cap=cap, verification=verification)

        presenter = DiagnosticReportPresenter(report)
        self.assertIn("DIAGNOSTIC EVALUATION REPORT", presenter.render_console())
        self.assertIn("# Diagnostic Evaluation Report", presenter.render_markdown())
        self.assertIn("<html", presenter.render_html())

    @patch("src.runner.ClaimVerifier")
    @patch("src.runner.ClaimDecomposer")
    def test_pipeline_runner_wiring_with_mocks(self, mock_decomposer_cls, mock_verifier_cls):
        trace = make_trace()
        verification = make_verification(trace.trace_id, [VerificationStatus.SUPPORTED])

        mock_decomposer_cls.return_value.decompose.return_value = MagicMock()
        mock_decomposer_cls.return_value.decompose.return_value.trace_id = trace.trace_id
        mock_decomposer_cls.return_value.decompose.return_value.metadata = {}
        mock_decomposer_cls.return_value.decompose.return_value.candidate_claims = []
        mock_verifier_cls.return_value.verify.return_value = verification

        runner = PipelineRunner()
        report = runner.run(trace)

        self.assertEqual(report.framework_metadata.trace_id, trace.trace_id)
        mock_decomposer_cls.return_value.decompose.assert_called_once_with(trace)
        mock_verifier_cls.return_value.verify.assert_called_once()
        # No gold_answer given -- answer correctness must not be computed.
        self.assertIsNone(report.answer_correctness)

    @patch("src.runner.ClaimVerifier")
    @patch("src.runner.ClaimDecomposer")
    def test_pipeline_runner_computes_answer_correctness_when_gold_answer_given(self, mock_decomposer_cls, mock_verifier_cls):
        trace = make_trace()
        verification = make_verification(trace.trace_id, [VerificationStatus.SUPPORTED])

        mock_decomposer_cls.return_value.decompose.return_value = MagicMock()
        mock_decomposer_cls.return_value.decompose.return_value.trace_id = trace.trace_id
        mock_decomposer_cls.return_value.decompose.return_value.metadata = {}
        mock_verifier_cls.return_value.verify.return_value = verification
        # AnswerCorrectnessEvaluator's decompose() call reuses this same mocked decomposer.
        mock_decomposer_cls.return_value.decompose.return_value.candidate_claims = []

        runner = PipelineRunner()
        report = runner.run(trace, gold_answer="The gold reference answer.")

        self.assertIsNotNone(report.answer_correctness)
        self.assertEqual(report.answer_correctness.total_gold_claims, 0)
        # Purely additive: health/primary_cause computed the same either way.
        self.assertEqual(report.executive_summary.primary_issue, "UNKNOWN")

        import os
        artifact_path = os.path.join("artifacts", "answer_correctness", f"TRACE_{trace.trace_id}.json")
        self.assertTrue(os.path.exists(artifact_path))
        os.remove(artifact_path)


if __name__ == "__main__":
    unittest.main()
