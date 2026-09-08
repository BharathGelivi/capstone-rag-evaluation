import unittest
import os
import tempfile
from src.root_cause_reasoner import RootCauseAnalysis, FailureType
from src.corrective_action_engine import CorrectiveActionEngine, ActionCategory
from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineState, PipelineStage, PipelineStatus

class TestCorrectiveActionEngine(unittest.TestCase):
    def setUp(self):
        self.engine = CorrectiveActionEngine()

    def test_healthy_pipeline(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_HEALTHY",
            primary_cause=FailureType.UNKNOWN,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=False
        )
        plan = self.engine.generate(rca)
        
        self.assertEqual(plan.trace_id, "TRACE_HEALTHY")
        self.assertEqual(len(plan.immediate_actions), 0)
        self.assertEqual(len(plan.short_term_actions), 0)
        self.assertEqual(len(plan.experimental_actions), 0)

    def test_retriever_failure(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_RETRIEVER",
            primary_cause=FailureType.RETRIEVAL_MISS,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        
        self.assertTrue(len(plan.immediate_actions) > 0)
        self.assertTrue(len(plan.short_term_actions) > 0)
        self.assertTrue(len(plan.experimental_actions) > 0)

        all_actions = plan.immediate_actions + plan.short_term_actions + plan.experimental_actions
        for act in all_actions:
            self.assertEqual(act.category, ActionCategory.RETRIEVAL)
            self.assertIsNotNone(act.observed_evidence)
            self.assertIsNotNone(act.root_cause)
            self.assertIsNotNone(act.expected_improvement)
            self.assertIsNotNone(act.success_metric)

    def test_mixed_failures(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_MIXED",
            primary_cause=FailureType.RETRIEVAL_MISS,
            secondary_effects=[FailureType.UNSUPPORTED_GENERATION],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        
        all_actions = plan.immediate_actions + plan.short_term_actions + plan.experimental_actions
        categories = [act.category for act in all_actions]
        self.assertIn(ActionCategory.RETRIEVAL, categories)
        self.assertIn(ActionCategory.GENERATION, categories)

    def test_parameterized_templates_differ_by_trace(self):
        def make_rca_and_psm(trace_id, max_score):
            rca = RootCauseAnalysis(
                trace_id=trace_id,
                primary_cause=FailureType.RETRIEVAL_MISS,
                secondary_effects=[],
                reasoning_chain=[],
                recommendations_needed=True
            )
            psm = PipelineStateMatrix(trace_id=trace_id, pipeline_states=[
                PipelineState(
                    stage=PipelineStage.RETRIEVER,
                    status=PipelineStatus.FAIL,
                    observation="Failed",
                    confidence=0.8,
                    metadata={"max_score": max_score, "threshold": 0.5}
                )
            ])
            return rca, psm

        rca1, psm1 = make_rca_and_psm("TRACE_A", max_score=0.42)
        rca2, psm2 = make_rca_and_psm("TRACE_B", max_score=0.19)

        plan1 = self.engine.generate(rca1, psm=psm1)
        plan2 = self.engine.generate(rca2, psm=psm2)

        action1 = plan1.immediate_actions[0]
        action2 = plan2.immediate_actions[0]

        self.assertIn("0.420", action1.observed_evidence)
        self.assertIn("0.190", action2.observed_evidence)
        self.assertNotEqual(action1.observed_evidence, action2.observed_evidence)

    def test_generate_without_psm_falls_back_to_na(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_NO_PSM",
            primary_cause=FailureType.RETRIEVAL_MISS,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        action = plan.immediate_actions[0]
        self.assertIn("N/A", action.observed_evidence)

    def test_low_utilization_produces_informational_advisory(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_LOW_UTIL",
            primary_cause=FailureType.UNKNOWN,  # healthy pipeline -- must not be gated on FAIL
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=False
        )
        psm = PipelineStateMatrix(trace_id="TRACE_LOW_UTIL", pipeline_states=[
            PipelineState(
                stage=PipelineStage.RETRIEVER,
                status=PipelineStatus.PASS,
                observation="OK",
                confidence=0.95,
                metadata={"chunk_utilization_rate": 0.25, "chunks_used": 1, "chunks_retrieved": 4}
            )
        ])
        plan = self.engine.generate(rca, psm=psm)

        self.assertEqual(len(plan.informational_actions), 1)
        action = plan.informational_actions[0]
        self.assertEqual(action.priority, "informational")
        self.assertIn("25%", action.description)
        # Must not leak into the causal tiers.
        self.assertEqual(len(plan.immediate_actions), 0)
        self.assertEqual(len(plan.short_term_actions), 0)
        self.assertEqual(len(plan.experimental_actions), 0)

    def test_high_utilization_produces_no_advisory(self):
        rca = RootCauseAnalysis(trace_id="TRACE_HIGH_UTIL", primary_cause=FailureType.UNKNOWN)
        psm = PipelineStateMatrix(trace_id="TRACE_HIGH_UTIL", pipeline_states=[
            PipelineState(
                stage=PipelineStage.RETRIEVER, status=PipelineStatus.PASS, observation="OK", confidence=0.95,
                metadata={"chunk_utilization_rate": 0.8, "chunks_used": 4, "chunks_retrieved": 5}
            )
        ])
        plan = self.engine.generate(rca, psm=psm)
        self.assertEqual(len(plan.informational_actions), 0)

    def test_missing_utilization_metadata_produces_no_advisory(self):
        rca = RootCauseAnalysis(trace_id="TRACE_NO_META", primary_cause=FailureType.UNKNOWN)
        psm = PipelineStateMatrix(trace_id="TRACE_NO_META", pipeline_states=[
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.PASS, observation="OK", confidence=0.95)
        ])
        plan = self.engine.generate(rca, psm=psm)
        self.assertEqual(len(plan.informational_actions), 0)

    def test_no_psm_produces_no_advisory(self):
        rca = RootCauseAnalysis(trace_id="TRACE_NO_PSM_UTIL", primary_cause=FailureType.UNKNOWN)
        plan = self.engine.generate(rca)
        self.assertEqual(len(plan.informational_actions), 0)

    def test_save_and_load(self):
        rca = RootCauseAnalysis(
            trace_id="TRACE_SAVE",
            primary_cause=FailureType.CHUNK_BOUNDARY,
            secondary_effects=[],
            reasoning_chain=[],
            recommendations_needed=True
        )
        plan = self.engine.generate(rca)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plan.save(tmpdir)
            self.assertTrue(os.path.exists(filepath))
            
            loaded_plan = plan.__class__.load(filepath)
            self.assertEqual(loaded_plan.trace_id, plan.trace_id)
            self.assertEqual(loaded_plan.primary_cause, plan.primary_cause)
            self.assertEqual(len(loaded_plan.immediate_actions), len(plan.immediate_actions))

if __name__ == '__main__':
    unittest.main()
