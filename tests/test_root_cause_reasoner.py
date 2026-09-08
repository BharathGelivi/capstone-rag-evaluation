import unittest
import os
import tempfile
from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineState, PipelineStage, PipelineStatus
from src.root_cause_reasoner import RootCauseReasoner, FailureType

class TestRootCauseReasoner(unittest.TestCase):
    def setUp(self):
        self.reasoner = RootCauseReasoner()
        
    def create_mock_psm(self, trace_id, states):
        return PipelineStateMatrix(
            trace_id=trace_id,
            pipeline_states=states
        )

    def test_healthy_pipeline(self):
        states = [
            PipelineState(stage=PipelineStage.CORPUS, status=PipelineStatus.UNKNOWN, observation="Unknown", confidence=1.0),
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.PASS, observation="Passed", confidence=0.9),
            PipelineState(stage=PipelineStage.GENERATOR, status=PipelineStatus.PASS, observation="Passed", confidence=0.9),
            PipelineState(stage=PipelineStage.GROUNDING, status=PipelineStatus.PASS, observation="Passed", confidence=0.9),
        ]
        psm = self.create_mock_psm("TRACE_HEALTHY", states)
        rca = self.reasoner.analyze(psm)
        
        self.assertEqual(rca.primary_cause, FailureType.UNKNOWN)
        self.assertEqual(len(rca.secondary_effects), 0)
        self.assertFalse(rca.recommendations_needed)
        self.assertIn("No failures detected", rca.reasoning_chain[-1])

    def test_retriever_failure(self):
        states = [
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
            PipelineState(stage=PipelineStage.GENERATOR, status=PipelineStatus.UNKNOWN, observation="Unknown", confidence=0.5),
            PipelineState(stage=PipelineStage.GROUNDING, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
        ]
        psm = self.create_mock_psm("TRACE_RETRIEVER_FAIL", states)
        rca = self.reasoner.analyze(psm)
        
        self.assertEqual(rca.primary_cause, FailureType.RETRIEVAL_MISS)
        self.assertEqual(len(rca.secondary_effects), 1)
        self.assertEqual(rca.secondary_effects[0], FailureType.GROUNDING_FAILURE)
        self.assertTrue(rca.recommendations_needed)
        
    def test_generator_failure(self):
        states = [
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.PASS, observation="Passed", confidence=0.9),
            PipelineState(stage=PipelineStage.GENERATOR, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
            PipelineState(stage=PipelineStage.GROUNDING, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
        ]
        psm = self.create_mock_psm("TRACE_GEN_FAIL", states)
        rca = self.reasoner.analyze(psm)
        
        self.assertEqual(rca.primary_cause, FailureType.UNSUPPORTED_GENERATION)
        self.assertEqual(len(rca.secondary_effects), 1)
        self.assertEqual(rca.secondary_effects[0], FailureType.GROUNDING_FAILURE)
        self.assertTrue(rca.recommendations_needed)
        
    def test_grounding_failure(self):
        states = [
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.PASS, observation="Passed", confidence=0.9),
            PipelineState(stage=PipelineStage.GENERATOR, status=PipelineStatus.PASS, observation="Passed", confidence=0.9),
            PipelineState(stage=PipelineStage.GROUNDING, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
        ]
        psm = self.create_mock_psm("TRACE_GROUND_FAIL", states)
        rca = self.reasoner.analyze(psm)
        
        self.assertEqual(rca.primary_cause, FailureType.GROUNDING_FAILURE)
        self.assertEqual(len(rca.secondary_effects), 0)
        self.assertTrue(rca.recommendations_needed)
        
    def test_earliest_upstream_failure_wins(self):
        # RETRIEVER fails with lower confidence than GENERATOR. Under causal
        # propagation, the earliest upstream failure (RETRIEVER) must win primary cause.
        states = [
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.FAIL, observation="Failed", confidence=0.5),
            PipelineState(stage=PipelineStage.GENERATOR, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
        ]
        psm = self.create_mock_psm("TRACE_CAUSAL_PROP", states)
        rca = self.reasoner.analyze(psm)

        self.assertEqual(rca.primary_cause, FailureType.RETRIEVAL_MISS)
        self.assertIn(FailureType.UNSUPPORTED_GENERATION, rca.secondary_effects)

    def test_save_and_load(self):
        states = [
            PipelineState(stage=PipelineStage.RETRIEVER, status=PipelineStatus.FAIL, observation="Failed", confidence=0.9),
        ]
        psm = self.create_mock_psm("TRACE_SAVE", states)
        rca = self.reasoner.analyze(psm)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = rca.save(tmpdir)
            self.assertTrue(os.path.exists(filepath))
            
            loaded_rca = rca.__class__.load(filepath)
            self.assertEqual(loaded_rca.trace_id, rca.trace_id)
            self.assertEqual(loaded_rca.primary_cause, rca.primary_cause)

if __name__ == '__main__':
    unittest.main()
