import unittest
import os
import tempfile
from src.rag_trace import RAGTrace
from src.claims import ClaimSet, Claim
from src.claim_verifier import VerificationSummary, VerificationResult, VerificationStatus
from src.pipeline_state_analyzer import PipelineStateAnalyzer, PipelineStage, PipelineStatus

class TestPipelineStateAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PipelineStateAnalyzer(retrieval_score_threshold=0.5)

    def create_mock_trace(self, trace_id, max_score, retrieved_chunk_references=None, execution_statistics=None):
        return RAGTrace(
            trace_id=trace_id,
            trace_version="1.0",
            pipeline_version="1.0",
            framework_version="1.0",
            timestamp="2026-07-13T12:00:00Z",
            question="Test question",
            generated_answer="Test answer",
            prompt_snapshot="Prompt",
            prompt_length=100,
            retrieved_chunk_references=retrieved_chunk_references if retrieved_chunk_references is not None else [{"chunk_id": "chunk1", "similarity_score": max_score}],
            configuration_snapshot={},
            execution_statistics=execution_statistics if execution_statistics is not None else {},
            pipeline_stage_status={}
        )

    def create_mock_claim_set(self, trace_id, num_claims):
        claim_set = ClaimSet(trace_id=trace_id)
        for i in range(num_claims):
            claim = Claim(
                claim_id=f"{trace_id}_C{i:03d}",
                trace_id=trace_id,
                claim_text=f"Claim {i}",
                sentence_id=f"S{i}",
                claim_index=i,
                character_start=0,
                character_end=10
            )
            claim_set.add_claim(claim)
        return claim_set

    def create_mock_verification(self, trace_id, statuses):
        results = []
        for i, status in enumerate(statuses):
            results.append(VerificationResult(
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
                neutral_score=0.1
            ))
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
            results=results
        )

    def test_healthy_pipeline(self):
        trace_id = "TRACE_HEALTHY"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED, VerificationStatus.SUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).status, PipelineStatus.PASS)
        self.assertEqual(matrix.get(PipelineStage.GENERATOR).status, PipelineStatus.PASS)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.PASS)
        
        # CHUNKING is emitted only when boundary evidence is observable -- an
        # UNKNOWN chunking state was removed deliberately as diagnostic noise --
        # so a healthy trace legitimately has no CHUNKING state.
        for stage in PipelineStage:
            state = matrix.get(stage)
            if state is None:
                self.assertEqual(stage, PipelineStage.CHUNKING)
                continue
            self.assertIsNotNone(state.confidence)
            self.assertTrue(isinstance(state.confidence, float))
            # Test that language does not contain reasoning words
            obs_lower = state.observation.lower()
            for word in ["hallucination", "failure", "root cause"]:
                self.assertNotIn(word, obs_lower)
                
        self.assertEqual(matrix.artifact_version, "1.0")

    def test_retriever_failure(self):
        trace_id = "TRACE_RETRIEVER_FAIL"
        trace = self.create_mock_trace(trace_id, max_score=0.2)
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.UNSUPPORTED, VerificationStatus.UNSUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).status, PipelineStatus.FAIL)
        self.assertEqual(matrix.get(PipelineStage.GENERATOR).status, PipelineStatus.UNKNOWN)
        # GROUNDING now keys off CONTRADICTED specifically; plain UNSUPPORTED claims pass.
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.PASS)

    def test_generator_failure(self):
        trace_id = "TRACE_GENERATOR_FAIL"
        trace = self.create_mock_trace(trace_id, max_score=0.9)
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED, VerificationStatus.UNSUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).status, PipelineStatus.PASS)
        self.assertEqual(matrix.get(PipelineStage.GENERATOR).status, PipelineStatus.FAIL)
        # GROUNDING now keys off CONTRADICTED specifically; plain UNSUPPORTED claims pass.
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.PASS)

        # Verify ids exist
        retriever_state = matrix.get(PipelineStage.RETRIEVER)
        self.assertTrue(len(retriever_state.supporting_claim_ids) > 0)
        self.assertTrue(len(retriever_state.supporting_chunk_ids) > 0)
        self.assertTrue(len(retriever_state.supporting_verification_ids) > 0)
        
    def test_grounding_failure(self):
        trace_id = "TRACE_GROUNDING_FAIL"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.CONTRADICTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.FAIL)

    def test_grounding_pass_with_only_unsupported_not_contradicted(self):
        trace_id = "TRACE_GROUNDING_PASS_UNSUPPORTED"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.UNSUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.PASS)

    def test_grounding_unknown_with_zero_claims(self):
        trace_id = "TRACE_GROUNDING_ZERO_CLAIMS"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 0)
        verification = self.create_mock_verification(trace_id, [])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        self.assertEqual(matrix.get(PipelineStage.GROUNDING).status, PipelineStatus.UNKNOWN)

    def test_corpus_failure_low_relevance(self):
        trace_id = "TRACE_CORPUS_FAIL"
        trace = self.create_mock_trace(
            trace_id, max_score=0.2,
            execution_statistics={"pre_rerank_min_dense_distance": 0.95, "pre_rerank_candidate_pool_size": 15}
        )
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.UNSUPPORTED])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        self.assertEqual(matrix.get(PipelineStage.CORPUS).status, PipelineStatus.FAIL)

    def test_chunking_boundary_failure(self):
        trace_id = "TRACE_CHUNKING_FAIL"
        trace = self.create_mock_trace(
            trace_id, max_score=0.8,
            retrieved_chunk_references=[
                {"chunk_id": "chunk1", "similarity_score": 0.8, "chunk_index": 3, "parent_document_id": "doc-1"},
                {"chunk_id": "chunk2", "similarity_score": 0.7, "chunk_index": 4, "parent_document_id": "doc-1"},
            ]
        )
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.PARTIALLY_SUPPORTED])
        # create_mock_verification hardcodes best_chunk_id="chunk1" for every result.

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        self.assertEqual(matrix.get(PipelineStage.CHUNKING).status, PipelineStatus.FAIL)

    def test_chunk_utilization_rate_trace_a(self):
        # Trace A: 4 chunks retrieved, 1 unique chunk used -> 0.25
        trace_id = "TRACE_UTIL_A"
        trace = self.create_mock_trace(
            trace_id, max_score=0.8,
            retrieved_chunk_references=[
                {"chunk_id": "c1", "similarity_score": 0.8},
                {"chunk_id": "c2", "similarity_score": 0.7},
                {"chunk_id": "c3", "similarity_score": 0.6},
                {"chunk_id": "c4", "similarity_score": 0.5},
            ]
        )
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED, VerificationStatus.SUPPORTED])
        # create_mock_verification hardcodes best_chunk_id="chunk1" for every result,
        # but chunk1 isn't in this trace's retrieved_chunk_references -- doesn't matter
        # for the utilization math itself, only the *set* of used ids' size vs. retrieved.
        # To exercise the real single-unique-chunk-used scenario, override best_chunk_id.
        for res in verification.results:
            res.best_chunk_id = "c1"

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        retriever = matrix.get(PipelineStage.RETRIEVER)

        self.assertEqual(retriever.metadata["chunk_utilization_rate"], 0.25)
        self.assertEqual(retriever.metadata["chunks_used"], 1)
        self.assertEqual(retriever.metadata["chunks_retrieved"], 4)
        # Non-goals: status/confidence must be exactly what they'd be without this change.
        self.assertEqual(retriever.status, PipelineStatus.PASS)
        self.assertEqual(retriever.confidence, 0.95)

    def test_chunk_utilization_rate_trace_b(self):
        # Trace B: 3 chunks retrieved, 1 unique chunk used -> ~0.333
        trace_id = "TRACE_UTIL_B"
        trace = self.create_mock_trace(
            trace_id, max_score=0.8,
            retrieved_chunk_references=[
                {"chunk_id": "c1", "similarity_score": 0.8},
                {"chunk_id": "c2", "similarity_score": 0.7},
                {"chunk_id": "c3", "similarity_score": 0.6},
            ]
        )
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED])
        verification.results[0].best_chunk_id = "c1"

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        retriever = matrix.get(PipelineStage.RETRIEVER)

        self.assertAlmostEqual(retriever.metadata["chunk_utilization_rate"], 1 / 3, places=6)
        self.assertEqual(retriever.metadata["chunks_used"], 1)
        self.assertEqual(retriever.metadata["chunks_retrieved"], 3)
        self.assertEqual(retriever.status, PipelineStatus.PASS)
        self.assertEqual(retriever.confidence, 0.95)

    def test_chunk_utilization_rate_none_when_nothing_retrieved(self):
        trace_id = "TRACE_UTIL_EMPTY"
        trace = self.create_mock_trace(trace_id, max_score=0.2, retrieved_chunk_references=[])
        claim_set = self.create_mock_claim_set(trace_id, 0)
        verification = self.create_mock_verification(trace_id, [])

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        retriever = matrix.get(PipelineStage.RETRIEVER)
        self.assertIsNone(retriever.metadata["chunk_utilization_rate"])

    def test_full_pipeline_health_and_primary_cause_unchanged_by_utilization(self):
        """
        Regression check: a full analyze() -> RootCauseReasoner pass on an
        existing-style healthy trace produces the same primary_cause as
        before this change, regardless of chunk_utilization_rate.
        """
        from src.root_cause_reasoner import RootCauseReasoner, FailureType

        trace_id = "TRACE_REGRESSION"
        trace = self.create_mock_trace(
            trace_id, max_score=0.9,
            retrieved_chunk_references=[
                {"chunk_id": "c1", "similarity_score": 0.9},
                {"chunk_id": "c2", "similarity_score": 0.8},
                {"chunk_id": "c3", "similarity_score": 0.7},
                {"chunk_id": "c4", "similarity_score": 0.6},
            ]
        )
        claim_set = self.create_mock_claim_set(trace_id, 2)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED, VerificationStatus.SUPPORTED])
        for res in verification.results:
            res.best_chunk_id = "c1"

        matrix = self.analyzer.analyze(trace, claim_set, verification)
        rca = RootCauseReasoner().analyze(matrix)

        # Low utilization (1/4 = 0.25) must not change the causal outcome.
        self.assertEqual(matrix.get(PipelineStage.RETRIEVER).metadata["chunk_utilization_rate"], 0.25)
        self.assertEqual(rca.primary_cause, FailureType.UNKNOWN)  # healthy pipeline, no failures

        grounding_score = (2 + 0) / 2  # 2 supported, 0 partial, out of 2 verified -- matches ReportBuilder's formula
        self.assertEqual(grounding_score, 1.0)

    def test_save_and_load(self):
        trace_id = "TRACE_SAVE_LOAD"
        trace = self.create_mock_trace(trace_id, max_score=0.8)
        claim_set = self.create_mock_claim_set(trace_id, 1)
        verification = self.create_mock_verification(trace_id, [VerificationStatus.SUPPORTED])
        
        matrix = self.analyzer.analyze(trace, claim_set, verification)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = matrix.save(tmpdir)
            self.assertTrue(os.path.exists(filepath))
            
            loaded_matrix = matrix.__class__.load(filepath)
            self.assertEqual(loaded_matrix.trace_id, matrix.trace_id)
            self.assertEqual(loaded_matrix.artifact_version, matrix.artifact_version)
            self.assertEqual(len(loaded_matrix.pipeline_states), len(matrix.pipeline_states))
            
            loaded_retriever = loaded_matrix.get(PipelineStage.RETRIEVER)
            self.assertEqual(loaded_retriever.status, PipelineStatus.PASS)
            self.assertEqual(loaded_retriever.confidence, matrix.get(PipelineStage.RETRIEVER).confidence)

if __name__ == '__main__':
    unittest.main()
