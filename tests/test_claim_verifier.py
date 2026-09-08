import os
import unittest
from unittest.mock import patch, MagicMock

from src.claim_verifier import ClaimVerifier, VerificationStatus, EvidenceSentence
from src.rag_trace import RAGTrace
from src.retriever import RetrievedChunk
from src.claim_decomposer import CandidateClaim, CandidateClaimSet

class TestClaimVerifier(unittest.TestCase):
    @patch('src.claim_verifier.pipeline')
    def setUp(self, mock_pipeline):
        # Create a mock pipeline that returns predefined scores based on the hypothesis
        self.mock_nli = MagicMock()
        mock_pipeline.return_value = self.mock_nli
        
        def side_effect(*args, **kwargs):
            # The NLI mock is called in two ways:
            #   1. Batch: nli_pipeline([{...}, {...}], batch_size=N)  -> return list of results
            #   2. Single: nli_pipeline({"text": ..., "text_pair": ...})  -> return one result
            raw = args[0] if args else []
            is_single = isinstance(raw, dict)
            batch = [raw] if is_single else raw
            output = []
            for item in batch:
                hypothesis = item.get("text_pair", "")
                if "supported" in hypothesis.lower():
                    output.append([{"label": "entailment", "score": 0.9}, {"label": "neutral", "score": 0.05}, {"label": "contradiction", "score": 0.05}])
                elif "contradict" in hypothesis.lower():
                    output.append([{"label": "entailment", "score": 0.05}, {"label": "neutral", "score": 0.05}, {"label": "contradiction", "score": 0.9}])
                elif "partial" in hypothesis.lower():
                    output.append([{"label": "entailment", "score": 0.5}, {"label": "neutral", "score": 0.4}, {"label": "contradiction", "score": 0.1}])
                else:  # unsupported / neutral
                    output.append([{"label": "entailment", "score": 0.1}, {"label": "neutral", "score": 0.85}, {"label": "contradiction", "score": 0.05}])
            return output[0] if is_single else output
                
        self.mock_nli.side_effect = side_effect
        # enable_llm_judge=False: several tests below deliberately produce
        # PARTIALLY_SUPPORTED/NOT_VERIFIABLE claims to test NLI aggregation
        # logic itself, which would otherwise trigger a real network call to
        # the LLM judge. The hybrid escalation path has its own tests below.
        self.verifier = ClaimVerifier(model_name="mock_model", enable_llm_judge=False)

    def test_sentence_splitting(self):
        text = "This is sentence one. This is sentence two! And three? Yes."
        sentences = self.verifier._split_into_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "This is sentence one.")

    def test_verification_status_supported(self):
        claim = CandidateClaim("c1", "t1", "This claim is supported.", "S1", 0, 0, 10, {})
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Some premise text.")]
        
        result = self.verifier.verify_claim(claim, chunks)
        
        self.assertEqual(result.verification_status, VerificationStatus.SUPPORTED)
        self.assertGreater(result.entailment_score, 0.7)
        self.assertIn("Evidence directly supports", result.verification_reason)
        
    def test_verification_status_contradicted(self):
        claim = CandidateClaim("c2", "t1", "This claim is contradicted.", "S1", 0, 0, 10, {})
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Some premise text.")]
        
        result = self.verifier.verify_claim(claim, chunks)
        self.assertEqual(result.verification_status, VerificationStatus.CONTRADICTED)
        self.assertGreater(result.contradiction_score, 0.7)

    def test_top_3_sentences(self):
        claim = CandidateClaim("c1", "t1", "This claim is supported.", "S1", 0, 0, 10, {})
        # Text with 4 sentences
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Sentence A. Sentence B. Sentence C. Sentence D.")]
        
        result = self.verifier.verify_claim(claim, chunks)
        self.assertEqual(len(result.top_evidence), 3)

    def test_top1_strategy_unchanged_default(self):
        self.assertEqual(self.verifier.aggregation_strategy, "top1")

    def test_invalid_aggregation_strategy_raises(self):
        with self.assertRaises(ValueError):
            ClaimVerifier(model_name="mock_model", aggregation_strategy="bogus")

    @patch('src.claim_verifier.pipeline')
    def test_concat_top3_resolves_supported_when_no_single_sentence_does(self, mock_pipeline):
        mock_nli = MagicMock()
        mock_pipeline.return_value = mock_nli

        def side_effect(*args, **kwargs):
            raw = args[0] if args else []
            is_single = isinstance(raw, dict)
            batch = [raw] if is_single else raw
            output = []
            for item in batch:
                premise = item.get("text", "")
                if "Alpha" in premise and "Beta" in premise and "Gamma" in premise:
                    output.append([{"label": "entailment", "score": 0.85}, {"label": "neutral", "score": 0.1}, {"label": "contradiction", "score": 0.05}])
                else:
                    output.append([{"label": "entailment", "score": 0.5}, {"label": "neutral", "score": 0.4}, {"label": "contradiction", "score": 0.1}])
            return output[0] if is_single else output

        mock_nli.side_effect = side_effect

        claim = CandidateClaim("c1", "t1", "Generic claim.", "S1", 0, 0, 10, {})
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Alpha fact one. Beta fact two. Gamma fact three.")]

        verifier_top1 = ClaimVerifier(model_name="mock_model", aggregation_strategy="top1", enable_llm_judge=False)
        result_top1 = verifier_top1.verify_claim(claim, chunks)
        self.assertEqual(result_top1.verification_status, VerificationStatus.PARTIALLY_SUPPORTED)

        verifier_concat = ClaimVerifier(model_name="mock_model", aggregation_strategy="concat_top3", enable_llm_judge=False)
        result_concat = verifier_concat.verify_claim(claim, chunks)
        self.assertEqual(result_concat.verification_status, VerificationStatus.SUPPORTED)

    @patch('src.claim_verifier.pipeline')
    def test_max_pool_top3_selects_argmax_entailment_triplet(self, mock_pipeline):
        mock_nli = MagicMock()
        mock_pipeline.return_value = mock_nli

        def side_effect(*args, **kwargs):
            raw = args[0] if args else []
            is_single = isinstance(raw, dict)
            batch = [raw] if is_single else raw
            output = []
            for item in batch:
                premise = item.get("text", "")
                if "Neutral" in premise:
                    output.append([{"label": "entailment", "score": 0.3}, {"label": "neutral", "score": 0.6}, {"label": "contradiction", "score": 0.1}])
                elif "Contradictory" in premise:
                    output.append([{"label": "entailment", "score": 0.1}, {"label": "neutral", "score": 0.0}, {"label": "contradiction", "score": 0.9}])
                else:
                    output.append([{"label": "entailment", "score": 0.05}, {"label": "neutral", "score": 0.9}, {"label": "contradiction", "score": 0.05}])
            return output[0] if is_single else output

        mock_nli.side_effect = side_effect

        claim = CandidateClaim("c1", "t1", "Generic claim.", "S1", 0, 0, 10, {})
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0,
                                  chunk_text="Neutral fact appears here. Contradictory fact appears here. Other fact appears here.")]

        verifier_top1 = ClaimVerifier(model_name="mock_model", aggregation_strategy="top1", enable_llm_judge=False)
        result_top1 = verifier_top1.verify_claim(claim, chunks)
        # top1 will pick "Neutral" because it's first and has highest entailment (0.3).
        # Neither entailment nor contradiction > threshold.
        self.assertEqual(result_top1.verification_status, VerificationStatus.NOT_VERIFIABLE)

        verifier_max_pool = ClaimVerifier(model_name="mock_model", aggregation_strategy="max_pool_top3", enable_llm_judge=False)
        result_max_pool = verifier_max_pool.verify_claim(claim, chunks)
        
        # max_pool_top3 should select the EXACT SAME triplet as the argmax entailment sentence ("Neutral").
        # The flawed logic took independent maxes resulting in 0.9 contradiction.
        # Now it correctly keeps the valid distribution [0.3, 0.6, 0.1]
        self.assertEqual(result_max_pool.verification_status, VerificationStatus.NOT_VERIFIABLE)
        self.assertEqual(result_max_pool.entailment_score, 0.3)
        self.assertEqual(result_max_pool.contradiction_score, 0.1)
        self.assertEqual(result_max_pool.neutral_score, 0.6)

    def test_verify_all_and_artifacts(self):
        claim_set = CandidateClaimSet("t1")
        claim_set.add_claim(CandidateClaim("c1", "t1", "Supported claim.", "S1", 0, 0, 10, {}))
        claim_set.add_claim(CandidateClaim("c2", "t1", "Contradicted claim.", "S2", 1, 0, 10, {}))
        
        trace_id = "t1"
        chunks = [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1", source_file="doc.pdf", chunk_index=0, chunk_text="Premise.")]
        
        summary = self.verifier.verify_all(claim_set, trace_id, chunks)
        
        self.assertEqual(summary.total_claims, 2)
        self.assertEqual(summary.supported_claims, 1)
        self.assertEqual(summary.contradicted_claims, 1)
        
        # Test artifact saving
        self.verifier.save_artifacts(summary)
        
        filepath = os.path.join("artifacts", "verification", "TRACE_t1.json")
        self.assertTrue(os.path.exists(filepath))
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)


def _make_nli_mock(entailment, neutral, contradiction):
    """A single-answer NLI mock: every pair scores the same triplet, so
    hybrid-escalation tests can pin down exactly which VerificationStatus
    the NLI fast path lands on without needing per-hypothesis branching."""
    mock_nli = MagicMock()

    def side_effect(*args, **kwargs):
        raw = args[0] if args else []
        is_single = isinstance(raw, dict)
        batch = [raw] if is_single else raw
        scored = [{"label": "entailment", "score": entailment},
                  {"label": "neutral", "score": neutral},
                  {"label": "contradiction", "score": contradiction}]
        output = [scored for _ in batch]
        return output[0] if is_single else output

    mock_nli.side_effect = side_effect
    return mock_nli


class TestHybridVerification(unittest.TestCase):
    """The LLM-judge escalation added on top of the NLI fast path (see
    AMBIGUOUS_STATUSES in claim_verifier.py). Confirms: clear-cut NLI
    verdicts never pay for a judge call, ambiguous ones do and get
    overridden on a clean judge response, and any judge failure (network
    error or unparseable output) falls back to the NLI verdict rather than
    breaking verification for the claim.
    """

    def _claim(self):
        return CandidateClaim("c1", "t1", "Some claim.", "S1", 0, 0, 10, {})

    def _chunks(self):
        return [RetrievedChunk(chunk_id="ch1", similarity_score=0.9, rank=1, page_number="1",
                                source_file="doc.pdf", chunk_index=0, chunk_text="Some evidence text.")]

    @patch('src.claim_verifier.OpenAILike')
    @patch('src.claim_verifier.pipeline')
    def test_clear_supported_skips_llm_judge(self, mock_pipeline, mock_openai_like):
        mock_pipeline.return_value = _make_nli_mock(0.95, 0.03, 0.02)
        mock_judge = MagicMock()
        mock_openai_like.return_value = mock_judge

        verifier = ClaimVerifier(model_name="mock_model")
        result = verifier.verify_claim(self._claim(), self._chunks())

        self.assertEqual(result.verification_status, VerificationStatus.SUPPORTED)
        self.assertEqual(result.verified_by, "nli")
        mock_judge.chat.assert_not_called()

    @patch('src.claim_verifier.OpenAILike')
    @patch('src.claim_verifier.pipeline')
    def test_ambiguous_escalates_and_uses_judge_verdict(self, mock_pipeline, mock_openai_like):
        mock_pipeline.return_value = _make_nli_mock(0.5, 0.4, 0.1)  # PARTIALLY_SUPPORTED
        mock_judge = MagicMock()
        judge_response = MagicMock()
        judge_response.message.content = "STATUS: SUPPORTED\nREASON: The evidence clearly backs this up."
        mock_judge.chat.return_value = judge_response
        mock_openai_like.return_value = mock_judge

        verifier = ClaimVerifier(model_name="mock_model")
        result = verifier.verify_claim(self._claim(), self._chunks())

        mock_judge.chat.assert_called_once()
        self.assertEqual(result.verification_status, VerificationStatus.SUPPORTED)
        self.assertEqual(result.verified_by, "llm")
        self.assertIn("[LLM judge]", result.verification_reason)

    @patch('src.claim_verifier.OpenAILike')
    @patch('src.claim_verifier.pipeline')
    def test_judge_network_failure_falls_back_to_nli(self, mock_pipeline, mock_openai_like):
        mock_pipeline.return_value = _make_nli_mock(0.5, 0.4, 0.1)  # PARTIALLY_SUPPORTED
        mock_judge = MagicMock()
        mock_judge.chat.side_effect = RuntimeError("connection refused")
        mock_openai_like.return_value = mock_judge

        verifier = ClaimVerifier(model_name="mock_model")
        result = verifier.verify_claim(self._claim(), self._chunks())

        self.assertEqual(result.verification_status, VerificationStatus.PARTIALLY_SUPPORTED)
        self.assertEqual(result.verified_by, "nli")

    @patch('src.claim_verifier.OpenAILike')
    @patch('src.claim_verifier.pipeline')
    def test_judge_unparseable_response_falls_back_to_nli(self, mock_pipeline, mock_openai_like):
        mock_pipeline.return_value = _make_nli_mock(0.1, 0.85, 0.05)  # NOT_VERIFIABLE-ish
        mock_judge = MagicMock()
        judge_response = MagicMock()
        judge_response.message.content = "I think this is probably fine, hard to say."
        mock_judge.chat.return_value = judge_response
        mock_openai_like.return_value = mock_judge

        verifier = ClaimVerifier(model_name="mock_model")
        result = verifier.verify_claim(self._claim(), self._chunks())

        self.assertEqual(result.verified_by, "nli")

    @patch('src.claim_verifier.OpenAILike')
    @patch('src.claim_verifier.pipeline')
    def test_enable_llm_judge_false_never_escalates(self, mock_pipeline, mock_openai_like):
        mock_pipeline.return_value = _make_nli_mock(0.5, 0.4, 0.1)  # PARTIALLY_SUPPORTED
        mock_judge = MagicMock()
        mock_openai_like.return_value = mock_judge

        verifier = ClaimVerifier(model_name="mock_model", enable_llm_judge=False)
        result = verifier.verify_claim(self._claim(), self._chunks())

        self.assertEqual(result.verified_by, "nli")
        mock_judge.chat.assert_not_called()


if __name__ == '__main__':
    unittest.main()
