"""
Tests for ClaimDecomposer.

All tests patch ``src.claim_decomposer.OpenAILike`` — Groq has been removed;
the NVIDIA endpoint is accessed exclusively via OpenAILike.
"""
import json
import unittest
import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.rag_trace import RAGTrace
from src.claim_decomposer import ClaimDecomposer


def make_trace(question="Q?", answer="The sky is blue. Water boils at 100 degrees Celsius.") -> RAGTrace:
    return RAGTrace(
        trace_id=str(uuid.uuid4()),
        trace_version="1.0",
        pipeline_version="1.0",
        framework_version="1.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        question=question,
        generated_answer=answer,
        prompt_snapshot="Mock prompt",
        prompt_length=100,
        retrieved_chunk_references=[],
        configuration_snapshot={},
        execution_statistics={},
        pipeline_stage_status={},
        diagnostics=None,
    )


def make_llm_response(text):
    response = MagicMock()
    # OpenAILike chat() returns ChatResponse with message.content
    response.message = MagicMock()
    response.message.content = text
    return response


class TestClaimDecomposer(unittest.TestCase):
    @patch("src.claim_decomposer.OpenAILike")
    def test_decompose_produces_candidate_claims(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_llm_response(
            json.dumps([
                {"claim_text": "The sky is blue.", "sentence_id": "S1"},
                {"claim_text": "Water boils at 100 degrees Celsius.", "sentence_id": "S2"},
            ])
        )
        mock_llm_cls.return_value = mock_llm

        decomposer = ClaimDecomposer(model_name="mock-model")
        trace = make_trace()
        claim_set = decomposer.decompose(trace)

        self.assertEqual(claim_set.total_candidates, 2)
        self.assertEqual(claim_set.candidate_claims[0].claim_text, "The sky is blue.")
        self.assertNotEqual(claim_set.candidate_claims[0].character_start, -1)
        self.assertEqual(claim_set.candidate_claims[0].metadata["match_type"], "exact")
        self.assertEqual(mock_llm.chat.call_count, 1)

    @patch("src.claim_decomposer.OpenAILike")
    def test_paraphrased_claim_resolves_via_fuzzy_match(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = make_llm_response(
            json.dumps([{"claim_text": "Water boils around 100 degrees Celsius", "sentence_id": "S1"}])
        )
        mock_llm_cls.return_value = mock_llm

        decomposer = ClaimDecomposer(model_name="mock-model")
        trace = make_trace(answer="The sky is blue. Water boils at 100 degrees Celsius when at sea level.")
        claim_set = decomposer.decompose(trace)

        claim = claim_set.candidate_claims[0]
        self.assertNotEqual(claim.character_start, -1)
        self.assertEqual(claim.metadata["match_type"], "fuzzy")
        self.assertGreaterEqual(claim.metadata["match_confidence"], 0.6)

    @patch("src.claim_decomposer.OpenAILike")
    def test_retry_path_on_invalid_first_response(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = [
            make_llm_response("This is not JSON at all."),
            make_llm_response(json.dumps([{"claim_text": "The sky is blue.", "sentence_id": "S1"}])),
        ]
        mock_llm_cls.return_value = mock_llm

        decomposer = ClaimDecomposer(model_name="mock-model")
        trace = make_trace()
        claim_set = decomposer.decompose(trace)

        self.assertEqual(mock_llm.chat.call_count, 2)
        self.assertEqual(claim_set.total_candidates, 1)
        self.assertTrue(claim_set.metadata["diagnostics"]["retry_attempt"])
        self.assertTrue(claim_set.metadata["diagnostics"]["success"])

    @patch("src.claim_decomposer.OpenAILike")
    def test_nvidia_uses_max_tokens_kwarg(self, mock_llm_cls):
        mock_llm_cls.return_value = MagicMock()
        ClaimDecomposer(model_name="mock-model")

        _, kwargs = mock_llm_cls.call_args
        self.assertIn("max_tokens", kwargs)

    @patch("src.claim_decomposer.OpenAILike")
    def test_nvidia_passes_api_base(self, mock_llm_cls):
        mock_llm_cls.return_value = MagicMock()
        ClaimDecomposer(model_name="mock-model")

        _, kwargs = mock_llm_cls.call_args
        self.assertIn("integrate.api.nvidia.com", kwargs.get("api_base", ""))

    @patch("src.claim_decomposer.OpenAILike")
    def test_both_attempts_fail_yields_zero_claims(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            make_llm_response("not json"),
            make_llm_response("still not json"),
        ]
        mock_llm_cls.return_value = mock_llm

        decomposer = ClaimDecomposer(model_name="mock-model")
        claim_set = decomposer.decompose(make_trace())

        self.assertEqual(claim_set.total_candidates, 0)
        self.assertFalse(claim_set.metadata["diagnostics"]["success"])


if __name__ == "__main__":
    unittest.main()
