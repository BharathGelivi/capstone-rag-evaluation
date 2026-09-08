import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from src.answer_correctness_evaluator import (
    AnswerCorrectnessEvaluator,
    AnswerCorrectnessSummary,
    GoldClaimResult,
)
from src.claim_decomposer import CandidateClaim, CandidateClaimSet
from src.claim_verifier import VerificationStatus, EvidenceSentence

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "answer_correctness_fixtures.json")


def make_gold_claim_set(trace_id, claim_texts):
    claim_set = CandidateClaimSet(trace_id=trace_id)
    for i, text in enumerate(claim_texts):
        claim_set.add_claim(CandidateClaim(
            candidate_id=f"gold_c{i}", trace_id=trace_id, claim_text=text,
            sentence_id=f"S{i}", claim_index=i,
            character_start=0, character_end=len(text), metadata={}
        ))
    return claim_set


def make_verify_outcome(status: VerificationStatus, sentence_text="Matching sentence.", confidence=0.9):
    best = EvidenceSentence(
        sentence_id="generated_answer_s0", chunk_id="generated_answer", chunk_rank=0,
        text=sentence_text, entailment_score=confidence, contradiction_score=0.0, neutral_score=0.05,
    ) if status != VerificationStatus.UNSUPPORTED or sentence_text else None
    return {
        "status": status, "reason": "mock", "confidence": confidence,
        "entailment": confidence, "contradiction": 0.0, "neutral": 0.05,
        "best_sentence": best, "top_3": [best] if best else [],
    }


class TestDecomposeGoldAnswer(unittest.TestCase):
    def test_builds_synthetic_trace_with_gold_answer_as_generated_answer(self):
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", [])
        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=MagicMock())

        evaluator._decompose_gold_answer("The gold answer text.", "T1")

        mock_decomposer.decompose.assert_called_once()
        synthetic_trace = mock_decomposer.decompose.call_args[0][0]
        self.assertEqual(synthetic_trace.generated_answer, "The gold answer text.")
        self.assertEqual(synthetic_trace.trace_id, "T1")


class TestEvaluate(unittest.TestCase):
    def test_claim_recall_all_supported(self):
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", ["Claim A.", "Claim B."])
        mock_verifier = MagicMock()
        mock_verifier._verify_against_sources.return_value = make_verify_outcome(VerificationStatus.SUPPORTED)

        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=mock_verifier)
        summary = evaluator.evaluate("Generated answer.", "Gold answer.", "T1")

        self.assertEqual(summary.total_gold_claims, 2)
        self.assertEqual(summary.claim_recall, 1.0)
        self.assertEqual(len(summary.results), 2)
        self.assertTrue(all(r.verification_status == "SUPPORTED" for r in summary.results))

    def test_claim_recall_partial(self):
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", ["Claim A.", "Claim B."])
        mock_verifier = MagicMock()
        mock_verifier._verify_against_sources.side_effect = [
            make_verify_outcome(VerificationStatus.SUPPORTED),
            make_verify_outcome(VerificationStatus.UNSUPPORTED, sentence_text=None),
        ]

        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=mock_verifier)
        summary = evaluator.evaluate("Generated answer.", "Gold answer.", "T1")

        self.assertEqual(summary.claim_recall, 0.5)

    def test_partially_supported_counts_toward_recall(self):
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", ["Claim A."])
        mock_verifier = MagicMock()
        mock_verifier._verify_against_sources.return_value = make_verify_outcome(VerificationStatus.PARTIALLY_SUPPORTED)

        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=mock_verifier)
        summary = evaluator.evaluate("Generated answer.", "Gold answer.", "T1")

        self.assertEqual(summary.claim_recall, 1.0)

    def test_contradicted_does_not_count_toward_recall(self):
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", ["Claim A."])
        mock_verifier = MagicMock()
        mock_verifier._verify_against_sources.return_value = make_verify_outcome(VerificationStatus.CONTRADICTED)

        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=mock_verifier)
        summary = evaluator.evaluate("Generated answer.", "Gold answer.", "T1")

        self.assertEqual(summary.claim_recall, 0.0)

    def test_zero_gold_claims_does_not_crash(self):
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", [])
        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=MagicMock())

        summary = evaluator.evaluate("Generated answer.", "Gold answer.", "T1")

        self.assertEqual(summary.total_gold_claims, 0)
        self.assertEqual(summary.claim_recall, 0.0)
        self.assertEqual(summary.results, [])

    def test_verifier_called_with_generated_answer_as_sole_evidence_source(self):
        """Step B: gold claims must be checked against the generated answer's
        sentences, not against any retrieved chunk set."""
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = make_gold_claim_set("T1", ["Claim A."])
        mock_verifier = MagicMock()
        mock_verifier._verify_against_sources.return_value = make_verify_outcome(VerificationStatus.SUPPORTED)

        evaluator = AnswerCorrectnessEvaluator(decomposer=mock_decomposer, verifier=mock_verifier)
        evaluator.evaluate("The generated answer text.", "Gold answer.", "T1")

        call_args = mock_verifier._verify_against_sources.call_args
        claim_text_arg, sources_arg = call_args[0]
        self.assertEqual(claim_text_arg, "Claim A.")
        self.assertEqual(sources_arg, [("generated_answer", 0, "The generated answer text.")])


class TestArtifactPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        summary = AnswerCorrectnessSummary(
            trace_id="T1", gold_answer="Gold.", total_gold_claims=1, claim_recall=1.0,
            results=[GoldClaimResult(claim_id="c1", claim_text="Claim.", verification_status="SUPPORTED",
                                      best_matching_sentence="Match.", confidence=0.9)]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = summary.save(base_dir=tmpdir)
            self.assertTrue(os.path.exists(path))
            self.assertIn("TRACE_T1.json", path)

            reloaded = AnswerCorrectnessSummary.load(path)
            self.assertEqual(reloaded.trace_id, "T1")
            self.assertEqual(reloaded.claim_recall, 1.0)
            self.assertEqual(reloaded.results[0].claim_text, "Claim.")


class TestFixturesFile(unittest.TestCase):
    def test_fixtures_load_and_have_expected_shape(self):
        with open(FIXTURES_PATH, encoding="utf-8") as f:
            fixtures = json.load(f)

        self.assertEqual(len(fixtures), 3)
        fixture_ids = {f["fixture_id"] for f in fixtures}
        self.assertEqual(fixture_ids, {
            "trace_a_private_defence_complete",
            "trace_b_abetment_complete",
            "trace_b_abetment_incomplete",
        })
        for fixture in fixtures:
            self.assertIn("trace_reference", fixture)
            self.assertIn("gold_answer", fixture)
            self.assertIn("expected_behavior", fixture)


if __name__ == "__main__":
    unittest.main()
