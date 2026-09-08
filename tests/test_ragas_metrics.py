import unittest
from unittest.mock import patch, MagicMock

from src.claim_verifier import ClaimVerifier, VerificationSummary
from src.retriever import RetrievedChunk
from src.ragas_metrics import (
    RagasEvaluator,
    compute_faithfulness,
    compute_faithfulness_weighted,
)


def make_verification_summary(total, supported=0, partial=0, contradicted=0, unsupported=0, not_verifiable=0):
    return VerificationSummary(
        trace_id="t1",
        total_claims=total,
        supported_claims=supported,
        partially_supported_claims=partial,
        contradicted_claims=contradicted,
        unsupported_claims=unsupported,
        not_verifiable_claims=not_verifiable,
        average_entailment_score=0.5,
        total_verification_latency_ms=0.0,
        results=[],
    )


def make_chunk(chunk_id, text, rank=1):
    return RetrievedChunk(
        chunk_id=chunk_id, similarity_score=0.9, rank=rank, page_number="1",
        source_file="doc.pdf", chunk_index=0, chunk_text=text,
    )


def make_llm(response_text):
    llm = MagicMock()
    response = MagicMock()
    response.message.content = response_text
    llm.chat.return_value = response
    return llm


class TestComputeFaithfulness(unittest.TestCase):
    def test_weighted_supported_and_partial(self):
        verification = make_verification_summary(total=4, supported=2, partial=1, unsupported=1)
        self.assertEqual(compute_faithfulness_weighted(verification), 0.625)

    def test_zero_claims_returns_none(self):
        verification = make_verification_summary(total=0)
        self.assertIsNone(compute_faithfulness(verification))


class TestRagasEvaluator(unittest.TestCase):
    def setUp(self):
        self.embed_model = MagicMock()
        # Deterministic embeddings: identical strings get identical vectors.
        vectors = {
            "What is X?": [1.0, 0.0],
            "What is X? (paraphrase)": [1.0, 0.0],
            "Unrelated question": [0.0, 1.0],
            "Answer text.": [1.0, 0.0],
            "Reference text.": [1.0, 0.0],
            "Different reference.": [0.0, 1.0],
        }
        self.embed_model.get_text_embedding.side_effect = lambda text: vectors.get(text, [0.5, 0.5])

    def test_compute_answer_relevancy_high_similarity(self):
        llm = make_llm("What is X?\nWhat is X? (paraphrase)")
        evaluator = RagasEvaluator(llm=llm, embed_model=self.embed_model)

        score = evaluator.compute_answer_relevancy("What is X?", "Answer text.")
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_compute_answer_relevancy_empty_answer_returns_none(self):
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model)
        self.assertIsNone(evaluator.compute_answer_relevancy("Q?", ""))

    def test_compute_context_precision_rank_weighted(self):
        # YES, NO, YES -> relevant at ranks 1 and 3. AP = (1/1 + 2/3) / 2
        llm = MagicMock()
        responses = [make_llm("YES").chat.return_value, make_llm("NO").chat.return_value, make_llm("YES").chat.return_value]
        llm.chat.side_effect = responses

        evaluator = RagasEvaluator(llm=llm, embed_model=self.embed_model)
        chunks = [make_chunk("c1", "text1", 1), make_chunk("c2", "text2", 2), make_chunk("c3", "text3", 3)]

        score = evaluator.compute_context_precision("Q?", "A.", chunks)
        expected = (1 / 1 + 2 / 3) / 2
        self.assertAlmostEqual(score, round(expected, 4), places=3)

    def test_compute_context_precision_no_relevant_chunks(self):
        llm = MagicMock()
        llm.chat.return_value = make_llm("NO").chat.return_value
        evaluator = RagasEvaluator(llm=llm, embed_model=self.embed_model)

        score = evaluator.compute_context_precision("Q?", "A.", [make_chunk("c1", "text1")])
        self.assertEqual(score, 0.0)

    def test_compute_context_relevancy_fraction(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            make_llm("YES").chat.return_value,
            make_llm("YES").chat.return_value,
            make_llm("NO").chat.return_value,
        ]
        evaluator = RagasEvaluator(llm=llm, embed_model=self.embed_model)
        chunks = [make_chunk("c1", "t1"), make_chunk("c2", "t2"), make_chunk("c3", "t3")]

        score = evaluator.compute_context_relevancy("Q?", chunks)
        self.assertAlmostEqual(score, 2 / 3, places=3)

    def test_compute_answer_similarity(self):
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model)
        score = evaluator.compute_answer_similarity("Answer text.", "Reference text.")
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_compute_answer_similarity_different_vectors(self):
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model)
        score = evaluator.compute_answer_similarity("Answer text.", "Different reference.")
        self.assertAlmostEqual(score, 0.0, places=3)

    def test_context_recall_and_answer_correctness_require_claim_verifier(self):
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model)
        with self.assertRaises(ValueError):
            evaluator.compute_context_recall("Reference sentence.", [make_chunk("c1", "text")])
        with self.assertRaises(ValueError):
            evaluator.compute_answer_correctness("Answer.", "Reference.")


class TestRagasEvaluatorWithClaimVerifier(unittest.TestCase):
    @patch("src.claim_verifier.pipeline")
    def setUp(self, mock_pipeline):
        self.mock_nli = MagicMock()
        mock_pipeline.return_value = self.mock_nli
        self.claim_verifier = ClaimVerifier(model_name="mock_model")
        self.embed_model = MagicMock()
        self.embed_model.get_text_embedding.side_effect = lambda text: [1.0, 0.0]

    def test_compute_context_recall_all_supported(self):
        # High entailment for every sentence.
        self.mock_nli.side_effect = lambda kwargs: [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "contradiction", "score": 0.05},
        ]
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model, claim_verifier=self.claim_verifier)
        chunks = [make_chunk("c1", "Some supporting context.")]

        score = evaluator.compute_context_recall("The reference answer is true.", chunks)
        self.assertEqual(score, 1.0)

    def test_compute_context_recall_none_supported(self):
        self.mock_nli.side_effect = lambda kwargs: [
            {"label": "entailment", "score": 0.05},
            {"label": "neutral", "score": 0.9},
            {"label": "contradiction", "score": 0.05},
        ]
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model, claim_verifier=self.claim_verifier)
        chunks = [make_chunk("c1", "Unrelated context.")]

        score = evaluator.compute_context_recall("The reference answer is true.", chunks)
        self.assertEqual(score, 0.0)

    def test_compute_answer_correctness_perfect_match(self):
        self.mock_nli.side_effect = lambda kwargs: [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "contradiction", "score": 0.05},
        ]
        evaluator = RagasEvaluator(llm=make_llm(""), embed_model=self.embed_model, claim_verifier=self.claim_verifier)

        score = evaluator.compute_answer_correctness("Answer text.", "Reference text.")
        # Perfect entailment both ways -> F1=1.0; embeddings identical -> similarity=1.0
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_evaluate_without_reference_leaves_reference_based_none(self):
        self.mock_nli.side_effect = lambda kwargs: [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "contradiction", "score": 0.05},
        ]
        llm = make_llm("Q1?\nQ2?\nQ3?")
        evaluator = RagasEvaluator(llm=llm, embed_model=self.embed_model, claim_verifier=self.claim_verifier)
        verification = make_verification_summary(total=2, supported=2)
        chunks = [make_chunk("c1", "Some context.")]

        metrics = evaluator.evaluate("Q?", "Answer text.", chunks, verification)

        self.assertIsNotNone(metrics.faithfulness)
        self.assertIsNotNone(metrics.answer_relevancy)
        self.assertIsNone(metrics.context_recall)
        self.assertIsNone(metrics.answer_similarity)
        self.assertIsNone(metrics.answer_correctness)

    def test_evaluate_with_reference_populates_all(self):
        self.mock_nli.side_effect = lambda kwargs: [
            {"label": "entailment", "score": 0.9},
            {"label": "neutral", "score": 0.05},
            {"label": "contradiction", "score": 0.05},
        ]
        llm = make_llm("Q1?\nQ2?\nQ3?")
        evaluator = RagasEvaluator(llm=llm, embed_model=self.embed_model, claim_verifier=self.claim_verifier)
        verification = make_verification_summary(total=2, supported=2)
        chunks = [make_chunk("c1", "Some context.")]

        metrics = evaluator.evaluate("Q?", "Answer text.", chunks, verification, reference="Reference text.")

        self.assertIsNotNone(metrics.context_recall)
        self.assertIsNotNone(metrics.answer_similarity)
        self.assertIsNotNone(metrics.answer_correctness)


if __name__ == "__main__":
    unittest.main()
