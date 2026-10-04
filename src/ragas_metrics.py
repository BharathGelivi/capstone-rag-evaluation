"""
RAGAS-style Evaluation Metrics Module.

Implements a subset of the RAGAS metric suite (https://github.com/explodinggradients/ragas)
against this framework's own data model, reusing the already-loaded LLM, embedding model,
and NLI model rather than introducing the `ragas` package itself.

Reference-free metrics (no ground-truth answer needed):
    - Faithfulness: fraction of answer claims supported by retrieved context.
    - Answer Relevancy: how closely the answer aligns with the original question.
    - Context Precision: rank-weighted precision of retrieved chunks.
    - Context Relevancy: fraction of retrieved chunks relevant to the question.

Reference-based metrics (require a `reference` / ground-truth answer):
    - Context Recall: fraction of reference-answer content covered by retrieved context.
    - Answer Similarity: embedding similarity between answer and reference.
    - Answer Correctness: claim-level F1 between answer and reference, blended with similarity.
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from llama_index.core.llms import ChatMessage, MessageRole

from src.claim_verifier import ClaimVerifier, VerificationSummary
from src.retriever import RetrievedChunk
from src import rate_limiter
from configs.ragas import (
    ANSWER_RELEVANCY_NUM_QUESTIONS,
    ANSWER_CORRECTNESS_WEIGHTS,
    RAGAS_ENTAILMENT_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class RagasMetrics:
    """RAGAS-style scores for a single RAG execution. None means "not computed"
    (e.g. a reference-based metric with no reference answer supplied)."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None


import numpy as np


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Uses numpy BLAS routines instead of a pure-Python loop — 50-100x faster
    for typical embedding dimensions (384, 768).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def compute_faithfulness_strict(verification: VerificationSummary) -> Optional[float]:
    """Standard RAGAS faithfulness: |supported claims| / |total claims|.

    This is the canonical RAGAS definition — integer supported count, no half-credit
    for partially supported claims. Use this for paper comparisons.
    """
    if verification.total_claims == 0:
        return None
    return round(verification.supported_claims / verification.total_claims, 4)


def compute_faithfulness_weighted(verification: VerificationSummary) -> Optional[float]:
    """Weighted faithfulness: partially supported claims count as 0.5.

    This is a softer variant that gives partial credit. Retained for diagnostic
    value but should NOT be reported as RAGAS faithfulness in papers.
    """
    if verification.total_claims == 0:
        return None
    weighted = verification.supported_claims + 0.5 * verification.partially_supported_claims
    return round(weighted / verification.total_claims, 4)


# Default alias used by RagasEvaluator.evaluate() — strict is the research standard.
compute_faithfulness = compute_faithfulness_strict


class RagasEvaluator:
    """
    Computes RAGAS-style metrics using dependency-injected model instances so the
    heavy LLM/embedding/NLI models are loaded once and shared with the rest of the
    pipeline (Generator, Retriever, ClaimVerifier), not reloaded here.
    """

    def __init__(self, llm: Any, embed_model: Any, claim_verifier: Optional[ClaimVerifier] = None):
        self.llm = llm
        self.embed_model = embed_model
        self.claim_verifier = claim_verifier

    def _call_llm(self, prompt: str) -> str:
        try:
            response = rate_limiter.call(self.llm.chat, [ChatMessage(role=MessageRole.USER, content=prompt)])
            return str(response.message.content)
        except Exception as e:
            logger.error(f"RAGAS metric LLM call failed: {e}")
            return ""

    def compute_answer_relevancy(self, question: str, answer: str) -> Optional[float]:
        """
        Generates N synthetic "reverse" questions the answer could be responding to,
        embeds them alongside the original question, and returns the mean cosine
        similarity. Reference-free: hybrid LLM-generation + embedding-similarity.
        """
        if not answer.strip():
            return None

        prompt = (
            f"Generate {ANSWER_RELEVANCY_NUM_QUESTIONS} distinct questions that the following "
            "answer could plausibly be responding to. Return ONLY the questions, one per line, "
            "with no numbering or extra commentary.\n\n"
            f"Answer:\n{answer}"
        )
        response_text = self._call_llm(prompt)
        generated_questions = [q.strip("-* ").strip() for q in response_text.split("\n") if q.strip()]
        if not generated_questions:
            return None

        question_embedding = self.embed_model.get_text_embedding(question)
        similarities = [
            _cosine_similarity(question_embedding, self.embed_model.get_text_embedding(gq))
            for gq in generated_questions[:ANSWER_RELEVANCY_NUM_QUESTIONS]
        ]
        return round(sum(similarities) / len(similarities), 4) if similarities else None

    def _judge_relevance(self, prompt: str) -> Optional[bool]:
        answer = self._call_llm(prompt).strip().upper().rstrip(".! ")
        return True if answer == "YES" else False if answer == "NO" else None

    def compute_context_precision(self, question: str, answer: str, retrieved_chunks: List[RetrievedChunk]) -> Optional[float]:
        """
        Rank-weighted precision (average precision) of retrieved chunks: an LLM judges
        each chunk's usefulness to the answer, in retrieval rank order. Reference-free
        (uses the generated answer as the relevance signal, matching ragas' no-reference variant).
        """
        if not retrieved_chunks:
            return None

        relevance_flags = []
        for chunk in retrieved_chunks:
            prompt = (
                "Given the question and answer below, determine if the following context chunk "
                "was useful for producing the answer. Respond with only YES or NO.\n\n"
                f"Question: {question}\nAnswer: {answer}\nContext: {chunk.chunk_text}"
            )
            relevance_flags.append(self._judge_relevance(prompt))

        if any(flag is None for flag in relevance_flags):
            return None
        total_relevant = sum(relevance_flags)
        if total_relevant == 0:
            return 0.0

        running_relevant = 0
        precision_sum = 0.0
        for rank, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                running_relevant += 1
                precision_sum += running_relevant / rank
        return round(precision_sum / total_relevant, 4)

    def compute_context_relevancy(self, question: str, retrieved_chunks: List[RetrievedChunk]) -> Optional[float]:
        """
        Fraction of retrieved chunks judged relevant to the question alone (no answer
        needed). A simplified, chunk-level stand-in for ragas' sentence-level Context
        Relevancy. Reference-free.
        """
        if not retrieved_chunks:
            return None

        flags = []
        for chunk in retrieved_chunks:
            prompt = (
                "Does the following context contain information relevant to answering the "
                "question? Respond with only YES or NO.\n\n"
                f"Question: {question}\nContext: {chunk.chunk_text}"
            )
            flags.append(self._judge_relevance(prompt))
        return None if any(flag is None for flag in flags) else round(sum(flags) / len(flags), 4)

    def compute_context_recall(self, reference: str, retrieved_chunks: List[RetrievedChunk]) -> Optional[float]:
        """Fraction of reference-answer sentences entailed by the retrieved context.

        Each reference sentence is scored against **each chunk individually**.
        A sentence is considered "supported" if any single chunk entails it above
        the threshold. This prevents the 512-token truncation problem that occurs
        when all chunks are concatenated into a single premise: for 5 chunks of
        512 tokens each, DeBERTa's attention window would only see chunk 1.

        Requires a `claim_verifier` (reuses its NLI model). Reference-based.
        """
        if not reference.strip() or not retrieved_chunks:
            return None
        if self.claim_verifier is None:
            raise ValueError("A ClaimVerifier instance is required to compute context_recall.")

        reference_sentences = self.claim_verifier._split_into_sentences(reference)
        if not reference_sentences:
            return None

        supported_count = 0
        for sentence in reference_sentences:
            # A reference sentence is "covered" if any retrieved chunk entails it.
            is_covered = any(
                self.claim_verifier.run_nli(
                    premise=chunk.chunk_text,
                    hypothesis=sentence,
                )["entailment"] >= RAGAS_ENTAILMENT_THRESHOLD
                for chunk in retrieved_chunks
            )
            if is_covered:
                supported_count += 1

        return round(supported_count / len(reference_sentences), 4)

    def compute_answer_similarity(self, answer: str, reference: str) -> Optional[float]:
        """Embedding cosine similarity between answer and reference. Reference-based."""
        if not answer.strip() or not reference.strip():
            return None
        answer_embedding = self.embed_model.get_text_embedding(answer)
        reference_embedding = self.embed_model.get_text_embedding(reference)
        return round(_cosine_similarity(answer_embedding, reference_embedding), 4)

    def compute_answer_correctness(
        self, answer: str, reference: str, answer_similarity: Optional[float] = None
    ) -> Optional[float]:
        """
        Claim-level F1 between answer and reference sentences (via NLI entailment,
        reusing claim_verifier's model), blended with Answer Similarity using
        ANSWER_CORRECTNESS_WEIGHTS. Requires a `claim_verifier`. Reference-based.
        """
        if not answer.strip() or not reference.strip():
            return None
        if self.claim_verifier is None:
            raise ValueError("A ClaimVerifier instance is required to compute answer_correctness.")

        answer_sentences = self.claim_verifier._split_into_sentences(answer)
        reference_sentences = self.claim_verifier._split_into_sentences(reference)
        if not answer_sentences or not reference_sentences:
            return None

        combined_reference = " ".join(reference_sentences)
        combined_answer = " ".join(answer_sentences)

        true_positives = sum(
            1 for s in answer_sentences
            if self.claim_verifier.run_nli(premise=combined_reference, hypothesis=s)["entailment"] >= RAGAS_ENTAILMENT_THRESHOLD
        )
        false_positives = len(answer_sentences) - true_positives
        false_negatives = sum(
            1 for s in reference_sentences
            if self.claim_verifier.run_nli(premise=combined_answer, hypothesis=s)["entailment"] < RAGAS_ENTAILMENT_THRESHOLD
        )

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        similarity = answer_similarity
        if similarity is None:
            similarity = self.compute_answer_similarity(answer, reference) or 0.0

        f1_weight, similarity_weight = ANSWER_CORRECTNESS_WEIGHTS
        return round(f1_weight * f1 + similarity_weight * similarity, 4)

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
        verification: VerificationSummary,
        reference: Optional[str] = None,
    ) -> RagasMetrics:
        """Computes all applicable metrics; reference-based ones are left as None if reference is not given."""
        metrics = RagasMetrics(
            faithfulness=compute_faithfulness_strict(verification),
            answer_relevancy=self.compute_answer_relevancy(question, answer),
            context_precision=self.compute_context_precision(question, answer, retrieved_chunks),
            context_relevancy=self.compute_context_relevancy(question, retrieved_chunks),
        )

        if reference:
            metrics.context_recall = self.compute_context_recall(reference, retrieved_chunks)
            metrics.answer_similarity = self.compute_answer_similarity(answer, reference)
            metrics.answer_correctness = self.compute_answer_correctness(answer, reference, metrics.answer_similarity)

        return metrics
