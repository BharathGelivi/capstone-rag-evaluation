"""
Claim Verification Module (Phase 2.3).

Responsible for independently verifying atomic claims against retrieved evidence chunks
using a Natural Language Inference (NLI) model at the sentence level.
"""

import os
import json
import time
import uuid
import re
import logging
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole

from src.rag_trace import RAGTrace
from src.retriever import RetrievedChunk
from src.claim_decomposer import CandidateClaim, CandidateClaimSet
from src import rate_limiter
from configs.models import (
    VERIFICATION_MODEL,
    VERIFICATION_BATCH_SIZE,
    VERIFICATION_GPU_BATCH_SIZE,
    NVIDIA_CLAIM_DECOMPOSER_MODEL,
    NVIDIA_BASE_URL,
    LLM_TEMPERATURE,
)
from configs.pipeline import CLAIM_JUDGE_MAX_TOKENS
from configs.prompts import CLAIM_JUDGE_SYSTEM_INSTRUCTION, DETAILED_THINKING_OFF
from src.device import get_hf_device_index, describe_device
from configs.thresholds import (
    ENTAILMENT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    PARTIAL_SUPPORT_THRESHOLD,
    NEUTRAL_UNSUPPORTED_THRESHOLD,
    EVIDENCE_AGGREGATION_STRATEGY
)

VALID_AGGREGATION_STRATEGIES = ("top1", "max_pool_top3", "concat_top3")

try:
    from transformers import pipeline
except ImportError:
    pass

try:
    import nltk
    # Ensure the sentence tokenizer corpus is available; download silently if not.
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


# The NLI model is reliable at the extremes (high entailment, high
# contradiction) and least reliable exactly where it lands in one of these
# two -- weak/ambiguous evidence. Only these statuses trigger the (much more
# expensive) LLM-judge escalation in verify_claim().
AMBIGUOUS_STATUSES = frozenset({
    VerificationStatus.PARTIALLY_SUPPORTED,
    VerificationStatus.NOT_VERIFIABLE,
})


@dataclass
class EvidenceSentence:
    sentence_id: str
    chunk_id: str
    chunk_rank: int
    text: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float


@dataclass
class VerificationResult:
    verification_id: str
    trace_id: str
    claim_id: str
    claim_text: str
    verification_status: VerificationStatus
    verification_reason: str
    confidence: float
    
    # Best single piece of evidence mapping
    best_chunk_id: Optional[str]
    best_chunk_rank: Optional[int]
    best_chunk_score: Optional[float]
    best_sentence_id: Optional[str]
    evidence_text: Optional[str]
    
    # Best evidence NLI scores
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    
    # Top 3 supporting sentences
    top_evidence: List[EvidenceSentence] = field(default_factory=list)
    
    verification_latency_ms: float = 0.0
    verification_model: str = VERIFICATION_MODEL
    # "nli" (fast path) or "llm" (escalated -- see AMBIGUOUS_STATUSES).
    verified_by: str = "nli"


@dataclass
class VerificationSummary:
    trace_id: str
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    not_verifiable_claims: int
    average_entailment_score: float
    total_verification_latency_ms: float
    results: List[VerificationResult] = field(default_factory=list)

    def to_json(self) -> str:
        # Convert enums to strings for json serialization
        data = asdict(self)
        for res in data["results"]:
            res["verification_status"] = res["verification_status"].value
        return json.dumps(data, indent=4)


class ClaimVerifier:
    def __init__(
        self,
        model_name: str = VERIFICATION_MODEL,
        aggregation_strategy: str = EVIDENCE_AGGREGATION_STRATEGY,
        enable_llm_judge: bool = True,
    ):
        if aggregation_strategy not in VALID_AGGREGATION_STRATEGIES:
            raise ValueError(
                f"Invalid aggregation_strategy '{aggregation_strategy}'. "
                f"Must be one of {VALID_AGGREGATION_STRATEGIES}."
            )
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        # Hybrid verification: NLI decides every claim; only claims it lands
        # in AMBIGUOUS_STATUSES get escalated to an LLM judge call (see
        # verify_claim()). False for tests that don't want a real network
        # call triggered by a deliberately-ambiguous fixture.
        self.enable_llm_judge = enable_llm_judge
        self.judge_llm = OpenAILike(
            model=NVIDIA_CLAIM_DECOMPOSER_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=CLAIM_JUDGE_MAX_TOKENS,
            api_key=os.environ.get("NVIDIA_API_KEY", ""),
            api_base=NVIDIA_BASE_URL,
            is_chat_model=True,
            # See src/rate_limiter.py: retries must go through call(), not the
            # client's own blind retry, or they burst past the shared budget.
            max_retries=0,
        ) if enable_llm_judge else None
        device_index = get_hf_device_index()
        logger.info(
            "Loading NLI Verification Model: %s on %s",
            self.model_name,
            describe_device(),
        )

        # Load zero-shot classification pipeline.
        # Note: For deberta-v3-large-zeroshot-v2, the labels are entailment, neutral, contradiction.
        # Using standard text-classification for NLI formats: [Premise][SEP][Hypothesis]
        #
        # deberta-v3-large is ~1.6 GB in fp32. On GPU it runs in fp16 — roughly
        # half the VRAM and materially faster, at no measurable cost to the
        # entailment/contradiction decision. CPU stays fp32 because fp16 on CPU
        # is emulated and slower than the baseline.
        pipeline_kwargs = {}
        if device_index >= 0:
            import torch

            pipeline_kwargs["dtype"] = torch.float16

        self.nli_pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            top_k=None,  # return scores for all labels (return_all_scores is deprecated in transformers>=5)
            truncation=True,
            device=device_index,
            **pipeline_kwargs,
        )

        self.batch_size = (
            VERIFICATION_GPU_BATCH_SIZE if device_index >= 0 else VERIFICATION_BATCH_SIZE
        )
            
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK's punkt tokenizer.

        NLTK sent_tokenize correctly handles:
        - Abbreviations ("Mr. Smith", "Rs. 500", "Sec. 399")
        - Decimal numbers ("Rs.500 is...")
        - Legal section references ("399.(1) This section...")
        - Numbered lists and multi-sentence paragraphs

        Falls back to the regex approach only if NLTK is unavailable.
        """
        text = text.strip()
        if not text:
            return []
        if _NLTK_AVAILABLE:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        # Fallback: regex (known limitations with abbreviations)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _determine_status_and_reason(self, entailment: float, contradiction: float, neutral: float) -> tuple[VerificationStatus, str]:
        """Resolves the scores into a final status and generates a reason."""
        if contradiction >= CONTRADICTION_THRESHOLD:
            return VerificationStatus.CONTRADICTED, f"Evidence explicitly contradicts the claim (Score: {contradiction:.2f})."
        elif entailment >= ENTAILMENT_THRESHOLD:
            return VerificationStatus.SUPPORTED, f"Evidence directly supports the claim (Score: {entailment:.2f})."
        elif entailment >= PARTIAL_SUPPORT_THRESHOLD:
            return VerificationStatus.PARTIALLY_SUPPORTED, f"Evidence provides partial/weak support (Score: {entailment:.2f})."
        elif neutral >= NEUTRAL_UNSUPPORTED_THRESHOLD:
            return VerificationStatus.UNSUPPORTED, f"Evidence is neutral and does not support the claim (Neutral Score: {neutral:.2f})."
        else:
            return VerificationStatus.NOT_VERIFIABLE, "Evidence is insufficient for conclusive verification."

    def run_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Runs the NLI pipeline on a single premise/hypothesis pair and returns scores."""
        # NLI Models usually expect premise and hypothesis combined or as text/text_pair
        # We use text=premise, text_pair=hypothesis in pipeline
        result = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
        
        # Extract scores (handling varying label names across different NLI models)
        scores = {item['label'].lower(): item['score'] for item in result}
        
        entailment = scores.get("entailment", scores.get("label_0", 0.0))
        neutral = scores.get("neutral", scores.get("label_1", 0.0))
        contradiction = scores.get("contradiction", scores.get("label_2", 0.0))
        
        return {
            "entailment": entailment,
            "neutral": neutral,
            "contradiction": contradiction
        }

    def _llm_judge(self, claim_text: str, evidence_text: str) -> Optional[tuple[VerificationStatus, str]]:
        """Escalation path for claims the NLI model left ambiguous.

        Returns (status, reason) on a clean parse, or None on any failure
        (bad response shape, unknown status label, network/API error) -- the
        caller falls back to the NLI verdict rather than ever letting a judge
        failure take down verification for the whole claim.
        """
        try:
            response = rate_limiter.call(self.judge_llm.chat, [
                ChatMessage(role=MessageRole.SYSTEM, content=DETAILED_THINKING_OFF),
                ChatMessage(role=MessageRole.SYSTEM, content=CLAIM_JUDGE_SYSTEM_INSTRUCTION),
                ChatMessage(
                    role=MessageRole.USER,
                    content=f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_text}",
                ),
            ])
            text = str(response.message.content)
        except Exception as e:
            logger.warning("LLM judge call failed, falling back to NLI verdict: %s", e)
            return None

        status_match = re.search(r"STATUS:\s*(\w+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE)
        if not status_match:
            logger.warning("LLM judge returned unparseable response, falling back to NLI verdict: %r", text)
            return None

        raw_status = status_match.group(1).strip().upper()
        try:
            status = VerificationStatus(raw_status)
        except ValueError:
            logger.warning("LLM judge returned unknown status '%s', falling back to NLI verdict", raw_status)
            return None

        reason = reason_match.group(1).strip() if reason_match else "LLM judge did not provide a reason."
        return status, f"[LLM judge] {reason}"

    def _score_text_sources_against_claim(self, claim_text: str, sources: List[tuple]) -> List[EvidenceSentence]:
        """
        Generic sentence-scoring core: splits each (source_id, source_rank, text)
        tuple into sentences and NLI-scores every sentence against claim_text.
        Shared by find_best_evidence (evidence = retrieved chunks) and
        AnswerCorrectnessEvaluator (evidence = the generated answer text) --
        neither should reimplement this NLI-calling loop.
        """
        inputs = []
        meta = []

        for source_id, source_rank, text in sources:
            sentences = self._split_into_sentences(text)
            for i, sentence in enumerate(sentences):
                inputs.append({"text": sentence, "text_pair": claim_text})
                meta.append((source_id, source_rank, sentence, f"{source_id}_s{i}"))

        if not inputs:
            return []

        # Run batched NLI. On GPU the batch is the whole point — sentence-level
        # scoring generates hundreds of short pairs per claim, and a batch of 8
        # leaves the card idle. CPU keeps the conservative default.
        results = self.nli_pipeline(inputs, batch_size=self.batch_size)
        
        all_scored_sentences: List[EvidenceSentence] = []
        for (source_id, source_rank, sentence, s_id), result in zip(meta, results):
            scores = {item['label'].lower(): item['score'] for item in result}
            
            entailment = scores.get("entailment", scores.get("label_0", 0.0))
            neutral = scores.get("neutral", scores.get("label_1", 0.0))
            contradiction = scores.get("contradiction", scores.get("label_2", 0.0))

            all_scored_sentences.append(
                EvidenceSentence(
                    sentence_id=s_id,
                    chunk_id=source_id,
                    chunk_rank=source_rank,
                    text=sentence,
                    entailment_score=entailment,
                    contradiction_score=contradiction,
                    neutral_score=neutral
                )
            )

        # Sort sentences by entailment score descending
        all_scored_sentences.sort(key=lambda x: x.entailment_score, reverse=True)
        return all_scored_sentences

    def find_best_evidence(self, claim: CandidateClaim, chunks: List[RetrievedChunk]) -> List[EvidenceSentence]:
        """Finds and scores all sentences in chunks against a claim, returning all scored sentences."""
        sources = [(chunk.chunk_id, chunk.rank, chunk.chunk_text) for chunk in chunks]
        return self._score_text_sources_against_claim(claim.claim_text, sources)

    def _aggregate_top3(self, top_3: List[EvidenceSentence], claim_text: str) -> Dict[str, float]:
        """Combine the top-3 scored sentences into a single NLI score triplet.

        max_pool_top3 strategy: use the sentence with the highest entailment score
        and return its complete (entailment, contradiction, neutral) triplet.
        Taking the maximum of each score independently across sentences produces
        values that can sum to >1.0, which are not valid probabilities and corrupt
        downstream threshold comparisons.

        concat_top3 strategy: concatenate the top-3 sentences and run NLI once
        on the combined premise. Valid distribution by construction.
        """
        if self.aggregation_strategy == "max_pool_top3":
            # Use the argmax-by-entailment sentence's full triplet.
            # This is identical to top1 for entailment, but we keep the strategy
            # distinct so ablation scripts can track it.
            best = max(top_3, key=lambda s: s.entailment_score)
            return {
                "entailment": best.entailment_score,
                "contradiction": best.contradiction_score,
                "neutral": best.neutral_score,
            }
        elif self.aggregation_strategy == "concat_top3":
            combined_premise = " ".join(s.text for s in top_3)
            return self.run_nli(premise=combined_premise, hypothesis=claim_text)
        raise ValueError(f"_aggregate_top3 called with unsupported strategy '{self.aggregation_strategy}'")

    def _verify_against_sources(self, claim_text: str, sources: List[tuple]) -> Dict[str, Any]:
        """
        Generic verification core shared by verify_claim (evidence = retrieved
        chunks) and AnswerCorrectnessEvaluator (evidence = the generated
        answer text): scores all sentences in `sources` against claim_text,
        aggregates the top-3 per self.aggregation_strategy, and resolves a
        final status/reason/confidence. Returns a plain dict rather than a
        VerificationResult since callers attach different identity fields
        (trace_id/claim_id vs. gold-claim-specific fields).
        """
        all_scored_sentences = self._score_text_sources_against_claim(claim_text, sources)

        top_3 = all_scored_sentences[:3]
        best_sentence = top_3[0] if top_3 else None

        if not best_sentence:
            return {
                "status": VerificationStatus.UNSUPPORTED,
                "reason": "No evidence sentences found.",
                "confidence": 0.0,
                "entailment": 0.0, "contradiction": 0.0, "neutral": 0.0,
                "best_sentence": None, "top_3": [],
            }

        if self.aggregation_strategy == "top1":
            entailment = best_sentence.entailment_score
            contradiction = best_sentence.contradiction_score
            neutral = best_sentence.neutral_score
        else:
            agg_scores = self._aggregate_top3(top_3, claim_text)
            entailment = agg_scores["entailment"]
            contradiction = agg_scores["contradiction"]
            neutral = agg_scores["neutral"]

        status, reason = self._determine_status_and_reason(entailment, contradiction, neutral)
        confidence = max(entailment, contradiction)

        return {
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "entailment": entailment, "contradiction": contradiction, "neutral": neutral,
            "best_sentence": best_sentence, "top_3": top_3,
        }

    def compute_all_aggregation_strategies(self, claim_text: str, sources: List[tuple]) -> Dict[str, Dict[str, Any]]:
        """
        Ablation helper (see scripts/ablate_aggregation_strategy.py): scores
        evidence sentences ONCE, then reports what each of the three
        aggregation strategies (top1 / max_pool_top3 / concat_top3) would
        have decided from that same scoring pass -- avoids re-running the
        expensive per-sentence NLI scoring loop three times just to compare
        strategies. Only concat_top3 needs one additional NLI call (on the
        concatenated top-3 premise); top1 and max_pool_top3 are free
        by-products of the same sentence scores.

        Returns {"best_sentence": EvidenceSentence or None,
        "top1": {...}, "max_pool_top3": {...}, "concat_top3": {...}}, each
        strategy dict with status/entailment/contradiction/neutral.
        best_sentence (and therefore its chunk_id/chunk_rank) is the same
        regardless of strategy -- only the final aggregated status differs.
        """
        all_scored_sentences = self._score_text_sources_against_claim(claim_text, sources)
        top_3 = all_scored_sentences[:3]
        best_sentence = top_3[0] if top_3 else None

        if not best_sentence:
            empty = {"status": VerificationStatus.UNSUPPORTED.value, "entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            return {"best_sentence": None, "top1": dict(empty), "max_pool_top3": dict(empty), "concat_top3": dict(empty)}

        results: Dict[str, Dict[str, Any]] = {}

        status, _ = self._determine_status_and_reason(
            best_sentence.entailment_score, best_sentence.contradiction_score, best_sentence.neutral_score
        )
        results["top1"] = {
            "status": status.value,
            "entailment": best_sentence.entailment_score,
            "contradiction": best_sentence.contradiction_score,
            "neutral": best_sentence.neutral_score,
        }

        max_pool = {
            "entailment": max(s.entailment_score for s in top_3),
            "contradiction": max(s.contradiction_score for s in top_3),
            "neutral": max(s.neutral_score for s in top_3),
        }
        status, _ = self._determine_status_and_reason(max_pool["entailment"], max_pool["contradiction"], max_pool["neutral"])
        results["max_pool_top3"] = {"status": status.value, **max_pool}

        combined_premise = " ".join(s.text for s in top_3)
        concat_scores = self.run_nli(premise=combined_premise, hypothesis=claim_text)
        status, _ = self._determine_status_and_reason(concat_scores["entailment"], concat_scores["contradiction"], concat_scores["neutral"])
        results["concat_top3"] = {"status": status.value, **concat_scores}

        results["best_sentence"] = best_sentence
        return results

    def verify_claim(self, claim: CandidateClaim, chunks: List[RetrievedChunk]) -> VerificationResult:
        """Verifies a single claim against all retrieved chunks at the sentence level."""
        t_start = time.time()

        sources = [(chunk.chunk_id, chunk.rank, chunk.chunk_text) for chunk in chunks]
        outcome = self._verify_against_sources(claim.claim_text, sources)
        best_sentence = outcome["best_sentence"]
        verified_by = "nli"

        if best_sentence:
            # Hybrid escalation: NLI resolves the clear-cut cases (high
            # entailment or high contradiction) essentially for free; only the
            # ambiguous middle -- exactly where NLI is least trustworthy --
            # pays for an LLM judge call. See AMBIGUOUS_STATUSES.
            if self.enable_llm_judge and outcome["status"] in AMBIGUOUS_STATUSES:
                evidence_text = " ".join(s.text for s in outcome["top_3"])
                judged = self._llm_judge(claim.claim_text, evidence_text)
                if judged is not None:
                    outcome["status"], outcome["reason"] = judged
                    verified_by = "llm"

            res = VerificationResult(
                verification_id=str(uuid.uuid4()),
                trace_id=claim.trace_id,
                claim_id=claim.candidate_id,
                claim_text=claim.claim_text,
                verification_status=outcome["status"],
                verification_reason=outcome["reason"],
                confidence=outcome["confidence"],
                best_chunk_id=best_sentence.chunk_id,
                best_chunk_rank=best_sentence.chunk_rank,
                best_chunk_score=None, # Needs to be passed if available in chunk metadata
                best_sentence_id=best_sentence.sentence_id,
                evidence_text=best_sentence.text,
                entailment_score=outcome["entailment"],
                contradiction_score=outcome["contradiction"],
                neutral_score=outcome["neutral"],
                top_evidence=outcome["top_3"],
                verification_latency_ms=(time.time() - t_start) * 1000,
                verified_by=verified_by,
            )
        else:
            # Fallback if no evidence exists
            res = VerificationResult(
                verification_id=str(uuid.uuid4()),
                trace_id=claim.trace_id,
                claim_id=claim.candidate_id,
                claim_text=claim.claim_text,
                verification_status=VerificationStatus.UNSUPPORTED,
                verification_reason="No evidence sentences found.",
                confidence=0.0,
                best_chunk_id=None, best_chunk_rank=None, best_chunk_score=None,
                best_sentence_id=None, evidence_text=None,
                entailment_score=0.0, contradiction_score=0.0, neutral_score=0.0,
                verification_latency_ms=(time.time() - t_start) * 1000
            )

        logger.info(f"Verified claim '{claim.candidate_id}': {res.verification_status.value} (Latency: {res.verification_latency_ms:.1f}ms)")
        return res

    @staticmethod
    def build_retrieved_chunks_from_trace(trace: RAGTrace, registry) -> List[RetrievedChunk]:
        """
        Reconstructs RetrievedChunk objects from a saved trace's
        retrieved_chunk_references + a loaded ChunkRegistry. Shared by
        verify() and any offline re-analysis (e.g.
        scripts/ablate_aggregation_strategy.py) that needs the same chunks
        without duplicating this reconstruction logic.
        """
        retrieved_chunks = []
        for ref in trace.retrieved_chunk_references:
            chunk_id = ref["chunk_id"]
            record = registry.get_chunk(chunk_id)
            if record:
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    similarity_score=ref.get("similarity_score", 0.0),
                    rank=ref.get("rank", 0),
                    page_number=str(ref.get("page_number", "")),
                    source_file=ref.get("source_file", ""),
                    chunk_index=ref.get("chunk_index", 0),
                    chunk_text=record.text,
                    dense_score=ref.get("dense_score", 0.0),
                    sparse_score=ref.get("sparse_score", 0.0),
                    dense_rank=ref.get("dense_rank", -1),
                    sparse_rank=ref.get("sparse_rank", -1),
                    rrf_score=ref.get("rrf_score", 0.0),
                    reranker_score=ref.get("reranker_score", 0.0)
                )
                retrieved_chunks.append(chunk)
        return retrieved_chunks

    def verify(self, trace: RAGTrace, claim_set: CandidateClaimSet) -> VerificationSummary:
        from src.chunk_registry import ChunkRegistry
        import os

        registry_path = "artifacts/chunk_registry.json"
        if not os.path.exists(registry_path):
            raise FileNotFoundError(f"Cannot find {registry_path} to load chunk texts.")

        registry = ChunkRegistry.load_from_json(registry_path)
        retrieved_chunks = self.build_retrieved_chunks_from_trace(trace, registry)
        return self.verify_all(claim_set, trace.trace_id, retrieved_chunks)

    def verify_all(self, claim_set: CandidateClaimSet, trace_id: str, retrieved_chunks: List[RetrievedChunk]) -> VerificationSummary:
        """Verifies all claims in a CandidateClaimSet."""
        t_start = time.time()
        logger.info(f"Starting verification for trace {trace_id} ({claim_set.total_candidates} claims)")
        
        results = []
        for claim in claim_set.candidate_claims:
            res = self.verify_claim(claim, retrieved_chunks)
            results.append(res)
            
        # Aggregate statistics
        total = len(results)
        supported = sum(1 for r in results if r.verification_status == VerificationStatus.SUPPORTED)
        partial = sum(1 for r in results if r.verification_status == VerificationStatus.PARTIALLY_SUPPORTED)
        contradicted = sum(1 for r in results if r.verification_status == VerificationStatus.CONTRADICTED)
        unsupported = sum(1 for r in results if r.verification_status == VerificationStatus.UNSUPPORTED)
        not_verif = sum(1 for r in results if r.verification_status == VerificationStatus.NOT_VERIFIABLE)
        
        avg_entailment = sum(r.entailment_score for r in results) / total if total > 0 else 0.0
        total_latency = (time.time() - t_start) * 1000
        
        summary = VerificationSummary(
            trace_id=trace_id,
            total_claims=total,
            supported_claims=supported,
            partially_supported_claims=partial,
            contradicted_claims=contradicted,
            unsupported_claims=unsupported,
            not_verifiable_claims=not_verif,
            average_entailment_score=avg_entailment,
            total_verification_latency_ms=total_latency,
            results=results
        )
        
        logger.info(f"Completed verification for trace {trace_id}. "
                    f"Supported: {supported}/{total}, Contradicted: {contradicted}/{total}")
        return summary

    def save_artifacts(self, summary: VerificationSummary):
        """Saves the verification summary to a JSON file."""
        dir_path = os.path.join("artifacts", "verification")
        os.makedirs(dir_path, exist_ok=True)
        
        filepath = os.path.join(dir_path, f"TRACE_{summary.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary.to_json())
        logger.info(f"Saved verification artifact to {filepath}")
