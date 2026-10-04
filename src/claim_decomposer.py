"""
Claim Decomposer Module.

Responsible for decomposing a generated answer from a RAGTrace into atomic,
verifiable facts (candidate claims).

Design decisions:
- Uses `.chat()` (not `.complete()`) for instruct-tuned models. Calling `.complete()`
  on chat-instruction-finetuned LLMs such as Llama-3.1 bypasses the chat template and
  produces unreliable, prose-formatted output instead of structured JSON.
- `sentence_id` is computed deterministically from character offsets on the Python side
  rather than being hallucinated by the LLM. This keeps statistics (average_claims_per
  _sentence) meaningful and controllable.
- Claim deduplication: identical claims (normalised lowercase) are silently dropped
  before adding to the CandidateClaimSet, preventing inflated faithfulness scores.
- `source_sentence` has been removed: it was always set to "" and is dead weight.
"""

import os
import json
import uuid
import re
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Set, Tuple
from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole

from src.rag_trace import RAGTrace
from src import rate_limiter
from configs.pipeline import CLAIM_DECOMPOSER_PROMPT_VERSION, CLAIM_DECOMPOSER_MAX_TOKENS
from configs.models import NVIDIA_CLAIM_DECOMPOSER_MODEL, NVIDIA_BASE_URL, LLM_TEMPERATURE, LLM_REQUEST_TIMEOUT
from configs.prompts import DETAILED_THINKING_OFF

logger = logging.getLogger(__name__)

# Minimum SequenceMatcher ratio for a fuzzy offset match to be considered usable.
FUZZY_MATCH_MIN_RATIO = 0.6


@dataclass
class CandidateClaim:
    candidate_id: str
    trace_id: str
    claim_text: str
    sentence_id: str        # deterministic: "S<index>" derived from character_start
    claim_index: int
    character_start: int
    character_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateClaimSet:
    trace_id: str
    candidate_claims: List[CandidateClaim] = field(default_factory=list)
    total_candidates: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_claim(self, claim: CandidateClaim):
        self.candidate_claims.append(claim)
        self.total_candidates += 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "CandidateClaimSet":
        data = json.loads(json_str)
        claims = [CandidateClaim(**c) for c in data.pop("candidate_claims", [])]
        return cls(candidate_claims=claims, **data)


class JSONRecoveryError(Exception):
    pass


class ClaimDecomposer:

    # System instruction given once; user turn contains only the text to decompose.
    _SYSTEM_INSTRUCTION = (
        "You are an expert fact extractor. Your task is to decompose text into atomic "
        "factual claims.\n\n"
        "Definition:\n"
        "An atomic claim is the smallest semantically complete factual assertion that "
        "can be independently verified.\n\n"
        "Rules:\n"
        "Rule 1: One fact per claim.\n"
        "Rule 2: Claims must be semantically complete.\n"
        "Rule 3: Claims must be independently verifiable.\n"
        "Rule 4: Do not infer information.\n"
        "Rule 5: Do not generate opinions.\n"
        "Rule 6: Do not rewrite meaning.\n"
        "Rule 7: Don't merge independently verifiable facts.\n"
        "Rule 8: Don't invent claims not explicitly stated.\n\n"
        "Return the output ONLY as a valid JSON array of objects. Do not include "
        "markdown formatting, explanations, code fences, or any additional text.\n"
        "Each object must have exactly one key: \"claim_text\".\n"
        "Example format:\n"
        "[\n"
        "  {\"claim_text\": \"The fine is Rs. 500.\"},\n"
        "  {\"claim_text\": \"Section 399 applies to robbery.\"}\n"
        "]"
    )

    def __init__(self, debug: Optional[bool] = None, model_name: Optional[str] = None):
        if debug is None:
            self.debug = os.environ.get("CLAIM_DECOMPOSER_DEBUG", "False").lower() == "true"
        else:
            self.debug = debug

        self.model_name = model_name or NVIDIA_CLAIM_DECOMPOSER_MODEL
        logger.info("Initialising ClaimDecomposer: model=%s", self.model_name)
        self.llm = OpenAILike(
            model=self.model_name,
            temperature=LLM_TEMPERATURE,
            max_tokens=CLAIM_DECOMPOSER_MAX_TOKENS,
            api_key=os.environ.get("NVIDIA_API_KEY", ""),
            api_base=NVIDIA_BASE_URL,
            is_chat_model=True,
            # See src/rate_limiter.py: retries must go through call(), not the
            # client's own blind retry, or they burst past the shared budget.
            max_retries=0,
            timeout=LLM_REQUEST_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, trace: RAGTrace) -> CandidateClaimSet:
        logger.info("Decomposing answer for trace_id: %s", trace.trace_id)
        answer = trace.generated_answer
        t0 = time.time()

        generation_latency = parsing_latency = recovery_latency = retry_latency = 0.0
        parsing_error: Optional[str] = None
        recovery_attempt = retry_attempt = success = False

        # First LLM call
        t_gen_start = time.time()
        response_str = self._call_llm(answer)
        generation_latency = time.time() - t_gen_start

        if self.debug:
            self._save_debug_artifacts(trace.trace_id, answer, response_str)

        # Parsing attempt
        t_parse_start = time.time()
        try:
            raw_claims = self._robust_json_parse(response_str)
            success = True
            parsing_latency = time.time() - t_parse_start
        except JSONRecoveryError as e:
            parsing_latency = time.time() - t_parse_start
            parsing_error = str(e)
            recovery_attempt = True
            retry_attempt = True
            logger.warning("Initial parse and recovery failed. Initiating retry. Error: %s", e)

            retry_prompt = (
                f"The previous response was not valid JSON.\n"
                f"Return ONLY the corrected JSON.\n"
                f"Do not change the content.\n"
                f"Do not add explanations.\n\n"
                f"Original Response:\n{response_str}"
            )

            t_retry_start = time.time()
            retry_response_str = self._call_llm(retry_prompt)
            retry_latency = time.time() - t_retry_start

            t_retry_parse_start = time.time()
            try:
                raw_claims = self._robust_json_parse(retry_response_str)
                success = True
                response_str = retry_response_str
            except JSONRecoveryError as e2:
                parsing_error = str(e2)
                logger.error("Retry parsing failed: %s", e2)
                raw_claims = []

            recovery_latency = time.time() - t_retry_parse_start

        total_latency = time.time() - t0
        logger.info(
            "Latencies — Generation: %.3fs, Parsing: %.3fs, Retry: %.3fs, "
            "Recovery: %.3fs, Total: %.3fs",
            generation_latency, parsing_latency, retry_latency,
            recovery_latency, total_latency,
        )

        return self._build_claim_set(
            trace_id=trace.trace_id,
            answer=answer,
            raw_claims=raw_claims,
            diagnostics={
                "raw_response": response_str,
                "parsing_error": parsing_error,
                "recovery_attempt": recovery_attempt,
                "retry_attempt": retry_attempt,
                "success": success,
                "latencies": {
                    "generation_latency": round(generation_latency, 3),
                    "parsing_latency": round(parsing_latency, 3),
                    "recovery_latency": round(recovery_latency, 3),
                    "retry_latency": round(retry_latency, 3),
                    "total_decomposition_latency": round(total_latency, 3),
                },
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_claim_set(
        self,
        trace_id: str,
        answer: str,
        raw_claims: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
    ) -> CandidateClaimSet:
        """Converts raw LLM claim dicts into a deduplicated CandidateClaimSet.

        Deduplication key: normalised claim text (stripped, lowercased).
        sentence_id is derived deterministically from the character_start offset —
        it is NOT requested from the LLM, which was producing inconsistent values.
        """
        claim_set = CandidateClaimSet(
            trace_id=trace_id,
            metadata={"CLAIM_DECOMPOSER_PROMPT_VERSION": CLAIM_DECOMPOSER_PROMPT_VERSION},
        )

        seen_texts: Set[str] = set()

        for i, raw_claim in enumerate(raw_claims):
            claim_text = raw_claim.get("claim_text", "").strip()
            if not claim_text:
                continue

            # Deduplicate
            norm_key = claim_text.lower()
            if norm_key in seen_texts:
                logger.debug("Skipping duplicate claim: '%s'", claim_text[:60])
                continue
            seen_texts.add(norm_key)

            # Locate character offset in answer
            start_idx = answer.find(claim_text)
            if start_idx != -1:
                end_idx = start_idx + len(claim_text)
                meta = {"match_type": "exact", "match_confidence": 1.0}
            else:
                fuzzy_start, fuzzy_end, ratio = self._fuzzy_match(answer, claim_text)
                if fuzzy_start != -1 and ratio >= FUZZY_MATCH_MIN_RATIO:
                    start_idx, end_idx = fuzzy_start, fuzzy_end
                    meta = {"match_type": "fuzzy", "match_confidence": round(ratio, 3)}
                else:
                    start_idx, end_idx = -1, -1
                    meta = {"match_type": "none", "match_confidence": 0.0}

            # Deterministic sentence_id from character position — avoids relying on
            # the LLM to produce consistent sentence identifiers across runs.
            sentence_id = f"S{start_idx}" if start_idx != -1 else f"S_unresolved_{i}"

            candidate = CandidateClaim(
                candidate_id=str(uuid.uuid4()),
                trace_id=trace_id,
                claim_text=claim_text,
                sentence_id=sentence_id,
                claim_index=i,
                character_start=start_idx,
                character_end=end_idx,
                metadata=meta,
            )
            claim_set.add_claim(candidate)

        # Compute sentence statistics from resolved offsets (not LLM output)
        resolved = [c for c in claim_set.candidate_claims if c.character_start != -1]
        unique_sentence_ids = {c.sentence_id for c in resolved}
        total_sentences = len(unique_sentence_ids)
        total_claims = claim_set.total_candidates
        avg_claims = (total_claims / total_sentences) if total_sentences > 0 else 0.0

        claim_set.metadata.update({
            "total_sentences": total_sentences,
            "total_candidate_claims": total_claims,
            "average_claims_per_sentence": round(avg_claims, 2),
            "diagnostics": diagnostics,
        })

        return claim_set

    def _fuzzy_match(self, answer: str, claim_text: str) -> Tuple[int, int, float]:
        """Locate the best-matching span for claim_text within answer.

        Uses difflib.SequenceMatcher.find_longest_match() for an O(n) anchor,
        then refines with a local window search.  More reliable than the previous
        stride-based approach which could miss the optimal offset between steps.
        Returns (start, end, ratio), or (-1, -1, 0.0) when unresolvable.
        """
        import difflib

        window_size = len(claim_text)
        if window_size == 0 or not answer:
            return -1, -1, 0.0

        # Fast anchor: find the longest matching block and centre the search window.
        matcher = difflib.SequenceMatcher(None, answer, claim_text, autojunk=False)
        match = matcher.find_longest_match(0, len(answer), 0, window_size)

        best_ratio = 0.0
        best_start = -1

        # Search in a neighbourhood around the anchor for the best window alignment.
        # Half-window of 30 chars is sufficient for paraphrased claims.
        search_radius = max(window_size, 30)
        lo = max(0, match.a - search_radius)
        hi = min(len(answer), match.a + search_radius + window_size)

        for start in range(lo, hi - window_size + 1):
            window = answer[start : start + window_size]
            ratio = difflib.SequenceMatcher(None, window, claim_text, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start

        if best_start == -1:
            return -1, -1, 0.0
        return best_start, best_start + window_size, best_ratio

    def _call_llm(self, user_content: str) -> str:
        """Call the NVIDIA NIM instruct model via the chat interface.

        Uses `.chat()` (not `.complete()`) because Llama-3.1 and other
        instruction-finetuned models require the chat template to reliably
        follow structured-output instructions. `.complete()` bypasses the
        template and degrades instruction adherence.
        """
        messages = [
            # Must be its own standalone first system message -- nemotron-3-super
            # only honours this toggle in that exact shape (see generator.py's
            # PromptBuilder.build_messages for the same fix and why). Without
            # it this call would occasionally narrate chain-of-thought instead
            # of emitting clean JSON, which _robust_json_parse's recovery
            # heuristics could then misparse into garbage "claims".
            ChatMessage(role=MessageRole.SYSTEM, content=DETAILED_THINKING_OFF),
            ChatMessage(role=MessageRole.SYSTEM, content=self._SYSTEM_INSTRUCTION),
            ChatMessage(role=MessageRole.USER, content=f"Text to decompose:\n{user_content}"),
        ]
        try:
            response = rate_limiter.call(self.llm.chat, messages)
            return str(response.message.content)
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return ""

    def _robust_json_parse(self, text: str) -> List[Dict[str, Any]]:
        claims = self._recover_json(text)
        if not isinstance(claims, list) or any(
            not isinstance(c, dict) or not isinstance(c.get("claim_text"), str)
            or not c["claim_text"].strip() for c in claims
        ):
            raise JSONRecoveryError("Expected an array of objects with nonempty claim_text strings.")
        return claims

    def _recover_json(self, text: str) -> Any:
        if not text or not text.strip():
            raise JSONRecoveryError("Empty string provided to JSON parser.")

        text = text.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # 3. Regex extraction of the largest JSON array
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            array_text = match.group(0).strip()
            try:
                return json.loads(array_text)
            except json.JSONDecodeError:
                text = array_text

        # 4. Bracket balancing
        balanced = text
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")
        if open_braces > 0:
            balanced += "}" * open_braces
        if open_brackets > 0:
            balanced += "]" * open_brackets
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

        # 5. Truncation recovery: drop back to last fully-closed object
        last_complete_end = text.rfind("}")
        if last_complete_end != -1:
            truncated = text[: last_complete_end + 1].rstrip().rstrip(",")
            if not truncated.lstrip().startswith("["):
                truncated = "[" + truncated
            truncated += "]"
            try:
                return json.loads(truncated)
            except json.JSONDecodeError as e:
                raise JSONRecoveryError(f"Failed to recover JSON: {e}") from e

        raise JSONRecoveryError("Failed to recover JSON: no valid object boundary found.")

    def _save_debug_artifacts(self, trace_id: str, prompt: str, response: str):
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dir_path = os.path.join("artifacts", "debug", date_str)
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, f"prompt_{trace_id}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(os.path.join(dir_path, f"response_{trace_id}.json"), "w", encoding="utf-8") as f:
            f.write(response)

        logger.info("Saved debug artifacts for trace %s", trace_id)
