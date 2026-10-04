"""
Generator Module.

Responsible exclusively for synthesizing answers using an LLM based on the
context provided by the Retriever.  All calls route through the NVIDIA NIM
OpenAI-compatible inference endpoint via ``llama_index.llms.openai_like.OpenAILike``.

Design notes
------------
- ``PromptBuilder`` is fully isolated from ``Generator`` so prompt variants can
  be swapped or A/B-tested without touching generation logic.
- Context chunks are presented in *reversed* rank order so the highest-scoring
  chunk appears immediately before the question (recency bias fix, Liu et al. 2023).
- ``generate()`` accepts an optional ``system_instructions`` kwarg for runtime
  domain switching — the API layer or a query router can inject domain-specific
  instructions per request without restarting the process.
"""

import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole

from src.retriever import RetrievalResult
from src import rate_limiter
from configs.models import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    NVIDIA_GENERATION_MODEL,
    NVIDIA_BASE_URL,
    LLM_REQUEST_TIMEOUT,
)
from configs.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    DETAILED_THINKING_OFF,
    QUERY_CONDENSER_PROMPT,
    THINK_ALOUD_INSTRUCTIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Captures the complete state of a single generation event.

    Everything the LLM received (``prompt``) and everything it produced
    (``generated_answer``) is preserved for downstream diagnostics.
    """
    question: str
    generated_answer: str
    prompt: str
    prompt_length: int
    model_name: str
    temperature: float
    max_tokens: int
    generation_time: float
    retrieved_chunk_ids: List[str]
    generation_metadata: Dict[str, Any]

    # Populated only when generate_stream(reasoning=True) elicited a
    # <thinking>...</thinking> block; never part of generated_answer.
    reasoning: str = ""

    # Set when the LLM call itself failed (timeout, rate limit, transport
    # error). Callers MUST check this before treating ``generated_answer`` as
    # an answer: an API failure previously became the answer string verbatim,
    # so a timeout rendered as a normal reply and was persisted to memory as
    # though the model had said it.
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Constructs the chat messages array from retrieval context and a question.

    Isolated from Generator to allow prompt experimentation without touching
    the generation / LLM-wiring code.
    """

    @staticmethod
    def build_messages(
        question: str,
        retrieved_chunks: List[Any],
        system_instructions: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[ChatMessage]:
        """Build ``[SYSTEM, *HISTORY, USER]`` chat messages.

        Chunks are presented in *reversed* rank order (lowest rank last) so
        the most relevant chunk is closest to the question in the context
        window — addressing the "lost in the middle" recency bias described
        in Liu et al. 2023.

        Prior turns are passed as genuine alternating user/assistant messages
        rather than being flattened into the system prompt. Models follow
        real conversational structure far more reliably than a transcript
        embedded in their instructions, which is what makes follow-up
        questions ("elaborate on that") resolvable at all.

        Args:
            question:            The user's question string.
            retrieved_chunks:    Chunks from the retriever, in rank order.
            system_instructions: Optional per-call override; falls back to
                                 ``DEFAULT_SYSTEM_INSTRUCTIONS``.
            chat_history:        Prior turns as ``[{"role": ..., "content": ...}]``
                                 in chronological order, excluding the current
                                 question.
        """
        instructions = system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS

        # Reverse so rank-1 chunk is last (closest to the question).
        ordered_chunks = list(reversed(retrieved_chunks))

        context_parts: List[str] = []
        for i, chunk in enumerate(ordered_chunks, start=1):
            context_parts.append(
                f"--- Context chunk {i} [Chunk-ID: {chunk.chunk_id}] ---\n{chunk.chunk_text}\n"
            )

        if context_parts:
            context = "\n".join(context_parts)
            user_message = (
                f"Retrieved context for this question:\n\n{context}\n\nQuestion: {question}"
            )
        else:
            # No chunks is not itself grounds for refusal — the conversation
            # may well carry the answer. Say so explicitly rather than
            # emitting a bare "No relevant context found." that reads to the
            # model as an instruction to give up.
            user_message = (
                "No document context was retrieved for this question. "
                "Answer from the conversation so far if you can.\n\n"
                f"Question: {question}"
            )

        # nemotron-3-super's reasoning toggle only takes effect as its own
        # standalone first system message with this exact content -- mixed
        # into a larger system prompt (even as a prefix line) it's ignored
        # and the model narrates its chain-of-thought into the answer anyway.
        messages: List[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=DETAILED_THINKING_OFF),
            ChatMessage(role=MessageRole.SYSTEM, content=instructions),
        ]

        for turn in chat_history or []:
            role = MessageRole.ASSISTANT if turn.get("role") == "assistant" else MessageRole.USER
            content = (turn.get("content") or "").strip()
            if content:
                messages.append(ChatMessage(role=role, content=content))

        messages.append(ChatMessage(role=MessageRole.USER, content=user_message))
        return messages


# ---------------------------------------------------------------------------
# Reasoning-mode tag splitter
# ---------------------------------------------------------------------------

class _ThinkingAnswerSplitter:
    """Incrementally splits a
    ``<thinking>...</thinking><answer>...</answer>`` completion, fed one
    stream delta at a time, into ``("reasoning", text)`` / ``("answer",
    text)`` pairs.

    A naive "re-match the whole buffer against a regex on every delta"
    approach leaks partial closing tags: if a delta boundary lands mid-tag
    (buffer ends "...done.</thin"), a lazy ``.*?</thinking>|$`` alternation
    matches the ``$`` branch and emits "...done.</thin" as content, since
    nothing yet proves ``</thin`` isn't just more reasoning text. The fix is
    to never emit a suffix that could still be the start of the tag being
    watched for -- hold it back until either the tag completes or enough
    further text arrives to prove it wasn't the tag.
    """

    _TAGS = ("<thinking>", "</thinking>", "<answer>", "</answer>")

    def __init__(self) -> None:
        self._buffer = ""
        self._phase = "pre"  # pre -> thinking -> between -> answer -> done

    def feed(self, delta: str):
        self._buffer += delta
        out: List[tuple] = []
        progressed = True
        while progressed:
            progressed = False
            if self._phase == "pre":
                progressed = self._consume_until("<thinking>", "thinking")
            elif self._phase == "thinking":
                progressed = self._emit_until("</thinking>", "reasoning", "between", out)
            elif self._phase == "between":
                progressed = self._consume_until("<answer>", "answer")
            elif self._phase == "answer":
                progressed = self._emit_until("</answer>", "answer", "done", out)
            elif self._phase == "done":
                self._buffer = ""
        return out

    def _consume_until(self, tag: str, next_phase: str) -> bool:
        idx = self._buffer.find(tag)
        if idx == -1:
            return False
        self._buffer = self._buffer[idx + len(tag):]
        self._phase = next_phase
        return True

    def _emit_until(self, close_tag: str, kind: str, next_phase: str, out: list) -> bool:
        idx = self._buffer.find(close_tag)
        if idx != -1:
            text = self._buffer[:idx]
            if text:
                out.append((kind, text))
            self._buffer = self._buffer[idx + len(close_tag):]
            self._phase = next_phase
            return True

        # No closing tag yet -- emit everything except a trailing suffix
        # that could still turn into the closing tag on the next delta.
        safe_len = len(self._buffer)
        for hold in range(min(len(close_tag) - 1, len(self._buffer)), 0, -1):
            if self._buffer.endswith(close_tag[:hold]):
                safe_len = len(self._buffer) - hold
                break
        if safe_len > 0:
            out.append((kind, self._buffer[:safe_len]))
            self._buffer = self._buffer[safe_len:]
        return False


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """Executes the generation phase using the NVIDIA NIM inference endpoint."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        system_instructions: Optional[str] = None,
    ) -> None:
        # Populated by generate_stream() once its iterator is exhausted.
        self.last_stream_result: Optional[GenerationResult] = None

        self.model_name = model_name or NVIDIA_GENERATION_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_instructions = system_instructions

        logger.info("Initialising Generator: model=%s", self.model_name)
        # An explicit timeout matters: without one a stalled free-tier request
        # can hang for minutes and surface to the user as a frozen UI. Bounded
        # retries cover the transient 5xx/rate-limit responses that the shared
        # NVIDIA endpoint returns under load.
        self.llm = OpenAILike(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=os.environ.get("NVIDIA_API_KEY", ""),
            api_base=NVIDIA_BASE_URL,
            is_chat_model=True,
            timeout=LLM_REQUEST_TIMEOUT,
            # Disable client retries: the client's own blind retry fires real
            # HTTP requests that rate_limiter.call() below never sees, letting
            # retries burst past the shared account-wide budget. All
            # retry/backoff decisions belong to rate_limiter.call() instead.
            max_retries=0,
        )

    def generate(
        self,
        retrieval_result: RetrievalResult,
        system_instructions: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        question_override: Optional[str] = None,
    ) -> GenerationResult:
        """Synthesize an answer from the retrieval context.

        Args:
            retrieval_result:    Output of ``Retriever.retrieve()``.
            system_instructions: Per-call system prompt override; falls back
                                 to the instance-level value, then to the
                                 default domain-agnostic instructions.
            chat_history:        Prior conversation turns as
                                 ``[{"role": ..., "content": ...}]``, so
                                 follow-up questions can be resolved.
        """
        logger.info("Generating answer for: '%s'", retrieval_result.question)
        start_time = time.time()

        effective_instructions = (
            system_instructions
            or self.system_instructions
            or DEFAULT_SYSTEM_INSTRUCTIONS
        )

        # Retrieval may have run on a condensed standalone query; the model
        # should still see the user's own wording as the final turn.
        display_question = question_override or retrieval_result.question

        messages = PromptBuilder.build_messages(
            question=display_question,
            retrieved_chunks=retrieval_result.retrieved_chunks,
            system_instructions=effective_instructions,
            chat_history=chat_history,
        )
        prompt_str = "\n".join(
            f"[{m.role.value.upper()}]: {m.content}" for m in messages
        )

        error: Optional[str] = None
        finish_reason = None
        try:
            response = rate_limiter.call(self.llm.chat, messages)
            generated_answer = str(response.message.content)
            raw = getattr(response, "raw", None)
            if isinstance(raw, dict):
                choices = raw.get("choices") or []
                finish_reason = choices[0].get("finish_reason") if choices else None
            elif raw is not None and getattr(raw, "choices", None):
                finish_reason = getattr(raw.choices[0], "finish_reason", None)
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc, exc_info=True)
            error = f"{type(exc).__name__}: {exc}"
            generated_answer = (
                "The language model did not respond in time. This is an "
                "infrastructure problem, not a limitation of your documents — "
                "please retry."
            )

        generation_time = time.time() - start_time

        result = GenerationResult(
            question=display_question,
            generated_answer=generated_answer,
            prompt=prompt_str,
            prompt_length=len(prompt_str),
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            generation_time=generation_time,
            retrieved_chunk_ids=retrieval_result.retrieved_chunk_ids,
            generation_metadata={
                "retrieval_time": retrieval_result.retrieval_time,
                "top_k_used": retrieval_result.top_k,
                "provider": "nvidia",
                "history_turns": len(chat_history or []),
                "finish_reason": finish_reason,
            },
            error=error,
        )

        logger.info("Generation complete in %.3fs.", generation_time)
        return result

    def generate_stream(
        self,
        retrieval_result: RetrievalResult,
        system_instructions: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        question_override: Optional[str] = None,
        reasoning: bool = False,
    ):
        """Yield answer text incrementally, then expose the full result.

        Generation is the longest stage in the pipeline (20-70s against the
        remote endpoint) and nothing local can make the model faster. Streaming
        does not reduce total time, but it moves time-to-first-token to ~1s, so
        the user reads the answer while it is still being produced instead of
        watching a spinner.

        With ``reasoning=False`` (the default) this yields plain ``str``
        deltas, unchanged from before. With ``reasoning=True`` it instructs
        the model to emit a ``<thinking>...</thinking><answer>...</answer>``
        completion (see ``THINK_ALOUD_INSTRUCTIONS``) and yields
        ``(kind, delta)`` tuples instead, ``kind`` being ``"reasoning"`` or
        ``"answer"``. ``last_stream_result.generated_answer`` always holds
        only the answer text either way -- callers that parse it for
        citations, or persist it to memory, never see the thinking block. If
        the model ignores the tag instruction and free-forms a response
        anyway, the whole output degrades to the answer rather than being
        lost.

        After the iterator is exhausted, ``last_stream_result`` holds the
        assembled :class:`GenerationResult` for the rest of the pipeline.
        """
        start_time = time.time()
        effective_instructions = (
            system_instructions
            or self.system_instructions
            or DEFAULT_SYSTEM_INSTRUCTIONS
        )
        if reasoning:
            effective_instructions = effective_instructions + THINK_ALOUD_INSTRUCTIONS
        display_question = question_override or retrieval_result.question
        messages = PromptBuilder.build_messages(
            question=display_question,
            retrieved_chunks=retrieval_result.retrieved_chunks,
            system_instructions=effective_instructions,
            chat_history=chat_history,
        )
        prompt_str = "\n".join(
            f"[{m.role.value.upper()}]: {m.content}" for m in messages
        )

        parts: List[str] = []
        think_parts: List[str] = []
        error: Optional[str] = None
        raw = ""
        finish_reason: Optional[str] = None
        splitter = _ThinkingAnswerSplitter() if reasoning else None
        try:
            # Best-effort only: rate_limiter.call() can't wrap a streamed
            # iterator (the request fires on first-chunk consumption, not on
            # this call), and this UI-only streaming path isn't exercised by
            # the experiment suite this limiter was hardened for.
            rate_limiter.acquire()
            for chunk in self.llm.stream_chat(messages):
                # The provider sets this on the final chunk only (None until
                # then); "length" means max_tokens was hit mid-generation --
                # a silent truncation that looks identical to a normal finish
                # unless checked. See the "length" handling below the loop.
                choices = getattr(chunk.raw, "choices", None) if chunk.raw else None
                if choices:
                    finish_reason = getattr(choices[0], "finish_reason", None) or finish_reason

                delta = chunk.delta or ""
                if not delta:
                    continue
                if not reasoning:
                    parts.append(delta)
                    yield delta
                    continue

                raw += delta
                for kind, text in splitter.feed(delta):
                    if kind == "reasoning":
                        think_parts.append(text)
                    else:
                        parts.append(text)
                    yield (kind, text)
        except Exception as exc:
            logger.error("LLM streaming failed: %s", exc, exc_info=True)
            error = f"{type(exc).__name__}: {exc}"
            # Surface the failure in-band too, otherwise a mid-stream error
            # would leave a silently truncated answer on screen.
            message = (
                "\n\n_The language model stopped responding partway through. "
                "This is an infrastructure problem, not a limitation of your "
                "documents — please retry._"
            )
            parts.append(message)
            yield (("answer", message) if reasoning else message)

        if not error and finish_reason == "length":
            logger.warning("Generation hit max_tokens (%d) before finishing.", self.max_tokens)
            message = (
                "\n\n_[Answer cut off: this question needed a longer response than the "
                "model's length limit allowed. Try asking a narrower question, or ask "
                "to continue.]_"
            )
            parts.append(message)
            yield (("answer", message) if reasoning else message)

        if reasoning and not parts and raw:
            # Either the model ignored the tag instruction and free-formed a
            # plain response, or it stopped mid-<thinking> before ever
            # reaching <answer> (early stop / truncation). Either way,
            # nothing was yielded as an "answer" delta above, so the SSE
            # stream has no token events and the frontend's answer bubble
            # would stay empty forever -- treat the raw text as the answer
            # and yield it now, stripped of any partial tag literals.
            fallback = raw
            for tag in _ThinkingAnswerSplitter._TAGS:
                fallback = fallback.replace(tag, "")
            parts = [fallback]
            yield ("answer", fallback)

        answer = "".join(parts)
        self.last_stream_result = GenerationResult(
            question=display_question,
            generated_answer=answer,
            prompt=prompt_str,
            prompt_length=len(prompt_str),
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            generation_time=time.time() - start_time,
            retrieved_chunk_ids=retrieval_result.retrieved_chunk_ids,
            generation_metadata={
                "retrieval_time": retrieval_result.retrieval_time,
                "top_k_used": retrieval_result.top_k,
                "provider": "nvidia",
                "history_turns": len(chat_history or []),
                "finish_reason": finish_reason,
                "streamed": True,
            },
            reasoning="".join(think_parts),
            error=error,
        )

    # Referring expressions that signal the question depends on prior turns.
    # Deliberately biased toward false positives: an unnecessary rewrite costs
    # one cheap 8B call, a missed one costs a wrong retrieval.
    _REFERRING_TERMS = frozenset(
        """that this it its they them those these he she his her their
        above former latter same one ones such said aforementioned""".split()
    )

    _CONVERSATIONAL_CUES = (
        "elaborate", "explain more", "more detail", "go on", "continue",
        "expand", "tell me more", "what about", "how about", "and ",
        "also", "instead", "why", "previous", "earlier", "just ask",
    )

    @classmethod
    def _needs_condensing(cls, question: str) -> bool:
        """Heuristic: does this question depend on the conversation?"""
        text = question.lower().strip()
        words = re.findall(r"[a-z']+", text)

        # Very short questions are almost never self-contained.
        if len(words) <= 4:
            return True
        if cls._REFERRING_TERMS.intersection(words):
            return True
        return any(cue in text for cue in cls._CONVERSATIONAL_CUES)

    def condense_query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Rewrite a follow-up question into a standalone retrieval query.

        Retrieval is stateless — embedding "can you elaborate on that?" matches
        nothing useful because the referent lives in the previous turn. This
        resolves the reference before retrieval runs.

        Falls back to the original question on any failure: a degraded query is
        strictly worse than the original, so condensation must never be able to
        break the pipeline.
        """
        if not chat_history:
            return question

        # Condensation costs a full LLM round-trip. Most questions are already
        # standalone, so only pay it when the text actually shows a dependency
        # on the previous turn: a referring expression, or a fragment so short
        # it cannot stand alone ("why?", "and the penalty?").
        if not self._needs_condensing(question):
            return question

        recent = chat_history[-6:]
        history_str = "\n".join(
            f"{'User' if t.get('role') != 'assistant' else 'Assistant'}: "
            f"{(t.get('content') or '')[:500]}"
            for t in recent
        )

        try:
            response = rate_limiter.call(self.llm.chat, [
                ChatMessage(role=MessageRole.SYSTEM, content=DETAILED_THINKING_OFF),
                ChatMessage(
                    role=MessageRole.USER,
                    content=QUERY_CONDENSER_PROMPT.format(
                        history=history_str, question=question
                    ),
                )
            ])
            condensed = str(response.message.content).strip().strip('"').strip()
        except Exception as exc:
            logger.warning("Query condensation failed, using original: %s", exc)
            return question

        # Guard against a rambling or empty rewrite.
        if not condensed or len(condensed) > 4 * len(question) + 200:
            return question

        if condensed != question:
            logger.info("Condensed query: '%s' -> '%s'", question, condensed)
        return condensed
