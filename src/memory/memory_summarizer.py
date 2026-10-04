"""
Memory Summarizer Module.

Automatically summarizes older interactions when the conversation
exceeds a configurable threshold. Summaries compress multiple
Q&A pairs into key topics, entities, and claims for efficient
long-term retrieval.
"""

import json
import logging
import os
from src import rate_limiter
from typing import List, Optional

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySummary,
)
from src.memory.memory_utils import (
    generate_summary_id,
    get_timestamp,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


class MemorySummarizer:
    """Summarizes batches of memory entries into compressed representations.

    When conversation size exceeds config.summarization_threshold,
    the oldest memories are summarized and the summary is stored
    for future retrieval. Uses the NVIDIA LLM for summarization
    when available, falls back to extractive summarization.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self._llm = None

    def _get_llm(self):
        """Lazily initialize the LLM for summarization."""
        if self._llm is None:
            try:
                from llama_index.llms.openai_like import OpenAILike
                from configs.models import (
                    NVIDIA_CLAIM_DECOMPOSER_MODEL,
                    NVIDIA_BASE_URL,
                    LLM_TEMPERATURE,
                    LLM_REQUEST_TIMEOUT,
                )

                self._llm = OpenAILike(
                    model=NVIDIA_CLAIM_DECOMPOSER_MODEL,
                    temperature=LLM_TEMPERATURE,
                    max_tokens=1024,
                    api_key=os.environ.get("NVIDIA_API_KEY", ""),
                    api_base=NVIDIA_BASE_URL,
                    is_chat_model=True,
                    timeout=LLM_REQUEST_TIMEOUT,
                    max_retries=0,
                )
            except Exception as e:
                logger.warning("LLM unavailable for summarization: %s", e)
        return self._llm

    def should_summarize(self, memory_count: int) -> bool:
        """Check if summarization is needed based on memory count."""
        return (
            self.config.auto_summary
            and memory_count >= self.config.summarization_threshold
        )

    def summarize(
        self,
        memories: List[MemoryEntry],
        session_id: str,
    ) -> Optional[MemorySummary]:
        """Summarize a batch of memory entries.

        Attempts LLM-based summarization first, falls back to
        extractive summarization if the LLM is unavailable.

        Args:
            memories:   List of MemoryEntry objects to summarize.
            session_id: The session these memories belong to.

        Returns:
            A MemorySummary object, or None if summarization fails.
        """
        if not memories:
            return None

        mem_logger.info(
            "Summarizing %d memories for session %s", len(memories), session_id
        )

        # Try LLM-based summarization
        llm = self._get_llm()
        if llm:
            return self._llm_summarize(memories, session_id, llm)

        # Fallback to extractive
        return self._extractive_summarize(memories, session_id)

    def _llm_summarize(
        self, memories: List[MemoryEntry], session_id: str, llm
    ) -> Optional[MemorySummary]:
        """Use LLM to generate a comprehensive summary."""
        from llama_index.core.llms import ChatMessage, MessageRole

        # Build conversation text for the LLM
        conversation_text = "\n\n".join(
            f"Q: {m.question}\nA: {m.answer}" for m in memories
        )

        system_prompt = (
            "You are a conversation summarizer. Given a series of Q&A interactions, "
            "produce a structured JSON summary with these fields:\n"
            '- "summary": A concise paragraph summarizing the key information exchanged.\n'
            '- "key_topics": A list of 3-7 key topics discussed.\n'
            '- "important_entities": A list of important entities mentioned '
            "(people, organizations, legal sections, etc.).\n"
            '- "important_claims": A list of the most important factual claims made.\n'
            "Return ONLY valid JSON. No markdown, no explanations."
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Summarize this conversation:\n\n{conversation_text}",
            ),
        ]

        try:
            response = rate_limiter.call(llm.chat, messages)
            response_text = str(response.message.content)

            # Parse JSON response
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    logger.warning("Failed to parse LLM summary response")
                    return self._extractive_summarize(memories, session_id)

            summary = MemorySummary(
                summary_id=generate_summary_id(),
                session_id=session_id,
                summary_text=data.get("summary", ""),
                key_topics=data.get("key_topics", []),
                important_entities=data.get("important_entities", []),
                important_claims=data.get("important_claims", []),
                source_memory_ids=[m.memory_id for m in memories],
                question_count=len(memories),
                created_at=get_timestamp(),
            )

            mem_logger.info(
                "LLM summary generated: %s (%d topics, %d entities)",
                summary.summary_id,
                len(summary.key_topics),
                len(summary.important_entities),
            )
            return summary

        except Exception as e:
            logger.error("LLM summarization failed: %s", e)
            return self._extractive_summarize(memories, session_id)

    def _extractive_summarize(
        self, memories: List[MemoryEntry], session_id: str
    ) -> MemorySummary:
        """Extractive fallback summarization (no LLM needed).

        Extracts key topics from questions and builds a simple summary.
        """
        # Extract topics from questions
        questions = [m.question for m in memories]
        answers = [m.answer for m in memories]

        # Simple topic extraction: most common significant words
        from collections import Counter
        import re

        stopwords = {
            "what", "does", "do", "is", "a", "an", "the", "in", "on", "at",
            "to", "for", "of", "and", "or", "with", "by", "as", "it", "this",
            "that", "are", "was", "were", "be", "has", "have", "had", "not",
            "how", "why", "who", "can", "about", "which", "would", "could",
        }

        all_text = " ".join(questions + answers)
        words = re.findall(r"\w+", all_text.lower())
        filtered = [w for w in words if w not in stopwords and len(w) > 2]
        word_counts = Counter(filtered)
        key_topics = [word for word, _ in word_counts.most_common(7)]

        # Build summary text
        summary_text = (
            f"Conversation with {len(memories)} interactions covering: "
            + ", ".join(key_topics[:5])
            + "."
        )

        summary = MemorySummary(
            summary_id=generate_summary_id(),
            session_id=session_id,
            summary_text=summary_text,
            key_topics=key_topics,
            important_entities=[],
            important_claims=[],
            source_memory_ids=[m.memory_id for m in memories],
            question_count=len(memories),
            created_at=get_timestamp(),
        )

        mem_logger.info(
            "Extractive summary generated: %s", summary.summary_id
        )
        return summary
