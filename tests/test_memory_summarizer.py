"""
Unit tests for Memory Summarizer.

Tests both LLM-based and extractive summarization.
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
)
from src.memory.memory_summarizer import MemorySummarizer


class TestMemorySummarizer(unittest.TestCase):
    """Tests for MemorySummarizer operations."""

    def setUp(self):
        self.config = MemoryConfig(
            summarization_threshold=5,
            auto_summary=True,
        )
        self.summarizer = MemorySummarizer(self.config)

    def test_should_summarize_below_threshold(self):
        self.assertFalse(self.summarizer.should_summarize(3))

    def test_should_summarize_at_threshold(self):
        self.assertTrue(self.summarizer.should_summarize(5))

    def test_should_summarize_above_threshold(self):
        self.assertTrue(self.summarizer.should_summarize(10))

    def test_should_summarize_disabled(self):
        config = MemoryConfig(auto_summary=False)
        summarizer = MemorySummarizer(config)
        self.assertFalse(summarizer.should_summarize(100))

    def test_extractive_summarize(self):
        memories = [
            MemoryEntry(
                memory_id=f"m{i}",
                session_id="s1",
                question=f"What is topic {i}?",
                answer=f"Topic {i} is about subject {i}.",
            )
            for i in range(5)
        ]

        summary = self.summarizer._extractive_summarize(memories, "s1")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.session_id, "s1")
        self.assertEqual(summary.question_count, 5)
        self.assertEqual(len(summary.source_memory_ids), 5)
        self.assertTrue(len(summary.key_topics) > 0)
        self.assertTrue(len(summary.summary_text) > 0)

    def test_summarize_empty_list(self):
        result = self.summarizer.summarize([], "s1")
        self.assertIsNone(result)

    def test_summarize_falls_back_to_extractive(self):
        """When no LLM is available, should use extractive summarization."""
        memories = [
            MemoryEntry(
                memory_id=f"m{i}",
                session_id="s1",
                question=f"Question about topic {i}",
                answer=f"Answer about topic {i} with details.",
            )
            for i in range(3)
        ]

        # _llm is None by default, so it should fall back to extractive
        summary = self.summarizer.summarize(memories, "s1")
        self.assertIsNotNone(summary)
        self.assertEqual(summary.question_count, 3)


if __name__ == "__main__":
    unittest.main()
