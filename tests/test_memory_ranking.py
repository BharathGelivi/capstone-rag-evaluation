"""
Unit tests for Memory Retriever ranking.

Tests the multi-factor scoring formula and search behavior.
"""

import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.memory.memory_models import MemoryConfig
from src.memory.memory_utils import (
    compute_frequency_score,
    compute_recency_score,
    generate_memory_id,
    generate_session_id,
    generate_summary_id,
    compute_text_hash,
)


class TestMemoryUtils(unittest.TestCase):
    """Tests for memory utility functions."""

    def test_generate_memory_id(self):
        mid = generate_memory_id()
        self.assertTrue(mid.startswith("mem_"))
        self.assertEqual(len(mid), 16)  # mem_ + 12 hex chars

    def test_generate_session_id(self):
        sid = generate_session_id()
        self.assertTrue(sid.startswith("session_"))

    def test_generate_summary_id(self):
        sid = generate_summary_id()
        self.assertTrue(sid.startswith("summary_"))

    def test_compute_text_hash(self):
        h1 = compute_text_hash("Hello World")
        h2 = compute_text_hash("hello world")
        h3 = compute_text_hash(" Hello World ")
        # Case-insensitive and strip-normalized
        self.assertEqual(h1, h2)
        self.assertEqual(h1, h3)

    def test_compute_text_hash_different(self):
        h1 = compute_text_hash("Hello")
        h2 = compute_text_hash("World")
        self.assertNotEqual(h1, h2)


class TestRecencyScore(unittest.TestCase):
    """Tests for recency score computation."""

    def test_recent_timestamp(self):
        """A timestamp from now should have high recency."""
        from datetime import datetime
        now = datetime.utcnow().isoformat() + "Z"
        score = compute_recency_score(now)
        self.assertGreater(score, 0.9)

    def test_old_timestamp(self):
        """A timestamp from a week ago should have low recency."""
        score = compute_recency_score("2020-01-01T00:00:00Z")
        self.assertLess(score, 0.01)

    def test_invalid_timestamp(self):
        """Invalid timestamp should return default 0.5."""
        score = compute_recency_score("not-a-date")
        self.assertEqual(score, 0.5)


class TestFrequencyScore(unittest.TestCase):
    """Tests for frequency score computation."""

    def test_zero_access(self):
        self.assertEqual(compute_frequency_score(0), 0.0)

    def test_some_access(self):
        score = compute_frequency_score(5, max_count=100)
        self.assertGreater(score, 0)
        self.assertLess(score, 1.0)

    def test_max_access(self):
        score = compute_frequency_score(100, max_count=100)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_over_max(self):
        """Frequency should be capped at 1.0."""
        score = compute_frequency_score(200, max_count=100)
        self.assertLessEqual(score, 1.0)


class TestRankingFormula(unittest.TestCase):
    """Tests for the complete ranking formula."""

    def test_ranking_weights_sum(self):
        """Default weights should sum to 1.0."""
        config = MemoryConfig()
        total = (
            config.semantic_weight
            + config.recency_weight
            + config.frequency_weight
            + config.importance_weight
        )
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_ranking_score_range(self):
        """Final score should be between 0 and 1 for normal inputs."""
        config = MemoryConfig()
        # All scores at maximum
        score = (
            config.semantic_weight * 1.0
            + config.recency_weight * 1.0
            + config.frequency_weight * 1.0
            + config.importance_weight * 1.0
        )
        self.assertAlmostEqual(score, 1.0, places=5)

        # All scores at minimum
        score = (
            config.semantic_weight * 0.0
            + config.recency_weight * 0.0
            + config.frequency_weight * 0.0
            + config.importance_weight * 0.0
        )
        self.assertAlmostEqual(score, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
