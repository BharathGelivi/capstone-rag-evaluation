"""
Unit tests for the Memory Store.

Tests short-term memory, long-term persistence, session management,
and CRUD operations on memory entries.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySummary,
    MemoryType,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore


class TestMemoryStoreShortTerm(unittest.TestCase):
    """Tests for the short-term (ring buffer) memory."""

    def setUp(self):
        self.config = MemoryConfig(
            short_memory_size=3,
            persistence_directory=tempfile.mkdtemp(),
        )
        self.store = MemoryStore(self.config)

    def tearDown(self):
        shutil.rmtree(self.config.persistence_directory, ignore_errors=True)

    def test_add_and_get_short_term(self):
        entry = MemoryEntry(
            memory_id="mem_1",
            session_id="s1",
            question="What is X?",
            answer="X is Y.",
        )
        self.store.add_to_short_term(entry)
        memories = self.store.get_short_term_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0].memory_id, "mem_1")

    def test_short_term_overflow(self):
        """Buffer size is 3, so adding 4 should evict the oldest."""
        for i in range(4):
            self.store.add_to_short_term(
                MemoryEntry(
                    memory_id=f"mem_{i}",
                    session_id="s1",
                    question=f"Q{i}",
                    answer=f"A{i}",
                )
            )
        memories = self.store.get_short_term_memories()
        self.assertEqual(len(memories), 3)
        # Oldest (mem_0) should be gone
        ids = [m.memory_id for m in memories]
        self.assertNotIn("mem_0", ids)
        self.assertIn("mem_3", ids)

    def test_clear_short_term(self):
        self.store.add_to_short_term(
            MemoryEntry(memory_id="mem_1", session_id="s1", question="Q", answer="A")
        )
        self.store.clear_short_term()
        self.assertEqual(len(self.store.get_short_term_memories()), 0)


class TestMemoryStoreLongTerm(unittest.TestCase):
    """Tests for the long-term (ChromaDB) persistence layer."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(
            persistence_directory=self.tmpdir,
            collection_name="test_memory",
        )
        self.store = MemoryStore(self.config)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_memory(self):
        entry = MemoryEntry(
            memory_id="mem_test",
            session_id="s1",
            question="What is Python?",
            answer="A programming language.",
        )
        self.store.save_memory(entry)
        retrieved = self.store.get_memory("mem_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.question, "What is Python?")

    def test_delete_memory(self):
        entry = MemoryEntry(
            memory_id="mem_del",
            session_id="s1",
            question="Q",
            answer="A",
        )
        self.store.save_memory(entry)
        self.assertTrue(self.store.delete_memory("mem_del"))
        self.assertIsNone(self.store.get_memory("mem_del"))

    def test_memory_count(self):
        self.assertEqual(self.store.memory_count(), 0)
        self.store.save_memory(
            MemoryEntry(memory_id="m1", session_id="s1", question="Q1", answer="A1")
        )
        self.assertEqual(self.store.memory_count(), 1)

    def test_clear_all_memory(self):
        for i in range(3):
            self.store.save_memory(
                MemoryEntry(
                    memory_id=f"m{i}", session_id="s1", question=f"Q{i}", answer=f"A{i}"
                )
            )
        self.assertEqual(self.store.memory_count(), 3)
        cleared = self.store.clear_all_memory()
        self.assertEqual(cleared, 3)
        self.assertEqual(self.store.memory_count(), 0)

    def test_get_session_memories(self):
        for i in range(3):
            self.store.save_memory(
                MemoryEntry(
                    memory_id=f"m_s1_{i}", session_id="s1", question=f"Q{i}", answer=f"A{i}"
                )
            )
        self.store.save_memory(
            MemoryEntry(memory_id="m_s2_0", session_id="s2", question="Q", answer="A")
        )
        s1_memories = self.store.get_session_memories("s1")
        self.assertEqual(len(s1_memories), 3)

    def test_clear_session_memory(self):
        self.store.save_memory(
            MemoryEntry(memory_id="m_a", session_id="s1", question="Q1", answer="A1")
        )
        self.store.save_memory(
            MemoryEntry(memory_id="m_b", session_id="s2", question="Q2", answer="A2")
        )
        cleared = self.store.clear_session_memory("s1")
        self.assertEqual(cleared, 1)
        self.assertEqual(self.store.memory_count(), 1)


class TestMemoryStoreSession(unittest.TestCase):
    """Tests for session persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(
            persistence_directory=self.tmpdir,
            collection_name="test_sessions",
        )
        self.store = MemoryStore(self.config)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_session(self):
        session = SessionInfo(session_id="s1", title="Test Session")
        self.store.save_session(session)
        retrieved = self.store.get_session("s1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Test Session")

    def test_list_sessions(self):
        self.store.save_session(SessionInfo(session_id="s1", title="Session 1"))
        self.store.save_session(SessionInfo(session_id="s2", title="Session 2"))
        sessions = self.store.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_delete_session(self):
        self.store.save_session(SessionInfo(session_id="s1", title="To Delete"))
        self.assertTrue(self.store.delete_session("s1"))
        self.assertIsNone(self.store.get_session("s1"))

    def test_rename_session(self):
        self.store.save_session(SessionInfo(session_id="s1", title="Old Name"))
        self.assertTrue(self.store.rename_session("s1", "New Name"))
        self.assertEqual(self.store.get_session("s1").title, "New Name")

    def test_session_persistence(self):
        """Sessions should survive store re-initialization."""
        self.store.save_session(SessionInfo(session_id="s1", title="Persistent"))

        # Create a new store pointing to the same directory
        store2 = MemoryStore(self.config)
        store2.initialize()
        retrieved = store2.get_session("s1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Persistent")


class TestMemoryStoreSummary(unittest.TestCase):
    """Tests for summary persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(persistence_directory=self.tmpdir, collection_name="test_sum")
        self.store = MemoryStore(self.config)
        self.store.initialize()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_get_summary(self):
        summary = MemorySummary(
            summary_id="sum_1",
            session_id="s1",
            summary_text="A summary",
            key_topics=["topic1", "topic2"],
        )
        self.store.save_summary(summary)
        summaries = self.store.get_summaries("s1")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].summary_text, "A summary")


if __name__ == "__main__":
    unittest.main()
