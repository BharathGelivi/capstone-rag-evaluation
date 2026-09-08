"""
Unit tests for Memory Persistence.

Tests that memory data survives store re-initialization
and that JSON serialization is consistent.
"""

import json
import shutil
import tempfile
import unittest

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySummary,
    MemoryType,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore


class TestPersistence(unittest.TestCase):
    """Tests for data persistence across store instances."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(
            persistence_directory=self.tmpdir,
            collection_name="test_persist",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_memory_persists_across_restart(self):
        """Memories should survive store re-initialization."""
        # Save with first store
        store1 = MemoryStore(self.config)
        store1.initialize()
        store1.save_memory(
            MemoryEntry(
                memory_id="persist_1",
                session_id="s1",
                question="Test question",
                answer="Test answer",
            )
        )
        count1 = store1.memory_count()
        self.assertEqual(count1, 1)

        # Load with second store
        store2 = MemoryStore(self.config)
        store2.initialize()
        count2 = store2.memory_count()
        self.assertEqual(count2, 1)

        retrieved = store2.get_memory("persist_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.question, "Test question")

    def test_relative_persistence_directory_resolves_to_project_root_not_cwd(self):
        """A relative persistence_directory (the dataclass default, and
        configs/memory.yaml's value, are both "./db/memory") must always
        resolve to <project_root>/db/memory regardless of the process's
        current working directory when MemoryStore was constructed --
        otherwise which database a launch sees (and can silently start
        writing into) depends on wherever the server happened to be started
        from, which reads exactly like "my deleted chats came back after a
        restart" (a different launcher just hit a different, untouched
        database) even though nothing was actually un-deleted.
        """
        import os

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        relative_config = MemoryConfig(
            persistence_directory="./db/memory_relative_path_test_marker",
            collection_name="test_relative_path",
        )
        store = MemoryStore(relative_config)
        try:
            self.assertEqual(
                store._persist_dir,
                os.path.join(project_root, "db", "memory_relative_path_test_marker"),
            )
        finally:
            shutil.rmtree(store._persist_dir, ignore_errors=True)

    def test_session_persists_across_restart(self):
        """Sessions should survive store re-initialization."""
        store1 = MemoryStore(self.config)
        store1.initialize()
        store1.save_session(
            SessionInfo(session_id="ps1", title="Persistent Session")
        )

        store2 = MemoryStore(self.config)
        store2.initialize()
        session = store2.get_session("ps1")
        self.assertIsNotNone(session)
        self.assertEqual(session.title, "Persistent Session")

    def test_summary_persists_across_restart(self):
        """Summaries should survive store re-initialization."""
        store1 = MemoryStore(self.config)
        store1.initialize()
        store1.save_summary(
            MemorySummary(
                summary_id="psum1",
                session_id="s1",
                summary_text="A persistent summary",
            )
        )

        store2 = MemoryStore(self.config)
        store2.initialize()
        summaries = store2.get_summaries("s1")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].summary_text, "A persistent summary")


class TestModelSerialization(unittest.TestCase):
    """Tests for JSON serialization of data models."""

    def test_memory_entry_roundtrip(self):
        entry = MemoryEntry(
            memory_id="m1",
            session_id="s1",
            question="Q",
            answer="A",
            memory_type=MemoryType.LONG_TERM,
            tags=["tag1", "tag2"],
            metadata={"key": "value"},
        )
        data = entry.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = MemoryEntry.from_dict(restored_data)
        self.assertEqual(restored.memory_id, "m1")
        self.assertEqual(restored.memory_type, MemoryType.LONG_TERM)
        self.assertEqual(restored.tags, ["tag1", "tag2"])

    def test_session_info_roundtrip(self):
        session = SessionInfo(
            session_id="s1",
            title="Test",
            question_count=5,
        )
        data = session.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = SessionInfo.from_dict(restored_data)
        self.assertEqual(restored.session_id, "s1")
        self.assertEqual(restored.title, "Test")
        self.assertEqual(restored.question_count, 5)

    def test_memory_summary_roundtrip(self):
        summary = MemorySummary(
            summary_id="sum1",
            session_id="s1",
            summary_text="Summary text",
            key_topics=["topic1"],
            important_entities=["entity1"],
        )
        data = summary.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = MemorySummary.from_dict(restored_data)
        self.assertEqual(restored.summary_id, "sum1")
        self.assertEqual(restored.key_topics, ["topic1"])


if __name__ == "__main__":
    unittest.main()
