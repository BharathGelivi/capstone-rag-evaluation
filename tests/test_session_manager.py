"""
Unit tests for Session Manager.

Tests session CRUD, switching, export/import, and activity tracking.
"""

import json
import shutil
import tempfile
import unittest

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore
from src.memory.session_manager import SessionManager


class TestSessionManager(unittest.TestCase):
    """Tests for SessionManager operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = MemoryConfig(
            persistence_directory=self.tmpdir,
            collection_name="test_sessions",
        )
        self.store = MemoryStore(self.config)
        self.store.initialize()
        self.sm = SessionManager(self.store, self.config)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_session(self):
        session = self.sm.create_session("My Session")
        self.assertIsNotNone(session)
        self.assertEqual(session.title, "My Session")
        self.assertEqual(self.sm.current_session_id, session.session_id)

    def test_delete_session(self):
        session = self.sm.create_session("To Delete")
        self.assertTrue(self.sm.delete_session(session.session_id))
        # Should auto-create a new session
        self.assertIsNotNone(self.sm.current_session_id)

    def test_rename_session(self):
        session = self.sm.create_session("Old")
        self.assertTrue(self.sm.rename_session(session.session_id, "New"))
        retrieved = self.store.get_session(session.session_id)
        self.assertEqual(retrieved.title, "New")

    def test_switch_session(self):
        s1 = self.sm.create_session("S1")
        s2 = self.sm.create_session("S2")
        self.assertEqual(self.sm.current_session_id, s2.session_id)

        self.sm.switch_session(s1.session_id)
        self.assertEqual(self.sm.current_session_id, s1.session_id)

    def test_list_sessions(self):
        self.sm.create_session("A")
        self.sm.create_session("B")
        sessions = self.sm.list_sessions()
        self.assertEqual(len(sessions), 2)

    def test_ensure_session(self):
        session_id = self.sm.ensure_session()
        self.assertIsNotNone(session_id)
        # Calling again should return the same session
        self.assertEqual(self.sm.ensure_session(), session_id)

    def test_update_activity(self):
        session = self.sm.create_session("Activity Test")
        old_count = session.question_count
        self.sm.update_session_activity(session.session_id)
        updated = self.store.get_session(session.session_id)
        self.assertEqual(updated.question_count, old_count + 1)

    def test_export_import_session(self):
        session = self.sm.create_session("Export Test")
        # Add a memory to the session
        self.store.save_memory(
            MemoryEntry(
                memory_id="m1",
                session_id=session.session_id,
                question="What is X?",
                answer="X is Y.",
            )
        )

        # Export
        export_data = self.sm.export_session(session.session_id, format="json")
        self.assertIn("session", export_data)
        self.assertIn("memories", export_data)
        self.assertEqual(len(export_data["memories"]), 1)

        # Import
        imported = self.sm.import_session(export_data)
        self.assertIsNotNone(imported)
        self.assertIn("imported", imported.title)

    def test_export_markdown(self):
        session = self.sm.create_session("MD Test")
        self.store.save_memory(
            MemoryEntry(
                memory_id="m1",
                session_id=session.session_id,
                question="Q1?",
                answer="A1.",
            )
        )
        export_data = self.sm.export_session(session.session_id, format="markdown")
        self.assertIn("markdown", export_data)
        self.assertIn("MD Test", export_data["markdown"])

    def test_export_csv(self):
        session = self.sm.create_session("CSV Test")
        self.store.save_memory(
            MemoryEntry(
                memory_id="m1",
                session_id=session.session_id,
                question="Q1?",
                answer="A1.",
            )
        )
        export_data = self.sm.export_session(session.session_id, format="csv")
        self.assertIn("csv", export_data)
        self.assertIn("Q1?", export_data["csv"])


if __name__ == "__main__":
    unittest.main()
