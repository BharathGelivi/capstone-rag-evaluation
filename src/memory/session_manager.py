"""
Session Manager Module.

Manages conversation sessions including creation, deletion, renaming,
switching, export, and import. Each session groups related interactions
and their memory entries.
"""

import logging
from typing import Any, Dict, List, Optional

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore, _locked
from src.memory.memory_utils import (
    generate_session_id,
    generate_memory_id,
    get_timestamp,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


class SessionManager:
    """Manages conversation sessions.

    Provides CRUD operations on sessions and handles session switching,
    export, and import functionality.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self.store = store
        self._lock = store._lock
        self.config = config or MemoryConfig()
        self._current_session_id: Optional[str] = None

    @property
    def current_session_id(self) -> Optional[str]:
        """Get the current active session ID."""
        return self._current_session_id

    @_locked
    def create_session(self, title: str = "New Session") -> SessionInfo:
        """Create a new session.

        Args:
            title: Human-readable session title.

        Returns:
            The newly created SessionInfo.
        """
        session = SessionInfo(
            session_id=generate_session_id(),
            title=title,
            created_at=get_timestamp(),
            last_activity=get_timestamp(),
        )
        self.store.save_session(session)
        self._current_session_id = session.session_id
        mem_logger.info("Created session: %s (%s)", session.session_id, title)
        return session

    @_locked
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its memories.

        If the deleted session is the current session, switches to the
        most recent remaining session or creates a new one.
        """
        success = self.store.delete_session(session_id)
        if success and self._current_session_id == session_id:
            sessions = self.store.list_sessions()
            if sessions:
                self._current_session_id = sessions[0].session_id
            else:
                new_session = self.create_session("New Session")
                self._current_session_id = new_session.session_id
        mem_logger.info("Deleted session: %s (success: %s)", session_id, success)
        return success

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a session."""
        return self.store.rename_session(session_id, new_title)

    @_locked
    def switch_session(self, session_id: str) -> Optional[SessionInfo]:
        """Switch to a different session.

        Returns the switched-to session info, or None if not found.
        """
        session = self.store.get_session(session_id)
        if session:
            self._current_session_id = session_id
            mem_logger.info("Switched to session: %s", session_id)
        return session

    def get_current_session(self) -> Optional[SessionInfo]:
        """Get the current active session."""
        if self._current_session_id:
            return self.store.get_session(self._current_session_id)
        return None

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions sorted by last activity."""
        return self.store.list_sessions()

    @_locked
    def update_session_activity(self, session_id: Optional[str] = None) -> None:
        """Update the last activity timestamp of a session.

        Used after each interaction to keep session metadata current.
        """
        sid = session_id or self._current_session_id
        if not sid:
            return

        session = self.store.get_session(sid)
        if session:
            session.last_activity = get_timestamp()
            session.question_count += 1
            self.store.save_session(session)

    @_locked
    def increment_memory_count(self, session_id: Optional[str] = None) -> None:
        """Increment the memory count for a session."""
        sid = session_id or self._current_session_id
        if not sid:
            return

        session = self.store.get_session(sid)
        if session:
            session.memory_count += 1
            self.store.save_session(session)

    @_locked
    def increment_trace_count(self, session_id: Optional[str] = None) -> None:
        """Increment the trace count for a session."""
        sid = session_id or self._current_session_id
        if not sid:
            return

        session = self.store.get_session(sid)
        if session:
            session.trace_count += 1
            self.store.save_session(session)

    @_locked
    def ensure_session(self) -> str:
        """Ensure a session exists, creating one if necessary.

        Returns the current session ID.
        """
        if self._current_session_id:
            session = self.store.get_session(self._current_session_id)
            if session:
                return self._current_session_id

        # Try to resume the most recent session
        sessions = self.store.list_sessions()
        if sessions:
            self._current_session_id = sessions[0].session_id
            return self._current_session_id

        # Create a new session
        session = self.create_session("New Session")
        return session.session_id

    def export_session(
        self, session_id: str, format: str = "json"
    ) -> Dict[str, Any]:
        """Export a session with all its memories.

        Args:
            session_id: The session to export.
            format:     Export format ('json', 'markdown', 'csv').

        Returns:
            Dictionary with export data and metadata.
        """
        session = self.store.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        memories = self.store.get_session_memories(session_id, limit=None)
        summaries = self.store.get_summaries(session_id)

        export_data = {
            "session": session.to_dict(),
            "memories": [m.to_dict() for m in memories],
            "summaries": [s.to_dict() for s in summaries],
            "export_timestamp": get_timestamp(),
            "memory_count": len(memories),
            "format": format,
        }

        if format == "markdown":
            md_parts = [
                f"# Session: {session.title}",
                f"**Created:** {session.created_at}",
                f"**Questions:** {session.question_count}",
                "",
                "---",
                "",
            ]
            for m in memories:
                md_parts.extend([
                    f"### Q: {m.question}",
                    "",
                    f"{m.answer}",
                    "",
                    f"*{m.timestamp}*",
                    "",
                    "---",
                    "",
                ])
            export_data["markdown"] = "\n".join(md_parts)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["timestamp", "question", "answer", "trace_id"])
            for m in memories:
                writer.writerow([m.timestamp, m.question, m.answer, m.trace_id or ""])
            export_data["csv"] = output.getvalue()

        mem_logger.info(
            "Exported session %s (%d memories, format: %s)",
            session_id, len(memories), format,
        )
        return export_data

    @_locked
    def import_session(self, data: Dict[str, Any], embed_text=None) -> SessionInfo:
        """Import a session from exported data.

        Creates a new session with a new ID and imports all memories.

        Args:
            data: Export data dictionary (from export_session).

        Returns:
            The newly created SessionInfo.
        """
        session_data = data.get("session", {})
        title = session_data.get("title", "Imported Session")
        new_session = self.create_session(f"{title} (imported)")

        # Import memories
        for mem_data in data.get("memories", []):
            entry = MemoryEntry.from_dict(dict(mem_data))
            existing = self.store._collection.get(ids=[entry.memory_id], include=["embeddings"])
            vectors = existing.get("embeddings")
            embedding = list(vectors[0]) if vectors is not None and len(vectors) else None
            if embedding is None and embed_text is not None:
                embedding = embed_text(f"{entry.question} {entry.answer}")
            entry.metadata["imported_from_memory_id"] = entry.memory_id
            entry.memory_id = generate_memory_id()
            entry.session_id = new_session.session_id
            self.store.save_memory(entry, embedding)

        # Update session counts
        new_session.question_count = len(data.get("memories", []))
        new_session.memory_count = len(data.get("memories", []))
        self.store.save_session(new_session)

        mem_logger.info(
            "Imported session: %s (%d memories)", new_session.session_id,
            len(data.get("memories", [])),
        )
        return new_session
