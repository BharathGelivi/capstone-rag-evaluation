"""
Memory Manager Module.

Top-level orchestrator for the memory system. Coordinates between
MemoryStore, MemoryRetriever, MemorySummarizer, and SessionManager.
Provides a unified API for the UI and pipeline integration.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySearchResult,
    MemoryType,
    SessionInfo,
)
from src.memory.memory_store import MemoryStore
from src.memory.memory_retriever import MemoryRetriever
from src.memory.memory_summarizer import MemorySummarizer
from src.memory.session_manager import SessionManager
from src.memory.memory_utils import (
    generate_memory_id,
    get_timestamp,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


class MemoryManager:
    """Unified interface for the X-RAG memory system.

    Coordinates all memory components and provides:
    - Memory save/search/delete operations
    - Session management
    - Automatic summarization
    - Integration with the RAG pipeline

    Usage:
        manager = MemoryManager()
        manager.initialize()
        session_id = manager.ensure_session()
        manager.save_interaction(question, answer, session_id=session_id)
        relevant_memories = manager.search_memory(new_question)
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or self._load_config()
        self.store = MemoryStore(self.config)
        self.retriever = MemoryRetriever(self.store, self.config)
        self.summarizer = MemorySummarizer(self.config)
        self.session_manager = SessionManager(self.store, self.config)
        self._initialized = False

    @staticmethod
    def _load_config() -> MemoryConfig:
        """Load configuration from YAML file if available.

        Resolved against the project root, not the process's current working
        directory: a relative ``"configs/memory.yaml"`` silently misses (and
        ``from_yaml`` silently falls back to defaults on any exception,
        including FileNotFoundError) whenever the server is launched from a
        directory other than the repo root -- which in turn falls back to
        the *default* ``persistence_directory``, itself relative, so the
        memory database silently moves depending on how the process was
        started. Same failure mode either way; anchoring the yaml lookup to
        the project root closes it here, and MemoryStore anchors
        ``persistence_directory`` the same way as a second line of defense.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(project_root, "configs", "memory.yaml")
        if os.path.exists(config_path):
            return MemoryConfig.from_yaml(config_path)
        return MemoryConfig()

    def initialize(self) -> None:
        """Initialize all memory components."""
        if self._initialized:
            return
        self.store.initialize()
        self._initialized = True
        mem_logger.info("MemoryManager fully initialized.")

    def ensure_session(self) -> str:
        """Ensure a session exists, returning the current session ID."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.ensure_session()

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def save_interaction(
        self,
        question: str,
        answer: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        claim_ids: Optional[List[str]] = None,
        importance_score: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Save a Q&A interaction to memory.

        This is the primary method called after each RAG pipeline run.
        Creates a MemoryEntry, computes its embedding, saves to the store,
        and triggers summarization if needed.

        Args:
            question:           The user's question.
            answer:             The generated answer.
            session_id:         Session to save to (uses current if None).
            trace_id:           Associated RAG trace ID.
            retrieved_chunk_ids: IDs of chunks used in retrieval.
            claim_ids:          IDs of extracted claims.
            importance_score:   User or system importance rating (0-1+).
            tags:               Optional tags for categorization.
            metadata:           Additional metadata.

        Returns:
            The saved MemoryEntry.
        """
        if not self._initialized:
            self.initialize()

        if not self.config.enable_memory:
            # Return a dummy entry if memory is disabled
            return MemoryEntry(
                memory_id=generate_memory_id(),
                session_id=session_id or "",
                question=question,
                answer=answer,
            )

        sid = session_id or self.session_manager.ensure_session()

        # Create memory entry
        entry = MemoryEntry(
            memory_id=generate_memory_id(),
            session_id=sid,
            question=question,
            answer=answer,
            timestamp=get_timestamp(),
            memory_type=MemoryType.LONG_TERM,
            retrieved_chunk_ids=retrieved_chunk_ids or [],
            claim_ids=claim_ids or [],
            trace_id=trace_id,
            importance_score=importance_score,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Compute embedding for the Q&A pair
        try:
            combined_text = f"{question} {answer}"
            embedding = self.retriever.embed_text(combined_text)
        except Exception as e:
            logger.warning("Failed to compute memory embedding: %s", e)
            embedding = None

        # Save to store
        self.store.save_memory(entry, embedding)

        # Update session metadata
        self.session_manager.update_session_activity(sid)
        self.session_manager.increment_memory_count(sid)
        if trace_id:
            self.session_manager.increment_trace_count(sid)

        # Check if summarization is needed
        self._check_summarization(sid)

        mem_logger.info(
            "Saved interaction: %s (session: %s, trace: %s)",
            entry.memory_id, sid, trace_id,
        )

        return entry

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        """Search memories by semantic similarity.

        Args:
            query:      Search query text.
            top_k:      Maximum results to return.
            session_id: Optional session filter.

        Returns:
            Ranked list of MemorySearchResult objects.
        """
        if not self._initialized:
            self.initialize()

        if not self.config.enable_memory:
            return []

        return self.retriever.search_memory(
            query=query, top_k=top_k, session_id=session_id
        )

    def get_memory_context(
        self,
        query: str,
        top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> str:
        """Get formatted memory context for prompt injection.

        Returns a string ready to be prepended to the RAG context.
        """
        if not self._initialized:
            self.initialize()

        if not self.config.enable_memory:
            return ""

        return self.retriever.get_relevant_context(
            query=query, top_k=top_k, session_id=session_id
        )

    def format_memory_context(
        self,
        results: List[MemorySearchResult],
        session_id: Optional[str] = None,
    ) -> str:
        """Format memories you already searched for, without re-running the search.

        Use this together with :meth:`search_memory` instead of calling
        :meth:`get_memory_context` as a second step — see
        ``MemoryRetriever.get_relevant_context`` for why the double call is
        both slower and wrong.
        """
        if not self.config.enable_memory:
            return ""
        return self.retriever.format_context(results, session_id=session_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory entry."""
        if not self._initialized:
            self.initialize()
        return self.store.delete_memory(memory_id)

    def clear_session_memory(self, session_id: Optional[str] = None) -> int:
        """Clear all memories for a session."""
        if not self._initialized:
            self.initialize()
        sid = session_id or self.session_manager.current_session_id
        if sid:
            return self.store.clear_session_memory(sid)
        return 0

    def clear_all_memory(self) -> int:
        """Clear all memories across all sessions."""
        if not self._initialized:
            self.initialize()
        return self.store.clear_all_memory()

    # ------------------------------------------------------------------
    # Session operations (delegated to SessionManager)
    # ------------------------------------------------------------------

    def create_session(self, title: str = "New Session") -> SessionInfo:
        """Create a new session."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.create_session(title)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.delete_session(session_id)

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a session."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.rename_session(session_id, new_title)

    def switch_session(self, session_id: str) -> Optional[SessionInfo]:
        """Switch to a different session."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.switch_session(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.list_sessions()

    def get_current_session(self) -> Optional[SessionInfo]:
        """Get the current active session."""
        return self.session_manager.get_current_session()

    def get_session_memories(
        self, session_id: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Get all memories for a session."""
        if not self._initialized:
            self.initialize()
        sid = session_id or self.session_manager.current_session_id
        if sid:
            return self.store.get_session_memories(sid, limit=None)
        return []

    def export_session(
        self, session_id: str, format: str = "json"
    ) -> Dict[str, Any]:
        """Export a session with its data."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.export_session(session_id, format)

    def import_session(self, data: Dict[str, Any]) -> SessionInfo:
        """Import a session from exported data."""
        if not self._initialized:
            self.initialize()
        return self.session_manager.import_session(data, embed_text=self.retriever.embed_text)

    # ------------------------------------------------------------------
    # Statistics and info
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self._initialized:
            self.initialize()

        sessions = self.store.list_sessions()
        return {
            "total_memories": self.store.memory_count(),
            "total_sessions": len(sessions),
            "short_term_count": len(self.store.get_short_term_memories()),
            "summaries_count": len(self.store.get_summaries()),
            "memory_enabled": self.config.enable_memory,
            "current_session": self.session_manager.current_session_id,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_summarization(self, session_id: str) -> None:
        """Check if summarization is needed and trigger if so."""
        if not self.config.auto_summary:
            return

        memories = self.store.get_session_memories(session_id)
        if self.summarizer.should_summarize(len(memories)):
            # Summarize the oldest half of memories
            half = len(memories) // 2
            to_summarize = memories[:half]

            summary = self.summarizer.summarize(to_summarize, session_id)
            if summary:
                self.store.save_summary(summary)
                mem_logger.info(
                    "Auto-summary triggered for session %s: %d memories summarized",
                    session_id, len(to_summarize),
                )
