"""
Memory Store Module.

Provides persistent storage for memory entries using ChromaDB for
long-term semantic memory and JSON files for session metadata.
Handles both short-term (in-memory ring buffer) and long-term
(ChromaDB-backed) storage.
"""

import json
import logging
import os
from collections import deque
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySummary,
    MemoryType,
    SessionInfo,
)
from src.memory.memory_utils import (
    ensure_directory,
    generate_memory_id,
    get_timestamp,
    load_json,
    save_json,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction


class _NullEmbeddingFunction(ChromaEmbeddingFunction):
    """Null embedding function to prevent ChromaDB from using its default 384-dim model."""
    def __call__(self, input: List[str]) -> List[List[float]]:
        return [[0.0] * 768 for _ in input]

    def name(self) -> str:
        return "null_embedding_function"


class MemoryStore:
    """Persistent storage layer for the memory system.

    Short-term memory: in-memory deque (last N interactions).
    Long-term memory: ChromaDB collection with embeddings.
    Session metadata: JSON files on disk.
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self._short_term: deque = deque(maxlen=self.config.short_memory_size)
        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[Any] = None
        self._sessions: Dict[str, SessionInfo] = {}
        self._summaries: Dict[str, MemorySummary] = {}
        self._initialized = False

        # Persistence paths. A relative persistence_directory (the dataclass
        # default, and configs/memory.yaml's value, are both "./db/memory")
        # is resolved against the project root, not the process's current
        # working directory -- otherwise which db/memory a launch sees (and
        # writes to) silently depends on where the server was started from,
        # which read like "deleted sessions come back after a restart" when
        # what actually happened was a different launcher hit a different,
        # untouched database.
        persist_dir = self.config.persistence_directory
        if not os.path.isabs(persist_dir):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            persist_dir = os.path.normpath(os.path.join(project_root, persist_dir))
        self._persist_dir = persist_dir
        self._sessions_file = os.path.join(self._persist_dir, "sessions.json")
        self._summaries_file = os.path.join(self._persist_dir, "summaries.json")

    def initialize(self) -> None:
        """Initialize the memory store, loading persisted data."""
        if self._initialized:
            return

        ensure_directory(self._persist_dir)
        mem_logger.info("Initializing MemoryStore at %s", self._persist_dir)

        # Initialize ChromaDB for long-term memory
        try:
            self._chroma_client = chromadb.PersistentClient(
                path=os.path.join(self._persist_dir, "chroma")
            )
            try:
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_NullEmbeddingFunction(),
                )
            except Exception as ve:
                if "Embedding function conflict" in str(ve) or "conflict" in str(ve).lower():
                    mem_logger.warning("Re-creating memory collection to match 768-dim embedding model.")
                    try:
                        self._chroma_client.delete_collection(name=self.config.collection_name)
                    except Exception:
                        pass
                    self._collection = self._chroma_client.create_collection(
                        name=self.config.collection_name,
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=_NullEmbeddingFunction(),
                    )
                else:
                    raise
            mem_logger.info(
                "ChromaDB memory collection initialized. Count: %d",
                self._collection.count(),
            )
        except Exception as e:
            logger.error("Failed to initialize ChromaDB for memory: %s", e)
            mem_logger.error("ChromaDB init failed: %s", e)
            raise

        # Load persisted sessions
        self._load_sessions()
        self._load_summaries()
        self._initialized = True
        mem_logger.info("MemoryStore initialized successfully.")

    # ------------------------------------------------------------------
    # Short-term memory
    # ------------------------------------------------------------------

    def add_to_short_term(self, entry: MemoryEntry) -> None:
        """Add an entry to the short-term memory buffer."""
        self._short_term.append(entry)
        mem_logger.debug("Added to short-term memory: %s", entry.memory_id)

    def get_short_term_memories(self) -> List[MemoryEntry]:
        """Get all entries in the short-term memory buffer."""
        return list(self._short_term)

    def clear_short_term(self) -> None:
        """Clear the short-term memory buffer."""
        self._short_term.clear()
        mem_logger.info("Short-term memory cleared.")

    # ------------------------------------------------------------------
    # Long-term memory (ChromaDB)
    # ------------------------------------------------------------------

    def save_memory(
        self,
        entry: MemoryEntry,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Save a memory entry to long-term storage.

        Args:
            entry: The memory entry to persist.
            embedding: Pre-computed embedding vector. If None, ChromaDB
                       will store only the document text (no vector search).

        Returns:
            The memory_id of the saved entry.
        """
        if not self._initialized:
            self.initialize()

        # Also add to short-term
        self.add_to_short_term(entry)

        # Save to ChromaDB
        metadata = {
            "session_id": entry.session_id,
            "timestamp": entry.timestamp,
            "memory_type": entry.memory_type.value,
            "importance_score": entry.importance_score,
            "access_count": entry.access_count,
            "trace_id": entry.trace_id or "",
            "tags": json.dumps(entry.tags),
            "chunk_ids": json.dumps(entry.retrieved_chunk_ids),
            "claim_ids": json.dumps(entry.claim_ids),
        }

        document = f"Q: {entry.question}\nA: {entry.answer}"

        try:
            upsert_kwargs: Dict[str, Any] = {
                "ids": [entry.memory_id],
                "documents": [document],
                "metadatas": [metadata],
            }
            if embedding is not None:
                upsert_kwargs["embeddings"] = [embedding]

            self._collection.upsert(**upsert_kwargs)

            mem_logger.info(
                "Saved memory %s to long-term store (session: %s)",
                entry.memory_id,
                entry.session_id,
            )
        except Exception as e:
            logger.error("Failed to save memory %s: %s", entry.memory_id, e)
            mem_logger.error("Memory save failed: %s — %s", entry.memory_id, e)
            raise

        return entry.memory_id

    def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a single memory entry by ID."""
        if not self._initialized:
            self.initialize()

        try:
            result = self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )
            if not result or not result["ids"]:
                return None
            return self._chroma_result_to_entry(result, 0)
        except Exception as e:
            logger.error("Failed to get memory %s: %s", memory_id, e)
            return None

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry by ID."""
        if not self._initialized:
            self.initialize()

        try:
            self._collection.delete(ids=[memory_id])
            mem_logger.info("Deleted memory: %s", memory_id)
            return True
        except Exception as e:
            logger.error("Failed to delete memory %s: %s", memory_id, e)
            return False

    def update_memory(self, entry: MemoryEntry, embedding: Optional[List[float]] = None) -> bool:
        """Update an existing memory entry."""
        try:
            self.save_memory(entry, embedding)
            mem_logger.info("Updated memory: %s", entry.memory_id)
            return True
        except Exception:
            return False

    def search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search long-term memory using an embedding vector.

        Returns raw ChromaDB results for further processing by MemoryRetriever.
        """
        if not self._initialized:
            self.initialize()

        where_filter = None
        if session_id:
            where_filter = {"session_id": session_id}

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(1, self._collection.count())),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.error("Memory search failed: %s", e)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_session_memories(
        self, session_id: str, limit: int = 100
    ) -> List[MemoryEntry]:
        """Get all memories for a specific session."""
        if not self._initialized:
            self.initialize()

        try:
            results = self._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"],
                limit=limit,
            )
            entries = []
            if results and results["ids"]:
                for i in range(len(results["ids"])):
                    entry = self._chroma_result_to_entry(results, i)
                    if entry:
                        entries.append(entry)
            # Sort by timestamp
            entries.sort(key=lambda e: e.timestamp)
            return entries
        except Exception as e:
            logger.error("Failed to get session memories: %s", e)
            return []

    def get_all_memories(self, limit: int = 1000) -> List[MemoryEntry]:
        """Get all memories across all sessions."""
        if not self._initialized:
            self.initialize()

        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.get(
                include=["documents", "metadatas"],
                limit=min(limit, count),
            )
            entries = []
            if results and results["ids"]:
                for i in range(len(results["ids"])):
                    entry = self._chroma_result_to_entry(results, i)
                    if entry:
                        entries.append(entry)
            entries.sort(key=lambda e: e.timestamp)
            return entries
        except Exception as e:
            logger.error("Failed to get all memories: %s", e)
            return []

    def clear_session_memory(self, session_id: str) -> int:
        """Clear all memories for a specific session. Returns count deleted."""
        if not self._initialized:
            self.initialize()

        try:
            # Get IDs for this session
            results = self._collection.get(
                where={"session_id": session_id},
                include=[],
            )
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                count = len(results["ids"])
                mem_logger.info("Cleared %d memories for session %s", count, session_id)
                return count
            return 0
        except Exception as e:
            logger.error("Failed to clear session memory: %s", e)
            return 0

    def clear_all_memory(self) -> int:
        """Clear all memories. Returns count deleted."""
        if not self._initialized:
            self.initialize()

        try:
            count = self._collection.count()
            if count > 0:
                # Delete and recreate collection
                self._chroma_client.delete_collection(self.config.collection_name)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            self._short_term.clear()
            mem_logger.info("Cleared all %d memories.", count)
            return count
        except Exception as e:
            logger.error("Failed to clear all memory: %s", e)
            return 0

    def memory_count(self) -> int:
        """Get the total number of memories stored."""
        if not self._initialized:
            self.initialize()
        try:
            return self._collection.count()
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Session management (persisted as JSON)
    # ------------------------------------------------------------------

    def save_session(self, session: SessionInfo) -> None:
        """Save or update a session."""
        self._sessions[session.session_id] = session
        self._persist_sessions()
        mem_logger.info("Session saved: %s (%s)", session.session_id, session.title)

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions, sorted by last activity (most recent first)."""
        sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its memories."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._persist_sessions()
            # Also clear memories for this session
            self.clear_session_memory(session_id)
            mem_logger.info("Session deleted: %s", session_id)
            return True
        return False

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Rename a session."""
        if session_id in self._sessions:
            self._sessions[session_id].title = new_title
            self._persist_sessions()
            mem_logger.info("Session renamed: %s → %s", session_id, new_title)
            return True
        return False

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def save_summary(self, summary: MemorySummary) -> None:
        """Save a memory summary."""
        self._summaries[summary.summary_id] = summary
        self._persist_summaries()
        mem_logger.info("Summary saved: %s", summary.summary_id)

    def get_summaries(self, session_id: Optional[str] = None) -> List[MemorySummary]:
        """Get summaries, optionally filtered by session."""
        summaries = list(self._summaries.values())
        if session_id:
            summaries = [s for s in summaries if s.session_id == session_id]
        return summaries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chroma_result_to_entry(self, result: Dict, index: int) -> Optional[MemoryEntry]:
        """Convert a ChromaDB result at the given index to a MemoryEntry."""
        try:
            memory_id = result["ids"][index]
            doc = result["documents"][index] if result.get("documents") else ""
            meta = result["metadatas"][index] if result.get("metadatas") else {}

            # Parse Q/A from document
            question = ""
            answer = ""
            if doc:
                if "\nA: " in doc:
                    parts = doc.split("\nA: ", 1)
                    question = parts[0].replace("Q: ", "", 1)
                    answer = parts[1]
                else:
                    question = doc

            return MemoryEntry(
                memory_id=memory_id,
                session_id=meta.get("session_id", ""),
                question=question,
                answer=answer,
                timestamp=meta.get("timestamp", ""),
                memory_type=MemoryType(meta.get("memory_type", "long_term")),
                importance_score=float(meta.get("importance_score", 1.0)),
                access_count=int(meta.get("access_count", 0)),
                trace_id=meta.get("trace_id") or None,
                tags=json.loads(meta.get("tags", "[]")),
                retrieved_chunk_ids=json.loads(meta.get("chunk_ids", "[]")),
                claim_ids=json.loads(meta.get("claim_ids", "[]")),
            )
        except Exception as e:
            logger.error("Failed to parse memory from ChromaDB result: %s", e)
            return None

    def _load_sessions(self) -> None:
        """Load sessions from the JSON persistence file."""
        data = load_json(self._sessions_file)
        if data and isinstance(data, list):
            for item in data:
                try:
                    session = SessionInfo.from_dict(item)
                    self._sessions[session.session_id] = session
                except Exception as e:
                    logger.warning("Skipping invalid session data: %s", e)
        mem_logger.info("Loaded %d sessions from disk.", len(self._sessions))

    def _persist_sessions(self) -> None:
        """Persist sessions to disk."""
        data = [s.to_dict() for s in self._sessions.values()]
        save_json(data, self._sessions_file)

    def _load_summaries(self) -> None:
        """Load summaries from the JSON persistence file."""
        data = load_json(self._summaries_file)
        if data and isinstance(data, list):
            for item in data:
                try:
                    summary = MemorySummary.from_dict(item)
                    self._summaries[summary.summary_id] = summary
                except Exception as e:
                    logger.warning("Skipping invalid summary data: %s", e)
        mem_logger.info("Loaded %d summaries from disk.", len(self._summaries))

    def _persist_summaries(self) -> None:
        """Persist summaries to disk."""
        data = [s.to_dict() for s in self._summaries.values()]
        save_json(data, self._summaries_file)
