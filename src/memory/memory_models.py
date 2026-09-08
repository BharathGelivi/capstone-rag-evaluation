"""
Memory Data Models.

Defines the core data structures for the memory system using dataclasses.
All models are JSON-serializable and designed for persistence.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """Types of memory entries."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SUMMARY = "summary"


@dataclass
class MemoryConfig:
    """Configuration for the memory system.

    Loaded from configs/memory.yaml or set programmatically.
    """
    short_memory_size: int = 10
    similarity_threshold: float = 0.6
    memory_weight: float = 0.3
    summarization_threshold: int = 20
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    persistence_directory: str = "./db/memory"
    auto_save: bool = True
    auto_summary: bool = True
    enable_memory: bool = True

    # Ranking weights
    semantic_weight: float = 0.5
    recency_weight: float = 0.2
    frequency_weight: float = 0.2
    importance_weight: float = 0.1

    # Memory collection name
    collection_name: str = "xrag_memory"

    @classmethod
    def from_yaml(cls, path: str) -> "MemoryConfig":
        """Load configuration from a YAML file."""
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception:
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryEntry:
    """Represents a single memory entry (one Q&A interaction).

    Contains all information needed to reconstruct and search
    past interactions.
    """
    memory_id: str
    session_id: str
    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    memory_type: MemoryType = MemoryType.LONG_TERM

    # RAG pipeline references
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    claim_ids: List[str] = field(default_factory=list)
    trace_id: Optional[str] = None

    # Metadata
    importance_score: float = 1.0
    access_count: int = 0
    last_accessed: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        if "memory_type" in data and isinstance(data["memory_type"], str):
            data["memory_type"] = MemoryType(data["memory_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MemorySearchResult:
    """Result from a memory search operation.

    Wraps a MemoryEntry with additional scoring/ranking information.
    """
    memory: MemoryEntry
    semantic_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0
    importance_score: float = 0.0
    final_score: float = 0.0
    retrieval_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "semantic_score": self.semantic_score,
            "recency_score": self.recency_score,
            "frequency_score": self.frequency_score,
            "importance_score": self.importance_score,
            "final_score": self.final_score,
            "retrieval_reason": self.retrieval_reason,
        }


@dataclass
class SessionInfo:
    """Metadata about a conversation session."""
    session_id: str
    title: str = "New Session"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    question_count: int = 0
    memory_count: int = 0
    trace_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class MemorySummary:
    """Summary of a batch of memories (auto-generated when threshold exceeded).

    Replaces older individual memories with a compressed representation.
    """
    summary_id: str
    session_id: str
    summary_text: str
    key_topics: List[str] = field(default_factory=list)
    important_entities: List[str] = field(default_factory=list)
    important_claims: List[str] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)
    question_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemorySummary":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
