"""
Memory Utility Functions.

Helper functions for memory ID generation, time calculations,
logging, and serialization used across the memory package.
"""

import hashlib
import logging
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_memory_id() -> str:
    """Generate a unique memory ID."""
    return f"mem_{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session_{uuid.uuid4().hex[:8]}"


def generate_summary_id() -> str:
    """Generate a unique summary ID."""
    return f"summary_{uuid.uuid4().hex[:8]}"


def compute_text_hash(text: str) -> str:
    """Compute a SHA-256 hash for deduplication."""
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()


def compute_recency_score(timestamp_str: str, decay_hours: float = 24.0) -> float:
    """Compute a recency score (0-1) based on how recent an entry is.

    Uses exponential decay with configurable half-life.
    Score of 1.0 = just now, decays towards 0.0 over time.
    """
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_ago = max(0, (now - ts).total_seconds() / 3600.0)
        import math
        return math.exp(-0.693 * hours_ago / max(decay_hours, 0.01))
    except (ValueError, AttributeError):
        return 0.5


def compute_frequency_score(access_count: int, max_count: int = 100) -> float:
    """Compute a frequency score (0-1) based on access count.

    Uses logarithmic scaling to prevent domination by heavily-accessed memories.
    """
    import math
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(max(max_count, 1)))


def get_timestamp() -> str:
    """Get the current UTC timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """Save data as a JSON file."""
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load data from a JSON file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_memory_logger(log_dir: str = "artifacts/memory_logs") -> logging.Logger:
    """Set up a dedicated file logger for memory operations."""
    ensure_directory(log_dir)
    mem_logger = logging.getLogger("xrag.memory")
    
    if not mem_logger.handlers:
        mem_logger.setLevel(logging.DEBUG)
        
        log_file = os.path.join(log_dir, f"memory_{datetime.utcnow().strftime('%Y-%m-%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        mem_logger.addHandler(file_handler)
    
    return mem_logger


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text for display purposes."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
