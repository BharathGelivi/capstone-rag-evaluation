"""
Chunk Registry Module.

Provides a canonical, independent registry for tracking every chunk produced
during ingestion.  The registry is the ground truth that all downstream
diagnostic modules (PipelineStateAnalyzer, RootCauseReasoner, RAGAS evaluator,
report generators) use to resolve a ``chunk_id`` back to its full text and
provenance metadata.

Performance characteristics:
  - ``get_chunk()`` / ``get_record()`` : O(1) — primary hash map
  - ``get_document_chunks()``          : O(k) where k = chunks in that document
                                         (O(1) doc-level index + k record lookups)
  - ``get_statistics()``               : O(n) — intentionally a full scan, only
                                         called during ingestion diagnostics
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChunkRecord:
    """Immutable, serialisable record for a single text chunk.

    Fields are populated by the registry from the LlamaIndex TextNode produced
    by the chunk engine.  All fields have defaults where a value may be absent
    (e.g., ``page_number`` is not available for plain-text files).
    """
    chunk_id: str
    parent_document_id: str
    source_file: str
    page_number: str
    chunk_index: int
    configured_chunk_size: int
    configured_chunk_overlap: int
    character_start: int
    character_end: int
    text: str
    text_length: int
    metadata: Dict


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ChunkRegistry:
    """In-memory registry of every chunk produced during ingestion.

    Backed by two internal dicts:
      ``_records``   — primary store: ``chunk_id → ChunkRecord``
      ``_doc_index`` — inverted index: ``document_id → [chunk_id, ...]``

    The inverted index makes ``get_document_chunks()`` an O(1) lookup instead
    of the previous O(n) full-table scan.
    """

    def __init__(self) -> None:
        self._records: Dict[str, ChunkRecord] = {}
        # Inverted index built during register() and restored by load_from_json().
        self._doc_index: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def register(self, nodes: List[BaseNode]) -> None:
        """Parse LlamaIndex nodes and store them as ChunkRecords.

        Args:
            nodes: TextNodes produced by the Chunk Engine.
        """
        logger.info("Registering %d chunk(s) into the registry.", len(nodes))

        for node in nodes:
            page_number, page_number_source_key = _extract_page_number(node)

            record = ChunkRecord(
                chunk_id=node.id_,
                parent_document_id=node.ref_doc_id or "unknown",
                source_file=node.metadata.get(
                    "file_name", node.metadata.get("source_file", "unknown")
                ),
                page_number=page_number,
                chunk_index=node.metadata.get("chunk_index", -1),
                configured_chunk_size=node.metadata.get("chunk_size_config", -1),
                configured_chunk_overlap=node.metadata.get("chunk_overlap_config", -1),
                character_start=node.metadata.get("character_start", -1),
                character_end=node.metadata.get("character_end", -1),
                text=node.text,
                text_length=len(node.text),
                metadata={
                    **node.metadata.copy(),
                    "page_number_source_key": page_number_source_key,
                },
            )
            self._records[record.chunk_id] = record
            self._doc_index.setdefault(record.parent_document_id, []).append(
                record.chunk_id
            )

        logger.info("Registry now contains %d record(s).", len(self._records))

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Return the ChunkRecord for *chunk_id*, or ``None`` if not found."""
        return self._records.get(chunk_id)

    def get_record(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Alias for ``get_chunk()`` — preferred name used by RAGAS evaluator."""
        return self._records.get(chunk_id)

    def get_document_chunks(self, document_id: str) -> List[ChunkRecord]:
        """Return all chunks belonging to *document_id*.

        O(1) index lookup + O(k) record fetches, where k is the number of
        chunks in the document (replaces the former O(n) full-table scan).
        """
        chunk_ids = self._doc_index.get(document_id, [])
        return [self._records[cid] for cid in chunk_ids if cid in self._records]

    def total_chunks(self) -> int:
        """Total number of registered chunks."""
        return len(self._records)

    def get_statistics(self) -> Dict:
        """Corpus-level statistics — called during ingestion diagnostics only."""
        if not self._records:
            return {
                "num_documents": 0,
                "num_chunks": 0,
                "avg_chunk_length": 0.0,
                "max_chunk_length": 0,
                "min_chunk_length": 0,
            }

        lengths = [r.text_length for r in self._records.values()]
        return {
            "num_documents": len(self._doc_index),
            "num_chunks": len(self._records),
            "avg_chunk_length": round(sum(lengths) / len(lengths), 2),
            "max_chunk_length": max(lengths),
            "min_chunk_length": min(lengths),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_json(self, file_path: str) -> None:
        """Serialise the registry to a JSON file at *file_path*.

        The ``_doc_index`` is **not** persisted — it is always reconstructed
        from the records on load, keeping the on-disk format minimal.
        """
        logger.info("Saving registry to '%s'.", file_path)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        data = {chunk_id: asdict(record) for chunk_id, record in self._records.items()}
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)

        logger.info("Registry saved (%d record(s)).", len(self._records))

    @classmethod
    def load_from_json(cls, file_path: str) -> "ChunkRegistry":
        """Deserialise a registry from *file_path* and rebuild the doc index.

        Args:
            file_path: Path to the JSON file produced by ``save_to_json()``.

        Returns:
            A fully initialised ChunkRegistry with the doc index populated.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
        """
        logger.info("Loading registry from '%s'.", file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Registry file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        registry = cls()
        for chunk_id, record_dict in data.items():
            record = ChunkRecord(**record_dict)
            registry._records[chunk_id] = record
            # Rebuild the inverted index that was not stored on disk.
            registry._doc_index.setdefault(record.parent_document_id, []).append(
                chunk_id
            )

        logger.info(
            "Successfully loaded %d record(s) across %d document(s).",
            len(registry._records),
            len(registry._doc_index),
        )
        return registry


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_page_number(node: BaseNode):
    """Return ``(page_number_str, source_key_str)`` from a node's metadata.

    Tries several metadata key names in preference order so the registry
    handles pages from different LlamaIndex readers (PyMuPDFReader uses
    ``page_label``; some readers use ``source`` or ``page_number``).
    """
    for key in ("source", "page_label", "page_number"):
        if key in node.metadata:
            return str(node.metadata[key]), key
    return "unknown", "none"
