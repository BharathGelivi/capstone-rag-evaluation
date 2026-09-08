"""
Embedding Engine Module.

Converts text chunks into dense vector embeddings using a local HuggingFace model.

Two key optimisations over the original sequential implementation:

1. **Batch processing** — all texts are embedded in a single batched forward pass
   via ``get_text_embedding_batch()``, which is 5–20× faster than a per-item loop
   (the exact speedup depends on whether a GPU is present).

2. **Persistent disk cache** — embeddings are stored as NumPy ``.npy`` files keyed
   by ``MD5(model_name + text)``. On subsequent runs identical chunks are loaded
   from disk with zero model calls, making re-ingestion after minor corpus edits
   nearly instantaneous.  The cache auto-invalidates on model change because the
   model name is part of the key.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import functools

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.device import get_device


@functools.lru_cache(maxsize=4)
def get_shared_embed_model(model_name: str) -> HuggingFaceEmbedding:
    """Return a process-wide shared embedding model for ``model_name``.

    The bi-encoder is used by document retrieval, memory retrieval, and
    ingestion. Each previously built its own instance, so the same weights were
    loaded two or three times — wasted VRAM and several seconds of startup for
    an object that is stateless at inference time and safe to share.
    """
    logger.info("Loading shared embedding model %s on %s.", model_name, get_device())
    return HuggingFaceEmbedding(model_name=model_name, device=get_device())

from configs.models import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_MODEL_NAME,
)
from src.chunk_registry import ChunkRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRecord:
    """Links a generated embedding vector back to its source chunk.

    Preserves the model name and dimensionality so that future model-comparison
    experiments can detect and reject stale, incompatible embeddings.
    """
    chunk_id: str
    parent_document_id: str
    embedding: List[float]
    embedding_model: str
    embedding_dimension: int
    timestamp: str


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class _EmbeddingCache:
    """Transparent, file-system-backed cache for embedding vectors.

    Cache entries are stored as NumPy ``.npy`` files under *cache_dir*.  The
    filename is the MD5 digest of ``f"{model_name}:{text}"`` which means:

    * Identical text embedded with the same model always hits the cache.
    * Changing ``EMBEDDING_MODEL_NAME`` produces a new digest → automatic miss.
    * The cache directory can be safely deleted to force a full re-embed.
    """

    def __init__(self, cache_dir: str, model_name: str) -> None:
        self._dir = cache_dir
        self._model = model_name
        os.makedirs(cache_dir, exist_ok=True)
        logger.debug("Embedding cache initialised at '%s'.", cache_dir)

    def _key(self, text: str) -> str:
        return hashlib.md5(f"{self._model}:{text}".encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Return the cached embedding for *text*, or ``None`` on a miss."""
        path = os.path.join(self._dir, f"{self._key(text)}.npy")
        if os.path.exists(path):
            return np.load(path).tolist()
        return None

    def put(self, text: str, vector: List[float]) -> None:
        """Persist *vector* to disk so future runs can skip re-embedding."""
        path = os.path.join(self._dir, f"{self._key(text)}.npy")
        np.save(path, np.array(vector, dtype=np.float32))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embeddings(
    registry: ChunkRegistry,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    cache_dir: str = EMBEDDING_CACHE_DIR,
) -> List[EmbeddingRecord]:
    """Generate embeddings for every chunk in *registry*.

    Chunks with a cached embedding are loaded from disk; remaining chunks are
    embedded in batches and then written to the cache before returning.

    Args:
        registry:   Populated ChunkRegistry produced by the ingestion stage.
        batch_size: Number of texts to embed per forward pass.
        cache_dir:  Directory used for the persistent embedding cache.

    Returns:
        One ``EmbeddingRecord`` per chunk, preserving the registry's order.
    """
    logger.info("Initialising embedding model: %s", EMBEDDING_MODEL_NAME)
    embed_model = get_shared_embed_model(EMBEDDING_MODEL_NAME)
    cache = _EmbeddingCache(cache_dir=cache_dir, model_name=EMBEDDING_MODEL_NAME)

    # Determine embedding dimension from a cheap test call.
    dimension = len(embed_model.get_text_embedding("probe"))
    logger.info("Embedding dimension: %d", dimension)

    records = list(registry._records.values())
    total = len(records)
    logger.info("Processing %d chunk(s) with batch_size=%d.", total, batch_size)

    # --- Pass 1: resolve cache hits -------------------------------------------
    cache_hits: List[Tuple[int, List[float]]] = []   # (original_index, vector)
    cache_misses: List[Tuple[int, str]] = []          # (original_index, text)

    for idx, record in enumerate(records):
        cached = cache.get(record.text)
        if cached is not None:
            cache_hits.append((idx, cached))
        else:
            cache_misses.append((idx, record.text))

    logger.info(
        "Cache: %d hit(s), %d miss(es) to embed.",
        len(cache_hits),
        len(cache_misses),
    )

    # --- Pass 2: batch-embed cache misses -------------------------------------
    miss_vectors: List[List[float]] = []
    if cache_misses:
        miss_texts = [text for _, text in cache_misses]
        # get_text_embedding_batch handles internal batching; we also chunk it
        # ourselves to surface progress logs at a human-readable frequency.
        for batch_start in range(0, len(miss_texts), batch_size):
            batch = miss_texts[batch_start : batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(miss_texts))
            logger.info(
                "Embedding batch %d–%d / %d ...",
                batch_start + 1,
                batch_end,
                len(miss_texts),
            )
            vectors = embed_model.get_text_embedding_batch(batch, show_progress=False)
            miss_vectors.extend(vectors)

        # Write newly computed vectors to cache.
        for (_, text), vector in zip(cache_misses, miss_vectors):
            cache.put(text, vector)

    # --- Pass 3: assemble results in original order ---------------------------
    # Merge hits and misses back into a single dict keyed by original index.
    index_to_vector: dict = {}
    for idx, vector in cache_hits:
        index_to_vector[idx] = vector
    for (idx, _), vector in zip(cache_misses, miss_vectors):
        index_to_vector[idx] = vector

    timestamp = datetime.utcnow().isoformat() + "Z"
    embedding_records: List[EmbeddingRecord] = []

    for idx, record in enumerate(records):
        vector = index_to_vector[idx]
        embedding_records.append(
            EmbeddingRecord(
                chunk_id=record.chunk_id,
                parent_document_id=record.parent_document_id,
                embedding=vector,
                embedding_model=EMBEDDING_MODEL_NAME,
                embedding_dimension=len(vector),
                timestamp=timestamp,
            )
        )

    logger.info(
        "Embedding generation complete. "
        "%d record(s) produced (%d from cache, %d freshly embedded).",
        len(embedding_records),
        len(cache_hits),
        len(cache_misses),
    )
    return embedding_records
