"""
Memory Retriever Module.

Responsible for searching memory using semantic similarity and
applying multi-factor ranking (semantic, recency, frequency, importance).
Memory retrieval augments — never replaces — document retrieval.
"""

import logging
from typing import List, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.memory.memory_models import (
    MemoryConfig,
    MemoryEntry,
    MemorySearchResult,
)
from src.memory.memory_store import MemoryStore
from src.memory.memory_utils import (
    compute_frequency_score,
    compute_recency_score,
    setup_memory_logger,
)

logger = logging.getLogger(__name__)
mem_logger = setup_memory_logger()


class MemoryRetriever:
    """Retrieves relevant memories based on semantic similarity and multi-factor ranking.

    Ranking formula:
        final_score = (semantic_weight × semantic_similarity)
                    + (recency_weight  × recency_score)
                    + (frequency_weight × frequency_score)
                    + (importance_weight × importance_score)

    Default weights: 0.5 / 0.2 / 0.2 / 0.1
    """

    def __init__(
        self,
        store: MemoryStore,
        config: Optional[MemoryConfig] = None,
        embed_model: Optional[HuggingFaceEmbedding] = None,
    ) -> None:
        self.store = store
        self.config = config or MemoryConfig()
        self._embed_model = embed_model

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """Lazily load the embedding model, reusing the shared instance.

        Memory and document retrieval are configured with the same model
        (memory.yaml pins ``embedding_model`` to match ``EMBEDDING_MODEL_NAME``
        precisely so their vectors are comparable). Constructing a second
        instance therefore loaded an identical ~0.5 GB model onto the GPU for
        no benefit, so this goes through the process-wide cache instead.
        """
        if self._embed_model is None:
            from src.embedding_engine import get_shared_embed_model

            self._embed_model = get_shared_embed_model(self.config.embedding_model)
        return self._embed_model

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        """Search memories by semantic similarity with multi-factor ranking.

        Args:
            query:      The search query text.
            top_k:      Maximum number of results to return.
            session_id: Optional filter to search within a single session.
            min_score:  Minimum final score threshold (defaults to config).

        Returns:
            Ranked list of MemorySearchResult objects.
        """
        if min_score is None:
            min_score = self.config.similarity_threshold

        # Embed the query
        from configs.models import BGE_QUERY_INSTRUCTION
        prefixed_query = BGE_QUERY_INSTRUCTION + query
        query_embedding = self.embed_model.get_text_embedding(prefixed_query)

        # Search ChromaDB
        raw_results = self.store.search_by_embedding(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch for re-ranking
            session_id=session_id,
        )

        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            return []

        # Build scored results
        scored_results: List[MemorySearchResult] = []
        ids = raw_results["ids"][0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        # Compute max access count for frequency normalization
        all_access_counts = [
            int(m.get("access_count", 0)) for m in metadatas
        ]
        max_access = max(all_access_counts) if all_access_counts else 1

        for i, (mem_id, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # Parse the memory entry
            entry = self.store.get_memory(mem_id)
            if entry is None:
                continue

            # Compute individual scores
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2)
            semantic = max(0.0, 1.0 - (dist / 2.0))
            recency = compute_recency_score(entry.timestamp)
            frequency = compute_frequency_score(entry.access_count, max_access)
            importance = min(1.0, entry.importance_score)

            # Multi-factor ranking
            final_score = (
                self.config.semantic_weight * semantic
                + self.config.recency_weight * recency
                + self.config.frequency_weight * frequency
                + self.config.importance_weight * importance
            )

            if final_score < min_score:
                continue

            # Generate retrieval reason
            reason_parts = []
            if semantic > 0.7:
                reason_parts.append(f"High semantic match ({semantic:.2f})")
            elif semantic > 0.5:
                reason_parts.append(f"Moderate semantic match ({semantic:.2f})")
            if recency > 0.8:
                reason_parts.append("Recent conversation")
            if frequency > 0.5:
                reason_parts.append("Frequently referenced")
            if importance > 0.8:
                reason_parts.append("High importance")
            reason = "; ".join(reason_parts) if reason_parts else "Relevant memory"

            scored_results.append(
                MemorySearchResult(
                    memory=entry,
                    semantic_score=semantic,
                    recency_score=recency,
                    frequency_score=frequency,
                    importance_score=importance,
                    final_score=final_score,
                    retrieval_reason=reason,
                )
            )

        # Sort by final score and truncate
        scored_results.sort(key=lambda r: r.final_score, reverse=True)
        results = scored_results[:top_k]

        # Update access counts for retrieved memories
        for result in results:
            result.memory.access_count += 1
            result.memory.last_accessed = (
                __import__("datetime").datetime.utcnow().isoformat() + "Z"
            )
            try:
                self.store.update_memory(result.memory)
            except Exception:
                pass

        mem_logger.info(
            "Memory search for '%s': %d results (top score: %.3f)",
            query[:50],
            len(results),
            results[0].final_score if results else 0.0,
        )

        return results

    def get_relevant_context(
        self,
        query: str,
        top_k: int = 3,
        session_id: Optional[str] = None,
    ) -> str:
        """Get a formatted context string from relevant memories for prompt injection.

        This is the main integration point with the RAG generator — the
        returned string is prepended to the retrieval context.

        Prefer :meth:`format_context` when you already hold search results:
        calling this in addition to ``search_memory`` runs the whole search
        twice (two embeddings, two vector queries) *and* double-increments
        every hit's ``access_count``, which skews the frequency term of the
        ranking against reality.
        """
        return self.format_context(
            self.search_memory(query, top_k=top_k, session_id=session_id),
            session_id=session_id,
        )

    def format_context(
        self,
        results: List[MemorySearchResult],
        session_id: Optional[str] = None,
    ) -> str:
        """Render already-retrieved memories into prompt-injectable text."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            is_current = session_id and result.memory.session_id == session_id
            header_type = "Current Session Interaction" if is_current else "Previous conversation"
            context_parts.append(
                f"--- {header_type} {i} (Relevance: {result.final_score:.2f}) ---\n"
                f"Q: {result.memory.question}\n"
                f"A: {result.memory.answer}\n"
            )
        return "\n".join(context_parts)

    def embed_text(self, text: str) -> List[float]:
        """Embed text using the configured embedding model.

        Used by MemoryManager to pre-compute embeddings before saving.
        """
        from configs.models import BGE_QUERY_INSTRUCTION
        return self.embed_model.get_text_embedding(BGE_QUERY_INSTRUCTION + text)
