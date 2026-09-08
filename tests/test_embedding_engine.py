"""
Tests for embedding_engine.generate_embeddings().

Key considerations for mocking:
  - get_shared_embed_model is the patch point, not HuggingFaceEmbedding.
    generate_embeddings() goes through that process-wide lru_cache, so patching
    the class leaves a real (or stale mock) model cached from an earlier test in
    the same process -- the same stale-patch bug already fixed in
    tests/test_retriever.py.
  - get_text_embedding() is used only for the dimension-probe call.
  - get_text_embedding_batch() is used for all real embeddings.
  - _EmbeddingCache is patched to a no-op so tests don't touch the filesystem
    and always force the batch-embed path (no spurious cache hits between runs).
"""
import unittest
from unittest.mock import MagicMock, patch

from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from src.chunk_registry import ChunkRegistry
from src.embedding_engine import generate_embeddings


def make_node(chunk_id, doc_id, text):
    node = TextNode(text=text, id_=chunk_id, metadata={"file_name": "doc.pdf"})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def _no_op_cache():
    """Return a cache stub that always misses (forces the batch-embed path)."""
    cache = MagicMock()
    cache.get.return_value = None  # always miss
    cache.put.return_value = None
    return cache


class TestEmbeddingEngine(unittest.TestCase):
    def setUp(self):
        # The shared model is lru_cached process-wide; a mock cached by one test
        # would otherwise leak into the next.
        from src.embedding_engine import get_shared_embed_model

        get_shared_embed_model.cache_clear()
        self.addCleanup(get_shared_embed_model.cache_clear)

        self.registry = ChunkRegistry()
        self.registry.register(
            [
                make_node("c1", "doc-1", "First chunk of text."),
                make_node("c2", "doc-1", "Second chunk of text."),
            ]
        )

    @patch("src.embedding_engine._EmbeddingCache")
    @patch("src.embedding_engine.get_shared_embed_model")
    def test_generate_embeddings_count_and_dimension(self, mock_embed_cls, mock_cache_cls):
        mock_model = MagicMock()
        # Probe call (dimension detection) and batch call both return fixed vectors.
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_model.get_text_embedding_batch.return_value = [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ]
        mock_embed_cls.return_value = mock_model
        mock_cache_cls.return_value = _no_op_cache()

        records = generate_embeddings(self.registry)

        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record.embedding_dimension, 3)
            self.assertEqual(record.embedding, [0.1, 0.2, 0.3])

    @patch("src.embedding_engine._EmbeddingCache")
    @patch("src.embedding_engine.get_shared_embed_model")
    def test_chunk_id_and_parent_id_passthrough(self, mock_embed_cls, mock_cache_cls):
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2]
        mock_model.get_text_embedding_batch.return_value = [
            [0.1, 0.2],
            [0.1, 0.2],
        ]
        mock_embed_cls.return_value = mock_model
        mock_cache_cls.return_value = _no_op_cache()

        records = generate_embeddings(self.registry)
        record_by_id = {r.chunk_id: r for r in records}

        self.assertEqual(record_by_id["c1"].parent_document_id, "doc-1")
        self.assertEqual(record_by_id["c2"].parent_document_id, "doc-1")

    @patch("src.embedding_engine._EmbeddingCache")
    @patch("src.embedding_engine.get_shared_embed_model")
    def test_cache_hit_skips_batch_embed(self, mock_embed_cls, mock_cache_cls):
        """When both chunks are cache hits, get_text_embedding_batch is never called."""
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.5, 0.5]
        mock_embed_cls.return_value = mock_model

        # Both chunks hit the cache.
        cache = MagicMock()
        cache.get.return_value = [0.5, 0.5]
        mock_cache_cls.return_value = cache

        records = generate_embeddings(self.registry)

        mock_model.get_text_embedding_batch.assert_not_called()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].embedding, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
