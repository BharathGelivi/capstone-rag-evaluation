import tempfile
import unittest
import uuid

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.chunk_registry import ChunkRegistry
from src.embedding_engine import EmbeddingRecord
from src.vector_store import ChromaVectorStore


def make_node(chunk_id, doc_id, text):
    node = TextNode(text=text, id_=chunk_id, metadata={"file_name": "doc.pdf", "chunk_index": 0})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        # ignore_cleanup_errors: chromadb keeps its sqlite file handle open on
        # Windows, which otherwise makes TemporaryDirectory cleanup raise.
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmpdir.cleanup)
        # Unique collection name per test run avoids colliding with ./db/chroma
        self.store = ChromaVectorStore(
            persist_dir=self.tmpdir.name,
            collection_name=f"test_collection_{uuid.uuid4().hex}",
        )
        self.store.initialize_collection()

        self.registry = ChunkRegistry()
        self.registry.register([make_node("c1", "doc-1", "Some chunk text.")])
        self.embedding_record = EmbeddingRecord(
            chunk_id="c1",
            parent_document_id="doc-1",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="mock-model",
            embedding_dimension=3,
            timestamp="2026-07-17T00:00:00Z",
        )

    def test_add_and_count(self):
        self.store.add_embeddings([self.embedding_record], self.registry)
        self.assertEqual(self.store.count(), 1)

    def test_get_by_chunk_id(self):
        self.store.add_embeddings([self.embedding_record], self.registry)
        retrieved = self.store.get_by_chunk_id("c1")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["document"], "Some chunk text.")
        self.assertEqual(retrieved["metadata"]["parent_document_id"], "doc-1")

    def test_get_by_chunk_id_missing_returns_none(self):
        self.store.add_embeddings([self.embedding_record], self.registry)
        self.assertIsNone(self.store.get_by_chunk_id("does-not-exist"))

    def test_search_returns_nearest(self):
        self.store.add_embeddings([self.embedding_record], self.registry)
        results = self.store.search(query_embedding=[0.1, 0.2, 0.3], top_k=1)

        self.assertEqual(results["ids"][0][0], "c1")


if __name__ == "__main__":
    unittest.main()
