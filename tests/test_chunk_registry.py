import os
import tempfile
import unittest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.chunk_registry import ChunkRegistry


def make_node(chunk_id, doc_id, text, metadata=None):
    node = TextNode(text=text, id_=chunk_id, metadata=metadata or {})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


class TestChunkRegistry(unittest.TestCase):
    def test_register_and_get_chunk(self):
        registry = ChunkRegistry()
        node = make_node("chunk-1", "doc-1", "Hello world.", {
            "file_name": "doc.pdf",
            "page_label": "3",
            "chunk_index": 0,
        })
        registry.register([node])

        record = registry.get_chunk("chunk-1")
        self.assertIsNotNone(record)
        self.assertEqual(record.parent_document_id, "doc-1")
        self.assertEqual(record.source_file, "doc.pdf")
        self.assertEqual(record.chunk_index, 0)
        self.assertEqual(record.text_length, len("Hello world."))
        self.assertIsNone(registry.get_chunk("missing"))

    def test_get_document_chunks(self):
        registry = ChunkRegistry()
        registry.register([
            make_node("c1", "doc-1", "A"),
            make_node("c2", "doc-1", "B"),
            make_node("c3", "doc-2", "C"),
        ])

        doc1_chunks = registry.get_document_chunks("doc-1")
        self.assertEqual(len(doc1_chunks), 2)
        self.assertEqual(registry.total_chunks(), 3)

    def test_get_statistics(self):
        registry = ChunkRegistry()
        registry.register([
            make_node("c1", "doc-1", "12345"),
            make_node("c2", "doc-1", "1234567890"),
        ])

        stats = registry.get_statistics()
        self.assertEqual(stats["num_documents"], 1)
        self.assertEqual(stats["num_chunks"], 2)
        self.assertEqual(stats["avg_chunk_length"], 7.5)
        self.assertEqual(stats["max_chunk_length"], 10)
        self.assertEqual(stats["min_chunk_length"], 5)

    def test_get_statistics_empty_registry(self):
        registry = ChunkRegistry()
        stats = registry.get_statistics()
        self.assertEqual(stats["num_chunks"], 0)

    def test_save_and_load_roundtrip(self):
        registry = ChunkRegistry()
        registry.register([make_node("c1", "doc-1", "Some text.", {"file_name": "doc.pdf"})])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "registry.json")
            registry.save_to_json(path)
            reloaded = ChunkRegistry.load_from_json(path)

            self.assertEqual(reloaded.total_chunks(), 1)
            self.assertEqual(reloaded.get_chunk("c1").source_file, "doc.pdf")

    def test_load_from_json_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ChunkRegistry.load_from_json("does_not_exist.json")

    def test_page_number_source_key_from_source(self):
        registry = ChunkRegistry()
        registry.register([make_node("c1", "doc-1", "Text.", {"source": "5", "page_label": "9"})])

        record = registry.get_chunk("c1")
        self.assertEqual(record.page_number, "5")
        self.assertEqual(record.metadata["page_number_source_key"], "source")

    def test_page_number_source_key_from_page_label(self):
        registry = ChunkRegistry()
        registry.register([make_node("c1", "doc-1", "Text.", {"page_label": "9"})])

        record = registry.get_chunk("c1")
        self.assertEqual(record.page_number, "9")
        self.assertEqual(record.metadata["page_number_source_key"], "page_label")

    def test_page_number_source_key_from_page_number(self):
        registry = ChunkRegistry()
        registry.register([make_node("c1", "doc-1", "Text.", {"page_number": "2"})])

        record = registry.get_chunk("c1")
        self.assertEqual(record.page_number, "2")
        self.assertEqual(record.metadata["page_number_source_key"], "page_number")

    def test_page_number_source_key_unknown(self):
        registry = ChunkRegistry()
        registry.register([make_node("c1", "doc-1", "Text.", {})])

        record = registry.get_chunk("c1")
        self.assertEqual(record.page_number, "unknown")
        self.assertEqual(record.metadata["page_number_source_key"], "none")


if __name__ == "__main__":
    unittest.main()
