import unittest

from llama_index.core.schema import Document
from src.chunk_engine import create_chunks


def make_document(num_sentences=30, **metadata):
    text = " ".join(f"This is sentence number {i} in the document." for i in range(1, num_sentences + 1))
    return Document(text=text, metadata=metadata)


class TestChunking(unittest.TestCase):
    def test_chunk_count_and_no_overlap(self):
        doc = make_document(file_path="doc.pdf", page_label="1")
        nodes = create_chunks([doc], chunk_size=50, chunk_overlap=0)

        self.assertGreater(len(nodes), 1)
        # chunk indices must be sequential starting at 0
        self.assertEqual([n.metadata["chunk_index"] for n in nodes], list(range(len(nodes))))

    def test_metadata_fields_populated(self):
        doc = make_document(file_path="doc.pdf", page_label="3")
        nodes = create_chunks([doc], chunk_size=50, chunk_overlap=0)

        first = nodes[0]
        self.assertEqual(first.metadata["chunk_size_config"], 50)
        self.assertEqual(first.metadata["chunk_overlap_config"], 0)
        self.assertEqual(first.metadata["source_file"], "doc.pdf")
        self.assertEqual(first.metadata["page_number"], "3")
        self.assertEqual(first.metadata["character_start"], 0)
        self.assertGreater(first.metadata["character_end"], 0)
        self.assertEqual(first.metadata["parent_document_id"], first.ref_doc_id)

    def test_multiple_documents_preserve_separate_parent_ids(self):
        doc1 = make_document(num_sentences=20, file_path="a.pdf")
        doc2 = make_document(num_sentences=20, file_path="b.pdf")
        nodes = create_chunks([doc1, doc2], chunk_size=50, chunk_overlap=0)

        parent_ids = {n.metadata["parent_document_id"] for n in nodes}
        self.assertEqual(len(parent_ids), 2)


if __name__ == "__main__":
    unittest.main()
