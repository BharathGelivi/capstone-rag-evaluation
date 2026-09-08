import unittest
from unittest.mock import patch, MagicMock

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.chunk_registry import ChunkRegistry
from src.vector_store import VectorStore


def make_node(chunk_id, doc_id, text, **metadata):
    node = TextNode(text=text, id_=chunk_id, metadata=metadata)
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


class FakeVectorStore(VectorStore):
    """Minimal in-memory VectorStore stub returning hand-crafted dense search results."""

    def __init__(self, search_result=None):
        self._search_result = search_result or {"ids": [[]], "distances": [[]]}

    def initialize_collection(self):
        pass

    def add_embeddings(self, records, chunk_registry):
        pass

    def search(self, query_embedding, top_k=5):
        return self._search_result

    def get_by_chunk_id(self, chunk_id):
        return None

    def count(self):
        return 0

    def delete_collection(self):
        pass


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.registry = ChunkRegistry()
        self.registry.register([
            make_node("c1", "doc-1", "Alpha bravo charlie.", chunk_index=0, page_label="1", file_name="doc.pdf"),
            make_node("c2", "doc-1", "Delta echo foxtrot.", chunk_index=1, page_label="2", file_name="doc.pdf"),
            # A third, unrelated chunk keeps the BM25 corpus non-degenerate (rank_bm25's
            # IDF collapses to 0 for a term appearing in exactly half of a 2-document corpus).
            make_node("c3", "doc-1", "Golf hotel india.", chunk_index=2, page_label="3", file_name="doc.pdf"),
        ])
        self.vector_store = FakeVectorStore({"ids": [["c1", "c2"]], "distances": [[0.05, 0.2]]})

        # Retriever gets its embedding model from embedding_engine's process-wide
        # cache, not by constructing HuggingFaceEmbedding directly -- so the
        # accessor is the correct patch point. (Patching the class instead left
        # call_count at 0 and made the singleton assertion below vacuously fail.)
        embed_patcher = patch("src.retriever.get_shared_embed_model")
        cross_encoder_patcher = patch("src.retriever.CrossEncoder")
        self.mock_get_embed_model = embed_patcher.start()
        self.mock_ce_cls = cross_encoder_patcher.start()
        self.addCleanup(embed_patcher.stop)
        self.addCleanup(cross_encoder_patcher.stop)

        self.mock_get_embed_model.return_value.get_text_embedding.return_value = [0.1, 0.2]
        self.mock_cross_encoder = MagicMock()
        self.mock_ce_cls.return_value = self.mock_cross_encoder

    def _make_retriever(self, top_k=2):
        from src.retriever import Retriever
        return Retriever(vector_store=self.vector_store, chunk_registry=self.registry, top_k=top_k)

    def test_hybrid_fusion_ranking(self):
        self.mock_cross_encoder.predict.return_value = [0.9, 0.1]
        retriever = self._make_retriever()

        result = retriever.retrieve("alpha bravo")

        self.assertEqual([c.chunk_id for c in result.retrieved_chunks], ["c1", "c2"])
        self.assertEqual(result.retrieved_chunks[0].dense_rank, 1)
        self.assertEqual(result.retrieved_chunks[0].sparse_rank, 1)

    def test_reranker_changes_order(self):
        # Reranker inverts the RRF-based ordering: c2 should end up ranked first.
        self.mock_cross_encoder.predict.return_value = [0.1, 0.9]
        retriever = self._make_retriever()

        result = retriever.retrieve("alpha bravo")

        self.assertEqual([c.chunk_id for c in result.retrieved_chunks], ["c2", "c1"])
        self.assertEqual(result.retrieved_chunks[0].rank, 1)
        self.assertEqual(result.retrieved_chunks[0].reranker_score, 0.9)

    def test_retrieved_chunk_fields_populated(self):
        self.mock_cross_encoder.predict.return_value = [0.9, 0.1]
        retriever = self._make_retriever()

        result = retriever.retrieve("alpha bravo")
        first = result.retrieved_chunks[0]

        self.assertEqual(first.source_file, "doc.pdf")
        self.assertEqual(first.page_number, "1")
        self.assertEqual(first.chunk_index, 0)
        self.assertEqual(first.chunk_text, "Alpha bravo charlie.")
        self.assertEqual(first.parent_document_id, "doc-1")
        self.assertEqual(result.retrieved_chunk_ids, [c.chunk_id for c in result.retrieved_chunks])

    def test_pre_rerank_candidate_metadata_populated(self):
        self.mock_cross_encoder.predict.return_value = [0.9, 0.1]
        retriever = self._make_retriever()

        result = retriever.retrieve("alpha bravo")

        self.assertEqual(result.retrieval_metadata["pre_rerank_candidate_pool_size"], 2)
        self.assertEqual(result.retrieval_metadata["pre_rerank_min_dense_distance"], 0.05)

    def test_get_retriever_singleton_avoids_reload(self):
        from src.retriever import get_retriever
        get_retriever.cache_clear()
        try:
            first = get_retriever(self.vector_store, self.registry, top_k=2)
            second = get_retriever(self.vector_store, self.registry, top_k=2)

            self.assertIs(first, second)
            self.assertEqual(self.mock_get_embed_model.call_count, 1)
            self.assertEqual(self.mock_ce_cls.call_count, 1)
        finally:
            get_retriever.cache_clear()


if __name__ == "__main__":
    unittest.main()
