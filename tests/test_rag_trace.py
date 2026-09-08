import os
import tempfile
import unittest

from src.retriever import RetrievalResult, RetrievedChunk
from src.generator import GenerationResult
from src.rag_trace import RAGTrace, RAGTraceBuilder


def make_retrieval_result(chunk_ids=("c1", "c2"), retrieval_metadata=None):
    chunks = [
        RetrievedChunk(
            chunk_id=cid,
            similarity_score=0.9,
            rank=i,
            page_number=str(i),
            source_file="doc.pdf",
            chunk_index=i,
            chunk_text=f"Text for {cid}.",
            parent_document_id="doc-1",
        )
        for i, cid in enumerate(chunk_ids, start=1)
    ]
    return RetrievalResult(
        question="What is the punishment for murder?",
        question_embedding_dimension=3,
        retrieved_chunks=chunks,
        retrieved_chunk_ids=list(chunk_ids),
        similarity_scores=[c.similarity_score for c in chunks],
        retrieval_time=0.5,
        top_k=len(chunk_ids),
        retrieval_metadata=retrieval_metadata if retrieval_metadata is not None else {},
    )


def make_generation_result(chunk_ids=("c1", "c2")):
    return GenerationResult(
        question="What is the punishment for murder?",
        generated_answer="The punishment is death or life imprisonment.",
        prompt="[SYSTEM]: ...\n[USER]: ...",
        prompt_length=42,
        model_name="mock-model",
        temperature=0.1,
        max_tokens=256,
        generation_time=0.3,
        retrieved_chunk_ids=list(chunk_ids),
        generation_metadata={},
    )


class TestRAGTrace(unittest.TestCase):
    def test_build_trace_from_results(self):
        retrieval_result = make_retrieval_result()
        generation_result = make_generation_result()

        trace = RAGTraceBuilder.build(retrieval_result, generation_result, total_pipeline_time=0.8)

        self.assertEqual(trace.question, "What is the punishment for murder?")
        self.assertEqual(len(trace.retrieved_chunk_references), 2)
        self.assertEqual(trace.execution_statistics["retrieval_time"], 0.5)
        self.assertEqual(trace.execution_statistics["generation_time"], 0.3)
        self.assertEqual(trace.retrieved_chunk_references[0]["parent_document_id"], "doc-1")

    def test_pre_rerank_stats_pass_through_to_execution_statistics(self):
        retrieval_result = make_retrieval_result(retrieval_metadata={
            "pre_rerank_candidate_pool_size": 12,
            "pre_rerank_min_dense_distance": 0.42,
        })
        trace = RAGTraceBuilder.build(retrieval_result, make_generation_result(), total_pipeline_time=1.0)

        self.assertEqual(trace.execution_statistics["pre_rerank_candidate_pool_size"], 12)
        self.assertEqual(trace.execution_statistics["pre_rerank_min_dense_distance"], 0.42)

    def test_build_raises_on_chunk_id_mismatch(self):
        retrieval_result = make_retrieval_result(chunk_ids=("c1", "c2"))
        generation_result = make_generation_result(chunk_ids=("c1", "c3"))

        with self.assertRaises(ValueError):
            RAGTraceBuilder.build(retrieval_result, generation_result, total_pipeline_time=0.1)

    def test_to_json_from_json_roundtrip(self):
        trace = RAGTraceBuilder.build(make_retrieval_result(), make_generation_result(), total_pipeline_time=1.0)

        json_str = trace.to_json()
        reloaded = RAGTrace.from_json(json_str)

        self.assertEqual(reloaded.trace_id, trace.trace_id)
        self.assertEqual(reloaded.generated_answer, trace.generated_answer)

    def test_save_to_json_with_tempdir(self):
        trace = RAGTraceBuilder.build(make_retrieval_result(), make_generation_result(), total_pipeline_time=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = RAGTraceBuilder.save_to_json(trace, base_dir=tmpdir)
            self.assertTrue(os.path.exists(saved_path))

            with open(saved_path, "r", encoding="utf-8") as f:
                reloaded = RAGTrace.from_json(f.read())
            self.assertEqual(reloaded.trace_id, trace.trace_id)


if __name__ == "__main__":
    unittest.main()
