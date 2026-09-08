import os
import tempfile
import unittest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.chunk_registry import ChunkRegistry
from src.rag_trace import RAGTrace
from scripts.baseline_adapters.common import resolve_examples, save_resolved_examples, load_resolved_examples, ResolvedExample


def make_node(chunk_id, doc_id, text):
    node = TextNode(text=text, id_=chunk_id, metadata={})
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
    return node


def make_trace(trace_id, question, answer, chunk_ids):
    return RAGTrace(
        trace_id=trace_id,
        trace_version="1.0",
        pipeline_version="1.0",
        framework_version="1.0",
        timestamp="2026-07-18T00:00:00Z",
        question=question,
        generated_answer=answer,
        prompt_snapshot="prompt",
        prompt_length=10,
        retrieved_chunk_references=[{"chunk_id": cid, "similarity_score": 0.9} for cid in chunk_ids],
        configuration_snapshot={},
        execution_statistics={},
        pipeline_stage_status={},
    )


class TestBaselineAdaptersCommon(unittest.TestCase):
    def setUp(self):
        self.registry = ChunkRegistry()
        self.registry.register([
            make_node("c1", "doc-1", "First chunk text."),
            make_node("c2", "doc-1", "Second chunk text."),
        ])
        self.trace1 = make_trace("T1", "Q1?", "A1.", ["c1", "c2"])
        self.trace2 = make_trace("T2", "Q2?", "A2.", ["c1"])

    def test_resolve_examples_shape(self):
        examples = resolve_examples([self.trace1, self.trace2], self.registry)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].trace_id, "T1")
        self.assertEqual(examples[0].contexts, ["First chunk text.", "Second chunk text."])
        self.assertIsNone(examples[0].gold_answer)

    def test_resolve_examples_with_gold_answers(self):
        gold_answers = {"T1": "Gold answer 1."}
        examples = resolve_examples([self.trace1, self.trace2], self.registry, gold_answers=gold_answers)

        by_id = {e.trace_id: e for e in examples}
        self.assertEqual(by_id["T1"].gold_answer, "Gold answer 1.")
        self.assertIsNone(by_id["T2"].gold_answer)

    def test_save_and_load_roundtrip(self):
        examples = resolve_examples([self.trace1], self.registry, gold_answers={"T1": "Gold."})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "examples.json")
            save_resolved_examples(examples, path)
            reloaded = load_resolved_examples(path)

            self.assertEqual(len(reloaded), 1)
            self.assertIsInstance(reloaded[0], ResolvedExample)
            self.assertEqual(reloaded[0].trace_id, "T1")
            self.assertEqual(reloaded[0].gold_answer, "Gold.")
            self.assertEqual(reloaded[0].contexts, ["First chunk text.", "Second chunk text."])


if __name__ == "__main__":
    unittest.main()
