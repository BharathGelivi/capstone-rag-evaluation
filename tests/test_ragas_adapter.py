import unittest

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from src.chunk_registry import ChunkRegistry
from src.rag_trace import RAGTrace
from scripts.baseline_adapters.ragas_adapter import to_ragas_dataset, resolve_contexts


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


class TestRagasAdapter(unittest.TestCase):
    def setUp(self):
        self.registry = ChunkRegistry()
        self.registry.register([
            make_node("c1", "doc-1", "Whoever commits murder shall be punished with death or imprisonment for life."),
            make_node("c2", "doc-1", "Section 105 covers culpable homicide not amounting to murder."),
            make_node("c3", "doc-2", "This chunk belongs to a different, unrelated trace."),
        ])
        self.trace1 = make_trace("T1", "What is the punishment for murder?", "Death or life imprisonment.", ["c1", "c2"])
        self.trace2 = make_trace("T2", "What is culpable homicide?", "See section 105.", ["c2"])
        self.trace3 = make_trace("T3", "Unrelated question?", "Unrelated answer.", ["c3"])

    def test_resolve_contexts_returns_actual_chunk_text(self):
        contexts = resolve_contexts(self.trace1, self.registry)
        self.assertEqual(contexts, [
            "Whoever commits murder shall be punished with death or imprisonment for life.",
            "Section 105 covers culpable homicide not amounting to murder.",
        ])

    def test_resolve_contexts_skips_missing_chunks(self):
        trace = make_trace("T4", "Q?", "A.", ["c1", "missing_id"])
        contexts = resolve_contexts(trace, self.registry)
        self.assertEqual(contexts, ["Whoever commits murder shall be punished with death or imprisonment for life."])

    def test_to_ragas_dataset_schema_without_gold_answers(self):
        dataset = to_ragas_dataset([self.trace1, self.trace2, self.trace3], self.registry)

        self.assertEqual(len(dataset), 3)
        sample = dataset.samples[0]
        self.assertEqual(sample.user_input, "What is the punishment for murder?")
        self.assertEqual(sample.response, "Death or life imprisonment.")
        self.assertEqual(sample.retrieved_contexts, [
            "Whoever commits murder shall be punished with death or imprisonment for life.",
            "Section 105 covers culpable homicide not amounting to murder.",
        ])
        self.assertIsNone(sample.reference)

    def test_to_ragas_dataset_with_gold_answers(self):
        gold_answers = {"T1": "Death or imprisonment for life, plus fine.", "T2": "Imprisonment for life or up to ten years."}
        dataset = to_ragas_dataset([self.trace1, self.trace2, self.trace3], self.registry, gold_answers=gold_answers)

        by_input = {s.user_input: s for s in dataset.samples}
        self.assertEqual(by_input["What is the punishment for murder?"].reference, "Death or imprisonment for life, plus fine.")
        self.assertEqual(by_input["What is culpable homicide?"].reference, "Imprisonment for life or up to ten years.")
        # trace3 has no matching gold answer -> reference stays None
        self.assertIsNone(by_input["Unrelated question?"].reference)


if __name__ == "__main__":
    unittest.main()
