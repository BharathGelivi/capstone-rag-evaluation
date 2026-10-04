"""Regression checks for the audit's data-loss, correctness and concurrency fixes."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from src.cache_utils import serialized_cache
from src.claim_decomposer import ClaimDecomposer, JSONRecoveryError
from src.claim_verifier import ClaimVerifier
from src.memory.memory_models import MemoryConfig, MemoryEntry
from src.memory.memory_store import MemoryStore
from src.memory.session_manager import SessionManager
from src.rag_trace import RAGTrace
from src.vector_store import ChromaVectorStore
from experiments.common import Checkpoint, ExperimentContext
from scripts.analyze_agreement import _pearson_r, _ragas_flags


def trace(**overrides):
    values = dict(trace_id="audit", trace_version="1.0", pipeline_version="1.0",
                  framework_version="1.0", timestamp="2026-10-02T00:00:00Z",
                  question="Q", generated_answer="A", prompt_snapshot="", prompt_length=0,
                  retrieved_chunk_references=[], configuration_snapshot={}, execution_statistics={},
                  pipeline_stage_status={})
    values.update(overrides)
    return RAGTrace(**values)


class TestAuditCorrectness(unittest.TestCase):
    def test_failed_ingestion_restores_reused_vectors(self):
        # Chroma keeps HNSW files mapped until process exit on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            store = ChromaVectorStore(directory)
            store.initialize_collection()
            store.collection.upsert(ids=["old"], embeddings=[[1., 0.]], documents=["Original"],
                                    metadatas=[{"parent_document_id": "original"}])
            def fail(records, registry):
                store.collection.upsert(ids=["old", "new"], embeddings=[[0., 1.], [0., 1.]],
                    documents=["Changed", "New"], metadatas=[{"parent_document_id":"changed"}]*2)
                raise RuntimeError("later batch failed")
            store.add_embeddings = fail
            registry = MagicMock(_records={"old": object(), "new": object()})
            with self.assertRaises(RuntimeError):
                store.publish_registry([MagicMock(chunk_id="old"), MagicMock(chunk_id="new")],
                                       registry, str(Path(directory)/"registry.json"))
            actual = store.collection.get(include=["embeddings", "documents", "metadatas"])
            self.assertEqual(actual["ids"], ["old"])
            self.assertEqual(actual["documents"], ["Original"])
            self.assertEqual(list(actual["embeddings"][0]), [1., 0.])
            self.assertEqual(actual["metadatas"], [{"parent_document_id": "original"}])
            registry.save_to_json.assert_not_called()

    def test_registry_fallback_clears_resolved_evidence_warning(self):
        t = trace(retrieved_chunk_references=[{"chunk_id": "a"}])
        self.assertEqual(ClaimVerifier.build_retrieved_chunks_from_trace(t, None), [])
        registry = MagicMock()
        registry.get_chunk.return_value = MagicMock(text="Recovered evidence", parent_document_id="doc")
        self.assertEqual(len(ClaimVerifier.build_retrieved_chunks_from_trace(t, registry)), 1)
        self.assertNotIn("unresolved_chunk_ids", t.diagnostics)

    def test_claim_parser_rejects_wrong_shapes(self):
        parser = object.__new__(ClaimDecomposer)
        for value in ('null', '{}', '[null]', '[{"claim_text": 42}]', '[{"claim_text": ""}]'):
            with self.subTest(value=value), self.assertRaises(JSONRecoveryError):
                parser._robust_json_parse(value)
        self.assertEqual(parser._robust_json_parse('[]'), [])

    def test_snapshot_is_preferred_and_parent_is_preserved(self):
        t = trace(prompt_snapshot="--- Context chunk 1 [Chunk-ID: a] ---\nOriginal evidence\n\nQuestion: Q",
                  retrieved_chunk_references=[{"chunk_id": "a", "parent_document_id": "doc"}])
        registry = MagicMock()
        registry.get_chunk.return_value.text = "Changed evidence"
        chunks = ClaimVerifier.build_retrieved_chunks_from_trace(t, registry)
        self.assertEqual(chunks[0].chunk_text, "Original evidence")
        self.assertEqual(chunks[0].parent_document_id, "doc")

    def test_missing_context_is_recorded(self):
        t = trace(retrieved_chunk_references=[{"chunk_id": "missing"}])
        self.assertEqual(ClaimVerifier.build_retrieved_chunks_from_trace(t, None), [])
        self.assertEqual(t.diagnostics["unresolved_chunk_ids"], ["missing"])

    def test_legacy_chat_trace_loads(self):
        payload = json.loads(trace().to_json())
        payload["claim_verification"] = {"claim_count": 0}
        self.assertEqual(RAGTrace.from_json(payload).diagnostics["claim_verification"]["claim_count"], 0)

    def test_failed_relevance_judge_is_unavailable(self):
        from src.ragas_metrics import RagasEvaluator
        evaluator = RagasEvaluator(MagicMock(), None)
        evaluator._call_llm = lambda _: "malformed"
        self.assertIsNone(evaluator.compute_context_precision("Q", "A", [MagicMock(chunk_text="text")]))
        self.assertIsNone(evaluator.compute_context_relevancy("Q", [MagicMock(chunk_text="text")]))

    def test_nonfinite_scores_do_not_mean_healthy(self):
        self.assertEqual(_ragas_flags([{"ragas_faithfulness": float("nan")}], .7), [None])
        self.assertAlmostEqual(_pearson_r([1, float("nan"), 2], [1, 9, 2]), 1)

    def test_api_rejects_unsafe_ids_and_options(self):
        from fastapi.testclient import TestClient
        from src.api import app
        from src.api_ui import app as ui
        self.assertEqual(TestClient(app).get("/artifacts/bad*id").status_code, 422)
        client = TestClient(ui)
        self.assertEqual(client.get("/ui/trace/bad*id").status_code, 422)
        self.assertEqual(client.post("/ui/chat", json={"question": " ", "session_id": "s"}).status_code, 422)
        self.assertEqual(client.post("/ui/chat", json={"question": "Q", "session_id": "s", "arm": "F_graphrag"}).status_code, 422)
        self.assertEqual(client.get("/ui/memory?limit=100000").status_code, 422)

    def test_lazy_cache_deduplicates_concurrent_misses(self):
        calls = []
        @serialized_cache()
        def load():
            time.sleep(.01)
            calls.append(1)
            return object()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: load(), range(8)))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(r is results[0] for r in results))

    def test_failed_cache_results_are_retried(self):
        calls = []
        @serialized_cache()
        def load():
            calls.append(1)
            return None
        load(); load()
        self.assertEqual(len(calls), 2)

    def test_collection_conflict_does_not_delete_data(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.vector_store.chromadb.PersistentClient") as client:
            client.return_value.get_or_create_collection.side_effect = ValueError("Embedding function conflict")
            with self.assertRaises(ValueError):
                ChromaVectorStore(directory).initialize_collection()
            client.return_value.delete_collection.assert_not_called()

    def test_insertion_failure_preserves_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text("original", encoding="utf-8")
            store = ChromaVectorStore(directory)
            store.collection = MagicMock()
            store.collection.get.return_value = {"ids": ["old"]}
            store.add_embeddings = MagicMock(side_effect=RuntimeError("failed"))
            registry = MagicMock(_records={"new": object()})
            with self.assertRaises(RuntimeError):
                store.publish_registry([MagicMock(chunk_id="new")], registry, str(path))
            self.assertEqual(path.read_text(encoding="utf-8"), "original")
            registry.save_to_json.assert_not_called()
            store.collection.delete.assert_not_called()

    def test_checkpoint_resume_repairs_final_fragment(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint("test", directory)
            checkpoint.append({"example_id": "a"})
            with open(checkpoint.records_path, "ab") as handle:
                handle.write(b'{"exam')
            checkpoint.append({"example_id": "b"})
            self.assertEqual(checkpoint.completed_ids(), ["a", "b"])

    def test_checkpoint_interior_corruption_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Checkpoint("test", directory)
            checkpoint.append({"example_id": "a"})
            Path(checkpoint.records_path).write_text('broken\n{"example_id":"b"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                checkpoint.load_records()


class TestAuditMemory(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.store = MemoryStore(MemoryConfig(persistence_directory=self.directory, collection_name="audit_memory"))
        self.store.initialize()

    def test_metadata_update_preserves_vector(self):
        entry = MemoryEntry(memory_id="m", session_id="s", question="Q", answer="A")
        vector = [1.] + [0.] * 767
        self.store.save_memory(entry, vector)
        entry.access_count = 5
        self.assertTrue(self.store.update_memory(entry))
        actual = self.store._collection.get(ids=["m"], include=["embeddings"])["embeddings"][0]
        self.assertEqual(list(actual), vector)
        self.assertEqual(len(self.store.get_short_term_memories()), 1)

    def test_text_update_without_embedding_is_refused(self):
        entry = MemoryEntry(memory_id="m", session_id="s", question="Q", answer="A")
        self.store.save_memory(entry, [1.] * 768)
        entry.answer = "Changed"
        self.assertFalse(self.store.update_memory(entry))
        self.assertEqual(self.store.get_memory("m").answer, "A")

    def test_import_does_not_overwrite_original(self):
        manager = SessionManager(self.store)
        original = manager.create_session("Original")
        entry = MemoryEntry(memory_id="m", session_id=original.session_id, question="Q", answer="A")
        self.store.save_memory(entry, [1.] * 768)
        imported = manager.import_session(manager.export_session(original.session_id))
        self.assertEqual(self.store.get_memory("m").session_id, original.session_id)
        copied = self.store.get_session_memories(imported.session_id)[0]
        self.assertNotEqual(copied.memory_id, "m")
        self.assertEqual(copied.metadata["imported_from_memory_id"], "m")

    def test_delete_clears_short_term(self):
        entry = MemoryEntry(memory_id="m", session_id="s", question="Q", answer="A")
        self.store.save_memory(entry, [1.] * 768)
        self.store.delete_memory("m")
        self.assertEqual(self.store.get_short_term_memories(), [])

    def test_concurrent_session_updates_are_not_lost(self):
        manager = SessionManager(self.store)
        session = manager.create_session()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: manager.update_session_activity(session.session_id), range(30)))
        self.assertEqual(self.store.get_session(session.session_id).question_count, 30)
        json.loads(Path(self.store._sessions_file).read_text(encoding="utf-8"))
