"""Fast, no-browser checks for src/api_ui.py -- run before the Playwright e2e suite."""

import unittest

from fastapi.testclient import TestClient

from src.api_ui import app


class TestApiUI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_config_lists_arms_and_corpora(self):
        r = self.client.get("/ui/config")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("D_ircot", body["arms"])
        self.assertIn("F_graphrag", body["arms"])
        self.assertIn("statutes", body["corpora"])

    def test_graph_memory_scope(self):
        r = self.client.get("/ui/graph", params={"scope": "memory", "max_nodes": 20})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        node_ids = {n["id"] for n in body["nodes"]}
        for edge in body["edges"]:
            self.assertIn(edge["source"], node_ids)
            self.assertIn(edge["target"], node_ids)

    def test_graph_rag_scope(self):
        r = self.client.get("/ui/graph", params={"scope": "rag", "max_nodes": 20})
        self.assertEqual(r.status_code, 200)
        self.assertIn("nodes", r.json())

    def test_graph_bad_scope_rejected(self):
        r = self.client.get("/ui/graph", params={"scope": "bogus"})
        self.assertEqual(r.status_code, 400)

    def test_sessions_list(self):
        r = self.client.get("/ui/sessions")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_memory_list_without_search(self):
        r = self.client.get("/ui/memory", params={"limit": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertFalse(body["searched"])
        self.assertIsInstance(body["memories"], list)

    def test_memory_search_returns_scored_results(self):
        r = self.client.get("/ui/memory", params={"search": "arrest", "limit": 5})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["searched"])
        for m in body["memories"]:
            self.assertIn("final_score", m)
            self.assertIn("memory", m)

    def test_trace_not_found_is_404(self):
        r = self.client.get("/ui/trace/does-not-exist")
        self.assertEqual(r.status_code, 404)

    def test_latest_trace_matches_shape_when_present(self):
        r = self.client.get("/ui/trace/latest")
        # Empty artifacts/rag_traces/ in a fresh checkout is a legitimate
        # 404, not a bug -- only assert the shape when one exists.
        if r.status_code == 200:
            body = r.json()
            self.assertIn("trace_id", body)
            self.assertIn("prompt_snapshot", body)


if __name__ == "__main__":
    unittest.main()
