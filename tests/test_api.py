import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api import app
from src.rag_trace import RAGTrace

client = TestClient(app, raise_server_exceptions=False)


class FakeReport:
    """A plain object (not a MagicMock) so report.__dict__ returns real attributes."""
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def save(self):
        pass


def make_valid_trace_payload():
    return {
        "trace_id": "TRACE_API_TEST",
        "trace_version": "1.0",
        "pipeline_version": "1.0",
        "framework_version": "1.0",
        "timestamp": "2026-07-17T12:00:00Z",
        "question": "Q?",
        "generated_answer": "A.",
        "prompt_snapshot": "Prompt",
        "prompt_length": 6,
        "retrieved_chunk_references": [],
        "configuration_snapshot": {},
        "execution_statistics": {},
        "pipeline_stage_status": {},
        "diagnostics": None,
    }


class TestAPI(unittest.TestCase):
    def test_get_root(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("framework_name", response.json())

    def test_get_health(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")

    def test_get_version(self):
        response = client.get("/version")
        self.assertEqual(response.status_code, 200)
        self.assertIn("framework_version", response.json())
        
    def test_get_report_not_found(self):
        response = client.get("/report/TRACE_INVALID")
        self.assertEqual(response.status_code, 404)
        
    def test_get_artifacts_not_found(self):
        response = client.get("/artifacts/TRACE_INVALID")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["artifacts"]), 0)

    def test_analyze_invalid_payload(self):
        response = client.post("/analyze", json={"invalid": "payload"})
        self.assertEqual(response.status_code, 422)

    def test_analyze_valid_payload_success(self):
        fake_report = FakeReport(analysis_status="COMPLETED")

        with patch("src.api.get_runner") as mock_get_runner:
            mock_get_runner.return_value.run.return_value = fake_report
            response = client.post("/analyze", json=make_valid_trace_payload())

        self.assertEqual(response.status_code, 200)

    def test_analyze_internal_error_returns_generic_message(self):
        with patch("src.api.get_runner") as mock_get_runner:
            mock_get_runner.return_value.run.side_effect = RuntimeError("db password abc123")
            response = client.post("/analyze", json=make_valid_trace_payload())

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertIn("reference_id", body)
        self.assertNotIn("abc123", response.text)

    def test_get_runner_is_singleton(self):
        from src.api import get_runner
        get_runner.cache_clear()
        try:
            with patch("src.api.PipelineRunner") as mock_runner_cls:
                first = get_runner()
                second = get_runner()
                self.assertIs(first, second)
                self.assertEqual(mock_runner_cls.call_count, 1)
        finally:
            get_runner.cache_clear()

if __name__ == '__main__':
    unittest.main()
