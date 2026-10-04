"""Contract and isolation tests for the external trace package."""
import copy
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from xrag import TraceEvaluator
from xrag.api import app
from xrag.schema import TraceInput


def trace():
    return {"question": "Capital of France?", "answer": "Paris is the capital of France.",
            "retrieved_chunks": [{"chunk_id": "a", "text": "Paris is the capital of France."}],
            "claims": ["Paris is the capital of France."]}


def test_accepts_canonical_trace_and_snapshot_evidence():
    payload = {"question": "Q", "generated_answer": "A", "retrieved_chunk_references": [{"chunk_id": "a"}],
               "prompt_snapshot": "--- Context chunk 1 [Chunk-ID: a] ---\nEvidence.\n\nQuestion: Q"}
    parsed = TraceInput.model_validate(payload)
    assert parsed.retrieved_chunks[0].text == "Evidence."
    assert "text" not in payload["retrieved_chunk_references"][0]


def test_preserves_chunk_provenance():
    payload = trace()
    payload["retrieved_chunks"][0].update(parent_document_id="doc", chunk_index=3, reranker_score=0.9)
    chunk = TraceInput.model_validate(payload).retrieved_chunks[0].model_dump()
    assert chunk["parent_document_id"] == "doc"
    assert chunk["chunk_index"] == 3
    assert chunk["reranker_score"] == 0.9


@pytest.mark.parametrize("change", [
    {"trace_id": "../escape"}, {"answer": " "}, {"claims": []}, {"claims": [" "]},
    {"retrieved_chunks": [{"chunk_id": "a"}]},
    {"retrieved_chunks": [{"chunk_id": "a", "text": " "}]},
    {"retrieved_chunks": [{"chunk_id": "a", "text": "A"}, {"chunk_id": "a", "text": "B"}]},
])
def test_rejects_invalid_input_before_model_load(change):
    evaluator = TraceEvaluator()
    with pytest.raises(ValidationError):
        evaluator.evaluate({**trace(), **change})
    assert evaluator._verifier is None


def test_missing_llm_credentials_fail_before_model_load(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    payload = trace()
    payload.pop("claims")
    evaluator = TraceEvaluator()
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        evaluator.evaluate(payload, claim_mode="llm")
    assert evaluator._verifier is None


def test_evaluates_reuses_model_and_never_reads_registry_or_saves():
    # Real verifier/diagnostic logic with only the costly model forward pass mocked.
    payload = trace()
    payload["configuration_snapshot"] = {"registry_path": "private/registry.json"}
    original = copy.deepcopy(payload)
    scores = [[{"label": "entailment", "score": 0.98},
               {"label": "neutral", "score": 0.01}, {"label": "contradiction", "score": 0.01}]]
    with patch("src.claim_verifier.pipeline") as pipeline, \
         patch("src.chunk_registry.ChunkRegistry.load_from_json") as registry, \
         patch("src.claim_verifier.ClaimVerifier.save_artifacts") as save, \
         patch("src.rag_trace.RAGTraceBuilder.save_to_json") as save_trace:
        pipeline.return_value.return_value = scores
        evaluator = TraceEvaluator()
        result = evaluator.evaluate(payload)
        evaluator.evaluate(json.dumps(payload))
        assert pipeline.call_count == 1
        registry.assert_not_called()
        save.assert_not_called()
        save_trace.assert_not_called()
    assert payload == original
    assert result["verification"]["supported_claims"] == 1
    assert result["metrics"]["claim_support_rate"] == 1.0
    assert result["metadata"]["llm_judge_enabled"] is False
    assert result["diagnostic_report"]["metadata"]["artifact_paths"] == {}


def test_empty_evidence_is_unverifiable():
    payload = trace()
    payload["retrieved_chunks"] = []
    with patch("src.claim_verifier.pipeline"):
        result = TraceEvaluator().evaluate(payload)
    assert result["verification"]["unsupported_claims"] == 0
    assert result["verification"]["not_verifiable_claims"] == 1
    assert result["diagnostic_report"]["analysis_status"] == "PARTIAL_ANALYSIS"


def test_llm_mode_disables_debug_artifact_writes(monkeypatch):
    from src.claim_decomposer import CandidateClaim, CandidateClaimSet
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("CLAIM_DECOMPOSER_DEBUG", "true")
    payload = trace()
    payload.pop("claims")
    candidates = CandidateClaimSet(trace_id="llm-test")
    candidates.add_claim(CandidateClaim("c1", "llm-test", "Paris is the capital of France.", "S1", 0, 0, 31))
    payload["trace_id"] = "llm-test"
    with patch("src.claim_decomposer.ClaimDecomposer") as decomposer, patch("src.claim_verifier.pipeline"):
        decomposer.return_value.decompose.return_value = candidates
        result = TraceEvaluator().evaluate(payload, claim_mode="llm")
        decomposer.assert_called_once_with(debug=False)
    assert result["metadata"]["claim_source"] == "llm"


def test_http_validation_and_forwarding():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/evaluate", json={"trace": {"question": "missing"}}).status_code == 422
    with patch("xrag.api.evaluate_trace", return_value={"trace_id": "ok"}) as evaluate:
        response = client.post("/evaluate", json={"trace": trace(), "claim_mode": "sentences"})
        assert response.json() == {"trace_id": "ok"}
        assert isinstance(evaluate.call_args.args[0], TraceInput)


def test_http_errors_do_not_expose_internal_details():
    with patch("xrag.api.evaluate_trace", side_effect=RuntimeError("secret-value")):
        response = TestClient(app).post("/evaluate", json={"trace": trace()})
    assert response.status_code == 500
    assert "secret-value" not in response.text
    assert response.json()["detail"]["reference_id"]
