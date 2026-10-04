"""Standalone trace API, independent of dashboard state and corpus registries."""
import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .evaluator import evaluate_trace
from .schema import EvaluationRequest

app = FastAPI(title="X-RAG Portable Trace API", version="0.1.0")
logger = logging.getLogger(__name__)


@app.get("/health")
def health():
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/evaluate")
def evaluate(payload: EvaluationRequest):
    try:
        return evaluate_trace(payload.trace, claim_mode=payload.claim_mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        reference = str(uuid4())
        logger.exception("Evaluation failed [ref=%s]", reference)
        raise HTTPException(status_code=500, detail={"error": "Evaluation failed", "reference_id": reference}) from exc
