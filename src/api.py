import os
import glob
import uuid
import re
import threading
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from src.logger import get_logger
from src.rag_trace import RAGTrace, RAGTraceBuilder
from src.runner import PipelineRunner
from src.report import DiagnosticEvaluationReport
from src.report_presenter import DiagnosticReportPresenter
from configs.api import API_VERSION, SUPPORTED_PIPELINE_VERSION


class RAGTraceRequest(BaseModel):
    """Mirrors the RAGTrace dataclass fields (src/rag_trace.py) for request validation."""
    trace_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    trace_version: str
    pipeline_version: str
    framework_version: str
    timestamp: str
    question: str
    generated_answer: str
    prompt_snapshot: str
    prompt_length: int
    retrieved_chunk_references: List[Dict[str, Any]]
    configuration_snapshot: Dict[str, Any]
    execution_statistics: Dict[str, Any]
    pipeline_stage_status: Dict[str, str]
    diagnostics: Optional[Dict[str, Any]] = None

    @field_validator("configuration_snapshot")
    @classmethod
    def validate_registry(cls, snapshot):
        if snapshot.get("registry_path"):
            root = Path("artifacts").resolve()
            if not Path(snapshot["registry_path"]).resolve().is_relative_to(root):
                raise ValueError("registry_path must be inside artifacts")
        return snapshot

logger = get_logger(__name__)

app = FastAPI(
    title="X-RAG Diagnostic Framework API",
    version=API_VERSION,
    description="API for running diagnostic pipelines on RAG execution traces."
)


_runner_lock = threading.Lock()

@lru_cache(maxsize=1)
def _get_runner() -> PipelineRunner:
    """Lazily construct the PipelineRunner (and its heavy models) on first use."""
    return PipelineRunner()


def get_runner() -> PipelineRunner:
    with _runner_lock:
        return _get_runner()


get_runner.cache_clear = _get_runner.cache_clear


def _safe_id(trace_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
        raise HTTPException(status_code=422, detail="Invalid trace identifier")
    return trace_id

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    reference_id = str(uuid.uuid4())
    logger.error(f"Unhandled error [ref={reference_id}]: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An internal error occurred.",
            "reference_id": reference_id
        }
    )

@app.get("/")
def get_root():
    return {
        "framework_name": "X-RAG Diagnostic Framework",
        "framework_version": API_VERSION,
        "status": "Online"
    }

@app.get("/health")
def get_health():
    return {
        "status": "healthy",
        "framework_version": API_VERSION
    }

@app.get("/version")
def get_version():
    return {
        "framework_version": API_VERSION,
        "artifact_versions": "1.0",
        "supported_pipeline_version": SUPPORTED_PIPELINE_VERSION
    }

@app.post("/analyze")
def analyze_trace(payload: RAGTraceRequest):
    """
    Takes a RAGTrace JSON payload, runs the complete diagnostic pipeline,
    and returns a DiagnosticEvaluationReport JSON.
    """
    trace = RAGTrace(**payload.model_dump())

    report = get_runner().run(trace)

    # Save artifacts
    report.save()

    # We can also save the original trace since it was passed here
    RAGTraceBuilder.save_to_json(trace)

    return report.__dict__

@app.get("/report/{trace_id}")
def get_report(trace_id: str):
    _safe_id(trace_id)
    path = f"artifacts/reports/{trace_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    report = DiagnosticEvaluationReport.load(path)
    return report.__dict__

@app.get("/report/{trace_id}/markdown")
def get_report_markdown(trace_id: str):
    _safe_id(trace_id)
    path = f"artifacts/reports/{trace_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    report = DiagnosticEvaluationReport.load(path)
    presenter = DiagnosticReportPresenter(report)
    return PlainTextResponse(presenter.render_markdown())

@app.get("/report/{trace_id}/html")
def get_report_html(trace_id: str):
    _safe_id(trace_id)
    path = f"artifacts/reports/{trace_id}.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    report = DiagnosticEvaluationReport.load(path)
    presenter = DiagnosticReportPresenter(report)
    return HTMLResponse(presenter.render_html())

@app.get("/artifacts/{trace_id}")
def get_artifacts(trace_id: str):
    _safe_id(trace_id)
    artifacts = {}
    base_dirs = {
        "RAGTrace": "artifacts/rag_traces",
        "ClaimSet": "artifacts/claims",
        "Verification": "artifacts/verification",
        "PipelineStateMatrix": "artifacts/pipeline_state_matrix",
        "RootCauseAnalysis": "artifacts/root_cause_analysis",
        "CorrectiveActionPlan": "artifacts/corrective_action_plan",
        "DiagnosticEvaluationReport": "artifacts/reports"
    }
    
    for name, directory in base_dirs.items():
        candidates = [f"{directory}/{trace_id}.json", f"{directory}/TRACE_{trace_id}.json"]
        if name == "RAGTrace":
            candidates += sorted(glob.glob(f"{directory}/*/trace_{trace_id}.json"))
        if name == "ClaimSet":
            candidates += [f"artifacts/claim_sets/{trace_id}.json"]
        for path in candidates:
            if os.path.isfile(path):
                artifacts[name] = path
                break
            
    return {"trace_id": trace_id, "artifacts": artifacts}
