"""
FastAPI backend for the React UI: chat (SSE), graphs, sessions.

Thin transport over existing implementations -- chat_service.run_chat_turn,
ui/components/graph.py's build_graph_data, src/memory_graph.py's
build_memory_graph_data, and MemoryManager. No pipeline logic lives here.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import chat_service
from src.memory_graph import build_memory_graph_data
from ui.components.graph import build_graph_data, build_legal_graph_data

app = FastAPI(title="X-RAG UI backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:5174", "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    arm: str = "C_hybrid_rerank"
    corpus: str = "statutes"
    chat_history: List[Dict[str, str]] = Field(default_factory=list, max_length=100)
    memory_enabled: bool = True
    deep_analysis: bool = False

    @model_validator(mode="after")
    def validate_options(self):
        from experiments.exp06_strategy_ablation import ARMS
        if not self.question.strip():
            raise ValueError("question must not be blank")
        if self.corpus not in chat_service.CORPUS_OPTIONS or self.arm not in ARMS:
            raise ValueError("Unknown corpus or retrieval arm")
        if self.corpus == "both" and self.arm != "C_hybrid_rerank":
            raise ValueError("Combined corpus requires C_hybrid_rerank")
        if self.corpus == "statutes" and (ARMS[self.arm].get("graph") or ARMS[self.arm].get("graph_expand")
                                         or self.arm == "C_fixed_chunking"):
            raise ValueError("This arm requires the judgments corpus")
        if any(t.get("role") not in ("user", "assistant") or not isinstance(t.get("content"), str)
               or len(t["content"]) > 20000 for t in self.chat_history):
            raise ValueError("Invalid conversation history")
        return self


class SessionCreateRequest(BaseModel):
    title: str = "New Session"


@app.get("/ui/config")
def get_config():
    from experiments.exp06_strategy_ablation import ARMS
    from src.device import describe_device

    return {
        "arms": list(ARMS.keys()),
        "corpora": chat_service.CORPUS_OPTIONS,
        "device": describe_device(),
        "arms_by_corpus": {c: [a for a, options in ARMS.items()
                              if (c != "both" or a == "C_hybrid_rerank")
                              and (c != "statutes" or not (options.get("graph") or options.get("graph_expand") or a == "C_fixed_chunking"))]
                           for c in chat_service.CORPUS_OPTIONS},
    }


@app.post("/ui/chat")
def chat(req: ChatRequest):
    """SSE stream. Declared as a sync `def` (not `async def`) so the blocking
    pipeline runs in Starlette's threadpool instead of the event loop --
    otherwise one slow turn would freeze every other route.
    """
    def sse_events():
        for event in chat_service.run_chat_turn(
            question=req.question,
            session_id=req.session_id,
            arm=req.arm,
            corpus=req.corpus,
            chat_history=req.chat_history,
            memory_enabled=req.memory_enabled,
            deep_analysis=req.deep_analysis,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_events(), media_type="text/event-stream")


@app.get("/ui/graph")
def graph(scope: str = "rag", session_id: Optional[str] = None, max_nodes: int = Query(150, ge=1, le=1000)):
    if scope == "memory":
        return build_memory_graph_data(max_observations=max_nodes)

    if scope == "legal":
        return build_legal_graph_data(chat_service.load_knowledge_graph(), max_nodes=max_nodes)

    if scope != "rag":
        raise HTTPException(status_code=400, detail="scope must be 'rag', 'legal' or 'memory'")

    from src.chunk_registry import ChunkRegistry

    registry_path = os.path.join(PROJECT_ROOT, "artifacts", "chunk_registry.json")
    registry = ChunkRegistry.load_from_json(registry_path) if os.path.exists(registry_path) else None

    mm = chat_service.get_memory_manager()
    sessions = mm.list_sessions()
    memories = mm.get_session_memories(session_id) if session_id else [
        m for s in sessions for m in mm.get_session_memories(s.session_id)
    ]

    return build_graph_data(
        registry=registry, memories=memories, sessions=sessions,
        last_result=None, max_chunks=max_nodes,
    )


@app.get("/ui/sessions")
def list_sessions():
    mm = chat_service.get_memory_manager()
    return [s.to_dict() for s in mm.list_sessions()]


@app.post("/ui/sessions")
def create_session(req: SessionCreateRequest):
    mm = chat_service.get_memory_manager()
    return mm.create_session(req.title).to_dict()


class SessionRenameRequest(BaseModel):
    title: str


@app.patch("/ui/sessions/{session_id}")
def rename_session(session_id: str, req: SessionRenameRequest):
    mm = chat_service.get_memory_manager()
    if not mm.rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "title": req.title}


@app.get("/ui/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Reload a session's turns for the message list.

    Only ``role``/``content`` survive a turn's live SSE stream into permanent
    storage -- the chunks, timings and eval scores the UI showed while the
    turn was in flight only ever existed in the frontend's in-memory state,
    so a page reload or navigating away from Chat and back used to make a
    turn's Details panel vanish for good even though the answer's own
    citations were still sitting right there in the text. ``trace_id`` is
    the one durable link back to that data (see ``/ui/trace/{trace_id}``,
    which reads the full RAGTrace JSON saved alongside it) -- returning it
    here lets the frontend re-fetch and rebuild the Details panel on demand.
    """
    mm = chat_service.get_memory_manager()
    memories = mm.get_session_memories(session_id)
    messages = []
    for mem in memories:
        messages.append({"role": "user", "content": mem.question})
        messages.append({
            "role": "assistant",
            "content": mem.answer,
            "trace_id": mem.trace_id or None,
        })
    return messages


@app.delete("/ui/sessions/{session_id}")
def delete_session(session_id: str):
    mm = chat_service.get_memory_manager()
    if not mm.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/ui/memory")
def list_memory(search: Optional[str] = None, session_id: Optional[str] = None, limit: int = Query(30, ge=1, le=1000)):
    """Memory page data. With ``search``, ranks via MemoryManager.search_memory
    (semantic/recency/frequency/importance all populated). Without it, this is
    a plain recency listing -- those per-query scores don't exist outside a
    search, so only ``importance_score`` (intrinsic to the memory) is set and
    the rest come back 0.
    """
    mm = chat_service.get_memory_manager()

    if search:
        results = mm.search_memory(search, top_k=limit, session_id=session_id)
        return {"memories": [r.to_dict() for r in results], "searched": True}

    memories = (
        mm.get_session_memories(session_id) if session_id
        else mm.store.get_all_memories(limit=limit)
    )
    memories = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:limit]
    return {
        "memories": [
            {
                "memory": m.to_dict(),
                "semantic_score": 0.0, "recency_score": 0.0, "frequency_score": 0.0,
                "importance_score": m.importance_score, "final_score": 0.0,
                "retrieval_reason": "",
            }
            for m in memories
        ],
        "searched": False,
    }


@app.get("/ui/trace/latest")
def latest_trace():
    trace = _read_latest_trace_file()
    if trace is None:
        raise HTTPException(status_code=404, detail="No traces recorded yet")
    return trace


@app.get("/ui/trace/{trace_id}")
def get_trace(trace_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", trace_id):
        raise HTTPException(status_code=422, detail="Invalid trace identifier")
    trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
    matches = glob.glob(os.path.join(trace_dir, "*", f"trace_{trace_id}.json"))
    matches += [p for p in (os.path.join(trace_dir, f"{trace_id}.json"),
                            os.path.join(trace_dir, f"trace_{trace_id}.json")) if os.path.isfile(p)]
    if not matches:
        raise HTTPException(status_code=404, detail="Trace not found")
    with open(matches[0], "r", encoding="utf-8") as f:
        trace = json.load(f)
    diagnostics = trace.get("diagnostics") or {}
    # Legacy UI wire fields are maintained while storage uses canonical diagnostics.
    for key in ("claim_verification", "claim_error"):
        if key in diagnostics:
            trace[key] = diagnostics[key]

    # RAGTraceBuilder records chunk_id/scores/provenance but never the chunk
    # text itself (keeps trace files small) -- the live SSE `chunks` event is
    # the only place that ever carried it, so a re-fetched trace would show
    # empty chunk cards. The trace doesn't record which corpus it ran
    # against either, so check both registries; a chunk_id collision across
    # corpora is not realistic (they're per-ingestion UUIDs).
    refs = trace.get("retrieved_chunk_references") or []
    if refs:
        # Captured prompt evidence remains valid when registries are replaced.
        snapshot = dict(re.findall(
            r"--- Context chunk \d+ \[Chunk-ID: ([^\]]*)\] ---\n(.*?)(?=\n--- Context chunk |\n\nQuestion: |\Z)",
            trace.get("prompt_snapshot") or "", re.DOTALL))
        for ref in refs:
            if ref.get("chunk_id") in snapshot:
                ref["text"] = snapshot[ref["chunk_id"]].strip()[:400]
        corpus = (trace.get("configuration_snapshot") or {}).get("corpus")
        corpora = [corpus] if corpus in chat_service.CORPORA else chat_service.CORPORA
        for corpus in corpora:
            registry = chat_service.load_registry(corpus)
            if registry is None:
                continue
            for ref in refs:
                if not ref.get("text"):
                    record = registry.get_chunk(ref["chunk_id"])
                    if record:
                        ref["text"] = record.text[:400]
        for ref in refs:
            ref.setdefault("text", "")

    return trace


def _read_latest_trace_file() -> Optional[Dict[str, Any]]:
    trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
    files = glob.glob(os.path.join(trace_dir, "*", "trace_*.json"))
    files += glob.glob(os.path.join(trace_dir, "*.json"))
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)

