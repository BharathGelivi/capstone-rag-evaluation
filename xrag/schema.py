"""Portable trace input; evidence is supplied by the caller."""
import re
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, AliasChoices, model_validator


class EvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="allow")
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1, validation_alias=AliasChoices("text", "chunk_text"))
    rank: int = Field(default=1, ge=0)
    similarity_score: float | None = None

    @model_validator(mode="after")
    def nonblank_text(self):
        if not self.text.strip():
            raise ValueError("Evidence text must not be blank")
        return self


class TraceInput(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid4()), pattern=r"^[A-Za-z0-9_-]{1,128}$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1, validation_alias=AliasChoices("answer", "generated_answer"))
    retrieved_chunks: list[EvidenceChunk] = Field(
        validation_alias=AliasChoices("retrieved_chunks", "retrieved_chunk_references"))
    claims: list[str] | None = Field(default=None, min_length=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    prompt_snapshot: str = ""
    configuration_snapshot: dict = Field(default_factory=dict)
    execution_statistics: dict = Field(default_factory=dict)
    pipeline_stage_status: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def recover_snapshot_evidence(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        key = "retrieved_chunks" if "retrieved_chunks" in value else "retrieved_chunk_references"
        refs = value.get(key)
        snapshot = dict(re.findall(
            r"--- Context chunk \d+ \[Chunk-ID: ([^\]]*)\] ---\n(.*?)(?=\n--- Context chunk |\n\nQuestion: |\Z)",
            value.get("prompt_snapshot") or "", re.DOTALL))
        if isinstance(refs, list):
            normalized = []
            for i, ref in enumerate(refs, 1):
                if isinstance(ref, str):
                    ref = {"text": ref}
                if isinstance(ref, dict):
                    ref = dict(ref)
                    ref.setdefault("chunk_id", f"chunk-{i}")
                    ref.setdefault("rank", i)
                    if not (ref.get("text") or ref.get("chunk_text")) and ref["chunk_id"] in snapshot:
                        ref["text"] = snapshot[ref["chunk_id"]]
                normalized.append(ref)
            value[key] = normalized
        return value

    @model_validator(mode="after")
    def validate_content(self):
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("Question and answer must not be blank")
        if self.claims is not None and any(not claim.strip() for claim in self.claims):
            raise ValueError("Claims must not be blank")
        ids = [chunk.chunk_id for chunk in self.retrieved_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence chunk IDs must be unique")
        return self


class EvaluationRequest(BaseModel):
    trace: TraceInput
    claim_mode: Literal["sentences", "llm"] = "sentences"
