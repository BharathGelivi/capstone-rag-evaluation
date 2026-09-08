import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

@dataclass
class FrameworkMetadata:
    trace_id: str
    framework_name: str = "X-RAG Diagnostic Framework"
    framework_version: str = "1.0"
    analysis_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutiveSummary:
    question: str
    generated_answer: str
    overall_health_score: float
    primary_issue: str
    supported_claims: int
    total_claims: int
    number_of_corrective_actions: int
    summary: str

@dataclass
class PipelineStageResult:
    stage: str
    status: str
    observation: str
    confidence: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineOverview:
    pipeline_stages: List[PipelineStageResult] = field(default_factory=list)

@dataclass
class EvaluationMetrics:
    retrieved_chunks: int
    verified_claims: int
    supported_claims: int
    partially_supported_claims: int
    unsupported_claims: int
    contradicted_claims: int
    grounding_score: float
    evidence_coverage: float
    average_entailment: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    verification_latency_ms: float

@dataclass
class EvidenceAnalysis:
    claim_id: str
    claim_text: str
    verification_status: str
    supporting_chunk_id: Optional[str]
    supporting_chunk_rank: Optional[int]
    supporting_evidence: str

@dataclass
class RootCauseSection:
    primary_cause: str
    secondary_effects: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    diagnosis_confidence: str = "HIGH"

@dataclass
class CorrectiveActionSection:
    priority: str
    title: str
    description: str
    observed_evidence: str
    expected_improvement: str
    success_metric: str
    tradeoff: str

@dataclass
class OverallAssessment:
    major_strength: str
    major_weakness: str
    next_priority: str
    overall_recommendation: str

@dataclass
class RagasMetricsSection:
    """RAGAS-style scores (see src/ragas_metrics.py). None means "not computed"."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None

@dataclass
class AnswerCorrectnessSection:
    """
    Answer correctness (claim recall) -- see src/answer_correctness_evaluator.py.
    None means "not computed" (no gold answer was available for this trace).
    Purely additive: does not affect overall_health_score or primary_issue.
    Full per-claim breakdown lives in artifacts/answer_correctness/TRACE_<id>.json,
    not duplicated here.
    """
    claim_recall: Optional[float] = None
    total_gold_claims: Optional[int] = None
    recalled_gold_claims: Optional[int] = None

@dataclass
class DiagnosticEvaluationReport:
    framework_metadata: FrameworkMetadata
    executive_summary: ExecutiveSummary
    pipeline_overview: PipelineOverview
    evaluation_metrics: EvaluationMetrics
    root_cause_analysis: RootCauseSection
    overall_assessment: OverallAssessment
    evidence_analysis: List[EvidenceAnalysis] = field(default_factory=list)
    corrective_actions: List[CorrectiveActionSection] = field(default_factory=list)
    ragas_metrics: Optional[RagasMetricsSection] = None
    answer_correctness: Optional[AnswerCorrectnessSection] = None
    artifact_version: str = "1.0"
    analysis_status: str = "COMPLETED"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'DiagnosticEvaluationReport':
        data = json.loads(data_str)
        
        data["framework_metadata"] = FrameworkMetadata(**data.get("framework_metadata", {}))
        data["executive_summary"] = ExecutiveSummary(**data.get("executive_summary", {}))
        
        stages = [PipelineStageResult(**s) for s in data.get("pipeline_overview", {}).get("pipeline_stages", [])]
        data["pipeline_overview"] = PipelineOverview(pipeline_stages=stages)
        
        data["evaluation_metrics"] = EvaluationMetrics(**data.get("evaluation_metrics", {}))
        data["root_cause_analysis"] = RootCauseSection(**data.get("root_cause_analysis", {}))
        data["overall_assessment"] = OverallAssessment(**data.get("overall_assessment", {}))
        
        data["evidence_analysis"] = [EvidenceAnalysis(**e) for e in data.get("evidence_analysis", [])]
        data["corrective_actions"] = [CorrectiveActionSection(**c) for c in data.get("corrective_actions", [])]

        if data.get("ragas_metrics"):
            data["ragas_metrics"] = RagasMetricsSection(**data["ragas_metrics"])

        if data.get("answer_correctness"):
            data["answer_correctness"] = AnswerCorrectnessSection(**data["answer_correctness"])

        return cls(**data)

    def save(self, base_dir: str = "artifacts/reports") -> str:
        os.makedirs(base_dir, exist_ok=True)
        trace_id = self.framework_metadata.trace_id
        filepath = os.path.join(base_dir, f"{trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'DiagnosticEvaluationReport':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
