import os
import json
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.root_cause_reasoner import RootCauseAnalysis, FailureType, RootCauseReasoner
from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineStage
from configs.thresholds import LOW_RETRIEVAL_EFFICIENCY_THRESHOLD


class _SafeDict(dict):
    """dict subclass that substitutes 'N/A' for any key missing at format_map time."""
    def __missing__(self, key):
        return "N/A"

class ActionCategory(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    CHUNKING = "CHUNKING"
    GENERATION = "GENERATION"
    CORPUS = "CORPUS"
    SYSTEM = "SYSTEM"

@dataclass
class CorrectiveAction:
    action_id: str
    category: ActionCategory
    title: str
    description: str
    observed_evidence: str
    root_cause: str
    expected_improvement: str
    success_metric: str
    tradeoff: str
    priority: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrectiveActionPlan:
    trace_id: str
    primary_cause: FailureType
    artifact_version: str = "1.0"
    immediate_actions: List[CorrectiveAction] = field(default_factory=list)
    short_term_actions: List[CorrectiveAction] = field(default_factory=list)
    experimental_actions: List[CorrectiveAction] = field(default_factory=list)
    # "informational" is a fourth priority tier, distinct from the three above:
    # standalone advisories (e.g. retrieval efficiency) sourced directly from
    # observed metadata, independent of RootCauseReasoner's causal failure path.
    informational_actions: List[CorrectiveAction] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = asdict(self)
        data["primary_cause"] = data["primary_cause"].value
        for action_list in ["immediate_actions", "short_term_actions", "experimental_actions", "informational_actions"]:
            for action in data[action_list]:
                action["category"] = action["category"].value
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'CorrectiveActionPlan':
        data = json.loads(data_str)
        data["primary_cause"] = FailureType(data["primary_cause"])

        for action_list in ["immediate_actions", "short_term_actions", "experimental_actions", "informational_actions"]:
            actions = []
            for act_data in data.get(action_list, []):
                act_data["category"] = ActionCategory(act_data["category"])
                actions.append(CorrectiveAction(**act_data))
            data[action_list] = actions

        return cls(**data)

    def save(self, base_dir: str = "artifacts/corrective_action_plan") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'CorrectiveActionPlan':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

class CorrectiveActionEngine:

    def __init__(self):
        self.lookup_table = self._initialize_lookup_table()

    # Maps a recommendation title to the configuration_snapshot keys/values that
    # indicate the capability is already deployed. If the snapshot matches, the
    # recommendation is suppressed as inapplicable.
    #
    # Format: {title_substring: {snapshot_key: value_that_means_enabled}}
    _CAPABILITY_GUARDS: Dict[str, Dict[str, Any]] = {
        "Hybrid Search":          {"bm25_enabled": True},
        "Cross-Encoder Reranker": {"reranker_enabled": True},
        "Lower Sampling Temperature": {"temperature": 0.0},
    }

    def _initialize_lookup_table(self) -> Dict[FailureType, List[Dict[str, str]]]:
        return {
            FailureType.MISSING_CORPUS: [
                {
                    "category": ActionCategory.CORPUS,
                    "title": "Ingest Missing Domain Knowledge",
                    "description": "Expand corpus to cover entities not found in the vector index.",
                    "observed_evidence": "All retrieved chunks have low similarity scores for given keywords (closest of {pool_size} candidates had cosine distance {pre_rerank_min_dense_distance}).",
                    "root_cause": "The vector store simply does not contain the required information.",
                    "expected_improvement": "Higher retrieval relevance scores for out-of-domain queries.",
                    "success_metric": "Reduce closest-candidate cosine distance below the corpus relevance threshold (currently {pre_rerank_min_dense_distance}).",
                    "tradeoff": "Increases index size and embedding costs.",
                    "priority": "immediate"
                }
            ],
            FailureType.RETRIEVAL_MISS: [
                {
                    "category": ActionCategory.RETRIEVAL,
                    "title": "Increase Retrieval Top-K",
                    "description": "Increase the number of retrieved documents before providing context to the LLM.",
                    "observed_evidence": "Ground truth or highly relevant chunks are present but ranked lower than current Top-K (observed max retrieval score: {max_score}, threshold: {threshold}).",
                    "root_cause": "Semantic similarity metric fails to rank the most contextually relevant chunks at the very top.",
                    "expected_improvement": "Higher recall; missing context is caught within the expanded window.",
                    "success_metric": "Verification support score increases above 0.8 (current retrieval confidence: {confidence}).",
                    "tradeoff": "Increases prompt token usage, latency, and cost.",
                    "priority": "immediate"
                },
                {
                    "category": ActionCategory.RETRIEVAL,
                    "title": "Implement Hybrid Search (Dense + Sparse)",
                    "description": "Add BM25 lexical search and combine with semantic vector search using Reciprocal Rank Fusion.",
                    "observed_evidence": "Queries containing specific IDs, acronyms, or names fail semantic retrieval.",
                    "root_cause": "Dense embeddings often fail at exact keyword matching.",
                    "expected_improvement": "Near-perfect recall for exact-match terminology.",
                    "success_metric": "Retrieval hit rate for exact entity queries approaches 1.0.",
                    "tradeoff": "Requires maintaining a secondary inverted index and fusion overhead.",
                    "priority": "short_term"
                },
                {
                    "category": ActionCategory.RETRIEVAL,
                    "title": "Deploy Cross-Encoder Reranker",
                    "description": "Pass the Top-N results through a Cross-Encoder to re-score query-document relevance.",
                    "observed_evidence": "Bi-encoder returns top-k chunks that share vocabulary but lack true semantic relevance to the question.",
                    "root_cause": "Bi-encoders calculate dot products independently and miss deep query-document interactions.",
                    "expected_improvement": "Drastically improved precision at Top-1 and Top-3.",
                    "success_metric": "Precision@3 increases by at least 20%.",
                    "tradeoff": "High computational cost; increases retrieval latency by 100-300ms.",
                    "priority": "experimental"
                }
            ],
            FailureType.CHUNK_BOUNDARY: [
                {
                    "category": ActionCategory.CHUNKING,
                    "title": "Increase Chunk Overlap",
                    "description": "Adjust text splitter to increase overlap between adjacent chunks by 10-20%.",
                    "observed_evidence": "LLM answers partially but misses critical context that was split into the next unretrieved chunk ({boundary_claim_ids_count} partially supported claim(s) affected).",
                    "root_cause": "Hard boundary splits separated a continuous semantic thought.",
                    "expected_improvement": "Context is preserved across chunks.",
                    "success_metric": "Reduction in 'Partially Supported' claims (currently {boundary_claim_ids_count}).",
                    "tradeoff": "Increases total chunk count, redundancy, and storage.",
                    "priority": "immediate"
                }
            ],
            FailureType.UNSUPPORTED_GENERATION: [
                {
                    "category": ActionCategory.GENERATION,
                    "title": "Lower Sampling Temperature",
                    "description": "Reduce generator temperature to 0.0 or 0.1 for deterministic outputs.",
                    "observed_evidence": "LLM generates facts that sound plausible but do not exist in the retrieved text ({unsupported_count} unsupported claim(s) observed).",
                    "root_cause": "High temperature allows the LLM to creatively sample from its parametric memory rather than strict context.",
                    "expected_improvement": "Output becomes rigid and closely aligned to the prompt context.",
                    "success_metric": "Hallucination rate (Unsupported claims) drops to near 0% (currently {unsupported_count}).",
                    "tradeoff": "Responses may become dry, repetitive, or overly cautious.",
                    "priority": "immediate"
                },
                {
                    "category": ActionCategory.GENERATION,
                    "title": "Strict Prompt Grounding",
                    "description": "Inject systemic instructions forcing the LLM to output 'I don't know' if context is insufficient.",
                    "observed_evidence": "The LLM guesses when retrieved context is irrelevant.",
                    "root_cause": "The prompt does not strongly penalize guessing.",
                    "expected_improvement": "LLM reliably declines to answer ungrounded queries.",
                    "success_metric": "Increase in 'Not Verifiable' traces rather than 'Unsupported' claims.",
                    "tradeoff": "May refuse to answer queries that could be logically deduced.",
                    "priority": "short_term"
                }
            ],
            FailureType.GROUNDING_FAILURE: [
                {
                    "category": ActionCategory.GENERATION,
                    "title": "Require Explicit Citations",
                    "description": "Force the LLM to append [Chunk-ID] to every sentence it generates.",
                    "observed_evidence": "LLM struggles to maintain factual fidelity across long generation spans ({contradicted_count} contradicted claim(s) observed).",
                    "root_cause": "Lack of forced intermediate reasoning (citations) causes drift from context.",
                    "expected_improvement": "Self-correction during generation ensures every fact is linked to evidence.",
                    "success_metric": "100% of claims map cleanly to a cited chunk (currently {contradicted_count} contradicted).",
                    "tradeoff": "Increases output token length and adds parsing overhead.",
                    "priority": "experimental"
                }
            ]
        }

    # Reverse of RootCauseReasoner.STAGE_TO_FAILURE_MAP, used to look up the
    # PipelineState (and its observed metadata) behind a given FailureType.
    FAILURE_TO_STAGE = {v: k for k, v in RootCauseReasoner.STAGE_TO_FAILURE_MAP.items()}

    def _build_format_values(self, psm: Optional[PipelineStateMatrix], failure: FailureType) -> Dict[str, str]:
        """Pre-formats observed trace values for interpolation into template strings."""
        if psm is None:
            return {}
        stage = self.FAILURE_TO_STAGE.get(failure)
        if stage is None:
            return {}
        state = psm.get(stage)
        if state is None:
            return {}

        values: Dict[str, Any] = {"confidence": state.confidence}
        values.update(state.metadata)

        format_values: Dict[str, str] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, float):
                format_values[key] = f"{value:.3f}"
            elif isinstance(value, list):
                format_values[f"{key}_count"] = str(len(value))
            else:
                format_values[key] = str(value)
        return format_values

    def _build_efficiency_advisories(self, psm: Optional[PipelineStateMatrix]) -> List[CorrectiveAction]:
        """
        Standalone advisory sourced directly from RETRIEVER's chunk_utilization_rate
        metadata (see PipelineStateAnalyzer) -- intentionally NOT produced via
        RootCauseReasoner/the failure-driven lookup table above, and NOT gated
        on any stage being FAIL. Purely informational.
        """
        if psm is None:
            return []
        retriever_state = psm.get(PipelineStage.RETRIEVER)
        if retriever_state is None:
            return []

        rate = retriever_state.metadata.get("chunk_utilization_rate")
        if rate is None or rate >= LOW_RETRIEVAL_EFFICIENCY_THRESHOLD:
            return []

        used = retriever_state.metadata.get("chunks_used")
        retrieved = retriever_state.metadata.get("chunks_retrieved")
        return [CorrectiveAction(
            action_id=str(uuid.uuid4()),
            category=ActionCategory.RETRIEVAL,
            title="Review Retrieval Top-K / Reranker Cutoff",
            description=(
                f"Retrieval efficiency is low ({rate * 100:.0f}%); consider reducing top-k "
                f"or tightening the reranker score cutoff."
            ),
            observed_evidence=f"{used}/{retrieved} retrieved chunks were used as supporting evidence for any verified claim.",
            root_cause="Retrieval breadth (top-k) may be wider than the corpus/question actually requires, adding unused context.",
            expected_improvement="Reduced prompt size and lower latency/cost with no loss of verified support.",
            success_metric=f"Chunk utilization rate above {LOW_RETRIEVAL_EFFICIENCY_THRESHOLD * 100:.0f}%.",
            tradeoff="Aggressively narrowing top-k risks losing recall for other, harder questions.",
            priority="informational",
            metadata={"chunk_utilization_rate": rate}
        )]

    def _is_already_deployed(self, title: str, config_snapshot: Dict[str, Any]) -> bool:
        """Return True if the recommendation's capability is already active in the pipeline.

        Checks the trace's configuration_snapshot against known capability guards so
        that the CAE never recommends, e.g., implementing hybrid search to a pipeline
        that already has BM25 enabled.
        """
        for title_fragment, guard in self._CAPABILITY_GUARDS.items():
            if title_fragment.lower() in title.lower():
                for key, expected_value in guard.items():
                    actual = config_snapshot.get(key)
                    if actual == expected_value:
                        return True
        return False

    def generate(
        self,
        rca: RootCauseAnalysis,
        psm: Optional[PipelineStateMatrix] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> CorrectiveActionPlan:
        """Generate a CorrectiveActionPlan from a RootCauseAnalysis.

        Args:
            rca:             Root cause analysis output from RootCauseReasoner.
            psm:             Pipeline state matrix used for evidence interpolation.
            config_snapshot: The pipeline's configuration_snapshot from RAGTrace.
                             Used to suppress recommendations for already-active
                             capabilities (e.g. no point recommending hybrid search
                             when bm25_enabled=True is in the snapshot).
        """
        immediate = []
        short_term = []
        experimental = []
        snapshot = config_snapshot or {}

        # Ordered dedup (not a set) so action ordering is deterministic across runs.
        failures_to_process = list(dict.fromkeys(
            ([rca.primary_cause] if rca.primary_cause != FailureType.UNKNOWN else [])
            + list(rca.secondary_effects)
        ))

        for failure in failures_to_process:
            templates = self.lookup_table.get(failure, [])
            format_values = _SafeDict(self._build_format_values(psm, failure))
            for template in templates:
                title = template["title"]

                # Skip recommendations for capabilities already deployed.
                if self._is_already_deployed(title, snapshot):
                    continue

                action = CorrectiveAction(
                    action_id=str(uuid.uuid4()),
                    category=template["category"],
                    title=title,
                    description=template["description"],
                    observed_evidence=template["observed_evidence"].format_map(format_values),
                    root_cause=template["root_cause"],
                    expected_improvement=template["expected_improvement"],
                    success_metric=template["success_metric"].format_map(format_values),
                    tradeoff=template["tradeoff"],
                    priority=template["priority"],
                )

                if action.priority == "immediate":
                    immediate.append(action)
                elif action.priority == "short_term":
                    short_term.append(action)
                else:
                    experimental.append(action)

        informational = self._build_efficiency_advisories(psm)

        return CorrectiveActionPlan(
            trace_id=rca.trace_id,
            primary_cause=rca.primary_cause,
            immediate_actions=immediate,
            short_term_actions=short_term,
            experimental_actions=experimental,
            informational_actions=informational,
        )
