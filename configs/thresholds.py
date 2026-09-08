# NLI Thresholds
ENTAILMENT_THRESHOLD = 0.7
CONTRADICTION_THRESHOLD = 0.7
PARTIAL_SUPPORT_THRESHOLD = 0.4
# Neutral score above which a claim with low entailment/contradiction is
# classified UNSUPPORTED rather than NOT_VERIFIABLE.
NEUTRAL_UNSUPPORTED_THRESHOLD = 0.8

# Similarity score below which retrieval is considered to have failed
# (used by PipelineStateAnalyzer's RETRIEVER/GENERATOR stages).
RETRIEVAL_SCORE_THRESHOLD = 0.5

# CORPUS diagnostic stage: cosine distance (lower = more similar) above which
# even the closest pre-rerank candidate is considered irrelevant to the query.
CORPUS_MAX_RELEVANT_DISTANCE = 0.75

# Strategy for combining the top-3 scored evidence sentences into a single
# verification decision. "top1" preserves the original single-sentence behavior.
# "max_pool_top3" takes the max of each score type across the top-3 sentences.
# "concat_top3" re-runs NLI once against the top-3 sentences concatenated.
EVIDENCE_AGGREGATION_STRATEGY = "top1"

# Below this chunk_utilization_rate (fraction of retrieved chunks that ended up
# as supporting evidence for any verified claim), PipelineStateAnalyzer's
# RETRIEVER metadata triggers an informational-priority corrective action
# advisory. Purely advisory -- does not affect RETRIEVER.status/.confidence
# or RootCauseReasoner's primary_cause selection.
LOW_RETRIEVAL_EFFICIENCY_THRESHOLD = 0.4
