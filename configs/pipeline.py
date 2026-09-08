# Pipeline configurations

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Strategy to use when splitting documents into chunks.
# Options:
#   "sentence"      - SentenceSplitter (default, sentence-boundary aware, fast)
#   "semantic"      - SemanticSplitterNodeParser (breakpoint-based, higher quality, slow)
#   "hierarchical"  - HierarchicalNodeParser (multi-granularity, best for complex docs)
# Changing strategy or chunk size requires a full re-ingest (delete db/chroma).
CHUNKING_STRATEGY = "sentence"

CHUNK_SIZE = 512

# Increased from 50 → 100 tokens (~20%) for better semantic continuity at chunk
# boundaries — the previous 10% overlap was too thin for legal text with long sentences.
CHUNK_OVERLAP = 100

# Vector Store
CHROMA_PERSIST_DIR = "./db/chroma"
CHROMA_COLLECTION_NAME = "rag_benchmark_collection"

# Retrieval
RETRIEVAL_TOP_K = 5

# How many candidates each retrieval arm (dense, BM25) contributes to RRF.
# Wider than the old value of 20 so the fusion pool is deep enough that the
# cross-encoder has genuinely different chunks to choose between. Fusion itself
# is cheap — only the reranker below costs real time.
FUSION_CANDIDATE_POOL = 40

# How many RRF-fused candidates are handed to the cross-encoder reranker.
# This MUST be substantially larger than RERANKER_TOP_N — otherwise the
# reranker only reorders chunks RRF already selected and cannot rescue a
# relevant chunk that RRF ranked 12th.
#
# LATENCY: bge-reranker-base on CPU costs roughly 1-1.5s per candidate at this
# corpus's chunk length, and the cost is linear. 20 lands around 20-30s on a
# laptop CPU. Raise it for better recall if you have a GPU; lower it to ~10 if
# you need snappier turns.
RERANK_INPUT_SIZE = 20

# Truncation length for cross-encoder input. Chunks average ~450 tokens, so at
# 512 nearly every pair pays full price. 384 keeps the passage lead (where the
# marginal heading and operative text live) at a meaningful speed saving.
RERANKER_MAX_LENGTH = 384

# Final number of chunks passed to the generator as context.
RERANKER_TOP_N = 6

# Claim Decomposer
CLAIM_DECOMPOSER_PROMPT_VERSION = "1.0"
CLAIM_DECOMPOSER_MAX_TOKENS = 4096

# Claim Verifier's LLM-judge escalation (see claim_verifier.py) -- only ever
# asked for a two-line verdict, so a small budget is plenty and keeps the
# rare escalation call cheap.
CLAIM_JUDGE_MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# Retrieval arm selection (ablation)
# ---------------------------------------------------------------------------
# Which retrieval arms run. "hybrid" is the shipped pipeline; the single-arm
# modes exist so the ablation can attribute a result to fusion rather than
# assert it. RRF over one arm reduces to that arm's own ranking, so all three
# share the same fusion code path.
#   "vector" - dense bi-encoder only
#   "bm25"   - lexical only
#   "hybrid" - both, fused with RRF
RETRIEVAL_MODE = "hybrid"
VALID_RETRIEVAL_MODES = ("vector", "bm25", "hybrid")

# Cross-encoder reranking on/off. Off makes the fusion pool the final context,
# which is arm B of the ablation.
RERANK_ENABLED = True
