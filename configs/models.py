# LLM and Model Configurations

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

# Upgraded from bge-small-en-v1.5 (384-dim, MTEB 62.2) to bge-base-en-v1.5
# (768-dim, MTEB 64.2) for a meaningful recall improvement at reasonable cost.
# NOTE: changing this invalidates the existing ChromaDB collection and the
# embedding cache — delete db/chroma and artifacts/chunk_registry.json, then
# re-run run_pipeline.py to rebuild.
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Number of chunks to embed in a single forward pass. 64 is a safe default
# that saturates CPU batch capacity without OOM risk on typical machines.
# Increase to 128-256 if you have a GPU.
EMBEDDING_BATCH_SIZE = 64

# Directory for the persistent embedding cache. Keyed by MD5(model_name+text)
# so it auto-invalidates when the model changes. Relative to project root.
EMBEDDING_CACHE_DIR = ".cache/embeddings"

# BGE models are trained with query/document instruction pairs. Prepending
# this prefix to *queries only* (not document chunks) improves retrieval
# precision. See: https://huggingface.co/BAAI/bge-base-en-v1.5
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching: "

# Reranker model
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# ---------------------------------------------------------------------------
# LLM — NVIDIA NIM (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
# All generation, claim decomposition, and baseline judge calls route through
# the NVIDIA free-tier inference endpoint.
#
# 2026-09-03: meta/llama-3.1-70b-instruct (this project's original model) and
# mistralai/mixtral-8x7b-instruct-v0.1 (the decomposer/planner/RAGChecker
# model) were both retired from NVIDIA's catalog (410 Gone). A follow-up
# attempt to move to nvidia/llama-3.1-nemotron-70b-instruct also fails -- not
# retired, but not entitled for this account's API key (404 "Function ...
# Not found for account"). Verified directly against
# https://integrate.api.nvidia.com/v1/chat/completions with this repo's
# NVIDIA_API_KEY: of ~80 catalog models tried, only
# nvidia/nemotron-3-super-120b-a12b returned a real completion, so every
# role below points at it for now. This is a stopgap, not a quality
# decision -- it means the "8B for high-throughput decomposition" split
# this file used to describe no longer holds; the small/fast roles
# (decomposer, planner, RAGChecker) are paying full 120B-class latency
# until a working small model is found for this account. Re-run the same
# probe against /v1/models if NVIDIA_API_KEY changes or 404s return.

LLM_PROVIDER = "nvidia"

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Generation: highest-quality model for the answer synthesis step.
NVIDIA_GENERATION_MODEL       = "nvidia/nemotron-3-super-120b-a12b"

# Claim decomposition: structured JSON output, high volume -- ideally a
# small model, but see the note above for why this is 120B for now.
NVIDIA_CLAIM_DECOMPOSER_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Retrieval control loops (IRCoT planner, agentic controller). These calls emit
# a short JSON object choosing the next query or action -- a routing decision,
# not answer synthesis. Was 8B (see note above); until a working small model
# is found for this account, multi-hop ablations will run at full-model
# planner latency rather than the ~49s/example the 8B split achieved.
NVIDIA_PLANNER_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Baseline judges — same split logic: RAGAS/ARES need strong instruction
# following; RAGChecker processes many short calls so 8B is sufficient
# (was, before the retirement/entitlement issue above).
NVIDIA_RAGAS_JUDGE_MODEL      = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_RAGCHECKER_JUDGE_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NVIDIA_ARES_JUDGE_MODEL       = "nvidia/nemotron-3-super-120b-a12b"

# Convenience alias used by code that only needs one model name (e.g. legacy
# references to LLM_MODEL_NAME before the NVIDIA migration).
LLM_MODEL_NAME = NVIDIA_GENERATION_MODEL

# Generation parameters
# temperature=0.0 is required for reproducibility: non-zero sampling means
# running the same trace twice yields different decompositions and different
# RAGAS scores, making results non-reproducible for a research benchmark.
LLM_TEMPERATURE = 0.0
# 1024 was too tight for genuinely multi-part questions (e.g. "what are the
# issues with X, how likely is Y, and are there justifications for Z") --
# citations alone can eat a third of the budget, and the model was hitting
# the cap mid-answer, mid-citation, with no indication to the user that it
# had been cut off rather than finished. See generate_stream()'s
# finish_reason=="length" check for the other half of that fix.
LLM_MAX_TOKENS  = 2048

# Per-request timeout in seconds. The free NVIDIA tier is shared capacity and
# occasionally stalls; without an explicit bound a single request can hang for
# minutes and reads to the user as a frozen UI. 90s is comfortably above a
# normal 70B completion (~40s) while still failing fast when the endpoint is
# not going to answer.
LLM_REQUEST_TIMEOUT = 90.0

# Retries for transient failures (429 rate limits, 5xx). Applied by the client
# with exponential backoff, so keep this small — the timeout above is per
# attempt, and the worst case is roughly timeout x (retries + 1).
LLM_MAX_RETRIES = 2

# NVIDIA's build.nvidia.com free tier caps requests at 40/minute per model,
# account-wide (confirmed via NVIDIA developer forums/docs, 2026). Every call
# in this project — generation, claim decomposition, and the claim verifier's
# LLM-judge escalation — shares one account and one model
# (nvidia/nemotron-3-super-120b-a12b), so they all share this one budget.
# Set below the documented cap, not at it: the limit is described by NVIDIA as
# varying with "model, use-case, and current overall traffic," and a proactive
# throttle that never hits 429 is more useful than one that races the cap.
NVIDIA_RPM_LIMIT = 32

# ---------------------------------------------------------------------------
# Verification (local NLI model — no API)
# ---------------------------------------------------------------------------
# cross-encoder/nli-deberta-v3-large is a proper NLI text-classification model
# (trained on SNLI+MultiNLI+FEVER) that natively accepts text/text_pair format
# with "entailment"/"neutral"/"contradiction" labels — exactly how claim_verifier.py
# calls it. Using a zero-shot-classification model (e.g. deberta-v3-large-zeroshot)
# with text-classification pipeline produces miscalibrated scores.
VERIFICATION_MODEL      = "cross-encoder/nli-deberta-v3-large"

# Batch size when the NLI model runs on CPU. Kept small: large CPU batches add
# latency to the first result without improving throughput.
VERIFICATION_BATCH_SIZE = 8

# Batch size when the NLI model runs on GPU. Verification scores every sentence
# of every retrieved chunk against every claim — hundreds of short pairs — so
# the card is only saturated at a much larger batch. 32 fits comfortably in 8 GB
# alongside the retrieval models at fp16; lower it if you hit CUDA OOM.
VERIFICATION_GPU_BATCH_SIZE = 32
