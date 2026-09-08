# X-RAG: An Explainable Diagnostic Framework for Retrieval-Augmented Generation Pipelines

**[DRAFT — Research Paper Pre-Submission Document]**

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are increasingly deployed for knowledge-intensive question answering, yet systematically diagnosing *where* and *why* a pipeline fails remains an open problem. Existing evaluation frameworks—RAGAS, RAGChecker, ARES—produce aggregate faithfulness or precision scores but provide no stage-level attribution: a low faithfulness score tells a developer that something is wrong but not whether the fault lies in the corpus, the retriever, the chunking strategy, or the generator. We present **X-RAG** (eXplainable RAG), a diagnostic framework that instruments the full RAG execution pipeline, decomposes generated answers into atomic verifiable claims, verifies each claim against retrieved evidence using Natural Language Inference (NLI), and attributes failures to specific pipeline stages using a structured Pipeline State Matrix (PSM). The PSM feeds a deterministic Root Cause Reasoner that selects a primary failure type from a typed taxonomy (MISSING\_CORPUS, RETRIEVAL\_MISS, CHUNK\_BOUNDARY, UNSUPPORTED\_GENERATION, GROUNDING\_FAILURE), and a Corrective Action Engine maps each diagnosis to actionable engineering recommendations. We evaluate X-RAG against RAGAS, RAGChecker, and ARES on a 40-question labeled dataset derived from three Indian legal codes, analyze inter-framework agreement using Cohen's κ, and report Pearson correlation between X-RAG's claim-level entailment scores and each baseline's faithfulness signal.

> [!WARNING]
> **Critical limitation the paper must address:** As of submission, only 3 of 40 evaluation examples have been run live end-to-end. The remaining 37 examples are unit-tested with fixture data. The paper's quantitative claims must be re-evaluated once the full 40-example run completes. All findings currently presented are preliminary and illustrative.

---

## 1. Introduction

### 1.1 The Problem

Large Language Models (LLMs) produce fluent text even when the underlying facts are wrong. RAG systems were designed to ground generation in retrieved evidence, reducing hallucination by restricting the model to retrieved context. In practice, RAG systems fail in at least five structurally distinct ways:

1. **Missing Corpus** — The answer simply does not exist in the indexed documents.
2. **Retrieval Miss** — The answer exists in the corpus but the retriever fails to surface the relevant chunk.
3. **Chunk Boundary** — The relevant fact is split across two chunks; neither chunk alone is sufficient.
4. **Unsupported Generation** — The retriever returns relevant context but the generator ignores it and produces parametric knowledge instead.
5. **Grounding Failure** — The generator produces claims that directly contradict the retrieved evidence.

Current evaluation frameworks conflate these cases. A faithfulness score of 0.4 could mean any of these failure types. The developer cannot act on such a score without manually inspecting every trace.

### 1.2 Our Contribution

X-RAG provides:
- A **RAGTrace** execution record that freezes every observable pipeline event (retrieved chunk IDs, similarity scores, prompts, generated answers) at inference time.
- A **Claim Decomposition** stage that fractures the generated answer into atomic, independently verifiable facts.
- An **NLI-based Claim Verification** stage that scores each claim against retrieved evidence at sentence granularity.
- A **Pipeline State Matrix** (PSM) that translates verification outcomes into per-stage PASS/FAIL/UNKNOWN verdicts with supporting evidence.
- A **Root Cause Reasoner** that performs causal inference across the PSM to identify the primary failure type.
- A **Corrective Action Engine** that maps diagnoses to parameterized, evidence-backed engineering recommendations.
- A **Baseline Comparison** against RAGAS, RAGChecker, and ARES on a hand-labeled evaluation dataset.

### 1.3 Why This Is Hard

> [!NOTE]
> **Critic's note:** The paper needs to make a stronger argument for *why* claim-level attribution is necessary, not just useful. The introduction currently reads as "RAGAS doesn't tell you which stage failed, we do." A peer reviewer will ask: can a developer not simply inspect the trace manually? The paper needs to argue scale, automation, and reproducibility as first-class requirements — not convenience.

---

## 2. Related Work

### 2.1 RAG Evaluation Frameworks

#### RAGAS (Es et al., 2023)
RAGAS defines four reference-free metrics: **Faithfulness** (fraction of answer claims entailed by retrieved context, verified via LLM), **Answer Relevancy** (reverse-question cosine similarity), **Context Precision** (rank-weighted LLM-judged precision), and **Context Recall** (reference-based). RAGAS produces aggregate scores per query with no stage attribution. Its LLM-as-judge approach makes it expensive per query and non-deterministic across runs.

**Where X-RAG differs:** X-RAG replaces the LLM-as-judge for Faithfulness with an NLI model (deterministic, reproducible, lower cost) and adds stage attribution entirely absent from RAGAS.

**Critical self-evaluation:** X-RAG's own RAGAS-style metric implementations deviate from RAGAS's definitions (e.g., partial-support half-credit for faithfulness, per-chunk vs. concatenated context for recall). These deviations must be explicitly documented in the paper as design choices with justification, or corrected.

#### RAGChecker (Ru et al., 2024)
RAGChecker decomposes both the retrieved context and the generated answer into atomic claims, then measures precision (generated claims supported by context) and recall (context claims covered by the answer) separately for the retrieval and generation stages. It is the closest prior work to X-RAG's claim-level approach.

**Where X-RAG differs:** RAGChecker uses an LLM for claim decomposition and verification (expensive, non-deterministic). X-RAG uses a local NLI model for verification, making it reproducible and free to run repeatedly. X-RAG additionally attributes failures to *five* pipeline stages vs. RAGChecker's two (retriever, generator).

**Critical gap:** RAGChecker has published benchmark results on multiple QA datasets. X-RAG does not yet have equivalent published benchmarks, which weakens the comparison. The paper needs to run the full 40-example dataset and report statistical significance.

#### ARES (Saad-Falcon et al., 2023)
ARES trains lightweight LLM judges on small labeled datasets using in-domain preference data, measuring Context Relevance, Answer Faithfulness, and Answer Relevance. It is designed for domain-specific deployment.

**Where X-RAG differs:** ARES requires labeled training data to fine-tune judges. X-RAG requires no fine-tuning. ARES provides no stage attribution.

#### TruLens (Trulera, 2023)
TruLens instruments LLM pipelines via decorators, measuring groundedness, answer relevance, and context relevance using a mix of LLM-as-judge and embedding similarity. It provides per-component tracing similar to X-RAG's stage attribution.

**Critical gap:** X-RAG does not compare against TruLens. A research paper should. TruLens is the closest existing system in terms of pipeline-level attribution.

#### DeepEval (Confident AI, 2023)
DeepEval provides a testing framework for LLM outputs with metrics including G-Eval, RAGAS-style metrics, and hallucination detection. No stage attribution.

### 2.2 Claim Decomposition and Verification

**FACTOOL (Chern et al., 2023)** decomposes LLM outputs into atomic claims and verifies them against external knowledge bases (Google Search, Wikipedia). X-RAG differs by verifying claims against *retrieved context only* (closed-book verification) rather than open-web search.

**FActScore (Min et al., 2023)** measures atomic factual precision of LLM generations against Wikipedia, defining atomicity similarly to X-RAG's claim decomposition rules.

**AtomicFacts (Guo et al., 2022)** introduces a formal definition of atomic facts that X-RAG's claim decomposition rules (Rules 1–8 in `ClaimDecomposer`) are directly derived from.

### 2.3 Root Cause Analysis in ML Systems

**Slice Finder (Chung et al., 2019)** and **Failing Loudly (Rabanser et al., 2019)** address post-deployment failure attribution in ML systems. X-RAG applies similar attribution philosophy specifically to RAG pipeline stages.

**Causal tracing (Meng et al., 2022)** identifies which MLP layers store factual associations in LLMs. X-RAG does not trace into the LLM internals but identifies the stage that *provided* (or failed to provide) the grounding context.

> [!IMPORTANT]
> **The paper needs a more comprehensive related works section.** Currently missing: (1) Survey of NLI models for text verification (beyond DeBERTa), (2) Chunk boundary detection literature, (3) Multi-hop RAG failure attribution, (4) Hybrid retrieval (BM25+dense) comparison literature. These gaps will be noticed in peer review.

---

## 3. Problem Formulation

### 3.1 Formal Definitions

Let $Q$ be a natural language question, $\mathcal{D}$ be a document corpus, and $\mathcal{C} = \{c_1, c_2, \ldots, c_n\}$ be the set of text chunks derived from $\mathcal{D}$ by a chunking strategy $\chi$ with parameters $(\sigma, \delta)$ (chunk size and overlap).

**Definition 1 (RAG Pipeline).** A RAG pipeline $\Pi$ is a function:
$$\Pi(Q, \mathcal{D}) = \langle R(Q, \mathcal{C}), G(Q, R(Q, \mathcal{C})) \rangle$$
where $R: Q \times \mathcal{C} \to \mathcal{C}_{top-k}$ is the retriever and $G: Q \times \mathcal{C}_{top-k} \to A$ is the generator, producing answer $A$.

**Definition 2 (Atomic Claim).** An atomic claim $\phi_i$ is the smallest semantically complete factual assertion extractable from $A$ that can be independently verified without requiring other claims.

**Definition 3 (Claim Verification).** Given claim $\phi_i$ and evidence set $E = \{s_j\}$ (sentences extracted from $\mathcal{C}_{top-k}$), the NLI verifier assigns:
$$V(\phi_i, E) \in \{\text{SUPPORTED}, \text{PARTIALLY\_SUPPORTED}, \text{CONTRADICTED}, \text{UNSUPPORTED}, \text{NOT\_VERIFIABLE}\}$$

**Definition 4 (Pipeline State Matrix).** The PSM is a function $\Psi: \langle \tau, \Phi, V \rangle \to \{s_1, s_2, s_3, s_4, s_5\}$ mapping a trace $\tau$, claim set $\Phi$, and verification summary $V$ to a vector of stage verdicts $(s_{\text{corpus}}, s_{\text{retriever}}, s_{\text{chunking}}, s_{\text{generator}}, s_{\text{grounding}})$ where each $s_i \in \{\text{PASS}, \text{FAIL}, \text{UNKNOWN}\}$.

**Definition 5 (Root Cause).** The primary root cause $\rho$ is the earliest-failing stage in causal order: $\rho = \arg\min_{s_i \in \text{FAIL}} \text{causal\_order}(s_i)$.

---

## 4. System Architecture

### 4.1 Overview

```
Input: Question Q
         │
  ┌──────▼──────────────────────────────────┐
  │         RAG PIPELINE (Execution)        │
  │  Ingest → Chunk → Embed → Store         │
  │  Query → Dense+Sparse → RRF → Rerank    │
  │  Generate Answer                        │
  └──────────────┬──────────────────────────┘
                 │ RAGTrace (frozen execution record)
  ┌──────────────▼──────────────────────────┐
  │    X-RAG DIAGNOSTIC FRAMEWORK          │
  │                                         │
  │  1. ClaimDecomposer (LLM)               │
  │     Answer → {φ₁, φ₂, ..., φₙ}         │
  │         │                               │
  │  2. ClaimVerifier (NLI)                 │
  │     ∀φᵢ: V(φᵢ, Evidence) → Status      │
  │         │                               │
  │  3. PipelineStateAnalyzer (Rules)       │
  │     VerificationSummary → PSM           │
  │         │                               │
  │  4. RootCauseReasoner (Deterministic)   │
  │     PSM → RCA (primary cause)           │
  │         │                               │
  │  5. CorrectiveActionEngine (Lookup)     │
  │     RCA → CorrectiveActionPlan          │
  │         │                               │
  │  6. RagasEvaluator (Aggregate Metrics)  │
  │     Faithfulness, AnswerRelevancy, etc. │
  └──────────────┬──────────────────────────┘
                 │
          DiagnosticEvaluationReport
         (JSON + PDF + HTML artifacts)
```

### 4.2 RAG Pipeline (Execution Layer)

The underlying RAG pipeline uses:
- **Ingestion:** PyMuPDF for PDF text extraction (1 Document per page)
- **Chunking:** LlamaIndex `SentenceSplitter` (chunk\_size=512, overlap=50)
- **Embedding:** `BAAI/bge-small-en-v1.5` (384-dim, local)
- **Vector Store:** ChromaDB (HNSW cosine similarity, persistent)
- **Retrieval:** Hybrid — ChromaDB dense (top-20) + BM25 sparse (top-20) fused via RRF (k=60), re-ranked by `BAAI/bge-reranker-base` cross-encoder (top-5)
- **Generation:** Groq (llama-3.3-70b-versatile) or HuggingFace Inference API (Qwen2.5-7B-Instruct)

The RAGTrace captures the frozen execution state: question, generated answer, full prompt snapshot, retrieved chunk references (IDs, ranks, all four score components), configuration snapshot, and latency statistics.

### 4.3 Stage 1 — Claim Decomposition

**Module:** [`claim_decomposer.py`](file:///c:/Users/geliv/OneDrive/Desktop/rag_benchmark/src/claim_decomposer.py)

The ClaimDecomposer sends the generated answer to an LLM with an 8-rule atomicity prompt, requesting a JSON array of `{claim_text, sentence_id}` objects. Eight rules enforce:
1. One fact per claim
2. Semantic completeness
3. Independent verifiability
4. No inference
5. No opinions
6. No meaning rewriting
7. No merging of independent facts
8. No invented claims

Character offsets (`character_start`, `character_end`) are computed Python-side via exact substring matching with a fuzzy fallback (sliding window SequenceMatcher, ratio ≥ 0.6), since LLMs hallucinate precise string indices.

A five-step JSON recovery pipeline handles malformed outputs: (1) direct parse, (2) markdown fence stripping, (3) regex array extraction, (4) bracket balancing, (5) truncation recovery (last closed `}` as recovery boundary). A single retry is issued if all five steps fail.

**Critical weakness for paper:** The decomposer uses `.complete()` (base completion) rather than `.chat()` (instruction-tuned interface), degrading instruction-following for JSON output. This must be fixed before results are presented as final.

### 4.4 Stage 2 — Claim Verification

**Module:** [`claim_verifier.py`](file:///c:/Users/geliv/OneDrive/Desktop/rag_benchmark/src/claim_verifier.py)

Each atomic claim $\phi_i$ is verified at sentence granularity. The retrieved chunks are sentence-split, and every (sentence, claim) pair is scored by an NLI model using the `transformers` pipeline:

$$\text{NLI}(\text{premise} = s_j, \text{hypothesis} = \phi_i) \to (p_{\text{ent}}, p_{\text{neu}}, p_{\text{con}})$$

The top-3 highest-entailment sentences are selected, aggregated by one of three strategies (top1, max\_pool\_top3, concat\_top3), and the final status is determined by threshold comparison:

| Condition | Status |
|---|---|
| $p_{\text{con}} \geq 0.7$ | CONTRADICTED |
| $p_{\text{ent}} \geq 0.7$ | SUPPORTED |
| $p_{\text{ent}} \geq 0.4$ | PARTIALLY\_SUPPORTED |
| $p_{\text{neu}} \geq 0.8$ | UNSUPPORTED |
| otherwise | NOT\_VERIFIABLE |

**NLI Model:** `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`

**Critical weakness for paper — must fix before submission:**
1. This model is a 2-label zero-shot model (`entailment`/`not_entailment`). It cannot produce `contradiction_score` — this score is always 0.0. The `CONTRADICTED` status can therefore never be assigned regardless of the evidence. The paper's GROUNDING\_FAILURE detection is currently broken.
2. The model should be replaced with a proper 3-class NLI model: `cross-encoder/nli-deberta-v3-large` which correctly produces entailment, neutral, and contradiction scores.

### 4.5 Stage 3 — Pipeline State Matrix

**Module:** [`pipeline_state_analyzer.py`](file:///c:/Users/geliv/OneDrive/Desktop/rag_benchmark/src/pipeline_state_analyzer.py)

The PSM evaluates five stages using deterministic rules over the verification summary and RAGTrace:

| Stage | PASS Condition | FAIL Condition | FAIL Signal |
|---|---|---|---|
| **CORPUS** | ≥1 supported claim | Min dense distance > 0.75 | Corpus doesn't contain this topic |
| **RETRIEVER** | ≥1 supported claim | All unsupported AND max\_score < 0.5 | Retriever returned irrelevant chunks |
| **CHUNKING** | No adjacent partial evidence | ≥1 partial claim with adjacent retrieved chunk | Fact split across chunk boundary |
| **GENERATOR** | All claims supported | ≥1 unsupported AND max\_score ≥ 0.5 | Generator hallucinated despite good context |
| **GROUNDING** | No contradicted claims | ≥1 contradicted claim | Generator produced actively wrong facts |

Each stage stores: status, observation text (strictly factual, no causal interpretation), confidence score, supporting claim/chunk/verification IDs, and raw metadata for downstream consumption.

**Critical weakness for paper:**
- CHUNKING stage returns UNKNOWN with confidence=1.0 in the non-failure case — semantically contradictory. The chunking stage is effectively blind unless the specific adjacency heuristic fires.
- Confidence scores (0.85, 0.95, etc.) are hardcoded constants, not computed from evidence. A paper reviewer will immediately ask how these were determined.

### 4.6 Stage 4 — Root Cause Reasoner

**Module:** [`root_cause_reasoner.py`](file:///c:/Users/geliv/OneDrive/Desktop/rag_benchmark/src/root_cause_reasoner.py)

The RootCauseReasoner traverses the PSM in causal order (CORPUS → RETRIEVER → CHUNKING → GENERATOR → GROUNDING), collects all FAIL stages, and selects the **highest-confidence failure** as the primary cause, with causal order as tiebreaker. Secondary effects are all remaining failures.

Causal order is defined by domain knowledge: an upstream failure (e.g., RETRIEVAL\_MISS) causally propagates to downstream failures (e.g., UNSUPPORTED\_GENERATION) because the generator is forced to hallucinate when the retriever provides irrelevant context.

**FailureType taxonomy:**
- `MISSING_CORPUS` — answer not in indexed documents
- `RETRIEVAL_MISS` — answer in corpus but not retrieved
- `CHUNK_BOUNDARY` — relevant fact split across chunks
- `UNSUPPORTED_GENERATION` — generator produced parametric knowledge
- `GROUNDING_FAILURE` — generator contradicted evidence
- `MULTI_HOP_REASONING_FAILURE` — [defined but not yet implemented]

**Critical weakness for paper:** The highest-confidence selection can pick a downstream symptom (e.g., GROUNDING) as the primary cause when an upstream cause (e.g., RETRIEVAL\_MISS) exists. This is causally wrong. The paper must either: (a) implement true propagation-aware causal selection (pick the earliest FAIL in causal order when multiple exist), or (b) explicitly present the confidence-based selection as a design choice and justify it.

### 4.7 Stage 5 — Corrective Action Engine

**Module:** [`corrective_action_engine.py`](file:///c:/Users/geliv/OneDrive/Desktop/rag_benchmark/src/corrective_action_engine.py)

A static lookup table maps each FailureType to a prioritized set of CorrectiveActions (immediate/short\_term/experimental). Each action includes: title, description, observed\_evidence (parameterized with real trace values), root\_cause, expected\_improvement, success\_metric, and tradeoff.

Actions are not generated by an LLM — this is intentional: RAG failures have well-known engineering solutions, and deterministic mapping guarantees consistent, reproducible recommendations.

A separate informational tier surfaces chunk utilization advisories (fraction of retrieved chunks that contributed to verified claims) independently of the failure-driven path.

**Critical weakness for paper:** The lookup table does not check the current system configuration. It can recommend "Implement Hybrid Search" to a system that already uses hybrid search. Action filtering based on `configuration_snapshot` is needed.

### 4.8 Evaluation Metrics (RAGAS-style)

**Module:** [`ragas_metrics.py`](file:///c:/Users/geliv/OneDrive/Desktop/rag_benchmark/src/ragas_metrics.py)

X-RAG computes seven metrics, grouped by reference requirement:

**Reference-free (computed every run):**
- **Faithfulness** — weighted claim support rate (SUPPORTED + 0.5 × PARTIALLY\_SUPPORTED) / total claims. Reuses existing NLI verification; zero additional cost.
- **Answer Relevancy** — generates N synthetic reverse questions via LLM, computes cosine similarity of embeddings to original question.
- **Context Precision** — rank-weighted average precision; LLM judges each chunk's usefulness to the answer.
- **Context Relevancy** — fraction of chunks judged relevant to the question by LLM.

**Reference-based (only when gold answer provided):**
- **Context Recall** — fraction of reference answer sentences entailed by retrieved context (via NLI).
- **Answer Similarity** — cosine similarity of answer and reference embeddings.
- **Answer Correctness** — claim-level F1 between answer and reference, blended with similarity via configurable weights.

**Critical deviation from RAGAS standard:** RAGAS faithfulness uses strict binary support (no partial credit). X-RAG's half-credit scheme for PARTIALLY\_SUPPORTED must be disclosed as a deviation, not described as "RAGAS faithfulness."

---

## 5. Evaluation Dataset

### 5.1 Corpus

Three Indian legal codes: **Bharatiya Nyaya Sanhita (BNS)**, **Bharatiya Nagarik Suraksha Sanhita (BNSS)**, and **Bharatiya Sakshya Adhiniyam (BSA)**. The corpus presents challenging characteristics:
- Legal text with bare section numbers as references (e.g., `399.(1)`)
- Cross-references between sections
- Nested conditional clauses
- Dense numerical content (fines, imprisonment terms, section numbers)

### 5.2 Labeled Evaluation Dataset

**File:** `eval/eval_dataset.csv` — 40 hand-authored question/gold-answer pairs, each verified against extracted PDF text.

| Failure Category | Count | Description |
|---|---|---|
| Healthy | 25 | Correct answers expected |
| MISSING\_CORPUS | 4 | Topics genuinely outside all three codes (crypto regulation, corporate tax, trademark) |
| RETRIEVAL\_MISS | 3 | Paraphrased away from statute wording |
| CHUNK\_BOUNDARY | 3 | Short provisions split at chunk boundaries |
| UNSUPPORTED\_GENERATION | 2 | Asking for specific numbers not stated in source |
| GROUNDING\_FAILURE | 3 | False-premise questions contradicting retrieved text |
| **Total** | **40** | |

### 5.3 Diagnostic Accuracy Measurement

For each example in the dataset with a labeled failure category, X-RAG's diagnosed `primary_cause` is compared against the ground-truth label. We report:
- **Detection Rate** — fraction of failure examples correctly flagged as non-UNKNOWN
- **Attribution Accuracy** — fraction of failure examples where `primary_cause` matches ground truth label
- **False Positive Rate** — fraction of healthy examples flagged as failures

> [!CAUTION]
> **Major paper weakness:** Only 3 of 40 examples have been run live at time of writing. The remaining 37 are validated against unit test fixtures, not real model outputs. **The paper cannot present final results until the full dataset is run.** Every quantitative claim in this document must be treated as preliminary. This is the single most important thing to fix before submission.

---

## 6. Baseline Comparison Methodology

### 6.1 Baselines

| Framework | Metrics | Judge | Stage Attribution |
|---|---|---|---|
| **X-RAG** | 7 metrics + PSM + RCA | NLI + LLM | ✅ CORPUS/RETRIEVER/CHUNKING/GENERATOR/GROUNDING |
| **RAGAS 0.2.15** | 4–7 metrics | LLM | ❌ |
| **RAGChecker** | Precision, Recall, 10 sub-metrics | LLM | ⚠️ Retriever vs. Generator only |
| **ARES** | Context Relevance, Faithfulness, Answer Relevance | Fine-tuned LLM | ❌ |

### 6.2 Infrastructure

All baselines share the same retrieved chunks from X-RAG's hybrid retrieval pipeline, ensuring a fair comparison on identical context. RAGAS runs in-process in the main venv. RAGChecker and ARES run in isolated Python 3.10 venvs and communicate via JSON file handoff to avoid dependency conflicts.

LLM: Groq free tier (llama-3.3-70b-versatile for generation and RAGAS; llama-3.1-8b-instant for RAGChecker; openai/gpt-oss-20b for ARES, required by ARES's internal model routing that checks `"gpt" in model_choice`).

### 6.3 Agreement Analysis

**Script:** `scripts/analyze_agreement.py`

- **Pearson correlation** between X-RAG's `average_entailment_score` and each baseline's faithfulness-style metric.
- **Cohen's κ** between X-RAG's binary failure verdict (`primary_cause != UNKNOWN`) and each baseline's threshold-based failure flag.
- **Disagreement analysis** — `disagreements.csv` lists every example where X-RAG and a baseline disagree, with X-RAG's full reasoning chain exported for qualitative analysis.

**The key differentiator claim:** When X-RAG and RAGAS disagree, X-RAG provides a specific stage attribution explaining *why* scores differ. This localization is the paper's primary novelty claim.

---

## 7. Results (Preliminary — 3 of 40 Examples)

> [!WARNING]
> The following results are based on 3 live examples only. They are illustrative, not statistically significant. All numbers must be updated before submission.

### 7.1 Qualitative Case Study

**Example:** Query about Section 103 of BNS (murder penalties).

| Stage | Status | Observation |
|---|---|---|
| CORPUS | PASS | Dense distance 0.23 < 0.75 threshold |
| RETRIEVER | PASS | 2 of 5 chunks contributed supported claims |
| CHUNKING | UNKNOWN | No adjacent partial evidence detected |
| GENERATOR | FAIL | 2 of 7 claims unsupported despite high retrieval scores |
| GROUNDING | ❌ NOT FUNCTIONAL | DeBERTa model has 2 labels; CONTRADICTED never produced |

**Root cause:** UNSUPPORTED\_GENERATION

**Corrective actions generated:** Lower sampling temperature, Strict prompt grounding

### 7.2 Known Infrastructure Bugs Affecting Results

1. **CONTRADICTED never produced** — DeBERTa-v3-zeroshot has 2 output labels. All 3 GROUNDING\_FAILURE probes in the dataset will be misclassified.
2. **NLI pipeline type mismatch** — Zero-shot model used as text-classification model. Entailment scores may be poorly calibrated.
3. **Context recall truncation** — Concatenated chunks exceed 512-token NLI limit. Recall scores for multi-chunk queries are unreliable.

These bugs **must be fixed before final results are collected**.

---

## 8. Critical Analysis: What Makes This a Better Paper

### 8.1 What Is Genuinely Novel

The paper has real novelty if it can demonstrate:

1. **Fine-grained stage attribution beyond retriever/generator split.** RAGChecker splits at retriever vs. generator. X-RAG adds CORPUS, CHUNKING, and GROUNDING as separate stages. The CORPUS stage in particular (using pre-reranking candidate pool distance) is novel.

2. **Deterministic, reproducible diagnostics at zero marginal cost.** RAGAS/RAGChecker use LLM-as-judge for verification, making results non-reproducible. X-RAG's NLI-based verification is deterministic (temperature=0) and runs locally, enabling per-query diagnostics at scale without API costs.

3. **Causal action recommendations with live trace parameterization.** The CorrectiveActionEngine injects real observed values (actual distances, actual counts) into recommendation templates, producing evidence-backed advice rather than generic best practices.

4. **The labeled failure injection dataset.** A 40-example dataset with explicitly labeled failure categories (MISSING\_CORPUS, RETRIEVAL\_MISS, CHUNK\_BOUNDARY, UNSUPPORTED\_GENERATION, GROUNDING\_FAILURE) for evaluating diagnostic accuracy is itself a contribution, especially for legal domain RAG.

### 8.2 What Must Be Fixed Before Submission

**Non-negotiable fixes:**

| Fix | Why Critical | Effort |
|---|---|---|
| Replace DeBERTa with `cross-encoder/nli-deberta-v3-large` (3-class NLI) | CONTRADICTED never fires; GROUNDING stage broken | Low — 2 config lines |
| Use `.chat()` in ClaimDecomposer | Instruct model hallucination in JSON output | Low — same as Generator |
| Fix context recall: score per-chunk, not concatenated | Silent truncation produces wrong recall | Medium |
| Run full 40 examples | No statistical results possible without this | High — compute time |
| Threshold calibration experiment | Reviewer will ask: how were 0.7/0.4/0.8 chosen? | Medium |
| Fix root cause selection to earliest-fail causal order | Current selection picks downstream symptom | Low |

**Important improvements:**

| Improvement | Why Important | Effort |
|---|---|---|
| Remove dead ClaimSet conversion pipeline | Dead code weakens paper's architecture description | Low |
| Populate ClaimType classifier | Claimed in architecture but never implemented | Medium |
| Composite health score (not just grounding) | Single-metric health score is oversimplified | Medium |
| Add TruLens to baseline comparison | It's the closest prior work — must compare | Medium |
| Formal false-positive analysis on 25 healthy examples | Current results only discuss failures | Low |
| Compute 95% confidence intervals on agreement metrics | Required for statistical validity | Low |

### 8.3 Structural Paper Weaknesses

**No ablation study.** The paper claims five diagnostic stages. Which stages actually contribute? An ablation removing each stage (e.g., what accuracy is achievable with only RETRIEVER+GENERATOR, no CORPUS/CHUNKING/GROUNDING?) would strengthen the contribution claim substantially.

**No error analysis.** Where does X-RAG's diagnosis fail? What are the false positive and false negative patterns? Without this, the paper reads as a system description, not a scientific evaluation.

**Domain specificity.** The evaluation corpus is exclusively Indian legal text. Reviewer will ask: does the framework generalize? A second domain (e.g., medical QA, Wikipedia) with 20–30 questions would address this concern.

**Claim decomposer is a black box.** The paper cannot claim "atomic claim decomposition" without measuring decomposition quality. How many claims does an 8-sentence answer decompose into? What is the LLM's consistency across runs (same answer → same claims)? These must be measured.

**The NLI verification is not validated.** The threshold values (0.7 entailment, 0.4 partial, 0.7 contradiction) are never empirically calibrated against a labeled (claim, evidence, status) dataset. This is the second most important thing to fix — a reviewer who runs the code and sees arbitrary magic numbers will reject.

---

## 9. Future Work

### 9.1 Near-Term (Required for Paper Completeness)

1. **NLI Model Replacement** — Replace `deberta-v3-large-zeroshot-v2.0` with `cross-encoder/nli-deberta-v3-large`, enabling proper 3-class NLI and unblocking GROUNDING\_FAILURE detection.
2. **Threshold Calibration Study** — Create a labeled (claim, sentence, status) dataset of ~100 examples and optimize thresholds via precision-recall curve analysis.
3. **Full 40-Example Evaluation** — Run complete baseline comparison. Currently blocked by compute time (10–20 minutes/example for CPU-only NLI).
4. **Causal Root Cause Selection** — Fix the primary cause selection to use earliest-in-causal-order FAIL when multiple FAIL stages exist.

### 9.2 Medium-Term (Strengthens Contribution)

5. **GPU-accelerated NLI batching** — Batch all (premise, hypothesis) pairs for a trace and run them in a single forward pass. Estimated 5–10x speedup from current sequential CPU inference.
6. **Claim Type Classification** — Automatically classify claims into ENTITY/ATTRIBUTE/RELATIONSHIP/NUMERICAL/TEMPORAL types. Different claim types may require different verification strategies.
7. **Semantic Chunking** — Replace `SentenceSplitter` with `SemanticSplitterNodeParser` to reduce CHUNK\_BOUNDARY failure rates.
8. **Multi-Domain Evaluation** — Add a second domain (medical or Wikipedia-based QA) to validate generalization.
9. **HyDE Query Expansion** — Add Hypothetical Document Embeddings to improve retrieval recall for vague queries.

### 9.3 Long-Term (Research Directions)

10. **LLM-based Root Cause Reasoning** — Replace the deterministic reasoner with an LLM that can perform nuanced causal inference, handling edge cases the rule-based system cannot.
11. **Adaptive Corrective Actions** — Train a recommendation model on historical (failure type, configuration, recommendation, outcome) tuples to generate context-aware, configuration-aware suggestions.
12. **Online Diagnostic Mode** — Integrate X-RAG as a real-time middleware layer that diagnoses and re-queries before returning an answer to the user (self-healing RAG).
13. **Multi-turn Conversation Tracing** — Extend RAGTrace to capture conversation history and diagnose failures that emerge from context drift across turns.

---

## 10. Conclusion

X-RAG addresses the stage attribution gap in RAG evaluation by introducing a five-stage diagnostic pipeline that localizes failures to specific components (corpus, retriever, chunking strategy, generator, or grounding quality). Unlike aggregate metrics that report *how much* a pipeline fails, X-RAG identifies *where* it fails and *why*, generating parameterized, evidence-backed corrective actions.

The framework is reproducible (deterministic NLI verification at zero marginal cost), explainable (full artifact chain from retrieved chunks to verified claims to root cause), and extensible (abstract VectorStore interface, configurable aggregation strategies, pluggable thresholds).

**The critical path to a publishable paper is:** (1) fix the NLI model, (2) run the full 40-example dataset, (3) calibrate thresholds empirically, (4) add TruLens comparison, (5) run an ablation study, (6) add a second evaluation domain.

---

## Appendix A: Module Inventory

| Module | Responsibility | Lines | Status |
|---|---|---|---|
| `ingestion.py` | PDF loading | 61 | ✅ Functional |
| `chunk_engine.py` | Text splitting | 81 | ✅ Functional |
| `chunk_registry.py` | Chunk provenance tracking | 173 | ✅ Functional |
| `embedding_engine.py` | Vector generation | 80 | ⚠️ Sequential (no batching) |
| `vector_store.py` | ChromaDB abstraction | 207 | ✅ Functional |
| `retriever.py` | Hybrid search + reranking | 245 | ✅ Functional |
| `generator.py` | LLM answer synthesis | 152 | ✅ Functional |
| `rag_trace.py` | Execution record | 174 | ✅ Functional |
| `claim_decomposer.py` | Answer → atomic claims | 387 | ⚠️ Uses `.complete()` not `.chat()` |
| `claims.py` | Claim data model | 148 | ⚠️ ClaimType never populated |
| `claim_verifier.py` | NLI-based verification | 460 | 🔴 Wrong NLI model type |
| `pipeline_state_analyzer.py` | Stage verdict computation | 287 | ⚠️ CHUNKING rarely activates |
| `root_cause_reasoner.py` | Causal attribution | 139 | ⚠️ Confidence-not-causal selection |
| `corrective_action_engine.py` | Recommendation generation | 306 | ⚠️ Config-unaware |
| `ragas_metrics.py` | Aggregate metrics | 263 | 🔴 Context recall truncated |
| `answer_correctness_evaluator.py` | Gold-claim recall | 142 | ✅ Functional |
| `runner.py` | Pipeline orchestration | 86 | ✅ Functional |
| `report_builder.py` | Report assembly | 218 | ⚠️ Health = single metric |
| `report.py` | Report data model | 167 | ✅ Functional |

**Legend:** ✅ Functional | ⚠️ Has known limitations | 🔴 Has critical bugs

---

## Appendix B: Dataset Statistics

| Metric | Value |
|---|---|
| Corpus PDFs | 3 (BNS, BNSS, BSA) |
| Total document pages | TBD |
| Total chunks (512/50) | TBD |
| Evaluation questions | 40 |
| Healthy examples | 25 (62.5%) |
| Failure examples | 15 (37.5%) |
| Labeled failure categories | 5 |
| Live runs completed | 3 |
| Remaining runs | 37 |

---

## Appendix C: Implementation Decisions Log

| Decision | Rationale | Alternative Considered |
|---|---|---|
| NLI over LLM-as-judge for verification | Deterministic, reproducible, zero marginal cost | LLM judge (used by RAGAS) |
| Static CAE lookup table over LLM recommendations | No hallucinated advice; guaranteed engineering correctness | LLM-generated recommendations |
| Deterministic PSM rules over LLM | Reproducibility; known calibration properties | LLM judge for stage verdicts |
| RRF (k=60) for hybrid search fusion | Score-agnostic; no normalization needed | Learned fusion weights |
| Per-page Document objects from PDF | LlamaIndex default; preserves page metadata | Full-document loading |
| ChromaDB for vector store | Local, persistent, free; abstract interface allows swap | Pinecone, Qdrant, Weaviate |
| Groq free tier as LLM backend | Zero cost; enables complete live runs | OpenAI API, Bedrock |
