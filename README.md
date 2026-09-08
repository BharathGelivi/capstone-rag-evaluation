# X-RAG Diagnostic Framework

## Run the React UI

Two servers, both from the repo root. Requires the Python env set up (see
[Quick Start](#quick-start) below) and Node.js/npm installed.

```bash
# Terminal 1 -- backend (SSE chat + graph API) on http://127.0.0.1:8010
python run_api_ui.py

# Terminal 2 -- frontend on http://localhost:5173 (first run: cd frontend && npm install)
cd frontend
npm run dev
```

Open <http://localhost:5173>. **Chat** tab: pick a retrieval arm (`D_ircot`
for IRCoT, `F_graphrag`/`G_ircot_graph`/`H_agentic_graph` for GraphRAG) and
ask a question against the statutes or judgments corpus. **Graph** tab:
toggle between the RAG provenance graph and the interactive claude-mem memory
graph.

Run the test suites from the repo root:

```bash
python -m pytest tests/test_api_ui.py -v      # fast, no browser
python -m pytest tests/test_ui_e2e.py -v      # real Playwright e2e (needs both servers running)
```

---

## Project Overview

The X-RAG Diagnostic Framework is a comprehensive evaluation tool designed to diagnose, trace, and recommend fixes for Retrieval-Augmented Generation (RAG) pipelines. It explicitly isolates execution from reasoning, allowing deterministic analysis of where exactly a pipeline failed (e.g., retrieval miss vs. hallucination vs. unsupported claim).

---

## Quick Start

Get the chat UI running from a fresh clone. Windows paths shown; on
macOS/Linux use `venv/bin/` instead of `venv\Scripts\`.

### 1. Create the environment

```bash
py -3.13 -m venv venv
venv\Scripts\pip install -r requirements.txt
```

### 2. Add your API key

Copy `.env.example` to `.env` and set the NVIDIA key — this is the only key the
core pipeline needs:

```ini
NVIDIA_API_KEY=nvapi-...
```

Get a free one at <https://build.nvidia.com/> (no credit card). All generation,
claim decomposition, and judging route through the NVIDIA NIM
OpenAI-compatible endpoint — see `configs/models.py`.

### 3. Enable your GPU (strongly recommended)

`pip install -r requirements.txt` installs a **CPU-only** build of torch. It
works, but claim verification costs 30–115 s *per claim* instead of ~3 s. Check
what you have:

```bash
venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If it prints `+cpu` or `False` and you have an NVIDIA card, install the CUDA
wheel matching your torch version:

```bash
venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"
```

Confirm the pipeline sees it:

```bash
venv\Scripts\python -c "from src.device import describe_device; print(describe_device())"
```

Full details, VRAM budget, and measured numbers: **`docs/gpu_setup.md`**.

### 4. Ingest your documents

Put PDFs in `data/`, then build the chunk registry and vector store. Re-run this
whenever the source documents change:

```bash
venv\Scripts\python run_pipeline.py
```

This writes `artifacts/chunk_registry.json` and `db/chroma/`. It downloads the
embedding and reranker models on first run.

### 5. Launch the chat UI

```bash
venv\Scripts\python -m streamlit run ui/app.py
```

Opens at <http://localhost:8501>.

---

## Using the App

A ChatGPT-style interface over your corpus, with the diagnostic pipeline
attached to every answer.

**Sidebar**

| Control | What it does |
|---|---|
| **➕ New Session** | Start a fresh conversation. Sessions persist across restarts. |
| **🧠 Memory** | Semantic recall of past turns, including from other sessions. |
| **⚡ Stream Answers** | Render tokens as they arrive. Same total time; first token in ~1s instead of a 20–70s blank wait. |
| **🔬 Deep Analysis** | Extract atomic claims and verify each against retrieved chunks with an NLI model. Defaults **on** with a GPU, **off** on CPU (where it adds minutes per turn). |
| **🐛 Debug Mode** | Show fusion scores, the raw trace, and the exact prompt sent to the model. |
| **📊 Details Panel** | Toggle the right-hand inspector. |
| **Device** | Shows whether models are on GPU or CPU — a CPU-only torch install is visible here. |

**Details panel tabs** — Chunks (what was retrieved, with dense/sparse/RRF/reranker
scores), Memory (which past turns were recalled and why), Claims (per-claim
verification verdicts), Trace (stage-by-stage timings), Search (query your
memory store).

**Pages**

- **🧠 Memory** — browse and manage the long-term memory store.
- **🐛 Debug** — pipeline internals.
- **🕸️ Knowledge Graph** — interactive provenance graph linking documents →
  chunks → questions → claims. Drag nodes, scroll to zoom, click for detail,
  click a legend row to filter, search to highlight.

### Follow-up questions

Follow-ups work: prior turns are passed as real conversation history, and a
condensation step rewrites references into standalone retrieval queries — so
*"can you elaborate on that?"* retrieves against the resolved topic, not the
literal words. A `↻ follow-up resolved` badge appears under answers where this
fired.

---

## Architecture Diagram

```
Client
      ↓
FastAPI (API Layer)
      ↓
X-RAG Diagnostic Framework
      ↓
Artifacts
      ↓
Reports
```

## Repository Layout

- `src/`: Core framework and API source code.
- `ui/`: Streamlit chat application (`app.py`), extra pages, components, styles.
- `tests/`: Unit test suite.
- `configs/`: Modular configuration properties.
- `artifacts/`: Generated traces, verification results, and reports.
- `data/`: Sample input files and source datasets.
- `eval/`: Labeled evaluation dataset used for baseline benchmark comparisons.
- `scripts/`: Standalone tools (threshold calibration, baseline comparison benchmark).
- `docs/`: Research logs and architectural decisions.
  - `gpu_setup.md` — enabling CUDA, VRAM budget, measured speedups.
  - `nli_finetuning.md` — adapting the NLI verifier to your domain on 8 GB.
  - `research_directions.md` — candidate contributions for a paper.

## Evaluation & Experimentation (In Progress)

We have finalized the design of a 15-experiment validation framework that objectively tests X-RAG against RAGAS, RAGChecker, and ARES AI. 
This framework isolates granular RAG failures (Retrieval Miss, Embedding Drift, Context Truncation, Hard Boundaries, Multi-Stage Cascading Failures) to definitively prove and measure which evaluation systems can trace true root causes. The evaluation runner and 750-example static dataset are currently in active development.

## Installation

For the core pipeline and chat UI, see **[Quick Start](#quick-start)** above.
The project targets Python 3.13; the optional evaluation baselines below need
their own Python 3.10 environments.

### Optional: Baseline Comparison Benchmark

Comparing X-RAG against RAGAS/RAGChecker/ARES (see `docs/RESEARCH_LOG.md` for
the full methodology and investigation) needs extra, heavier dependencies not
required for the core pipeline, split across **three separate virtual
environments** -- `ragchecker` and `ares-ai` have pinned dependencies that
conflict with each other and with this project's own Python 3.13 venv, so each
gets its own **Python 3.10** venv. No Rust or C/C++ compiler install is needed
for any of them -- read `docs/RESEARCH_LOG.md` for how each blocker that
initially looked like it needed one was actually resolved (short version: old
pinned dependencies like `numpy<2.0`/`litellm==1.91.x` have prebuilt wheels for
Python 3.10, just not for 3.13).

1. **ragas** installs into this project's *main* venv:
   ```bash
   pip install -r requirements-eval.txt
   ```

   Also read `requirements-eval.txt` first -- it documents a required local
   compatibility shim for `ragas` (an upstream packaging issue, not something
   this project can fix).
2. **ragchecker** needs its own Python 3.10 venv:
   ```bash
   py -3.10 -m venv venv_eval_ragchecker
   venv_eval_ragchecker\Scripts\pip install -r requirements-eval-ragchecker.txt
   ```
3. **ares-ai** needs its own, separate Python 3.10 venv:
   ```bash
   py -3.10 -m venv venv_eval_ares
   venv_eval_ares\Scripts\pip install -r requirements-eval-ares.txt
   ```

   This one reliably fails on its *first* run with a Windows long-path
   `OSError` on a jupyterlab asset -- just run the same install command again
   and it completes using the cached downloads.

The comparison (and the core pipeline's own generation/claim decomposition)
routes through NVIDIA NIM's free tier (`configs/models.py`
`LLM_PROVIDER = "nvidia"`) -- get a free key at https://build.nvidia.com/ and
put it in `.env` as `NVIDIA_API_KEY=...`. No paid OpenAI/Bedrock key is
required. Each stage picks an appropriately-sized model: 70B for
quality-critical generation and judging, 8B for high-throughput claim
decomposition, to stay inside free-tier rate limits.

**Running the comparison** (after the environments above are set up):

```bash
# 1. Run all examples in eval/eval_dataset.csv through X-RAG + all three baselines.
#    Resumable: safe to re-run after an interruption or an HF credit/rate-limit error --
#    it skips eval rows already recorded in the manifest.
#    (Note: NLI Verification is now batched to speed up the X-RAG evaluation step significantly.)
python -m scripts.run_baseline_comparison

# Preview what would run without calling any model or LLM API:
python -m scripts.run_baseline_comparison --dry-run

# 2. Compute cross-framework correlation, agreement (Cohen's kappa), and disagreements.
python -m scripts.analyze_agreement

# 3. Render the self-contained paper-ready report.
python -m scripts.generate_comparison_report
```

Outputs land in `artifacts/benchmark_comparison/`: `results.json`/`results.csv`
(per-example scores), `correlations.json`, `agreement.json`,
`disagreements.csv`, `failures.log`, and finally `comparison_report.md` -- the
one file to read for the full methodology and findings.

**Per-example detailed report** -- a single self-contained HTML page for one
trace (full RAG trace, claim decomposition, chunk-level provenance, root
cause reasoning, and a side-by-side comparison against real RAGAS/RAGChecker/
ARES scores for that same trace, each heavy section tucked into a collapsible
`<details>` disclosure so the page opens compact):

```bash
python -m scripts.generate_diagnostic_report --trace-id <trace_id>
```

Writes to `artifacts/diagnostic_reports/<trace_id>.html`. Requires the trace
to have already been run once through `run_baseline_comparison.py` (or
`query.py`) so `artifacts/reports/<trace_id>.json` and the trace file exist;
the RAGAS/RAGChecker/ARES comparison section is included automatically if a
matching row exists in `artifacts/benchmark_comparison/results.json`.

## Configuration

Configuration variables are located in the `configs/` directory:

- `models.py`: Embedding, reranker, NLI, and LLM model names; verification batch sizes.
- `pipeline.py`: Chunk size/overlap, and the retrieval knobs below.
- `thresholds.py`: NLI entailment/contradiction thresholds that map to verdicts.
- `prompts.py`: Generator system prompts and the follow-up query condenser.
- `memory.yaml`: Memory ranking weights, similarity threshold, persistence path.
- `api.py`: API ports and hosts.

### Retrieval knobs worth knowing (`configs/pipeline.py`)

| Setting | Default | Effect |
|---|---|---|
| `FUSION_CANDIDATE_POOL` | 40 | Candidates each arm (dense, BM25) contributes to RRF. Cheap. |
| `RERANK_INPUT_SIZE` | 20 | Candidates handed to the cross-encoder. **Must exceed `RERANKER_TOP_N`**, or the reranker can only reorder what RRF already chose instead of rescuing a chunk RRF ranked 12th. This is the main latency/recall dial: ~1–1.5 s per candidate on CPU, negligible on GPU. |
| `RERANKER_MAX_LENGTH` | 384 | Truncation for cross-encoder input. |
| `RERANKER_TOP_N` | 6 | Chunks passed to the generator as context. |

## Running the Pipeline (CLI)

1. Ingest documents from `data/` into the chunk registry and vector store (run this first, and again any time the source PDFs change):
   ```bash
   python run_pipeline.py
   ```
2. Ask a question against the ingested corpus and run the full diagnostic pipeline (retrieval, generation, claim decomposition, verification, root cause analysis, corrective actions, and a timestamped PDF report):
   ```bash
   python query.py "your question here"
   ```

   If `NVIDIA_API_KEY` isn't set in `.env` or the environment, you'll be prompted for it interactively.

## Running Unit Tests

To verify the installation and the diagnostic framework integrity:

```bash
python -m unittest discover tests
```

## Running FastAPI

To launch the API server locally:

```bash
python run_api.py
```

The server will start on `http://127.0.0.1:8000`.

## Swagger Documentation

Once the API is running, access the interactive auto-generated Swagger UI at:
`http://127.0.0.1:8000/docs`

## Example API Calls

**Healthcheck:**

```bash
curl http://127.0.0.1:8000/health
```

**Analyze a RAGTrace:**

```bash
curl -X POST http://127.0.0.1:8000/analyze \
     -H 'Content-Type: application/json' \
     -d @artifacts/rag_traces/TRACE_123.json
```

## Example Diagnostic Report

The `DiagnosticEvaluationReport` represents the final analysis output. Rendered formats are available via:

- `GET /report/{trace_id}/markdown`
- `GET /report/{trace_id}/html`

## Future Work

- Integration with major RAG deployment frameworks (LlamaIndex, LangChain).
- Enhanced feedback loops directly returning Corrective Action Plans to the generation model.
- Fine-tuning the NLI verifier on in-domain claim/evidence pairs — see `docs/nli_finetuning.md`.

## Performance Notes

Measured on an RTX 4060 Laptop (8 GB). The models are unchanged throughout —
these are all pipeline-level fixes.

| Issue | Before | After |
|---|---|---|
| NLI verifier reloaded every turn (VRAM leak) | 6s → 25s → 46s per claim, VRAM exhausted | flat 2.4 GB, 2.7s → 1.4s per claim |
| Embedding model loaded 2–3× | duplicate ~0.5 GB copies | one shared instance |
| Memory searched twice per turn | 2 embeds + 2 vector queries, `access_count` double-counted | one search |
| Condensation on every turn | +1 LLM round-trip always | skipped for standalone questions |
| Answer rendering | blank until complete | first token in ~1s |

**What dominates now:** remote LLM generation (20–70s on the free NVIDIA tier).
Retrieval is 0.5–1.0s, of which ~95% is cross-encoder reranking; embedding
(7ms), Chroma (2ms), and BM25 (0.2ms) are noise. Nothing local is worth
optimising further without changing models.

Two behaviours worth knowing:

- **`st.cache_resource` is what prevents the VRAM leak.** `ClaimVerifier` and
  `ClaimDecomposer` must never be constructed inside the request path —
  building a fresh verifier per turn loads another ~1.6 GB of NLI weights
  without releasing the previous one.
- **Editing `ui/app.py` while the server is running invalidates those caches**,
  so the next turn reloads every model. Restart after editing rather than
  relying on hot-reload if you are timing anything.

## Troubleshooting

**Every answer is "I do not have enough information to answer this."**
Almost always retrieval, not the LLM. Check in this order:

1. Is the corpus actually ingested? `artifacts/chunk_registry.json` must exist
   and `db/chroma/` must be non-empty. If not, run `run_pipeline.py`.
2. Turn on **🐛 Debug Mode** and open the **Chunks** tab. If the top chunk's
   reranker score is high but the text is off-topic, it is a retrieval miss —
   raise `RERANK_INPUT_SIZE`.
3. If the refusal only happens on *follow-up* questions, check that the
   `↻ follow-up resolved` badge appears; if not, condensation failed and the
   raw pronoun query went to the retriever.

**Everything is extremely slow (minutes per message).**
You are on a CPU-only torch build. Check the **Device** readout in the sidebar
and see step 3 of the Quick Start.

**Answers are fine but the app feels sluggish on CPU.**
Turn **🔬 Deep Analysis** off. Claim extraction and NLI verification are the
expensive stages; retrieval and generation alone are ~30 s.

**`CUDA out of memory`.**
Lower, in order: `VERIFICATION_GPU_BATCH_SIZE` (`configs/models.py`), then
`RERANK_INPUT_SIZE`, then `RERANKER_MAX_LENGTH` (`configs/pipeline.py`).

**Changing the embedding model returns nonsense.**
Embeddings and the stored vectors must match. Delete `db/chroma/` and
`artifacts/chunk_registry.json`, then re-run `run_pipeline.py`.

**Force CPU to compare timings.**

```bash
set XRAG_DEVICE=cpu && venv\Scripts\python -m streamlit run ui/app.py
```

## Open Source Contribution Guide

We welcome community contributions! Please review `CONTRIBUTING.md` for guidelines on coding standards, folder structure, adding new diagnostic modules, and submitting Pull Requests.

---

## Retrieval-Strategy Research Platform

The framework above diagnoses *one* pipeline. This layer compares *strategies*:
baseline RAG, hybrid retrieval, IRCoT, agentic RAG, GraphRAG and their
combinations, over a corpus of Indian Supreme Court judgments — asking which
technique actually fixes which RAG failure mode.

Full component-by-component documentation, with trade-offs and industry
comparison, is in **[`docs/COMPONENTS.md`](docs/COMPONENTS.md)**.

### What is in it

| Piece | Module | What it adds |
|---|---|---|
| Corpus acquisition | `scripts/fetch_judgments.py` | SC/HC judgments from the AWS Open Data mirrors, with manifests |
| Parsing + provenance | `src/legal_corpus.py` | section/paragraph structure, citation extraction, per-document provenance |
| Legal-aware chunking | `src/legal_corpus.py` | structure-aware chunks vs the fixed-size baseline |
| Hybrid retrieval arms | `src/retriever.py` | `mode="vector"\|"bm25"\|"hybrid"`, `rerank=True\|False` |
| IRCoT | `src/ircot.py` | interleaved retrieval/reasoning (arXiv:2212.10509) |
| Agentic controller | `src/agentic.py` | bounded action selection, incl. contradiction search |
| Knowledge graph | `src/legal_graph.py` | NetworkX citation graph that participates in retrieval |
| Citation validation | `src/citation_check.py` | does the cited authority exist, and was it retrieved? |
| Benchmark | `scripts/build_benchmark.py` | multi-hop questions derived from the corpus, with gold evidence |
| Evaluation | `src/rag_eval.py` | retrieval / generation / cost metrics, and the 13-metric panel |
| Ablation | `experiments/exp06_*`, `exp07_*` | 11 retrieval arms, 4 generation arms, resumable |

### Running it

```bash
# 1. Acquire judgments (resumable, never re-downloads)
python scripts/fetch_judgments.py sc --years 2015-2024 --limit 70

# 2. Parse, chunk (both strategies), embed, store, write the manifest
python -m scripts.build_legal_corpus --limit 400 --rebuild

# 3. Build the citation graph and the benchmark
python -m scripts.build_benchmark --per-type 6

# 4. Ablations (resumable; interrupt and re-run to continue)
python -m experiments.run_all --only 6     # retrieval, 11 arms
python -m experiments.run_all --only 7     # generation, 4 arms
```

Results land in `artifacts/experiments/exp06_strategy_ablation/` and
`exp07_generation_ablation/` (`records.jsonl` per example, `summary.json` per
experiment), with every headline collected in
`artifacts/experiments/SUITE_REPORT.md`.

### In the chat UI

The sidebar exposes a **corpus** selector (statutes or judgments) and a
**retrieval strategy** selector carrying the same eleven arms the ablation
measures — the UI calls the same `execute_arm` function the experiments call, so
the two cannot drift apart. Two tabs were added to the details panel:

- **📐 Metrics** — faithfulness, answer relevancy, context precision, context
  recall, answer correctness, precision, recall, F1, hallucination, context
  relevance, answer relevance, answer faithfulness, and citation correctness,
  with a provenance table stating how each is computed and by which judge.
  Computed on demand (≈10 judge calls) and cached per trace.
- **🧭 Strategy** — the IRCoT step log or the agentic action log for the turn:
  each query, what it was looking for, new evidence, graph relation paths, and
  the termination reason.

### Module self-checks

Every new module carries a runnable self-check needing no corpus, GPU or API key:

```bash
python -m src.legal_corpus && python -m src.legal_graph && python -m src.ircot
python -m src.agentic && python -m src.citation_check && python -m src.rag_eval
```

---

## Research Experiment Suite (E1–E5)

Five studies backing the paper live in [`experiments/`](experiments/). They are the
research half of this repository: the pipeline above is the object of study, and these
are the measurements.

**One command runs all five, resumably:**

```powershell
venv\Scripts\python.exe -m experiments.run_all
```

Takes a few seconds in the default offline mode — no corpus, no API key, no GPU — and
writes everything to `artifacts/experiments/`, with every headline collected in
`artifacts/experiments/SUITE_REPORT.md`.

| # | Experiment | Question it answers |
|---|---|---|
| E1 | Causal fault injection | Inject a known fault; does stage-attributed diagnosis recover it? |
| E2 | Reranker-window pathology | At what `rerank_input / top_n` ratio does the cross-encoder stop being a no-op? |
| E3 | Refusal calibration | How much does a faithfulness metric reward an answer that says nothing? |
| E4 | Corpus quality | What fraction of the corpus can answer anything, and what does the rest cost? |
| E5 | Diagnostic agreement | Frameworks correlate on scores — do they agree on *causes*? |

### Resuming

Interrupt it at any point and re-run the same command: it continues where it stopped.
Resumption works both **between** experiments (a finished experiment is skipped) and
**within** one (a run killed at example 27 of 72 resumes at 27). Every example's
randomness is seeded from its own id, so a resumed run produces byte-identical records
to an uninterrupted one.

```powershell
venv\Scripts\python.exe -m experiments.run_all --status      # what's done, what isn't
venv\Scripts\python.exe -m experiments.run_all --from 5      # continue at 5; leave 1-4 alone
venv\Scripts\python.exe -m experiments.run_all --only 2 4    # just these
venv\Scripts\python.exe -m experiments.run_all --force       # discard checkpoints, start over
```

### Live mode

The default offline mode is deterministic and self-contained. To run E2/E3/E4 against
the real corpus and LLM instead (requires `run_pipeline.py` to have been run, plus an
API key for E3's generation A/B):

```powershell
venv\Scripts\python.exe -m experiments.run_all --mode live
venv\Scripts\python.exe -m experiments.run_all --only 3 --extra verifier=nli
```

Each experiment runs at least 50 examples; the suite refuses to run one below that
floor. Tests: `venv\Scripts\python.exe -m pytest tests/test_experiments.py -q`.

**Full write-up — design rationale, what each experiment can and cannot show, measured
results, and the publication path — is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).**
