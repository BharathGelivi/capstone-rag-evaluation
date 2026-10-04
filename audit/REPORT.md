# X-RAG implementation audit

Implemented on 2 October 2026 against revision `300b7873f4a0ae38e0524bcfb51be8b287da054c`.
The existing architecture, pipeline stages, retrieval budgets, diagnostic thresholds and research methods remain in place. No new dependency or indexing service was introduced. This is an evidence-based audit, not a guarantee that every possible bug has been found.

## Handoff

- Frontend: http://127.0.0.1:5173 — npm development server, listening PID **45712**.
- Chat backend: http://127.0.0.1:8010/docs — repository virtual environment, listening PID **20756**.
- Diagnostic API: http://127.0.0.1:8001/docs — repository virtual environment, listening PID **37400**.

Port 8000 belongs to an unrelated `dashboard.py --port 8000` process, PID 35256. It was left running. You approved temporary port 8001 for the diagnostic API. All three project services responded with HTTP 200 and are left running. Listening PIDs differ from Windows launcher PIDs; the values above identify the actual servers.

Runtime stdout/stderr logs are under `.cache/xrag-audit/`: `frontend.*.log`, `chat-api.*.log`, and `diagnostic-api.*.log`. Commands are `npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`, `venv/Scripts/python.exe run_api_ui.py`, and `venv/Scripts/python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8001`. Startup uses hidden windows. Process details are recorded in `.cache/xrag-audit/services.json`.

## Repository map and baseline

The local semantic inventory records Python symbols, imports and call sites, and frontend symbols/imports. Root entrypoints, `src`, `configs`, `scripts`, `experiments`, `tests`, evaluator tests, Streamlit `ui`, and `frontend/src` are indexed. Python call sites are syntax-level evidence, not a complete dynamic call graph. The final inventory includes root entrypoints that were missing from the first inventory.

```
run_pipeline / build_legal_corpus -> ingestion -> embedding -> Chroma + registry
query / chat service -> retrieval -> generation -> RAGTrace
RAGTrace -> claim decomposition -> verification -> PSM -> RCA -> corrective actions -> report
diagnostic API -> PipelineRunner; chat API -> chat service + session/memory storage
React and Streamlit -> existing APIs/services; experiments -> existing components + checkpoints
```

Artifact writers/readers and compatibility layouts were checked across runner, CLI, APIs, experiment scripts and tests before removing obsolete code. Supported providers and standalone tools were retained.

Before source implementation, **2,077 data/database/artifact files** were copied and hashed under `.cache/xrag-audit/baseline/protected`. The frozen revision, initial Git status, hashes and measurements are in `baseline/baseline.json`; source structure is in `baseline/semantic-index.json`. The only pre-existing tracked modification was the two synthetic audit lines in `failures.log`. The inventory helper itself was then an untracked audit addition.

Preserved audit baseline:

| Measurement | Before |
|---|---:|
| Focused Python suite | 476 tests; 4 stale failures; 4 skips; 48.407 s |
| Frontend build | Passed; recorded bundling time 1.43 s |
| Frontend lint | 8 warnings |
| Approved bounded live diagnostic | 43.979 s |
| In-process health median, 10 samples | 0.703 ms |

The approved live observation included retrieval 1.249 s, generation 18.059 s, decomposition 14.534 s and verification 5.763 s; component initialization was recorded separately. It disabled judge escalation and did not run full RAGAS or the production chat request. It must not be described as full production latency. No additional live LLM request was made during implementation.

The prior audit exercised real memory updates. All 28 inspected active memory vectors were zero at the implementation baseline; their state before the audit is unknown. The two synthetic `ragas exploded` entries added to `failures.log` during that audit were identified by exact timestamp and removed; the three earlier research log entries were retained.

The final environment snapshot records Python 3.13.3, Windows 11, Intel family-6/model-183 CPU, NVIDIA GeForce RTX 4060 Laptop GPU, installed package versions, and configuration hashes. It was captured after implementation; dependency versions were not changed. Secrets and `.env` contents are excluded. Baseline database hashes, the configured collection, and final vector counts identify the corpus/index used for the local benchmark.

## Confirmed findings and minimal fixes

Each row names the affected behavior, concrete source evidence and regression coverage. Test names below refer to repository test modules. Related defects are grouped where they share one minimal repair.

| ID / severity | Evidence and affected behavior | Implemented fix | Acceptance coverage |
|---|---|---|---|
| D01 High | `vector_store.py` and `memory_store.py` deleted collections after embedding-function initialization errors | Preserve collections and raise the initialization error | Audit regression: conflict never deletes; vector-store/memory suites |
| D02 High | Memory metadata updates used the null embedding function and overwrote real embeddings | Retrieve and explicitly reuse the stored vector; reject changed text without a replacement vector | Audit metadata-update and changed-text tests |
| D03 High | Active memory baseline had 28 zero vectors | Re-embed stored Q/A text locally after backup; update vectors only | Backup/current logical comparison: 28 records, identical documents/metadata, zero invalid vectors |
| D04 High | Imported sessions reused exported memory IDs, overwriting existing memories | Generate fresh IDs, retain source ID as metadata, reuse source vector or embed imported text; copy input records | Audit import collision test; memory/session suites |
| D05 High | Session read-modify-write updates and JSON persistence were unsynchronized/non-atomic | Existing-store RLock covers mutations; atomic fsynced JSON replacement | Audit 30 concurrent session updates; session/store tests |
| D06 Medium | Memory delete/clear left deque entries, summaries or session state behind | Clean associated short-term memory, summaries and sessions; surface persistence failures | Audit delete test; memory/session suites |
| D07 High | Ingestion published registries early and deleted the usable collection before storage succeeded | Insert/verify first, atomically publish registry, then delete obsolete IDs; roll back reused vectors and partial new IDs on pre-publication failure | Audit failed-publication and real-Chroma reused-ID rollback tests; vector-store suite |
| D08 Medium | Registry and embedding-cache writes could leave partial files | Atomic registry/cache replacement; reject corrupt/non-finite cached arrays | Registry, embedding and ingestion tests |
| C01 High | Runner treated dict trace references as objects; verification reconstruction lost parent IDs and could use the wrong corpus/current text | Shared reconstruction prefers captured prompt evidence, preserves parent metadata, then uses selected registry | Snapshot/parent/fallback audit tests; verifier and runner suites |
| C02 High | Missing original chunks could be judged unsupported as though all evidence were available | Record unresolved IDs, mark inconclusive unsupported claims not verifiable, report partial; clear warnings after successful fallback | Missing-evidence and resolved-fallback audit tests |
| C03 High | Malformed decomposition JSON could appear as a healthy zero-claim response | Validate array/items/nonempty claim strings; propagate extraction success; distinguish failure from valid empty output | Parser wrong-shape/empty tests; decomposer, PSM/report tests |
| C04 Medium | Canonical claim conversion generated IDs unrelated to candidate verification IDs | Preserve candidate IDs and metadata across conversion | Claims, verifier, runner/end-to-end tests |
| C05 Medium | Empty/all-unverifiable claims gave misleading generator success and grounding weight | Generator UNKNOWN when unassessable; omit empty grounding contribution to health | PSM/report suites |
| C06 Medium | Chat claim fields were appended outside the trace schema | Store fields in `diagnostics`; read legacy top-level fields compatibly | Legacy trace audit test; API/UI tests |
| C07 Medium | Runner/CLI omitted canonical artifacts and discovery assumed only one layout | Persist existing canonical names; discover current date directories and legacy flat layouts; record paths | API, runner/end-to-end and report suites |
| C08 Medium | Report appendix invented artifacts and verification identifiers | Use actual verification UUIDs and recorded produced paths; show unrecorded paths explicitly | Report-builder/presenter suites |
| C09 Medium | Report stage labels were derived from failure-name suffixes, producing non-stage names | Map causes through the existing reasoner's stage map | Report-presenter suite and source mapping check |
| C10 Medium | Corrective guards used defaults instead of the actual generation configuration | Record actual temperature/token/retrieval settings; forward snapshots; support old temperature key | Trace, corrective-action and runner tests |
| C11 Medium | Truncated generation looked completed; condensed retrieval query could replace the user's question in the trace | Capture finish reason for sync/stream; mark length-limited generation partial; keep displayed question | Generator/trace suites; interrupted-stream browser test |
| C12 Medium | Chat metrics included hardcoded similarity/relevancy numbers and cross-encoder scores presented as evaluated metrics | Compute available semantic relevance with the existing encoder; use unavailable values otherwise; label lexical overlap as heuristic | Chat/API review, frontend build, answer/evidence browser fixture |
| C13 Medium | Failed RAGAS relevance checks became false negatives; NaN/Inf polluted agreement | Tri-state relevance and unavailable aggregates; finite-input filtering | Audit RAGAS/finite-input tests; RAGAS/agreement/research tests |
| A01 High | User identifiers/registry paths could influence filesystem paths | Constrain IDs and enforce registry containment under artifacts | Audit request/path tests; API suite |
| A02 High | Report HTML interpolated untrusted report text | HTML-escape report title/content | Presenter suite and generated-HTML implementation check |
| A03 High | `lru_cache` alone allowed multiple concurrent cold model loads | Serialize existing cached loaders and runner initialization; do not cache failed loads | Eight-thread single-initialization audit test; retry-after-failure test |
| A04 High | Cached generator's `last_stream_result` was shared between conversations | Per-turn shallow generator copy, retaining shared model/client | Chat and Streamlit caller review; generator tests; browser request-identity checks |
| A05 Medium | Missing credentials could prompt on chat request; some diagnostic/summary calls bypassed shared limits/timeouts | Noninteractive chat credential check; explicit timeouts; shared existing limiter for structured calls, summaries and metrics | Generator/IRCOT/summary/decomposer/limiter tests |
| A06 Medium | Unsupported combinations, malformed history, blank questions and unbounded pagination reached execution | Request validation, corpus/arm compatibility and bounded limits | API/UI/audit request tests; compatible-arm browser test |
| A07 Medium | Internal exceptions were exposed, and persistence/recall failures could disappear | Generic error references in client responses, server details in logs, explicit warnings/failures | API error tests; stream/error browser checks |
| A08 Medium | Registry caches stayed stale after ingestion | Cache signature includes registry mtime and size; unsuccessful loads are retried | Cache audit tests and loader call-site review |
| A09 Low | Timing/device/config UI data did not represent actual work/hardware | Correct elapsed memory time and expose actual device/corpus-compatible arms | API config test; browser device/arm test |
| P01 Medium | Diagnostic initialization constructed a retriever/reranker solely to obtain the encoder | Reuse the existing shared encoder directly | Runner/end-to-end tests; removed constructor/import reference check |
| P02 Medium | Repeated checkpoint append repair would rescan the entire growing file | Validate/repair when file signature changes; ordinary sequential appends avoid repeated scans | Checkpoint/audit/experiment suites |
| U01 High | HTTP errors or EOF without completion could look like a successful chat | Check status/body, require terminal SSE event, cancel/release reader | Interrupted-stream and visible-memory-error browser checks |
| U02 High | React state lag allowed duplicate sends/session creation | Immediate busy ref; handle session creation failure before send | Double-submission browser check |
| U03 High | Old session/unmounted requests could update the new conversation | Abort obsolete requests; apply events by controller/session/message identity | Session-switch and navigation-during-turn browser checks |
| U04 Medium | Hydration could replace an active conversation; retry reused failed turn history | Guard hydration and retry with the intended prior-message snapshot | Retry-history/session-switch browser checks |
| U05 Medium | Memory/Graph/Debug responses could arrive out of order without visible failure state | Abort obsolete loads; guard responses; loading/error states | Browser memory error and graph checks; build/lint |
| U06 Medium | UI offered incompatible arms and showed assumed hardware/arm counts | Use returned compatibility/device data and actual arm count | Browser configuration check; API config tests |
| U07 Low | Interactive controls lacked keyboard/ARIA support and effect dependencies were stale | Keyboard navigation, accessible labels, stable callbacks, correct dependencies | Browser role-based interaction; zero-warning lint/build |
| U08 Medium | Memory limits were applied before recency sorting; session hydration silently truncated history | Sort before limit; explicitly request all session memories | Memory/session suites and caller review |
| R01 Medium | Four test expectations covered obsolete 1–12 registry/mode behavior | Update assertions to supported 1–13 experiment registry and hybrid mode | Full Python suite |
| R02 High | Tests wrote real memory/artifacts or initialized remote models; integration included live experiments | Temporary storage, mocked clients, opt-in live/browser tests; offline integration selects E1–E5 | Full suite, protected-file hashes, mocked browser suite |
| R03 High | `--dry-run` executed live models and wrote outputs for two examples | Return a JSON execution plan before credentials/model imports/writes | Actual dry-run smoke invocation with limit 2; baseline comparison tests |
| R04 Medium | Completed experiment runs were skipped before configuration compatibility was checked | Validate plan fingerprint before skipping; honor explicit complete rerun | Experiment checkpoint/run-all suites |
| R05 Medium | A truncated final checkpoint record could corrupt the next append; interior corruption was silently ignored | Repair only final fragment; reject interior corruption; fsync appended record | Audit checkpoint recovery/corruption tests; experiment suite |
| R06 Medium | Equal-score retrieval and graph traversal depended on incidental iteration order | Deterministic chunk-ID tie break and sorted graph traversal | Retriever/graph/research suites; six benchmark retrievals returned identical IDs |
| R07 Medium | Strategy ablation ranks and retrieval-call counts omitted later retrievals | Reassign final ranks and account for initial/contradiction/additional calls | Strategy/research-platform tests |
| R08 Medium | Naive timestamps and assumed judgment URLs could misrepresent recency/provenance | Normalize timestamp handling; record exact bucket/key source URL and prefer it downstream; migrate manifest columns atomically | Memory-utils and 23 fetch/research tests |
| R09 Low | Unused configuration/imports, broken mapping and obsolete browser test setup obscured supported behavior | Remove proven unused paths; retain supported CLI/tools/providers | AST/import/caller/reference checks; full tests/build |

## Data and research integrity

The final hash check found **no missing protected file**. All original corpus files, registries, sessions, historical traces, reports and research outputs retained their baseline bytes. Exactly five baseline files changed:

1. Active memory Chroma SQLite database.
2. Active memory HNSW `data_level0.bin`.
3. Active memory HNSW `length.bin`.
4. `artifacts/benchmark_comparison/failures.log`: only the two precisely identified synthetic audit lines removed.
5. `artifacts/memory_logs/memory_2026-10-02.log`: runtime/repair logging.

An isolated copy of the backed-up memory database was compared with the current active collection. Both contain 28 IDs; documents and metadata are identical; invalid vectors changed from 28 to zero. An idempotent repair rerun found zero invalid entries and performed zero repairs. The first repair successfully updated vectors but its JSON report encountered a NumPy integer serialization error; that reporting bug was corrected and the independent baseline comparison verifies the repair outcome.

Historical experiment outputs were not rerun or overwritten. E1–E5 regression runs used temporary output directories. Expected future diagnostic differences include UNKNOWN/partial states for unavailable evidence/extraction, actual claim/verification links, no fabricated report references, truthful unavailable metrics, and preserved embeddings improving future memory recall. Corrected results should be recorded as new runs; they are not interchangeable with historical results.

The baseline backup intentionally retains the audit-added lines as evidence. Private baseline copies and memory content remain in the ignored local `.cache` directory, not the report/manifest. New transient test artifacts are not research results.

Two new `TRACE_TRACE_E2E.json` files from the intermediate, failing test fixtures were copied to `.cache/xrag-audit/synthetic-test-artifacts` and removed from live `artifacts/claims` and `artifacts/answer_correctness`. Neither existed in the baseline; the exact cleanup is recorded in validation evidence. No historical artifact was deleted.

## Dead-code and obsolete-path removal

No standalone script/provider/research artifact was deleted merely because it lacked an import. Removed items were checked against the semantic inventory, runtime/CLI/experiment/test references and repository text searches:

- Unused `LLM_MAX_RETRIES` setting: clients deliberately use zero internal retries and the shared limiter handles retry/backoff; no remaining reference.
- Runner's temporary retriever/vector-store/registry construction used only to obtain the encoder; its unused imports and locals were removed when replaced by the existing shared encoder.
- Broken object-style RAGAS reference mapping; both verifier and runner now use the same reconstruction routine.
- Presenter code that fabricated verification names and a fixed generated-artifact appendix; real metadata replaces it.
- Unused imports in memory manager, verifier, trace/vector-store modules and related removed paths.
- Obsolete browser warm-up/model calls and stale selectors; tests now use explicit opt-in and deterministic mocked backend fixtures.

The `HuggingFaceEmbedding` compatibility/test seam, supported credential/provider helpers, public experiment helper exports, `groq_judge`, comparison tools and standalone experiments were retained. No evidence established that those supported paths were dead.

## Validation and measurement limits

Detailed test counts, benchmarks, integrity results and service checks are in `validation.json`; every changed file and purpose is listed in `modified-files.json`. Local full logs are under `.cache/xrag-audit`.

- Isolated Python regression suite: **505 tests, zero failures/errors, seven opt-in browser skips**, 247.038 s. The final audit-specific suite additionally covers the last ingestion rollback repair: **20 passed**, 5.094 s.
- Dedicated evaluator environments: ARES **4 passed**; RAGChecker **3 passed**.
- Final affected suites: vector store **4 passed**, generator **20 passed**, report presenter **14 passed**, including explicit HTML-injection and traceability regressions.
- Additional pytest function suites: **23 passed** (fetch judgments and research platform).
- Headless Chromium: **7 passed**, with every backend call mocked; no live LLM request or memory mutation.
- Frontend TypeScript/Vite production build passes; lint has **zero warnings**, down from eight.
- `pip check`, final Python compilation and Git whitespace checks pass.
- Actual local service GET checks: diagnostic health, chat configuration and frontend return HTTP 200.

The local retrieval benchmark uses an isolated copy of the frozen 893-vector statutory collection, the same BNS question, configured hybrid retrieval/reranking, a 20-candidate reranker window and six final context chunks. It records cold initialization, first retrieval, five warm retrievals, working-set memory and peak allocated GPU memory, with Hugging Face networking disabled. Chunk IDs stayed identical across all six calls. Retrieval budgets were not reduced.

| After observation | Result |
|---|---:|
| Cold dependency imports | 25.999 s |
| Cold local retrieval-component initialization, imports excluded | 13.895 s |
| First local retrieval | 2.780 s |
| Five warm retrievals, median | 0.678 s |
| Benchmark process working set | 1.745 GB |
| Retrieval-only peak allocated GPU memory | 1.831 GB |
| In-process health median, persistent client | 4.162 ms |
| Frontend final bundling time | 0.329 s |
| Additional live full-pipeline measurement | Not performed |

Cold initialization and health observations are slower than the earlier bounded measurements; differing component boundaries, imports/client lifetime and audit workloads prevent attributing those differences to a code regression or claiming a speedup. The full suite now includes offline integration and additional regression coverage, so its elapsed time is not comparable to the original focused 476-test run.

The original single live observation and the new local measurements have different cache/workload/component boundaries. They are observations, **not a controlled production speedup claim**. Health timing also varies with test-client lifetime and concurrent audit workload. Generation, decomposition and verification after-latency are unavailable because the prior approval covered only one specific baseline request. The actual simultaneous-live-conversation latency and GPU-lock contention profile are also unmeasured. The GPU critical section remains unchanged.

Retained resource improvements are the removal of unnecessary retriever/reranker initialization and prevention of duplicate cold model loading; the concurrent regression proves one initialization across eight callers. No speculative retrieval-budget change or shortened GPU locking was introduced.

## Issues encountered and remaining limitations

- Windows sandbox restrictions initially blocked temporary Chroma writes and Vite subprocesses. Authorized checks were rerun with reviewed execution permissions; successful results are recorded.
- Four baseline failures were stale test expectations rather than product defects. Two intermediate end-to-end fixture errors came from nonserializable mock metadata and were fixed by realistic isolated fixtures.
- A missing optional `psutil` package and then a Windows handle binding error affected only the temporary profiling helper. Profiling now uses standard-library Windows memory counters; no dependency was added.
- A Chroma HNSW file remained mapped during Windows temporary-directory cleanup in the new ingestion test. Cleanup accommodates that platform behavior; the regression assertions pass and storage remains isolated.
- Port 8000 is occupied; the approved temporary diagnostic API is on 8001. No unrelated process was stopped.
- The semantic index is local/static; runtime-generated imports and external extensions may require additional investigation.
- Browser tests validate UI state and request behavior with fixtures; they do not prove remote provider availability or live answer quality. Health/config startup does not eagerly initialize every model.
- No additional live before/after benchmark or live concurrent conversation was authorized or performed. Full production performance comparison remains deferred.
- Chroma and registry publication are separate persistence operations. Failed pre-publication insertion rolls back reused IDs and partial new IDs while preserving the prior registry. This is not a cross-process transaction: a database outage during rollback or process death mid-ingestion still requires inspection/recovery from the protected backup. A transaction/redesign was not introduced.
- Session synchronization is within the existing single-worker process. Cross-process session writers are outside the tested deployment model.
- Existing historical judgment rows without a stored exact source URL retain legacy fallback behavior. Existing research metadata was not rewritten to guess missing bench paths.
- Model output, scores and network timings remain nondeterministic. Correctness changes may produce different future diagnostic results; historical artifacts were preserved rather than relabeled.

Implementation proceeded in Default mode after explicit approval of the plan. Remaining limitations are reported separately from completed fixes; no architecture change was needed.
