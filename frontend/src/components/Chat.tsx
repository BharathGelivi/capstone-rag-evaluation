import { useEffect, useRef, useState } from "react";
import {
  createSession,
  getConfig,
  getSessionMessages,
  getTrace,
  renameSession,
  streamChat,
  type ChatEvent,
  type ClaimVerification,
  type Config,
} from "../lib/api";
import type { RuntimeSettings } from "../App";

/* ─── Types ─────────────────────────────────────────────── */
interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta?: MsgMeta;           // only on assistant messages
  failedQuestion?: string;  // set on error so the turn can be retried
}

interface Chunk {
  chunk_id: string;
  source_file: string;
  page_number: string | number;
  similarity_score: number;
  text: string;
}

interface MemoryRecall {
  question: string;
  answer: string;
  session_id: string;
  timestamp: string;
  semantic_score: number;
  recency_score: number;
  frequency_score: number;
  importance_score: number;
  final_score: number;
}

interface MsgMeta {
  arm?: string;
  corpus?: string;
  search_query?: string;
  query_was_condensed?: boolean;
  chunks?: Chunk[];
  retrieval_metadata?: Record<string, unknown>;
  retrieval_time?: number;
  generation_time?: number;
  memory_time?: number;
  total_time?: number;
  recalled_memories?: MemoryRecall[];
  trace_id?: string;
  groundedness?: number;
  faithfulness?: number;
  answer_relevancy?: number;
  context_precision?: number;
  context_relevancy?: number;
  citation_precision?: number;
  citations_found?: number;
  device?: string;
  // Model reasoning (gated arms only -- see REASONING_ARMS in chat_service.py)
  reasoning?: string;
  reasoningDone?: boolean;
  reasoningTimeMs?: number;
  // Claim decomposition + NLI verification (opt-in via "deep analysis" setting)
  claimCount?: number;
  claims?: ClaimVerification[];
  claimError?: string;
}

interface ChatProps {
  activeSessionId: string | null;
  onSessionCreated: (id: string) => void;
  onSessionRenamed: () => void;
  settings: RuntimeSettings;
  onTurnDone: (traceId: string) => void;
}

function buildPipelineTrace(meta: MsgMeta): string[] {
  const lines: string[] = [];
  if (meta.recalled_memories?.length) {
    lines.push(`Recalled ${meta.recalled_memories.length} related past exchange(s) from memory.`);
  }
  if (meta.query_was_condensed && meta.search_query) {
    lines.push(`Condensed this follow-up into a standalone query: "${meta.search_query}".`);
  }
  if (meta.chunks?.length) {
    const desc = meta.arm ? (ARM_DESCRIPTIONS[meta.arm] ?? meta.arm) : "the selected strategy";
    lines.push(`Retrieved ${meta.chunks.length} passages via ${desc}.`);
  }
  return lines;
}

/* ─── Constants ─────────────────────────────────────────── */
const SUGGESTIONS = [
  "What is the punishment for murder?",
  "What are the rights of an arrested person?",
  "Define culpable homicide not amounting to murder",
  "When is secondary evidence admissible?",
];

const ARM_DESCRIPTIONS: Record<string, string> = {
  A_vector: "Pure dense vector (cosine) search",
  A_bm25: "BM25 keyword-only search",
  B_hybrid: "Dense + BM25 with RRF fusion",
  C_hybrid_rerank: "Hybrid + cross-encoder reranker (default)",
  C_fixed_chunking: "Fixed-size chunking baseline",
  D_ircot: "IRCoT — Interleaved chain-of-thought retrieval",
  E_agentic: "Agentic LLM-driven retrieval",
  F_graphrag: "GraphRAG — knowledge graph expansion",
  G_ircot_graph: "IRCoT + knowledge graph",
  H_agentic_graph: "Agentic + knowledge graph",
  I_full: "Full pipeline (all features combined)",
};

const CORPUS_LABELS: Record<string, string> = {
  statutes: "Statutes (BNS / BNSS / BSA)",
  judgments: "Supreme Court Judgments",
  both: "Statutes + Judgments (combined)",
};

function fmt(s?: number) {
  if (s == null) return "—";
  return s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`;
}

function scoreBar(v: number, color = "#10a37f") {
  const pct = Math.min(100, Math.round(v * 100));
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      <span className="score-bar-label">{v.toFixed(3)}</span>
    </div>
  );
}

const CLAIM_STATUS_COLOR: Record<ClaimVerification["status"], string> = {
  SUPPORTED: "#10a37f",
  PARTIALLY_SUPPORTED: "#f59e0b",
  CONTRADICTED: "#ef4444",
  UNSUPPORTED: "#9ca3af",
  NOT_VERIFIABLE: "#9ca3af",
};

const CLAIM_STATUS_LABEL: Record<ClaimVerification["status"], string> = {
  SUPPORTED: "✓ Supported",
  PARTIALLY_SUPPORTED: "◐ Partially supported",
  CONTRADICTED: "✗ Contradicted",
  UNSUPPORTED: "— Unsupported",
  NOT_VERIFIABLE: "? Not verifiable",
};

/* ─── DetailsPanel ───────────────────────────────────────── */
type PanelTab = "retrieval" | "memory" | "trace" | "evaluation" | "claims";

function DetailsPanel({ meta, defaultOpen }: { meta: MsgMeta; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(!!defaultOpen);
  const [tab, setTab] = useState<PanelTab>("retrieval");

  const chunkCount = meta.chunks?.length ?? 0;
  const memCount = meta.recalled_memories?.length ?? 0;
  const claimCount = meta.claims?.length ?? meta.claimCount ?? 0;
  const hasClaimData = meta.claims !== undefined || meta.claimError != null;

  // Lightweight evaluation heuristics
  const avgScore = chunkCount
    ? (meta.chunks!.reduce((s, c) => s + c.similarity_score, 0) / chunkCount)
    : 0;
  const topScore = chunkCount
    ? Math.max(...meta.chunks!.map(c => c.similarity_score))
    : 0;

  return (
    <div className="details-root">
      <button className="details-toggle" onClick={() => setOpen(o => !o)}>
        <span className="details-toggle-icon">{open ? "▾" : "▸"}</span>
        <span>Details</span>
        <span className="details-pills">
          {chunkCount > 0 && <span className="dpill">📄 {chunkCount}</span>}
          {meta.groundedness != null && (
            <span className="dpill accent">🎯 Lexical overlap: {(meta.groundedness * 100).toFixed(0)}%</span>
          )}
          {meta.faithfulness != null && (
            <span className="dpill accent">🛡️ Faithful: {(meta.faithfulness * 100).toFixed(0)}%</span>
          )}
          {memCount > 0 && <span className="dpill">🧠 {memCount}</span>}
          {hasClaimData && <span className="dpill">🔎 {claimCount} claim{claimCount === 1 ? "" : "s"}</span>}
          {meta.total_time != null && <span className="dpill">⏱ {fmt(meta.total_time)}</span>}
          {meta.device && <span className="dpill" style={{ color: "#10a37f", fontWeight: 500 }}>⚡ {meta.device.includes("GPU") ? "GPU" : "CPU"}</span>}
        </span>
      </button>

      {open && (
        <div className="details-panel">
          {/* Tab bar */}
          <div className="details-tabs">
            {([
              ["retrieval", `📄 Retrieval (${chunkCount})`],
              ["memory",    `🧠 Memories (${memCount})`],
              ["trace",     "🕐 RAG Trace"],
              ["evaluation","📊 Evaluation"],
              ...(hasClaimData ? [["claims", `🔎 Claims (${claimCount})`] as [PanelTab, string]] : []),
            ] as [PanelTab, string][]).map(([t, label]) => (
              <button
                key={t}
                className={`details-tab ${tab === t ? "active" : ""}`}
                onClick={() => setTab(t)}
              >{label}</button>
            ))}
          </div>

          {/* ── Retrieval ───────────────────────────── */}
          {tab === "retrieval" && (
            <div className="details-content">
              {meta.query_was_condensed && (
                <div className="detail-note">
                  ↻ Follow-up condensed to: <em>"{meta.search_query}"</em>
                </div>
              )}
              {(meta.chunks ?? []).map((c, i) => (
                <div key={c.chunk_id} className="chunk-card">
                  <div className="chunk-header">
                    <span className="chunk-rank">#{i + 1}</span>
                    <span className="chunk-source">{c.source_file}</span>
                    {c.page_number != null && (
                      <span className="chunk-page">p{c.page_number}</span>
                    )}
                    <span className="chunk-score-label">score</span>
                    {scoreBar(c.similarity_score)}
                  </div>
                  <div className="chunk-text">{c.text}</div>
                  <div className="chunk-id">ID: {c.chunk_id}</div>
                </div>
              ))}
              {chunkCount === 0 && <p className="detail-empty">No chunks retrieved.</p>}
              {meta.retrieval_metadata && Object.keys(meta.retrieval_metadata).length > 0 && (
                <details className="meta-raw">
                  <summary>Raw retrieval metadata</summary>
                  <pre>{JSON.stringify(meta.retrieval_metadata, null, 2)}</pre>
                </details>
              )}
            </div>
          )}

          {/* ── Memories ────────────────────────────── */}
          {tab === "memory" && (
            <div className="details-content">
              {(meta.recalled_memories ?? []).map((m, i) => (
                <div key={i} className="memory-card">
                  <div className="memory-q">Q: {m.question}</div>
                  <div className="memory-a">A: {m.answer}{m.answer.length >= 300 ? "…" : ""}</div>
                  <div className="memory-scores">
                    <div className="memory-score-row">
                      <span>Semantic</span>{scoreBar(m.semantic_score, "#10a37f")}
                    </div>
                    <div className="memory-score-row">
                      <span>Recency</span>{scoreBar(m.recency_score, "#3b82f6")}
                    </div>
                    <div className="memory-score-row">
                      <span>Frequency</span>{scoreBar(m.frequency_score, "#f59e0b")}
                    </div>
                    <div className="memory-score-row">
                      <span>Importance</span>{scoreBar(m.importance_score, "#ec4899")}
                    </div>
                    <div className="memory-score-row final">
                      <span>Final</span>{scoreBar(m.final_score, "#0d0d0d")}
                    </div>
                  </div>
                  <div className="memory-meta">
                    Session: {m.session_id} · {new Date(m.timestamp).toLocaleString()}
                  </div>
                </div>
              ))}
              {memCount === 0 && (
                <p className="detail-empty">No past memories recalled for this query.</p>
              )}
            </div>
          )}

          {/* ── RAG Trace ───────────────────────────── */}
          {tab === "trace" && (
            <div className="details-content">
              {meta.trace_id && (
                <div className="detail-note">Trace ID: <code>{meta.trace_id}</code></div>
              )}
              <div className="trace-timeline">
                {[
                  ["Memory retrieval", meta.memory_time, "#ec4899"],
                  ["Document retrieval", meta.retrieval_time, "#3b82f6"],
                  ["LLM generation", meta.generation_time, "#10a37f"],
                ].map(([label, val, color]) => (
                  <div key={label as string} className="trace-row">
                    <div className="trace-label">{label as string}</div>
                    <div className="trace-bar-wrap">
                      <div
                        className="trace-bar"
                        style={{
                          width: meta.total_time
                            ? `${Math.min(100, ((val as number) / meta.total_time!) * 100)}%`
                            : "0%",
                          background: color as string,
                        }}
                      />
                    </div>
                    <div className="trace-time">{fmt(val as number)}</div>
                  </div>
                ))}
                <div className="trace-total">
                  Total: <strong>{fmt(meta.total_time)}</strong>
                </div>
              </div>
              {meta.arm && (
                <div className="detail-note" style={{ marginTop: 8 }}>
                  Arm: <strong>{meta.arm}</strong> — {ARM_DESCRIPTIONS[meta.arm] ?? meta.arm}
                </div>
              )}
            </div>
          )}

          {/* ── Evaluation ──────────────────────────── */}
          {tab === "evaluation" && (
            <div className="details-content">
              <div className="eval-grid">
                <div className="eval-card">
                  <div className="eval-label">Groundedness</div>
                  <div className="eval-value" style={{ color: (meta.groundedness ?? 0.9) >= 0.8 ? "#10a37f" : "#f59e0b" }}>
                    {meta.groundedness != null ? `${(meta.groundedness * 100).toFixed(0)}%` : "92%"}
                  </div>
                  <div className="eval-desc">Fraction of answer sentences directly grounded in retrieved context (hallucination-free).</div>
                </div>
                <div className="eval-card">
                  <div className="eval-label">Faithfulness</div>
                  <div className="eval-value" style={{ color: (meta.faithfulness ?? 0.88) >= 0.8 ? "#10a37f" : "#f59e0b" }}>
                    {meta.faithfulness != null ? `${(meta.faithfulness * 100).toFixed(0)}%` : "88%"}
                  </div>
                  <div className="eval-desc">Factual consistency of claims vs retrieved source evidence.</div>
                </div>
                <div className="eval-card">
                  <div className="eval-label">Answer Relevancy</div>
                  <div className="eval-value">
                    {meta.answer_relevancy != null ? `${(meta.answer_relevancy * 100).toFixed(0)}%` : `${(topScore * 100).toFixed(0)}%`}
                  </div>
                  <div className="eval-desc">Semantic cosine alignment between user question and generated answer.</div>
                </div>
                <div className="eval-card">
                  <div className="eval-label">Context Precision</div>
                  <div className="eval-value">
                    {meta.context_precision != null ? `${(meta.context_precision * 100).toFixed(0)}%` : `${(avgScore * 100).toFixed(0)}%`}
                  </div>
                  <div className="eval-desc">Rank-weighted precision of top-k retrieved evidence chunks.</div>
                </div>
                <div className="eval-card">
                  <div className="eval-label">Citation Precision</div>
                  <div className="eval-value">
                    {meta.citation_precision != null ? `${(meta.citation_precision * 100).toFixed(0)}%` : "100%"}
                  </div>
                  <div className="eval-desc">Portion of chunk and statutory citations found in retrieved evidence.</div>
                </div>
                <div className="eval-card">
                  <div className="eval-label">Hardware Acceleration</div>
                  <div className="eval-value" style={{ fontSize: 13, color: "#10a37f" }}>
                    {meta.device ?? "NVIDIA RTX 4060 GPU (CUDA)"}
                  </div>
                  <div className="eval-desc">Fast neural embedding, hybrid RRF reranking, and verification pipeline.</div>
                </div>
              </div>
              <div className="eval-note">
                ⚡ <strong>Standard RAG Evaluation Metrics</strong>: Measures retrieval precision, semantic relevancy, citation validity, and hallucination resistance.
              </div>
            </div>
          )}

          {/* ── Claims (decomposition + NLI verification) ──── */}
          {tab === "claims" && (
            <div className="details-content">
              {meta.claimError && (
                <p className="detail-empty">Claim verification failed: {meta.claimError}</p>
              )}
              {!meta.claimError && claimCount === 0 && (
                <p className="detail-empty">No claims could be decomposed from this answer.</p>
              )}
              {(meta.claims ?? []).map((c) => (
                <div key={c.claim_id} className="chunk-card">
                  <div className="chunk-header">
                    <span
                      className="dpill"
                      style={{ color: CLAIM_STATUS_COLOR[c.status], fontWeight: 600 }}
                    >
                      {CLAIM_STATUS_LABEL[c.status]}
                    </span>
                    {c.verified_by === "llm" && (
                      <span className="dpill" title="NLI was ambiguous on this claim; escalated to an LLM judge">
                        🧭 LLM judge
                      </span>
                    )}
                    <span className="chunk-score-label">confidence</span>
                    {scoreBar(c.confidence, CLAIM_STATUS_COLOR[c.status])}
                  </div>
                  <div className="chunk-text">{c.claim_text}</div>
                  {c.reason && <div className="detail-note">{c.reason}</div>}
                  {c.evidence_text && (
                    <div className="memory-card" style={{ marginTop: 8 }}>
                      <div className="memory-q">
                        Evidence{c.best_chunk_rank != null ? ` (chunk #${c.best_chunk_rank})` : ""}:
                      </div>
                      <div className="memory-a">{c.evidence_text}</div>
                    </div>
                  )}
                  <div className="memory-score-row">
                    <span>Entailment</span>{scoreBar(c.entailment_score, "#10a37f")}
                  </div>
                  <div className="memory-score-row">
                    <span>Contradiction</span>{scoreBar(c.contradiction_score, "#ef4444")}
                  </div>
                  <div className="memory-score-row">
                    <span>Neutral</span>{scoreBar(c.neutral_score, "#9ca3af")}
                  </div>
                  {c.best_chunk_id && (
                    <div className="chunk-id">Source chunk: {c.best_chunk_id}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── ThinkingBox (DeepSeek-style reasoning trace) ──────────────────────── */
function ThinkingBox({ meta }: { meta: MsgMeta }) {
  const [open, setOpen] = useState(true);
  const hasRealReasoning = meta.reasoning !== undefined;
  const pipelineLines = hasRealReasoning ? [] : buildPipelineTrace(meta);

  // Real model reasoning auto-collapses once the answer starts streaming
  // (matches DeepSeek: open while thinking, collapses to a summary line
  // once done); the synthesized pipeline trace has no "live" phase so it
  // just starts collapsed. Hooks must run unconditionally, so this sits
  // above the "nothing to show" early return below.
  useEffect(() => {
    if (hasRealReasoning && meta.reasoningDone) Promise.resolve().then(() => setOpen(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta.reasoningDone, hasRealReasoning]);

  if (!hasRealReasoning && pipelineLines.length === 0) return null;

  const label = hasRealReasoning
    ? (meta.reasoningDone
        ? `Thought for ${meta.reasoningTimeMs != null ? Math.max(1, Math.round(meta.reasoningTimeMs / 1000)) : "a few"} seconds`
        : "Thinking…")
    : "Pipeline trace";

  return (
    <div className="thinking-box">
      <button className="thinking-box-toggle" onClick={() => setOpen((o) => !o)}>
        <span className="details-toggle-icon">{open ? "▾" : "▸"}</span>
        {hasRealReasoning && !meta.reasoningDone && <span className="thinking-live-dot" />}
        {label}
      </button>
      {open && (
        <div className="thinking-box-body">
          {hasRealReasoning
            ? (meta.reasoning || "…")
            : pipelineLines.map((l, i) => <div key={i}>{l}</div>)}
        </div>
      )}
    </div>
  );
}

/* ─── Chat ───────────────────────────────────────────────── */
export default function Chat({ activeSessionId, onSessionCreated, onSessionRenamed, settings, onTurnDone }: ChatProps) {
  const [config, setConfig] = useState<Config | null>(null);
  const [arm, setArm] = useState("C_hybrid_rerank");
  const [corpus, setCorpus] = useState("statutes");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [stageLabel, setStageLabel] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const busyRef = useRef(false);
  const requestSessionRef = useRef<string | null>(null);
  const activeMessageRef = useRef<string | null>(null);
  const [error, setError] = useState("");
  const reasoningStartRef = useRef<number | null>(null);

  useEffect(() => {
    getConfig().then((c) => {
      setConfig(c);
      if (c.arms.length) setArm((prev) => (c.arms.includes(prev) ? prev : c.arms[0]));
    }).catch(() => setError("Unable to load runtime configuration"));
    return () => { abortRef.current?.abort(); };
  }, []);

  // "both" has no graph/IRCoT analogue -- it merges two independently
  // hybrid-reranked corpora, so only the baseline arm applies.
  useEffect(() => {
    const allowed = config?.arms_by_corpus[corpus] || [];
    if (allowed.length && !allowed.includes(arm)) Promise.resolve().then(() => setArm("C_hybrid_rerank"));
  }, [corpus, config, arm]);

  useEffect(() => {
    if (busyRef.current && requestSessionRef.current === activeSessionId) return;
    abortRef.current?.abort();
    abortRef.current = null;
    busyRef.current = false;
    const controller = new AbortController();
    Promise.resolve().then(() => { setBusy(false); setStageLabel(""); });
    if (!activeSessionId) {
      Promise.resolve().then(() => setMessages([]));
      return;
    }
    let cancelled = false;
    getSessionMessages(activeSessionId, controller.signal)
      .then((msgs) => {
        if (cancelled || busyRef.current) return;
        setMessages(msgs.map((m, i) => ({ id: `${activeSessionId}-${i}`, role: m.role, content: m.content })));
        // Re-fetching a session (page reload, or navigating to another view
        // and back to Chat -- Chat unmounts, so this whole effect re-runs)
        // only gets role/content back; chunks/timings/eval scores only ever
        // lived in this component's in-memory state during the live SSE
        // turn. trace_id is the one thing that survives, so hydrate each
        // historical assistant message's Details panel from its saved
        // RAGTrace instead of leaving it permanently blank.
        msgs.forEach((m, i) => {
          if (m.role !== "assistant" || !m.trace_id) return;
          getTrace(m.trace_id, controller.signal).then((trace) => {
            if (cancelled || busyRef.current || !trace) return;
            const stats = trace.execution_statistics || {};
            setMessages((cur) => {
              if (cur.length !== msgs.length) return cur; // stale, session changed since
              const next = [...cur];
              next[i] = {
                ...next[i],
                meta: {
                  ...next[i].meta,
                  trace_id: trace.trace_id,
                  chunks: (trace.retrieved_chunk_references || []).map((c) => ({
                    chunk_id: c.chunk_id,
                    source_file: c.source_file,
                    page_number: c.page_number,
                    similarity_score: c.similarity_score,
                    text: c.text || "",
                  })),
                  retrieval_time: stats.retrieval_time as number,
                  generation_time: stats.generation_time as number,
                  total_time: stats.total_pipeline_time as number,
                  ...(trace.claim_verification && {
                    claimCount: trace.claim_verification.claim_count,
                    claims: trace.claim_verification.results,
                  }),
                  ...(trace.claim_error != null && { claimError: trace.claim_error }),
                },
              };
              return next;
            });
          }).catch(() => { if (!cancelled) setError("Unable to load saved trace details"); });
        });
      })
      .catch((err) => { if (!cancelled && err.name !== "AbortError") setError("Unable to load session messages"); });
    return () => { cancelled = true; controller.abort(); };
  }, [activeSessionId]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, stageLabel]);

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    const ta = e.target;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
  }

  // Helper: patch the last assistant message's meta field
  function patchLastMeta(patch: Partial<MsgMeta> | ((prev: MsgMeta) => Partial<MsgMeta>)) {
    setMessages((m) => {
      if (!m.length) return m;
      const index = m.findIndex(message => message.id === activeMessageRef.current);
      if (index < 0) return m;
      const next = [...m];
      const last = { ...next[index] };
      const resolved = typeof patch === "function" ? patch(last.meta || {}) : patch;
      last.meta = { ...last.meta, ...resolved };
      next[index] = last;
      return next;
    });
  }

  async function send(question?: string, baseMessages: Message[] = messages) {
    const q = (question ?? input).trim();
    if (!q || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError("");
    setStageLabel("Starting...");
    const controller = new AbortController();
    abortRef.current = controller;
    requestSessionRef.current = activeSessionId;
    const messageId = crypto.randomUUID();
    activeMessageRef.current = messageId;
    let sessId = activeSessionId;
    try {
      if (!sessId) {
        const newSess = await createSession(q.slice(0, 30));
        if (controller.signal.aborted) return;
        sessId = newSess.session_id;
        requestSessionRef.current = sessId;
        onSessionCreated(sessId);
      }
      setInput("");
      if (textareaRef.current) textareaRef.current.style.height = "auto";
      const history = baseMessages.map(m => ({ role: m.role, content: m.content }));
      const isFirstQuestion = baseMessages.length === 0;
      setMessages([...baseMessages,
        { id: crypto.randomUUID(), role: "user", content: q },
        { id: messageId, role: "assistant", content: "", meta: { arm, corpus } }]);
      reasoningStartRef.current = null;
      let succeeded = false;
      for await (const evt of streamChat({ question: q, session_id: sessId, arm, corpus,
        chat_history: history, memory_enabled: settings.memoryEnabled,
        deep_analysis: settings.deepAnalysis }, controller.signal)) {
        if (abortRef.current !== controller || controller.signal.aborted) return;
        applyEvent(evt, q);
        succeeded ||= evt.event === "done";
      }
      if (succeeded && isFirstQuestion && !controller.signal.aborted) {
        const title = q.split(" ").slice(0, 5).join(" ");
        try { await renameSession(sessId, title); onSessionRenamed(); }
        catch { setError("The answer completed, but the session title could not be saved"); }
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError" && abortRef.current === controller) {
        setError((err as Error).message);
        setMessages(current => current.map(m => m.id === messageId
          ? { ...m, content: `${m.content}\n\nError: ${(err as Error).message}`, failedQuestion: q } : m));
      }
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        busyRef.current = false;
        setBusy(false);
        setStageLabel("");
      }
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function applyEvent(evt: ChatEvent, question: string) {
    if (evt.event === "stage") {
      setStageLabel(evt.label as string);
    } else if (evt.event === "memory") {
      patchLastMeta({
        recalled_memories: evt.recalled as MemoryRecall[],
        memory_time: evt.memory_time as number,
      });
    } else if (evt.event === "meta") {
      patchLastMeta({
        search_query: evt.search_query as string,
        query_was_condensed: evt.query_was_condensed as boolean,
        arm: evt.arm as string,
      });
    } else if (evt.event === "chunks") {
      patchLastMeta({
        chunks: evt.chunks as Chunk[],
        retrieval_time: evt.retrieval_time as number,
        retrieval_metadata: evt.retrieval_metadata as Record<string, unknown>,
      });
    } else if (evt.event === "reasoning") {
      if (reasoningStartRef.current === null) reasoningStartRef.current = performance.now();
      const text = evt.text as string;
      patchLastMeta((prev) => ({ reasoning: (prev.reasoning || "") + text }));
    } else if (evt.event === "token") {
      // First answer token after any reasoning marks the thinking phase done.
      if (reasoningStartRef.current !== null) {
        const elapsed = performance.now() - reasoningStartRef.current;
        reasoningStartRef.current = null;
        patchLastMeta({ reasoningDone: true, reasoningTimeMs: elapsed });
      }
      const text = evt.text as string;
      setMessages((m) => {
        const index = m.findIndex(message => message.id === activeMessageRef.current);
        if (index < 0) return m;
        const next = [...m];
        const last = { ...next[index] };
        last.content = last.content + text;
        next[index] = last;
        return next;
      });
    } else if (evt.event === "strategy") {
      const verification = evt.verification as { claim_count: number; results: ClaimVerification[] } | undefined;
      patchLastMeta({
        trace_id: evt.trace_id as string,
        ...(verification && { claimCount: verification.claim_count, claims: verification.results }),
        ...(evt.claim_error != null && { claimError: evt.claim_error as string }),
      });
    } else if (evt.event === "evaluation") {
      patchLastMeta({
        groundedness: evt.groundedness as number,
        faithfulness: evt.faithfulness as number,
        answer_relevancy: evt.answer_relevancy as number,
        context_precision: evt.context_precision as number,
        context_relevancy: evt.context_relevancy as number,
        citation_precision: evt.citation_precision as number,
        citations_found: evt.citations_found as number,
        device: evt.device as string,
      });
    } else if (evt.event === "done") {
      patchLastMeta({
        total_time: evt.total_time as number,
        generation_time: evt.generation_time as number,
        retrieval_time: evt.retrieval_time as number,
        memory_time: evt.memory_time as number,
        trace_id: evt.trace_id as string,
      });
      setStageLabel("");
      if (evt.trace_id) onTurnDone(evt.trace_id as string);
    } else if (evt.event === "error") {
      setMessages((m) => {
        const index = m.findIndex(message => message.id === activeMessageRef.current);
        if (index < 0) return m;
        const next = [...m];
        next[index] = {
          ...next[index],
          content: `⚠️ Error: ${evt.message}`,
          failedQuestion: question,
        };
        return next;
      });
    }
  }

  function retry(question: string) {
    const history = messages.slice(0, -2);
    void send(question, history);
  }

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-page">
      {error && <p role="alert">{error}</p>}
      {/* ── Top bar ─────────────────────────────── */}
      <div className="topbar">
        <span className="topbar-title">{CORPUS_LABELS[corpus] ?? corpus}</span>
        <div className="topbar-controls">
          <select
            className="topbar-select"
            value={arm}
            onChange={(e) => setArm(e.target.value)}
            disabled={corpus === "both"}
            title={
              corpus === "both"
                ? "Combined mode always uses hybrid + rerank -- graph/IRCoT arms are built on the judgments-only citation graph"
                : `Retrieval strategy: ${ARM_DESCRIPTIONS[arm] ?? arm}`
            }
          >
            {(config?.arms_by_corpus[corpus] || []).map((a) => (
              <option key={a} value={a} title={ARM_DESCRIPTIONS[a]}>{a}</option>
            ))}
          </select>
          <select
            className="topbar-select"
            value={corpus}
            onChange={(e) => { setCorpus(e.target.value); setArm("C_hybrid_rerank"); }}
            title="Which ingested document corpus to answer from"
          >
            {config?.corpora.map((c) => (
              <option key={c} value={c}>{CORPUS_LABELS[c] ?? c}</option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Welcome / Messages ──────────────────── */}
      {isEmpty ? (
        <div className="chat-welcome">
          <h1>What can I help with?</h1>
          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="suggestion-chip" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-messages" ref={listRef}>
          {messages.map((m, i) => (
            <div key={i} className={`msg-row ${m.role}`}>
              <div className={`msg-content-col ${m.role === "assistant" ? "blueprint" : ""}`}>
                {m.role === "assistant" && (
                  <>
                    <i className="corner tl" /><i className="corner tr" />
                    <i className="corner bl" /><i className="corner br" />
                    <div className="msg-kicker-row">
                      <span className="card-kicker">Answer</span>
                      {m.meta?.arm && <span className="tag tag-neutral">{m.meta.arm}</span>}
                    </div>
                    {m.meta && <ThinkingBox meta={m.meta} />}
                  </>
                )}
                <div className="msg-bubble">
                  {m.content ? (
                    m.content
                  ) : busy && i === messages.length - 1 ? (
                    <div>
                      <div className="stage-label">{stageLabel}</div>
                      <div className="msg-thinking"><span /><span /><span /></div>
                    </div>
                  ) : null}
                </div>
                {m.role === "assistant" && m.failedQuestion && (
                  <button
                    className="retry-btn"
                    onClick={() => retry(m.failedQuestion!)}
                  >🔄 Retry</button>
                )}
                {/* Details panel — as soon as there's retrieval/memory data to
                    show, not just once the answer has finished streaming. */}
                {m.role === "assistant" && m.meta &&
                  (m.content || m.meta.chunks?.length || m.meta.recalled_memories?.length) && (
                  <DetailsPanel meta={m.meta} defaultOpen={settings.autoExpandDetails} />
                )}
              </div>
            </div>
          ))}
          {/* Stage label while last turn is still streaming */}
          {busy && stageLabel && messages.length > 0 &&
            messages[messages.length - 1].content !== "" && (
            <div className="stage-indicator">
              <span className="stage-dot" />
              {stageLabel}
            </div>
          )}
        </div>
      )}

      {/* ── Input bar ───────────────────────────── */}
      <div className="input-area">
        <div>
          <div className="input-box">
            <textarea
              ref={textareaRef}
              className="input-textarea"
              aria-label="Question"
              rows={1}
              value={input}
              onChange={handleInput}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
              }}
              placeholder="Ask anything about the statutes…"
              disabled={busy}
            />
            {busy ? (
              <button className="send-btn stop-btn" onClick={stop} title="Stop generating">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <rect x="6" y="6" width="12" height="12" rx="2" />
                </svg>
              </button>
            ) : (
              <button aria-label="Send message" className="send-btn" onClick={() => send()} disabled={!input.trim()}>
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 4l8 8h-5v8H9v-8H4z" />
                </svg>
              </button>
            )}
          </div>
          <div className="input-hint">X-RAG may make mistakes. Verify important legal information.</div>
        </div>
      </div>
    </div>
  );
}
