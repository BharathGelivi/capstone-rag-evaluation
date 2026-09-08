export const API_BASE = "http://127.0.0.1:8010";

export interface Config {
  arms: string[];
  corpora: string[];
}

export async function getConfig(): Promise<Config> {
  const r = await fetch(`${API_BASE}/ui/config`);
  return r.json();
}

export interface ChatEvent {
  event: "stage" | "memory" | "meta" | "chunks" | "reasoning" | "token" | "strategy" | "evaluation" | "done" | "error";
  [key: string]: unknown;
}

export interface ChatRequest {
  question: string;
  session_id: string;
  arm: string;
  corpus: string;
  chat_history: { role: string; content: string }[];
  memory_enabled: boolean;
  deep_analysis: boolean;
}

/**
 * Stream a chat turn's SSE events. Reads split across chunk boundaries are
 * common with fetch + ReadableStream, so buffer until the last "\n\n" frame
 * separator rather than assuming one read == one event.
 */
export async function* streamChat(req: ChatRequest, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API_BASE}/ui/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.body) return;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      if (frame.startsWith("data: ")) {
        yield JSON.parse(frame.slice(6)) as ChatEvent;
      }
    }
  }
}

export interface GraphData {
  nodes: Array<{
    id: string;
    type: string;
    label: string;
    degree: number;
    detail?: string;
    meta?: string;
  }>;
  edges: Array<{ source: string; target: string; kind: string }>;
  truncated: boolean;
}

export async function getGraph(scope: "rag" | "legal" | "memory"): Promise<GraphData> {
  const r = await fetch(`${API_BASE}/ui/graph?scope=${scope}`);
  return r.json();
}

export interface Session {
  session_id: string;
  title: string;
  created_at: string;
  last_activity: string;
  interaction_count: number;
}

export async function getSessions(): Promise<Session[]> {
  const r = await fetch(`${API_BASE}/ui/sessions`);
  return r.json();
}

export async function createSession(title: string = "New Session"): Promise<Session> {
  const r = await fetch(`${API_BASE}/ui/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return r.json();
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API_BASE}/ui/sessions/${id}`, { method: "DELETE" });
}

export async function renameSession(id: string, title: string): Promise<void> {
  await fetch(`${API_BASE}/ui/sessions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
}

export async function getSessionMessages(
  id: string
): Promise<{ role: "user" | "assistant"; content: string; trace_id?: string | null }[]> {
  const r = await fetch(`${API_BASE}/ui/sessions/${id}/messages`);
  return r.json();
}

export interface MemoryEntry {
  memory_id: string;
  session_id: string;
  question: string;
  answer: string;
  timestamp: string;
  importance_score: number;
  access_count: number;
  trace_id: string | null;
  retrieved_chunk_ids: string[];
  claim_ids: string[];
  tags: string[];
}

export interface ScoredMemory {
  memory: MemoryEntry;
  semantic_score: number;
  recency_score: number;
  frequency_score: number;
  importance_score: number;
  final_score: number;
  retrieval_reason: string;
}

export async function getMemory(
  opts: { search?: string; sessionId?: string; limit?: number } = {}
): Promise<{ memories: ScoredMemory[]; searched: boolean }> {
  const params = new URLSearchParams();
  if (opts.search) params.set("search", opts.search);
  if (opts.sessionId) params.set("session_id", opts.sessionId);
  if (opts.limit) params.set("limit", String(opts.limit));
  const r = await fetch(`${API_BASE}/ui/memory?${params}`);
  return r.json();
}

export interface TraceChunkReference {
  chunk_id: string;
  rank: number;
  similarity_score: number;
  page_number: string | number;
  source_file: string;
  text?: string;
}

export interface ClaimVerification {
  claim_id: string;
  claim_text: string;
  status: "SUPPORTED" | "PARTIALLY_SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED" | "NOT_VERIFIABLE";
  reason: string;
  confidence: number;
  entailment_score: number;
  contradiction_score: number;
  neutral_score: number;
  best_chunk_id: string | null;
  best_chunk_rank: number | null;
  evidence_text: string | null;
  verified_by: "nli" | "llm";
}

export interface TraceData {
  trace_id: string;
  question: string;
  timestamp: string;
  prompt_snapshot: string;
  execution_statistics: Record<string, unknown>;
  configuration_snapshot: Record<string, unknown>;
  retrieved_chunk_references: TraceChunkReference[];
  // Only present when the turn ran with deep_analysis=true -- see
  // chat_service.py's post-hoc patch of the saved trace file.
  claim_verification?: { claim_count: number; results: ClaimVerification[] };
  claim_error?: string;
}

export async function getLatestTrace(): Promise<TraceData | null> {
  const r = await fetch(`${API_BASE}/ui/trace/latest`);
  if (!r.ok) return null;
  return r.json();
}

export async function getTrace(traceId: string): Promise<TraceData | null> {
  const r = await fetch(`${API_BASE}/ui/trace/${traceId}`);
  if (!r.ok) return null;
  return r.json();
}

