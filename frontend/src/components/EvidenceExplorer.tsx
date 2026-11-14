import { useState } from "react";
import type { TraceData } from "../lib/api";

const verdictLabels = {
  SUPPORTED: "Supported",
  PARTIALLY_SUPPORTED: "Partially supported",
  CONTRADICTED: "Contradicted",
  UNSUPPORTED: "Unsupported",
  NOT_VERIFIABLE: "Not verifiable",
} as const;

export default function EvidenceExplorer({ trace }: { trace: TraceData | null }) {
  const [filter, setFilter] = useState("ALL");
  const claims = trace?.claim_verification?.results ?? [];
  return <section className="card elev-sm" aria-label="Claim evidence explorer">
    <h2>Claim evidence explorer</h2>
    <p>Inspect each claim against the evidence used by the verifier. Scores are model outputs, not proof of correctness.</p>
    {trace?.claim_error && <p role="alert">Verification failed: {trace.claim_error}</p>}
    {!claims.length && <p>No verified claims available. Enable Deep analysis above, then ask a question in Chat.</p>}
    {!!claims.length && <>
      <div role="group" aria-label="Verdict counts for all claims" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
        {Object.entries(verdictLabels).map(([status, label]) => (
          <span key={status} className="dpill">
            {claims.filter(claim => claim.status === status).length} {label}
          </span>
        ))}
      </div>
      <label>Verdict <select value={filter} onChange={e => setFilter(e.target.value)}>
        <option value="ALL">All claims ({claims.length})</option>
        {[...new Set(claims.map(c => c.status))].map(s => <option key={s} value={s}>{s.replaceAll("_", " ")}</option>)}
      </select></label>
      {claims.filter(c => filter === "ALL" || c.status === filter).map(c => {
        const source = trace?.retrieved_chunk_references.find(r => r.chunk_id === c.best_chunk_id);
        const evidence = c.evidence_text || source?.text;
        return <details key={c.claim_id} style={{ marginTop: 16, padding: 16, border: "1px solid #94a3b8", borderRadius: 12 }}>
          <summary style={{ cursor: "pointer" }}><strong>{c.status.replaceAll("_", " ")}</strong> - {c.claim_text}</summary>
          <p>{c.reason || "No explanation supplied by the verifier."}</p>
          <p><strong>Evidence</strong> � {source?.source_file || c.best_chunk_id || "No source identified"}{source?.page_number != null ? ` � Page ${source.page_number}` : ""}</p>
          {evidence ? <blockquote style={{ borderLeft: "3px solid #10a37f", paddingLeft: 16, whiteSpace: "pre-wrap" }}>{evidence}</blockquote> : <p>No evidence text available. This does not establish that the claim is false.</p>}
          <table style={{ width: "100%", textAlign: "left" }}><thead><tr><th>Model signal</th><th>Score</th></tr></thead><tbody>
            {[["Entailment", c.entailment_score], ["Contradiction", c.contradiction_score], ["Neutral", c.neutral_score]].map(([label, value]) => <tr key={String(label)}><td>{label}</td><td>{typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "Unavailable"}</td></tr>)}
          </tbody></table>
          <p>Verified by: {c.verified_by === "llm" ? "LLM judge" : "NLI model"}</p>
        </details>;
      })}
    </>}
  </section>;
}
