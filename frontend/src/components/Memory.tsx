import { useEffect, useState } from "react";
import { getMemory, type ScoredMemory } from "../lib/api";

function scoreBar(v: number, color = "var(--color-accent)") {
  const pct = Math.min(100, Math.round(v * 100));
  return (
    <div className="mem-score-row">
      <div className="mem-score-track"><div className="mem-score-fill" style={{ width: `${pct}%`, background: color }} /></div>
      <span className="mem-score-val">{v.toFixed(2)}</span>
    </div>
  );
}

export default function Memory() {
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [memories, setMemories] = useState<ScoredMemory[]>([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const t = setTimeout(() => setDebounced(search), 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    setLoading(true);
    getMemory({ search: debounced || undefined, limit: 30 })
      .then((r) => { setMemories(r.memories); setSearched(r.searched); })
      .catch(() => setMemories([]))
      .finally(() => setLoading(false));
  }, [debounced]);

  return (
    <div className="memory-page">
      <div className="topbar">
        <span className="topbar-title">Memory</span>
        <div className="memory-search">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="11" cy="11" r="8" /><path d="M21 21l-4.3-4.3" /></svg>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search memories…"
          />
        </div>
      </div>

      <div className="memory-grid xrag-scroll">
        {!loading && memories.length === 0 && (
          <p className="text-muted">
            {search ? `No memories match "${search}".` : "No memories yet. Start a conversation to create some."}
          </p>
        )}
        {memories.map((sr) => (
          <div key={sr.memory.memory_id} className="memory-page-card blueprint">
            <i className="corner tl" /><i className="corner tr" />
            <i className="corner bl" /><i className="corner br" />
            <div className="memory-card-head">
              <span className="tag tag-accent">{sr.memory.session_id.slice(0, 12)}</span>
              <span className="memory-card-time">{new Date(sr.memory.timestamp).toLocaleString()}</span>
            </div>
            <div className="memory-card-q"><strong>Q:</strong> {sr.memory.question}</div>
            <div className="memory-card-a"><strong>A:</strong> {sr.memory.answer.slice(0, 240)}{sr.memory.answer.length > 240 ? "…" : ""}</div>
            {searched ? (
              <div className="memory-card-scores">
                <div>Semantic{scoreBar(sr.semantic_score, "#10a37f")}</div>
                <div>Recency{scoreBar(sr.recency_score, "#3b82f6")}</div>
                <div>Frequency{scoreBar(sr.frequency_score, "#f59e0b")}</div>
                <div>Final{scoreBar(sr.final_score, "var(--color-accent-900)")}</div>
              </div>
            ) : (
              <div className="memory-card-scores">
                <div>Importance{scoreBar(sr.memory.importance_score)}</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
