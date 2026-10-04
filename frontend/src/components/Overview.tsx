import { useEffect, useState } from "react";
import { getConfig, type Config } from "../lib/api";
import type { Tab } from "../App";

interface OverviewProps {
  sessionCount: number;
  onNavigate: (tab: Tab) => void;
}

const CARDS: Array<{ tab: Tab; title: string; desc: string; icon: React.ReactNode }> = [
  {
    tab: "chat", title: "Chat",
    desc: "Ask questions across the available retrieval arms and read grounded, citable answers.",
    icon: <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />,
  },
  {
    tab: "graph", title: "Knowledge Graph",
    desc: "Trace provenance from document to chunk to question to claim.",
    icon: <><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></>,
  },
  {
    tab: "memory", title: "Memory",
    desc: "Browse recalled past questions, scored by recency, frequency and importance.",
    icon: <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></>,
  },
  {
    tab: "debug", title: "Debug",
    desc: "Inspect raw prompts, retrieval metadata and runtime toggles.",
    icon: <><rect x="7" y="8" width="10" height="10" rx="1" />
      <path d="M12 2v4M8 8 6 5M16 8l2-3M9 18l-2 3M15 18l2 3M4 13H2M22 13h-2" /></>,
  },
];

export default function Overview({ sessionCount, onNavigate }: OverviewProps) {
  const [config, setConfig] = useState<Config | null>(null);

  useEffect(() => {
    getConfig().then(setConfig).catch(() => setConfig(null));
  }, []);

  const stats = [
    { label: "Sessions", value: String(sessionCount) },
    { label: "Retrieval arms", value: String(config?.arms.length ?? "—") },
    { label: "Corpora", value: String(config?.corpora.length ?? "—") },
  ];

  return (
    <div className="overview-page">
      <span className="tag tag-outline">Diagnostic framework</span>
      <h1 className="overview-title">Inspect how X-RAG answers, not just what it answers.</h1>
      <p className="overview-lede">
        {config ? `${config.arms.length} retrieval strategies` : "Retrieval strategies"}, one interface. Trace every answer back to its chunks, its
        recalled memories, and its timing — across the statutes and judgments corpora.
      </p>

      <div className="overview-cards">
        {CARDS.map((c) => (
          <div role="button" tabIndex={0} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") onNavigate(c.tab); }} key={c.tab} className="overview-card blueprint" onClick={() => onNavigate(c.tab)}>
            <i className="corner tl" /><i className="corner tr" />
            <i className="corner bl" /><i className="corner br" />
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-700)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              {c.icon}
            </svg>
            <div className="card-title">{c.title}</div>
            <p className="card-body">{c.desc}</p>
            <span className="overview-card-open">Open →</span>
          </div>
        ))}
      </div>

      <div className="overview-stats">
        {stats.map((s) => (
          <div key={s.label} className="card elev-sm">
            <div className="card-kicker">{s.label}</div>
            <div className="overview-stat-value">{s.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
