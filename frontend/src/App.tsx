import { useEffect, useState } from "react";
import Chat from "./components/Chat";
import Graph from "./components/Graph";
import Overview from "./components/Overview";
import Memory from "./components/Memory";
import Debug from "./components/Debug";
import { createSession, deleteSession, getSessions, type Session } from "./lib/api";
import "./App.css";

export type Tab = "overview" | "chat" | "graph" | "memory" | "debug";

export interface RuntimeSettings {
  memoryEnabled: boolean;
  deepAnalysis: boolean;
  autoExpandDetails: boolean;
}

const NAV: Array<{ id: Tab; label: string; icon: React.ReactNode }> = [
  {
    id: "overview", label: "Overview",
    icon: <path d="M3 10.5 12 3l9 7.5M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5" />,
  },
  {
    id: "chat", label: "Chat",
    icon: <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />,
  },
  {
    id: "graph", label: "Knowledge Graph",
    icon: <><circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" /></>,
  },
  {
    id: "memory", label: "Memory",
    icon: <><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></>,
  },
  {
    id: "debug", label: "Debug",
    icon: <><rect x="7" y="8" width="10" height="10" rx="1" />
      <path d="M12 2v4M8 8 6 5M16 8l2-3M9 18l-2 3M15 18l2 3M4 13H2M22 13h-2" /></>,
  },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [lastTraceId, setLastTraceId] = useState<string | null>(null);
  const [settings, setSettings] = useState<RuntimeSettings>({
    memoryEnabled: true, deepAnalysis: true, autoExpandDetails: false,
  });

  const loadSessions = async () => {
    try {
      const list = await getSessions();
      setSessions(list);
      if (list.length > 0 && !activeSessionId) {
        setActiveSessionId(list[0].session_id);
      }
    } catch (e) {
      console.error("Failed to load sessions:", e);
    }
  };

  useEffect(() => {
    loadSessions();
  }, []);

  const handleNewChat = async () => {
    try {
      const newSession = await createSession("New Session");
      setSessions((prev) => [newSession, ...prev]);
      setActiveSessionId(newSession.session_id);
      setTab("chat");
    } catch (e) {
      console.error("Failed to create session:", e);
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await deleteSession(id);
      const updated = sessions.filter((s) => s.session_id !== id);
      setSessions(updated);
      if (activeSessionId === id) {
        setActiveSessionId(updated.length > 0 ? updated[0].session_id : null);
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  };

  return (
    <div className="app">
      {/* ── Sidebar ─────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header" style={{ cursor: "pointer" }} onClick={() => setTab("overview")}>
          <span className="sidebar-logo">X-RAG</span>
          <span className="sidebar-tagline">Diagnostic</span>
        </div>

        <button className="sidebar-new-btn" onClick={handleNewChat}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 5v14M5 12h14" />
          </svg>
          New session
        </button>

        <span className="sidebar-section-label">Sessions</span>
        <div className="sidebar-sessions-list">
          {sessions.map((s) => (
            <div
              key={s.session_id}
              className={`sidebar-session-item ${activeSessionId === s.session_id && tab === "chat" ? "active" : ""}`}
              onClick={() => {
                setActiveSessionId(s.session_id);
                setTab("chat");
              }}
            >
              <span className="sidebar-session-title">{s.title || "Untitled Chat"}</span>
              <button
                className="sidebar-session-delete"
                onClick={(e) => handleDeleteSession(e, s.session_id)}
                title="Delete session"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>
            </div>
          ))}
        </div>

        <span className="sidebar-section-label">Views</span>
        {NAV.map((n) => (
          <button
            key={n.id}
            className={`sidebar-nav-item ${tab === n.id ? "active" : ""}`}
            onClick={() => setTab(n.id)}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              {n.icon}
            </svg>
            {n.label}
          </button>
        ))}

        <div className="sidebar-footer">
          Device: <span>RTX 4060 · CUDA</span>
        </div>
      </aside>

      {/* ── Main content ────────────────────────── */}
      <div className="main">
        {tab === "overview" && <Overview sessionCount={sessions.length} onNavigate={setTab} />}
        {tab === "chat" && (
          <Chat
            activeSessionId={activeSessionId}
            onSessionCreated={(id) => {
              setActiveSessionId(id);
              loadSessions();
            }}
            onSessionRenamed={loadSessions}
            settings={settings}
            onTurnDone={setLastTraceId}
          />
        )}
        {tab === "graph" && <Graph />}
        {tab === "memory" && <Memory />}
        {tab === "debug" && (
          <Debug settings={settings} onSettingsChange={setSettings} lastTraceId={lastTraceId} />
        )}
      </div>
    </div>
  );
}
