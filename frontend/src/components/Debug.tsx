import { useEffect, useState } from "react";
import { getLatestTrace, getTrace, type TraceData } from "../lib/api";
import type { RuntimeSettings } from "../App";

interface DebugProps {
  settings: RuntimeSettings;
  onSettingsChange: (s: RuntimeSettings) => void;
  lastTraceId: string | null;
}

function toggle(label: string, on: boolean, onClick: () => void) {
  return (
    <div key={label} className="debug-toggle-row">
      <span>{label}</span>
      <button role="switch" aria-checked={on} aria-label={label} className={`debug-toggle ${on ? "on" : ""}`} onClick={onClick}>
        <span className="debug-toggle-knob" />
      </button>
    </div>
  );
}

export default function Debug({ settings, onSettingsChange, lastTraceId }: DebugProps) {
  const [trace, setTrace] = useState<TraceData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    Promise.resolve().then(() => { if (!controller.signal.aborted) { setLoading(true); setError(""); } });
    const load = lastTraceId ? getTrace(lastTraceId, controller.signal) : getLatestTrace(controller.signal);
    load.then(value => { if (!controller.signal.aborted) setTrace(value); })
      .catch(err => { if (err.name !== "AbortError") setError("Unable to load trace"); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [lastTraceId]);

  const stats = (trace?.execution_statistics || {}) as Record<string, number>;
  const total = stats.total_pipeline_time || (stats.retrieval_time || 0) + (stats.generation_time || 0) || 1;
  const traceRows: Array<[string, number | undefined]> = [
    ["Retrieval", stats.retrieval_time],
    ["Generation", stats.generation_time],
  ];

  return (
    <div className="debug-page xrag-scroll">
      <span className="topbar-title">Debug</span>
      {error && <p role="alert">{error}</p>}

      <div className="card elev-sm">
        <div className="card-kicker">Runtime toggles</div>
        <div className="debug-toggles">
          {toggle("Memory recall", settings.memoryEnabled, () =>
            onSettingsChange({ ...settings, memoryEnabled: !settings.memoryEnabled }))}
          {toggle("Deep analysis (claim verification)", settings.deepAnalysis, () =>
            onSettingsChange({ ...settings, deepAnalysis: !settings.deepAnalysis }))}
          {toggle("Auto-expand answer details", settings.autoExpandDetails, () =>
            onSettingsChange({ ...settings, autoExpandDetails: !settings.autoExpandDetails }))}
        </div>
      </div>

      <div className="card elev-sm">
        <div className="card-kicker">
          {trace ? `Last trace — ${trace.trace_id}` : loading ? "Last trace" : "Last trace — none recorded yet"}
        </div>
        {trace && (
          <div className="debug-trace-rows">
            {traceRows.map(([label, val]) => (
              <div key={label} className="debug-trace-row">
                <span className="debug-trace-label">{label}</span>
                <div className="debug-trace-track">
                  <div className="debug-trace-fill" style={{ width: `${Math.min(100, ((val || 0) / total) * 100)}%` }} />
                </div>
                <span className="debug-trace-time">{val != null ? (val < 1 ? `${Math.round(val * 1000)} ms` : `${val.toFixed(2)} s`) : "—"}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="card elev-sm">
        <div className="card-kicker">Prompt sent to model</div>
        <pre className="debug-pre">{trace?.prompt_snapshot || (loading ? "Loading…" : "No trace available.")}</pre>
      </div>

      <div className="card elev-sm">
        <div className="card-kicker">Raw retrieval metadata</div>
        <pre className="debug-pre">
          {trace ? JSON.stringify(trace.retrieved_chunk_references, null, 2) : (loading ? "Loading…" : "No trace available.")}
        </pre>
      </div>
    </div>
  );
}
