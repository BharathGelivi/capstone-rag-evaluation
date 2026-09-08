import { useEffect, useRef, useState } from "react";
import { getGraph, type GraphData } from "../lib/api";

// Union of both graphs' node types (RAG provenance graph + claude-mem memory
// graph) so one legend/color map covers either scope.
const NODE_TYPES: Record<string, { label: string; color: string }> = {
  document: { label: "Document", color: "#f59e0b" },
  chunk: { label: "Chunk", color: "#3b82f6" },
  session: { label: "Session", color: "#ec4899" },
  question: { label: "Question", color: "#7c3aed" },
  claim: { label: "Claim", color: "#10b981" },
  observation: { label: "Observation", color: "#3b82f6" },
  concept: { label: "Concept", color: "#10b981" },
  file: { label: "File", color: "#f59e0b" },
  // Legal citation graph (scope="legal") -- a separate scope, never mixed
  // with the provenance/memory types above on the same render.
  case: { label: "Case", color: "#f59e0b" },
  court: { label: "Court", color: "#ec4899" },
  statute: { label: "Statute", color: "#7c3aed" },
  section: { label: "Section", color: "#3b82f6" },
  article: { label: "Article", color: "#10b981" },
};

interface Node {
  id: string;
  type: string;
  label: string;
  degree: number;
  detail?: string;
  meta?: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
}

interface Link {
  s: Node;
  t: Node;
  kind: string;
}

/**
 * Force-directed canvas graph, ported from the vanilla JS in
 * ui/components/graph.py's render_graph_html() (physics, pan/zoom, search,
 * click-to-filter legend, hover/click detail panel). Kept as hand-rolled
 * canvas rather than a library (e.g. react-force-graph-2d) because the
 * working implementation already exists there.
 */
export default function Graph() {
  const [scope, setScope] = useState<"rag" | "legal" | "memory">("rag");
  const [data, setData] = useState<GraphData | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [info, setInfo] = useState<Node | null>(null);
  const [stats, setStats] = useState("");

  useEffect(() => {
    setInfo(null);
    getGraph(scope).then(setData);
  }, [scope]);

  useEffect(() => {
    if (!data || !canvasRef.current || !rootRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const root = rootRef.current;

    let rect = root.getBoundingClientRect();
    function size() {
      rect = root.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    size();

    const byId: Record<string, Node> = {};
    const nodes: Node[] = data!.nodes
      .filter((n) => !hidden.has(n.type))
      .map((n) => {
        const o: Node = {
          ...n,
          x: rect.width / 2 + (Math.random() - 0.5) * 380,
          y: rect.height / 2 + (Math.random() - 0.5) * 380,
          vx: 0,
          vy: 0,
          r: Math.min(16, 4.5 + Math.sqrt(n.degree || 1) * 1.9),
        };
        byId[n.id] = o;
        return o;
      });
    const links: Link[] = data!.edges
      .map((e) => ({ s: byId[e.source], t: byId[e.target], kind: e.kind }))
      .filter((e) => e.s && e.t);

    setStats(`${nodes.length} nodes · ${links.length} edges${data!.truncated ? " (sample)" : ""}`);
    // testability hook, per the e2e plan
    (window as unknown as Record<string, unknown>).__graphNodeCount = nodes.length;

    let alpha = 1.0;
    let tx = 0,
      ty = 0,
      scale = 1;
    let dragNode: Node | null = null,
      panning = false,
      lastX = 0,
      lastY = 0;
    let hoverNode: Node | null = null,
      selected: Node | null = null;
    let raf = 0;

    function step() {
      if (alpha < 0.005) return;
      alpha *= 0.994;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          let dx = b.x - a.x,
            dy = b.y - a.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1) {
            dx = Math.random() - 0.5;
            dy = Math.random() - 0.5;
            d2 = 1;
          }
          if (d2 > 90000) continue;
          const f = 900 / d2;
          const d = Math.sqrt(d2);
          const fx = (dx / d) * f,
            fy = (dy / d) * f;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }
      }
      for (const l of links) {
        const dx = l.t.x - l.s.x,
          dy = l.t.y - l.s.y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        const f = (d - 62) * 0.0038;
        const fx = (dx / d) * f,
          fy = (dy / d) * f;
        l.s.vx += fx;
        l.s.vy += fy;
        l.t.vx -= fx;
        l.t.vy -= fy;
      }
      const cx = rect.width / 2,
        cy = rect.height / 2;
      for (const n of nodes) {
        n.vx += (cx - n.x) * 0.0016;
        n.vy += (cy - n.y) * 0.0016;
        if (n === dragNode) continue;
        n.vx *= 0.86;
        n.vy *= 0.86;
        n.x += n.vx * alpha * 5.5;
        n.y += n.vy * alpha * 5.5;
      }
    }

    function matches(n: Node) {
      return query && (n.label + " " + (n.meta || "") + " " + (n.detail || "")).toLowerCase().includes(query.toLowerCase());
    }

    function neighbours(n: Node | null) {
      const s = new Set<Node>();
      if (!n) return s;
      for (const l of links) {
        if (l.s === n) s.add(l.t);
        if (l.t === n) s.add(l.s);
      }
      return s;
    }

    function draw() {
      ctx.save();
      ctx.clearRect(0, 0, rect.width, rect.height);
      ctx.translate(tx, ty);
      ctx.scale(scale, scale);
      const focus = selected || hoverNode;
      const near = neighbours(focus);

      for (const l of links) {
        const active = !!(focus && (l.s === focus || l.t === focus));
        ctx.strokeStyle = active ? "rgba(0,0,0,0.55)" : "rgba(0,0,0,0.12)";
        ctx.lineWidth = (active ? 1.8 : 0.7) / scale;
        ctx.beginPath();
        ctx.moveTo(l.s.x, l.s.y);
        ctx.lineTo(l.t.x, l.t.y);
        ctx.stroke();
      }

      for (const n of nodes) {
        const dim = !!((focus && n !== focus && !near.has(n)) || (query && !matches(n)));
        const color = (NODE_TYPES[n.type] || {}).color || "#94a3b8";
        ctx.globalAlpha = dim ? 0.18 : 1;

        if (!dim && (n === focus || matches(n))) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, n.r + 6 / scale, 0, Math.PI * 2);
          ctx.fillStyle = color + "33";
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 1.4 / scale;
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.stroke();

        if (!dim && (scale > 0.75 || n.r > 9 || n === focus)) {
          ctx.globalAlpha = dim ? 0.2 : 0.92;
          ctx.fillStyle = "#1a1a1a";
          ctx.font = `${10.5 / scale}px 'Segoe UI',system-ui,sans-serif`;
          ctx.textAlign = "center";
          ctx.fillText(n.label, n.x, n.y + n.r + 11 / scale);
        }
        ctx.globalAlpha = 1;
      }
      ctx.restore();
    }

    function loop() {
      step();
      draw();
      raf = requestAnimationFrame(loop);
    }

    function toWorld(ev: MouseEvent) {
      const r = canvas.getBoundingClientRect();
      return { x: (ev.clientX - r.left - tx) / scale, y: (ev.clientY - r.top - ty) / scale };
    }
    function pick(p: { x: number; y: number }) {
      let best: Node | null = null,
        bestD = Infinity;
      for (const n of nodes) {
        const d = Math.hypot(n.x - p.x, n.y - p.y);
        if (d < n.r + 6 && d < bestD) {
          best = n;
          bestD = d;
        }
      }
      return best;
    }

    const onDown = (ev: MouseEvent) => {
      const p = toWorld(ev);
      dragNode = pick(p);
      if (!dragNode) {
        panning = true;
        lastX = ev.clientX;
        lastY = ev.clientY;
      }
    };
    const onMove = (ev: MouseEvent) => {
      const p = toWorld(ev);
      if (dragNode) {
        dragNode.x = p.x;
        dragNode.y = p.y;
        dragNode.vx = 0;
        dragNode.vy = 0;
        alpha = Math.max(alpha, 0.35);
      } else if (panning) {
        tx += ev.clientX - lastX;
        ty += ev.clientY - lastY;
        lastX = ev.clientX;
        lastY = ev.clientY;
      } else {
        hoverNode = pick(p);
      }
    };
    const onUp = (ev: MouseEvent) => {
      if (dragNode && Math.abs(dragNode.vx) < 1e-6) {
        selected = dragNode;
        setInfo(dragNode);
      } else if (panning) {
        const p = toWorld(ev);
        if (!pick(p)) {
          selected = null;
          setInfo(null);
        }
      }
      dragNode = null;
      panning = false;
    };
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const r = canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left,
        my = ev.clientY - r.top;
      const k = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
      const ns = Math.min(4, Math.max(0.18, scale * k));
      tx = mx - (mx - tx) * (ns / scale);
      ty = my - (my - ty) * (ns / scale);
      scale = ns;
    };
    const onResize = () => size();

    canvas.addEventListener("mousedown", onDown);
    canvas.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    window.addEventListener("resize", onResize);

    loop();
    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("mousedown", onDown);
      canvas.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      canvas.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
    };
  }, [data, hidden, query]);

  const counts: Record<string, number> = {};
  data?.nodes.forEach((n) => (counts[n.type] = (counts[n.type] || 0) + 1));

  return (
    <div className="graph-page">
      <div className="graph-toolbar">
        <select value={scope} onChange={(e) => setScope(e.target.value as "rag" | "legal" | "memory")}>
          <option value="rag">RAG provenance</option>
          <option value="legal">Legal citation graph</option>
          <option value="memory">Claude-mem memory</option>
        </select>
        <input placeholder="Search nodes…" value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>
      <div className="graph-canvas-wrap blueprint" ref={rootRef}>
        <i className="corner tl" /><i className="corner tr" />
        <i className="corner bl" /><i className="corner br" />
        <canvas ref={canvasRef} />
        <div className="graph-legend">
          {Object.entries(NODE_TYPES)
            .filter(([t]) => counts[t])
            .map(([t, meta]) => (
              <div
                key={t}
                className="legend-row"
                style={{ opacity: hidden.has(t) ? 0.35 : 1, cursor: "pointer" }}
                onClick={() =>
                  setHidden((h) => {
                    const next = new Set(h);
                    next.has(t) ? next.delete(t) : next.add(t);
                    return next;
                  })
                }
              >
                <span className="legend-dot" style={{ background: meta.color }} />
                <span>{meta.label}</span>
                <span className="legend-count">{counts[t]}</span>
              </div>
            ))}
        </div>
        {info && (
          <div className="graph-info">
            <strong>{(NODE_TYPES[info.type] || {}).label || info.type}</strong> {info.label}
            {info.meta && <div className="meta">{info.meta}</div>}
            {info.detail && <div className="detail">{info.detail}</div>}
          </div>
        )}
        <div className="graph-stats">{stats}</div>
      </div>
    </div>
  );
}
