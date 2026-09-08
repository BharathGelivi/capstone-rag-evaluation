"""
Interactive knowledge-graph renderer.

Builds a provenance graph over the RAG artifacts — documents, chunks, sessions,
questions and verified claims — and renders it as a self-contained force-directed
canvas widget.

The physics and rendering are hand-rolled vanilla JS rather than pulled from a
CDN: Streamlit components are sandboxed iframes and a CDN fetch fails silently
when the machine is offline or behind a proxy, which would leave a blank panel
with no error. Everything here ships inline.
"""

from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional

# Node type -> (display label, colour). Kept in one place so the legend, the
# filter chips and the canvas can never drift out of sync.
NODE_TYPES: Dict[str, Dict[str, str]] = {
    "document": {"label": "Document", "color": "#f59e0b"},
    "chunk": {"label": "Chunk", "color": "#3b82f6"},
    "session": {"label": "Session", "color": "#ec4899"},
    "question": {"label": "Question", "color": "#7c3aed"},
    "claim": {"label": "Claim", "color": "#10b981"},
    # Legal citation graph (src/legal_graph.py) -- a different scope, never
    # mixed with the provenance types above on the same render.
    "case": {"label": "Case", "color": "#f59e0b"},
    "court": {"label": "Court", "color": "#ec4899"},
    "statute": {"label": "Statute", "color": "#7c3aed"},
    "section": {"label": "Section", "color": "#3b82f6"},
    "article": {"label": "Article", "color": "#10b981"},
}


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_graph_data(
    registry: Any = None,
    memories: Optional[List[Any]] = None,
    sessions: Optional[List[Any]] = None,
    last_result: Optional[Dict[str, Any]] = None,
    max_chunks: int = 150,
    include_uncited: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Assemble nodes and edges from the available RAG artifacts.

    Every argument is optional — the graph degrades gracefully to whatever
    subset of the pipeline has actually run.

    Args:
        registry:    ChunkRegistry, for document/chunk structure.
        memories:    MemoryEntry list, for question nodes and their provenance.
        sessions:    SessionInfo list, for session grouping.
        last_result: The most recent pipeline result, for claim nodes.
        max_chunks:  Cap on chunk nodes. The full corpus is ~900 chunks, which
                     renders as an unreadable hairball and pins the CPU; chunks
                     actually cited by a question are always kept, and the
                     remainder is sampled evenly across documents.

    Returns:
        ``{"nodes": [...], "edges": [...], "truncated": bool}``
    """
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, label: str, **extra: Any) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "label": label,
                "degree": 0,
                **extra,
            }

    def add_edge(source: str, target: str, kind: str) -> None:
        if source in nodes and target in nodes:
            edges.append({"source": source, "target": target, "kind": kind})
            nodes[source]["degree"] += 1
            nodes[target]["degree"] += 1

    # ------------------------------------------------------------------
    # Which chunks matter: those cited by a question always survive the cap.
    # ------------------------------------------------------------------
    cited_chunk_ids: set = set()
    for mem in memories or []:
        for cid in getattr(mem, "retrieved_chunk_ids", None) or []:
            cited_chunk_ids.add(cid)

    # ------------------------------------------------------------------
    # Documents and chunks
    # ------------------------------------------------------------------
    truncated = False
    if registry is not None:
        records = list(getattr(registry, "_records", {}).values())

        if not include_uncited:
            # Cited-only view. These chunks still need nodes — dropping the
            # registry entirely would leave every question's edges pointing at
            # targets that do not exist, and add_edge would silently discard
            # them, rendering a graph with zero chunks.
            records = [r for r in records if r.chunk_id in cited_chunk_ids]
        elif len(records) > max_chunks:
            truncated = True
            cited = [r for r in records if r.chunk_id in cited_chunk_ids]
            rest = [r for r in records if r.chunk_id not in cited_chunk_ids]
            slots = max(0, max_chunks - len(cited))
            if slots and rest:
                # Even stride keeps the sample spread across the whole corpus
                # instead of clustering in whichever document happens to be first.
                step = max(1, len(rest) // slots)
                rest = rest[::step][:slots]
            else:
                rest = []
            records = cited + rest

        for record in records:
            source = os.path.basename(record.source_file or "unknown")
            doc_id = f"doc::{source}"
            add_node(doc_id, "document", source, detail=f"Source document: {source}")

            chunk_id = f"chunk::{record.chunk_id}"
            add_node(
                chunk_id,
                "chunk",
                f"p{record.page_number}",
                detail=_truncate(record.text, 400),
                meta=f"{source} · page {record.page_number} · {record.chunk_id[:8]}",
            )
            add_edge(doc_id, chunk_id, "contains")

    # ------------------------------------------------------------------
    # Sessions and questions
    # ------------------------------------------------------------------
    for session in sessions or []:
        sid = f"session::{session.session_id}"
        add_node(
            sid,
            "session",
            _truncate(session.title, 28),
            detail=f"Session: {session.title}",
            meta=session.session_id,
        )

    for mem in memories or []:
        qid = f"q::{mem.memory_id}"
        add_node(
            qid,
            "question",
            _truncate(mem.question, 30),
            detail=f"Q: {mem.question}\n\nA: {_truncate(mem.answer, 500)}",
            meta=str(getattr(mem, "timestamp", "")),
        )

        sid = f"session::{mem.session_id}"
        if sid in nodes:
            add_edge(sid, qid, "asked")

        for cid in getattr(mem, "retrieved_chunk_ids", None) or []:
            add_edge(qid, f"chunk::{cid}", "retrieved")

    # ------------------------------------------------------------------
    # Claims from the most recent run (with verification status if present)
    # ------------------------------------------------------------------
    if last_result:
        verification = last_result.get("verification")
        claims = last_result.get("claims")

        if verification is not None and getattr(verification, "results", None):
            for i, vr in enumerate(verification.results):
                cid = f"claim::{i}"
                status = getattr(vr.verification_status, "value", str(vr.verification_status))
                add_node(
                    cid,
                    "claim",
                    _truncate(vr.claim_text, 26),
                    detail=f"{vr.claim_text}\n\nStatus: {status}\nConfidence: {vr.confidence:.2f}",
                    meta=status,
                    status=status,
                )
                if getattr(vr, "best_chunk_id", None):
                    add_edge(cid, f"chunk::{vr.best_chunk_id}", "supported_by")
        elif claims is not None and getattr(claims, "candidate_claims", None):
            for i, claim in enumerate(claims.candidate_claims):
                cid = f"claim::{i}"
                add_node(
                    cid,
                    "claim",
                    _truncate(claim.claim_text, 26),
                    detail=claim.claim_text,
                    meta="unverified",
                )

    # Drop orphan chunks — a chunk connected only to its document adds noise
    # without adding provenance information, and there can be hundreds.
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "truncated": truncated,
    }


def _legal_node_label(node_id: str, data: Dict[str, Any]) -> str:
    node_type = data.get("type")
    if node_type == "case":
        return data.get("case_name") or data.get("citation") or node_id.split(":", 1)[-1]
    if node_type == "court":
        return data.get("name") or node_id.split(":", 1)[-1]
    return data.get("label") or node_id.split(":", 1)[-1]


def _legal_node_detail(node_id: str, data: Dict[str, Any]) -> str:
    node_type = data.get("type")
    if node_type == "case":
        lines = [data.get("case_name") or node_id]
        if data.get("citation"):
            lines.append(f"Citation: {data['citation']}")
        if data.get("neutral_citation"):
            lines.append(f"Neutral citation: {data['neutral_citation']}")
        if data.get("court"):
            lines.append(f"Court: {data['court']}")
        if data.get("date"):
            lines.append(f"Date: {data['date']}")
        lines.append("In corpus" if data.get("in_corpus") else "Cited, not in corpus")
        return "\n".join(lines)
    if node_type == "court":
        return f"Court: {data.get('name', node_id)}"
    return data.get("label", node_id)


def build_legal_graph_data(
    graph: Any = None,
    max_nodes: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convert the legal citation graph (``src/legal_graph.py``, a NetworkX
    ``MultiDiGraph`` of cases/courts/statutes/sections/articles) into the same
    ``{nodes, edges, truncated}`` shape ``build_graph_data`` produces, so
    ``render_graph_html`` needs no changes to render either.

    The full graph is ~10^4 nodes -- unreadable and slow to lay out. When over
    ``max_nodes``, in-corpus cases are prioritised (they're the ones with
    retrievable text), then their direct citation/statute neighbours, up to
    the cap; everything else is dropped rather than sampled, since an
    out-of-corpus case with no kept neighbour contributes no visible edge.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "truncated": False}

    all_ids = list(graph.nodes(data=True))
    truncated = len(all_ids) > max_nodes

    if truncated:
        in_corpus_cases = [n for n, d in all_ids if d.get("type") == "case" and d.get("in_corpus")]
        keep_ids: set = set(in_corpus_cases[:max_nodes])
        for node_id in list(keep_ids):
            if len(keep_ids) >= max_nodes:
                break
            keep_ids.update(list(graph.successors(node_id))[: max(0, max_nodes - len(keep_ids))])
            keep_ids.update(list(graph.predecessors(node_id))[: max(0, max_nodes - len(keep_ids))])
        kept = [(n, d) for n, d in all_ids if n in keep_ids]
    else:
        kept = all_ids

    nodes = []
    kept_ids = set()
    for node_id, data in kept:
        node_type = data.get("type", "case")
        nodes.append({
            "id": node_id,
            "type": node_type,
            "label": _truncate(_legal_node_label(node_id, data), 28),
            "degree": graph.degree(node_id),
            "detail": _legal_node_detail(node_id, data),
            "meta": data.get("court") or data.get("citation") or "",
        })
        kept_ids.add(node_id)

    edges = [
        {"source": u, "target": v, "kind": data.get("relation", "")}
        for u, v, data in graph.edges(data=True)
        if u in kept_ids and v in kept_ids
    ]

    return {"nodes": nodes, "edges": edges, "truncated": truncated}


def render_graph_html(
    data: Dict[str, Any],
    height: int = 680,
    dark: bool = True,
) -> str:
    """Render the graph as a self-contained HTML document for st.components."""
    payload = json.dumps(data)
    bg = "#0f0f23" if dark else "#ffffff"
    fg = "#e5e7eb" if dark else "#1f2937"
    panel = "rgba(26,26,46,0.94)" if dark else "rgba(255,255,255,0.96)"
    border = "rgba(124,58,237,0.35)" if dark else "rgba(0,0,0,0.12)"
    types_json = json.dumps(NODE_TYPES)

    return f"""
<div id="kg-root" style="position:relative;width:100%;height:{height}px;
     background:{bg};border-radius:14px;overflow:hidden;border:1px solid {border};
     font-family:'Segoe UI',system-ui,-apple-system,sans-serif;">

  <canvas id="kg-canvas" style="display:block;width:100%;height:100%;cursor:grab;"></canvas>

  <div id="kg-legend" style="position:absolute;top:12px;left:12px;background:{panel};
       border:1px solid {border};border-radius:10px;padding:10px 12px;backdrop-filter:blur(8px);
       font-size:11px;color:{fg};max-width:190px;"></div>

  <div style="position:absolute;top:12px;right:12px;display:flex;gap:6px;align-items:center;">
    <input id="kg-search" placeholder="Search nodes…"
           style="background:{panel};border:1px solid {border};border-radius:8px;
                  padding:7px 10px;color:{fg};font-size:12px;width:170px;outline:none;">
    <button id="kg-reset" title="Reset view"
            style="background:{panel};border:1px solid {border};border-radius:8px;
                   padding:7px 10px;color:{fg};font-size:12px;cursor:pointer;">Reset</button>
  </div>

  <div id="kg-info" style="position:absolute;bottom:12px;left:12px;background:{panel};
       border:1px solid {border};border-radius:10px;padding:11px 13px;backdrop-filter:blur(8px);
       font-size:11.5px;color:{fg};max-width:330px;display:none;line-height:1.5;
       max-height:190px;overflow-y:auto;white-space:pre-wrap;"></div>

  <div id="kg-stats" style="position:absolute;bottom:12px;right:12px;font-size:10.5px;
       color:{fg};opacity:0.55;"></div>
</div>

<script>
(function() {{
  const DATA  = {payload};
  const TYPES = {types_json};
  const FG    = "{fg}";

  const canvas = document.getElementById("kg-canvas");
  const ctx    = canvas.getContext("2d");
  const root   = document.getElementById("kg-root");

  // ---- state -------------------------------------------------------------
  const hidden = new Set();
  let nodes = [], links = [];
  let tx = 0, ty = 0, scale = 1;
  let dragNode = null, panning = false, lastX = 0, lastY = 0;
  let hoverNode = null, selected = null, query = "";
  let alpha = 1.0;

  function size() {{
    const r = root.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = r.width  * dpr;
    canvas.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return r;
  }}
  let rect = size();

  // ---- build -------------------------------------------------------------
  function build() {{
    const byId = {{}};
    nodes = DATA.nodes.filter(n => !hidden.has(n.type)).map(n => {{
      const prev = byId[n.id];
      const o = {{
        ...n,
        x: (prev && prev.x) || rect.width/2  + (Math.random()-0.5)*380,
        y: (prev && prev.y) || rect.height/2 + (Math.random()-0.5)*380,
        vx: 0, vy: 0,
        r: Math.min(16, 4.5 + Math.sqrt(n.degree || 1) * 1.9)
      }};
      byId[n.id] = o;
      return o;
    }});
    const idx = {{}};
    nodes.forEach(n => idx[n.id] = n);
    links = DATA.edges
      .map(e => ({{ s: idx[e.source], t: idx[e.target], kind: e.kind }}))
      .filter(e => e.s && e.t);
    alpha = 1.0;
    document.getElementById("kg-stats").textContent =
      nodes.length + " nodes · " + links.length + " edges" +
      (DATA.truncated ? " (chunk sample)" : "");
  }}

  // ---- physics -----------------------------------------------------------
  function step() {{
    if (alpha < 0.005) return;
    alpha *= 0.994;

    // Repulsion. O(n^2) is fine at this scale (few hundred nodes) and avoids
    // the complexity of a quadtree for no perceptible gain.
    for (let i = 0; i < nodes.length; i++) {{
      const a = nodes[i];
      for (let j = i+1; j < nodes.length; j++) {{
        const b = nodes[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let d2 = dx*dx + dy*dy;
        if (d2 < 1) {{ dx = Math.random()-0.5; dy = Math.random()-0.5; d2 = 1; }}
        if (d2 > 90000) continue;
        const f = 900 / d2;
        const d = Math.sqrt(d2);
        const fx = (dx/d)*f, fy = (dy/d)*f;
        a.vx -= fx; a.vy -= fy;
        b.vx += fx; b.vy += fy;
      }}
    }}

    // Spring attraction along edges
    for (const l of links) {{
      const dx = l.t.x - l.s.x, dy = l.t.y - l.s.y;
      const d = Math.sqrt(dx*dx + dy*dy) || 1;
      const f = (d - 62) * 0.0038;
      const fx = (dx/d)*f, fy = (dy/d)*f;
      l.s.vx += fx; l.s.vy += fy;
      l.t.vx -= fx; l.t.vy -= fy;
    }}

    // Gravity toward centre keeps disconnected components on screen
    const cx = rect.width/2, cy = rect.height/2;
    for (const n of nodes) {{
      n.vx += (cx - n.x) * 0.0016;
      n.vy += (cy - n.y) * 0.0016;
      if (n === dragNode) continue;
      n.vx *= 0.86; n.vy *= 0.86;
      n.x += n.vx * alpha * 5.5;
      n.y += n.vy * alpha * 5.5;
    }}
  }}

  // ---- draw --------------------------------------------------------------
  function matches(n) {{
    return query && (n.label + " " + (n.meta||"") + " " + (n.detail||""))
      .toLowerCase().includes(query);
  }}

  function neighbours(n) {{
    const s = new Set();
    if (!n) return s;
    for (const l of links) {{
      if (l.s === n) s.add(l.t);
      if (l.t === n) s.add(l.s);
    }}
    return s;
  }}

  function draw() {{
    ctx.save();
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.translate(tx, ty);
    ctx.scale(scale, scale);

    const focus = selected || hoverNode;
    const near = neighbours(focus);

    for (const l of links) {{
      const active = focus && (l.s === focus || l.t === focus);
      ctx.strokeStyle = active ? "rgba(124,58,237,0.85)" : "rgba(148,163,184,0.16)";
      ctx.lineWidth = (active ? 1.8 : 0.7) / scale;
      ctx.beginPath();
      ctx.moveTo(l.s.x, l.s.y);
      ctx.lineTo(l.t.x, l.t.y);
      ctx.stroke();
    }}

    for (const n of nodes) {{
      const dim = (focus && n !== focus && !near.has(n)) || (query && !matches(n));
      const color = (TYPES[n.type] || {{}}).color || "#94a3b8";
      ctx.globalAlpha = dim ? 0.18 : 1;

      if (!dim && (n === focus || matches(n))) {{
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 6/scale, 0, Math.PI*2);
        ctx.fillStyle = color + "33";
        ctx.fill();
      }}

      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.lineWidth = 1.4/scale;
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.stroke();

      // Labels only where they will be legible and not overlapping everything
      if (!dim && (scale > 0.75 || n.r > 9 || n === focus)) {{
        ctx.globalAlpha = dim ? 0.2 : 0.92;
        ctx.fillStyle = FG;
        ctx.font = (10.5/scale) + "px 'Segoe UI',system-ui,sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(n.label, n.x, n.y + n.r + 11/scale);
      }}
      ctx.globalAlpha = 1;
    }}
    ctx.restore();
  }}

  function loop() {{ step(); draw(); requestAnimationFrame(loop); }}

  // ---- interaction -------------------------------------------------------
  function toWorld(ev) {{
    const r = canvas.getBoundingClientRect();
    return {{ x: (ev.clientX - r.left - tx)/scale, y: (ev.clientY - r.top - ty)/scale }};
  }}

  function pick(p) {{
    let best = null, bestD = Infinity;
    for (const n of nodes) {{
      const d = Math.hypot(n.x - p.x, n.y - p.y);
      if (d < n.r + 6 && d < bestD) {{ best = n; bestD = d; }}
    }}
    return best;
  }}

  canvas.addEventListener("mousedown", ev => {{
    const p = toWorld(ev);
    dragNode = pick(p);
    if (!dragNode) {{ panning = true; lastX = ev.clientX; lastY = ev.clientY; }}
    canvas.style.cursor = "grabbing";
  }});

  canvas.addEventListener("mousemove", ev => {{
    const p = toWorld(ev);
    if (dragNode) {{
      dragNode.x = p.x; dragNode.y = p.y;
      dragNode.vx = 0; dragNode.vy = 0;
      alpha = Math.max(alpha, 0.35);
    }} else if (panning) {{
      tx += ev.clientX - lastX; ty += ev.clientY - lastY;
      lastX = ev.clientX; lastY = ev.clientY;
    }} else {{
      const h = pick(p);
      if (h !== hoverNode) {{
        hoverNode = h;
        canvas.style.cursor = h ? "pointer" : "grab";
      }}
    }}
  }});

  window.addEventListener("mouseup", ev => {{
    // A press with no movement is a click: select the node and show its detail.
    if (dragNode && Math.abs(dragNode.vx) < 1e-6) {{
      selected = dragNode;
      showInfo(dragNode);
    }} else if (panning) {{
      const p = toWorld(ev);
      if (!pick(p)) {{ selected = null; hideInfo(); }}
    }}
    dragNode = null; panning = false;
    canvas.style.cursor = "grab";
  }});

  canvas.addEventListener("wheel", ev => {{
    ev.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const k = ev.deltaY < 0 ? 1.12 : 1/1.12;
    const ns = Math.min(4, Math.max(0.18, scale * k));
    tx = mx - (mx - tx) * (ns/scale);
    ty = my - (my - ty) * (ns/scale);
    scale = ns;
  }}, {{ passive: false }});

  const info = document.getElementById("kg-info");
  function showInfo(n) {{
    const t = (TYPES[n.type] || {{}}).label || n.type;
    info.style.display = "block";
    info.textContent = "[" + t + "] " + n.label
      + (n.meta ? "\\n" + n.meta : "")
      + "\\n\\n" + (n.detail || "");
  }}
  function hideInfo() {{ info.style.display = "none"; }}

  // ---- legend / filters --------------------------------------------------
  const legend = document.getElementById("kg-legend");
  function renderLegend() {{
    const counts = {{}};
    DATA.nodes.forEach(n => counts[n.type] = (counts[n.type]||0) + 1);
    legend.innerHTML = "<div style='font-weight:600;margin-bottom:6px;opacity:.75;"
      + "font-size:10px;letter-spacing:.06em;'>CLICK TO FILTER</div>"
      + Object.keys(TYPES).filter(t => counts[t]).map(t => {{
        const off = hidden.has(t);
        return "<div data-t='" + t + "' style='display:flex;align-items:center;gap:7px;"
          + "cursor:pointer;padding:3px 0;opacity:" + (off ? 0.35 : 1) + ";'>"
          + "<span style='width:9px;height:9px;border-radius:50%;background:"
          + TYPES[t].color + ";display:inline-block;flex:none;'></span>"
          + "<span style='" + (off ? "text-decoration:line-through;" : "") + "'>"
          + TYPES[t].label + "</span>"
          + "<span style='margin-left:auto;opacity:.55;'>" + counts[t] + "</span></div>";
      }}).join("");
    legend.querySelectorAll("[data-t]").forEach(el => {{
      el.onclick = () => {{
        const t = el.getAttribute("data-t");
        hidden.has(t) ? hidden.delete(t) : hidden.add(t);
        selected = null; hideInfo(); build(); renderLegend();
      }};
    }});
  }}

  document.getElementById("kg-search").addEventListener("input", e => {{
    query = e.target.value.trim().toLowerCase();
  }});

  document.getElementById("kg-reset").addEventListener("click", () => {{
    tx = 0; ty = 0; scale = 1; selected = null; hideInfo(); build();
  }});

  window.addEventListener("resize", () => {{ rect = size(); }});

  build();
  renderLegend();
  loop();
}})();
</script>
"""
