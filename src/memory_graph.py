"""
Interactive graph over claude-mem's own memory (observations/sessions/concepts).

Reads directly from claude-mem's SQLite DB rather than its HTTP worker: the
worker's port is dynamic (written to ~/.claude-mem/worker.pid) and the worker
may not be running, whereas the DB file is always there once claude-mem has
run once. Opened read-only via a `file:` URI so a live worker holding the
file does not block us and we never risk writing to it.

Same ``{"nodes": [...], "edges": [...], "truncated": bool}`` shape as
``ui/components/graph.py``'s ``build_graph_data()``, so both graphs share one
renderer on the frontend.
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

DB_PATH = os.path.expanduser("~/.claude-mem/claude-mem.db")

#: type -> display color, mirrors the legend convention in ui/components/graph.py
NODE_TYPES: Dict[str, Dict[str, str]] = {
    "session": {"label": "Session", "color": "#ec4899"},
    "observation": {"label": "Observation", "color": "#3b82f6"},
    "concept": {"label": "Concept", "color": "#10b981"},
    "file": {"label": "File", "color": "#f59e0b"},
}


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _connect() -> Optional[sqlite3.Connection]:
    if not os.path.exists(DB_PATH):
        return None
    uri = f"file:{DB_PATH.replace(os.sep, '/')}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def build_memory_graph_data(
    project: Optional[str] = None,
    max_observations: int = 150,
) -> Dict[str, Any]:
    """Assemble nodes/edges from claude-mem observations, sessions, concepts, files.

    Args:
        project:          Restrict to one project (matches the `project` column).
        max_observations: Cap, most-recent-first -- an unbounded pull renders as
                           a hairball the same way an uncapped chunk graph would.
    """
    conn = _connect()
    if conn is None:
        return {"nodes": [], "edges": [], "truncated": False}

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, label: str, **extra: Any) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "type": node_type, "label": label, "degree": 0, **extra}

    def add_edge(source: str, target: str, kind: str) -> None:
        if source in nodes and target in nodes:
            edges.append({"source": source, "target": target, "kind": kind})
            nodes[source]["degree"] += 1
            nodes[target]["degree"] += 1

    try:
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT id, memory_session_id, project, type, title, subtitle, "
            "concepts, files_read, files_modified, created_at "
            "FROM observations "
        )
        params: List[Any] = []
        if project:
            query += "WHERE project = ? "
            params.append(project)
        query += "ORDER BY created_at_epoch DESC LIMIT ?"
        params.append(max_observations)
        rows = conn.execute(query, params).fetchall()

        session_rows = {
            r["memory_session_id"]: r
            for r in conn.execute(
                "SELECT memory_session_id, custom_title, user_prompt FROM sdk_sessions"
            ).fetchall()
        }
    finally:
        conn.close()

    truncated = len(rows) >= max_observations

    for row in reversed(rows):  # oldest first, so layout reads chronologically
        obs_id = f"obs::{row['id']}"
        obs_type = row["type"] or "observation"
        add_node(
            obs_id, "observation", _truncate(row["title"] or f"#{row['id']}", 30),
            detail=f"{row['title'] or ''}\n\n{row['subtitle'] or ''}",
            meta=f"{obs_type} · {row['created_at']}",
            obs_type=obs_type,
        )

        sid = row["memory_session_id"]
        if sid:
            snode = f"session::{sid}"
            if snode not in nodes:
                srow = session_rows.get(sid)
                title = (srow["custom_title"] if srow else None) or sid[:8]
                add_node(snode, "session", _truncate(title, 28), detail=title, meta=sid)
            add_edge(snode, obs_id, "contains")

        for field in ("concepts",):
            try:
                for concept in json.loads(row[field] or "[]"):
                    cnode = f"concept::{concept}"
                    add_node(cnode, "concept", concept, detail=f"Concept: {concept}")
                    add_edge(obs_id, cnode, "tagged")
            except (json.JSONDecodeError, TypeError):
                pass

        for field in ("files_read", "files_modified"):
            try:
                for path in json.loads(row[field] or "[]"):
                    fname = os.path.basename(path)
                    fnode = f"file::{fname}"
                    add_node(fnode, "file", fname, detail=path)
                    add_edge(obs_id, fnode, "modified" if field == "files_modified" else "read")
            except (json.JSONDecodeError, TypeError):
                pass

    return {"nodes": list(nodes.values()), "edges": edges, "truncated": truncated}


def demo() -> None:
    """Self-check against whatever DB is actually on disk. No fixtures."""
    data = build_memory_graph_data(max_observations=50)
    assert set(data.keys()) == {"nodes", "edges", "truncated"}
    node_ids = {n["id"] for n in data["nodes"]}
    for edge in data["edges"]:
        assert edge["source"] in node_ids and edge["target"] in node_ids
    types = {n["type"] for n in data["nodes"]}
    assert types <= set(NODE_TYPES), types
    print(f"memory_graph demo OK  (nodes={len(data['nodes'])} edges={len(data['edges'])} "
          f"db_found={os.path.exists(DB_PATH)})")


if __name__ == "__main__":
    demo()
