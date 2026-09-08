"""
Knowledge Graph page.

Renders the provenance graph linking documents → chunks → questions → claims,
so you can see which sources actually backed which answers.
"""

import os
import sys

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.components.graph import build_graph_data, render_graph_html  # noqa: E402

st.set_page_config(page_title="Knowledge Graph — X-RAG", page_icon="🕸️", layout="wide")

CSS_PATH = os.path.join(PROJECT_ROOT, "ui", "styles", "main.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("## 🕸️ Knowledge Graph")
st.caption(
    "Provenance across the pipeline — drag nodes, scroll to zoom, click a node "
    "for detail, and click a legend entry to filter that type out."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_registry():
    """Load the chunk registry (cached — it is a few MB of JSON)."""
    from src.chunk_registry import ChunkRegistry

    path = os.path.join(PROJECT_ROOT, "artifacts", "chunk_registry.json")
    if not os.path.exists(path):
        return None
    return ChunkRegistry.load_from_json(path)


def get_memory_manager():
    """Reuse the app's MemoryManager if the chat page created one."""
    if "memory_manager" in st.session_state:
        return st.session_state.memory_manager

    from src.memory.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.initialize()
    st.session_state.memory_manager = mm
    return mm


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns([1.3, 1.3, 1.3, 1])

with col1:
    scope = st.selectbox(
        "Scope",
        ["Current session", "All sessions"],
        help="Which questions to include as nodes.",
    )
with col2:
    max_chunks = st.slider(
        "Max chunk nodes", 25, 400, 120, step=25,
        help="The full corpus is ~900 chunks. Chunks cited by a question are "
             "always included; the rest are sampled.",
    )
with col3:
    include_corpus = st.checkbox(
        "Include uncited chunks", value=False,
        help="Off shows only chunks that actually backed an answer — usually "
             "the more readable view.",
    )
with col4:
    st.write("")
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

mm = get_memory_manager()
registry = load_registry()

if registry is None:
    st.warning(
        "No chunk registry found. Run `python run_pipeline.py` to ingest the "
        "documents first — the graph will then show document and chunk nodes."
    )

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
try:
    sessions = mm.list_sessions()
except Exception:
    sessions = []

current_sid = st.session_state.get("current_session_id")
if scope == "Current session" and current_sid:
    sessions = [s for s in sessions if s.session_id == current_sid]

memories = []
for session in sessions:
    try:
        memories.extend(mm.get_session_memories(session.session_id))
    except Exception:
        pass

if not memories and not registry:
    st.info("Nothing to graph yet. Ask a question on the chat page first.")
    st.stop()

data = build_graph_data(
    registry=registry,
    memories=memories,
    sessions=sessions,
    last_result=st.session_state.get("last_pipeline_result"),
    max_chunks=max_chunks,
    include_uncited=include_corpus,
)

if not data["nodes"]:
    st.info(
        "No nodes to display with the current filters. Try enabling "
        "**Include uncited chunks** or switching scope to **All sessions**."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
counts = {}
for node in data["nodes"]:
    counts[node["type"]] = counts.get(node["type"], 0) + 1
m1.metric("Nodes", len(data["nodes"]))
m2.metric("Edges", len(data["edges"]))
m3.metric("Questions", counts.get("question", 0))
m4.metric("Chunks", counts.get("chunk", 0))

components.html(render_graph_html(data, height=680), height=700, scrolling=False)

with st.expander("How to read this graph"):
    st.markdown(
        """
- **🟠 Document** — a source PDF in `data/`.
- **🔵 Chunk** — a retrievable passage. Size scales with how many things connect to it,
  so large blue nodes are passages that repeatedly backed answers.
- **🟣 Question** — one turn you asked. Edges run to every chunk retrieved for it.
- **🩷 Session** — a conversation thread.
- **🟢 Claim** — an atomic claim extracted from the answer, linked to the chunk that
  verified it.

**Interactions:** drag to reposition · scroll to zoom · drag empty space to pan ·
click a node to pin its detail · click a legend row to filter that type ·
type in the search box to highlight matches.
        """
    )
