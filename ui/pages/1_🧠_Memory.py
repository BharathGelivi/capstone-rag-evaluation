"""
Memory Visualization Page.

Provides memory timeline, memory cards, search, statistics,
session overview, and memory graph visualization.
"""

import json
import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.memory.memory_manager import MemoryManager

st.set_page_config(
    page_title="Memory Visualization — X-RAG",
    page_icon="🧠",
    layout="wide",
)

# Load CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize memory manager
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()
    st.session_state.memory_manager.initialize()

mm: MemoryManager = st.session_state.memory_manager


def main():
    st.title("🧠 Memory Visualization")
    st.markdown("Explore and analyze your conversation memories.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Dashboard", "🗂 Memory Cards", "🔍 Search", "📈 Timeline", "🕸 Graph"]
    )

    # Tab 1: Dashboard
    with tab1:
        render_dashboard()

    # Tab 2: Memory Cards
    with tab2:
        render_memory_cards()

    # Tab 3: Search
    with tab3:
        render_search()

    # Tab 4: Timeline
    with tab4:
        render_timeline()

    # Tab 5: Graph
    with tab5:
        render_graph()


def render_dashboard():
    """Render memory system dashboard with statistics."""
    stats = mm.get_statistics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Memories", stats["total_memories"])
    with col2:
        st.metric("Sessions", stats["total_sessions"])
    with col3:
        st.metric("Short-term", stats["short_term_count"])
    with col4:
        st.metric("Summaries", stats["summaries_count"])

    st.markdown("---")

    # Session overview
    st.markdown("### 💬 Session Overview")
    sessions = mm.list_sessions()

    if sessions:
        for session in sessions:
            with st.expander(f"📁 {session.title} — {session.question_count} questions", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Created:** {session.created_at[:16]}")
                with col2:
                    st.markdown(f"**Last Active:** {session.last_activity[:16]}")
                with col3:
                    st.markdown(f"**Traces:** {session.trace_count}")

                # Memory count for session
                memories = mm.get_session_memories(session.session_id)
                st.markdown(f"**Memories in session:** {len(memories)}")

                if memories:
                    # Show a heatmap-style view of activity
                    st.markdown("**Recent interactions:**")
                    for mem in memories[-5:]:
                        st.markdown(f"- **Q:** {mem.question[:80]}...")
    else:
        st.info("No sessions yet. Start a conversation to create memories.")


def render_memory_cards():
    """Render individual memory cards with full details."""
    sessions = mm.list_sessions()
    selected_session = st.selectbox(
        "Filter by session",
        ["All Sessions"] + [f"{s.title} ({s.session_id})" for s in sessions],
    )

    session_id = None
    if selected_session != "All Sessions":
        session_id = selected_session.split("(")[-1].rstrip(")")

    if session_id:
        memories = mm.get_session_memories(session_id)
    else:
        memories = mm.store.get_all_memories()

    if not memories:
        st.info("No memories found.")
        return

    st.markdown(f"**Showing {len(memories)} memories**")

    for mem in reversed(memories):  # Most recent first
        with st.expander(
            f"💭 {mem.question[:60]}... — {mem.timestamp[:16]}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Question:** {mem.question}")
                st.markdown(f"**Answer:** {mem.answer}")

            with col2:
                st.markdown(f"**Memory ID:** `{mem.memory_id}`")
                st.markdown(f"**Session:** `{mem.session_id}`")
                st.markdown(f"**Timestamp:** {mem.timestamp}")
                st.markdown(f"**Importance:** {mem.importance_score}")
                st.markdown(f"**Access Count:** {mem.access_count}")

                if mem.trace_id:
                    st.markdown(f"**Trace ID:** `{mem.trace_id}`")
                if mem.retrieved_chunk_ids:
                    st.markdown(f"**Chunks:** {len(mem.retrieved_chunk_ids)}")
                if mem.claim_ids:
                    st.markdown(f"**Claims:** {len(mem.claim_ids)}")
                if mem.tags:
                    st.markdown(f"**Tags:** {', '.join(mem.tags)}")

            # Delete button
            if st.button(f"🗑 Delete", key=f"del_mem_{mem.memory_id}"):
                mm.delete_memory(mem.memory_id)
                st.rerun()


def render_search():
    """Render memory search interface."""
    st.markdown("### 🔍 Semantic Memory Search")
    st.markdown("Search across all memories using semantic similarity.")

    query = st.text_input("Enter search query", placeholder="Search questions, answers, claims...")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Results", 1, 20, 5)
    with col2:
        sessions = mm.list_sessions()
        session_filter = st.selectbox(
            "Session filter",
            ["All Sessions"] + [f"{s.title} ({s.session_id})" for s in sessions],
            key="search_session_filter",
        )

    session_id = None
    if session_filter != "All Sessions":
        session_id = session_filter.split("(")[-1].rstrip(")")

    if query:
        with st.spinner("Searching memories..."):
            results = mm.search_memory(query, top_k=top_k, session_id=session_id)

        if results:
            st.markdown(f"**Found {len(results)} results**")

            for i, sr in enumerate(results, 1):
                with st.expander(
                    f"#{i} — Score: {sr.final_score:.3f} — {sr.memory.question[:60]}...",
                    expanded=i <= 3,
                ):
                    st.markdown(f"**Q:** {sr.memory.question}")
                    st.markdown(f"**A:** {sr.memory.answer[:500]}...")
                    st.markdown(f"**Why Retrieved:** {sr.retrieval_reason}")

                    st.markdown("---")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Semantic", f"{sr.semantic_score:.3f}")
                    with col2:
                        st.metric("Recency", f"{sr.recency_score:.3f}")
                    with col3:
                        st.metric("Frequency", f"{sr.frequency_score:.3f}")
                    with col4:
                        st.metric("Importance", f"{sr.importance_score:.3f}")
        else:
            st.info("No matching memories found.")


def render_timeline():
    """Render a chronological timeline of interactions."""
    st.markdown("### 📈 Memory Timeline")

    memories = mm.store.get_all_memories()
    if not memories:
        st.info("No memories to display.")
        return

    # Group by date
    from collections import defaultdict
    date_groups = defaultdict(list)
    for mem in memories:
        date = mem.timestamp[:10]
        date_groups[date].append(mem)

    for date in sorted(date_groups.keys(), reverse=True):
        st.markdown(f"#### 📅 {date}")
        for mem in date_groups[date]:
            st.markdown(
                f"""
                <div style="border-left: 3px solid var(--accent, #7c3aed); padding-left: 1rem; margin-bottom: 0.75rem;">
                    <p style="font-size: 0.85rem; color: #e5e7eb; margin: 0;">
                        <strong>Q:</strong> {mem.question[:100]}...
                    </p>
                    <p style="font-size: 0.75rem; color: #9ca3af; margin: 0.25rem 0 0 0;">
                        {mem.timestamp[11:16]} UTC | Session: {mem.session_id[:15]}...
                        {f' | Trace: {mem.trace_id[:8]}...' if mem.trace_id else ''}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("")


def render_graph():
    """Render a memory graph using NetworkX + HTML visualization."""
    st.markdown("### 🕸 Memory Graph")
    st.markdown("Visualize relationships between questions, answers, chunks, and claims.")

    memories = mm.store.get_all_memories(limit=50)
    if not memories:
        st.info("No memories to display in graph.")
        return

    try:
        import networkx as nx

        G = nx.DiGraph()

        for mem in memories[-20:]:  # Last 20 for readability
            q_node = f"Q: {mem.question[:40]}..."
            a_node = f"A: {mem.answer[:40]}..."
            mem_node = f"Mem: {mem.memory_id[:10]}"

            G.add_node(q_node, type="question", color="#7c3aed")
            G.add_node(a_node, type="answer", color="#10b981")
            G.add_node(mem_node, type="memory", color="#3b82f6")

            G.add_edge(q_node, a_node, relation="generates")
            G.add_edge(a_node, mem_node, relation="stored_as")

            # Add chunk nodes
            for chunk_id in mem.retrieved_chunk_ids[:3]:
                c_node = f"Chunk: {chunk_id[:15]}..."
                G.add_node(c_node, type="chunk", color="#f59e0b")
                G.add_edge(c_node, q_node, relation="retrieved_for")

            # Add claim nodes
            for claim_id in mem.claim_ids[:3]:
                cl_node = f"Claim: {claim_id[:15]}..."
                G.add_node(cl_node, type="claim", color="#ef4444")
                G.add_edge(a_node, cl_node, relation="contains")

        # Display stats
        st.markdown(f"**Nodes:** {G.number_of_nodes()} | **Edges:** {G.number_of_edges()}")

        # Display as adjacency list since we can't render interactive graphs in Streamlit
        # without additional dependencies
        st.markdown("#### Node Types")
        type_counts = {}
        for _, data in G.nodes(data=True):
            t = data.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        cols = st.columns(len(type_counts))
        colors = {"question": "🟣", "answer": "🟢", "memory": "🔵", "chunk": "🟡", "claim": "🔴"}
        for i, (t, count) in enumerate(type_counts.items()):
            with cols[i]:
                st.metric(f"{colors.get(t, '⚪')} {t.title()}", count)

        # Show relationships
        st.markdown("#### Relationships")
        for u, v, data in list(G.edges(data=True))[:30]:
            st.markdown(f"- `{u}` → *{data.get('relation', '')}* → `{v}`")

    except ImportError:
        st.warning("Install `networkx` for graph visualization: `pip install networkx`")
        st.markdown("Showing memory connections as a list instead:")
        for mem in memories[-10:]:
            st.markdown(
                f"- **{mem.question[:50]}...** → {len(mem.retrieved_chunk_ids)} chunks, "
                f"{len(mem.claim_ids)} claims"
            )


if __name__ == "__main__":
    main()
