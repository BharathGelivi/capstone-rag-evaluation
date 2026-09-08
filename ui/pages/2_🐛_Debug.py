"""
Debug Tools Page.

Developer mode showing embedding vectors, similarity matrices,
memory ranking details, merged context, and prompt snapshots.
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
    page_title="Debug Tools — X-RAG",
    page_icon="🐛",
    layout="wide",
)

CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "styles", "main.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()
    st.session_state.memory_manager.initialize()

mm: MemoryManager = st.session_state.memory_manager


def main():
    st.title("🐛 Debug Tools")
    st.markdown("Developer tools for inspecting memory, embeddings, and pipeline internals.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔢 Embeddings", "📐 Similarity", "📊 Ranking", "📝 Prompts", "📁 Exports"]
    )

    with tab1:
        render_embedding_inspector()

    with tab2:
        render_similarity_matrix()

    with tab3:
        render_ranking_debug()

    with tab4:
        render_prompt_inspector()

    with tab5:
        render_exports()


def render_embedding_inspector():
    """Inspect embedding vectors for memories."""
    st.markdown("### 🔢 Embedding Inspector")
    st.markdown("Enter text to see its embedding vector.")

    text = st.text_area("Input text", height=100, placeholder="Enter text to embed...")

    if text and st.button("Generate Embedding"):
        with st.spinner("Computing embedding..."):
            try:
                embedding = mm.retriever.embed_text(text)
                st.markdown(f"**Dimension:** {len(embedding)}")
                st.markdown(f"**First 10 values:** `{[round(v, 4) for v in embedding[:10]]}`")
                st.markdown(f"**L2 Norm:** {sum(v**2 for v in embedding)**0.5:.4f}")

                # Show distribution
                import statistics
                st.markdown(f"**Mean:** {statistics.mean(embedding):.6f}")
                st.markdown(f"**Std Dev:** {statistics.stdev(embedding):.6f}")
                st.markdown(f"**Min:** {min(embedding):.6f}")
                st.markdown(f"**Max:** {max(embedding):.6f}")

                with st.expander("Full vector"):
                    st.json([round(v, 6) for v in embedding])
            except Exception as e:
                st.error(f"Error: {e}")


def render_similarity_matrix():
    """Compute and display similarity between texts."""
    st.markdown("### 📐 Similarity Matrix")
    st.markdown("Compare semantic similarity between multiple texts.")

    texts_input = st.text_area(
        "Enter texts (one per line)",
        height=150,
        placeholder="Text 1\nText 2\nText 3...",
    )

    if texts_input and st.button("Compute Similarity"):
        texts = [t.strip() for t in texts_input.strip().split("\n") if t.strip()]

        if len(texts) < 2:
            st.warning("Enter at least 2 texts.")
            return

        with st.spinner("Computing embeddings and similarities..."):
            try:
                embeddings = [mm.retriever.embed_text(t) for t in texts]

                # Compute cosine similarities
                import math

                def cosine_sim(a, b):
                    dot = sum(x * y for x, y in zip(a, b))
                    norm_a = math.sqrt(sum(x**2 for x in a))
                    norm_b = math.sqrt(sum(x**2 for x in b))
                    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

                matrix = []
                for i, e1 in enumerate(embeddings):
                    row = []
                    for j, e2 in enumerate(embeddings):
                        row.append(round(cosine_sim(e1, e2), 4))
                    matrix.append(row)

                # Display as table
                labels = [t[:30] + "..." if len(t) > 30 else t for t in texts]
                st.markdown("#### Cosine Similarity Matrix")

                header = "| | " + " | ".join(labels) + " |"
                separator = "|" + "|".join(["---"] * (len(labels) + 1)) + "|"
                rows = []
                for i, label in enumerate(labels):
                    values = " | ".join(f"{v:.4f}" for v in matrix[i])
                    rows.append(f"| {label} | {values} |")

                st.markdown("\n".join([header, separator] + rows))

            except Exception as e:
                st.error(f"Error: {e}")


def render_ranking_debug():
    """Debug memory ranking for a query."""
    st.markdown("### 📊 Memory Ranking Debug")
    st.markdown("See exactly how memories are scored and ranked for a query.")

    query = st.text_input("Query to debug", placeholder="Enter a question...")

    if query and st.button("Debug Ranking"):
        with st.spinner("Computing rankings..."):
            results = mm.search_memory(query, top_k=10)

            if results:
                st.markdown(f"**{len(results)} memories ranked**")

                # Show ranking details
                for i, sr in enumerate(results, 1):
                    with st.expander(f"Rank {i}: {sr.memory.question[:60]}... (Score: {sr.final_score:.4f})"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### Score Breakdown")
                            config = mm.config

                            st.markdown(f"""
                            | Component | Raw Score | Weight | Weighted |
                            |-----------|-----------|--------|----------|
                            | Semantic | {sr.semantic_score:.4f} | {config.semantic_weight} | {sr.semantic_score * config.semantic_weight:.4f} |
                            | Recency | {sr.recency_score:.4f} | {config.recency_weight} | {sr.recency_score * config.recency_weight:.4f} |
                            | Frequency | {sr.frequency_score:.4f} | {config.frequency_weight} | {sr.frequency_score * config.frequency_weight:.4f} |
                            | Importance | {sr.importance_score:.4f} | {config.importance_weight} | {sr.importance_score * config.importance_weight:.4f} |
                            | **Final** | | | **{sr.final_score:.4f}** |
                            """)

                        with col2:
                            st.markdown("#### Memory Details")
                            st.markdown(f"**ID:** `{sr.memory.memory_id}`")
                            st.markdown(f"**Session:** `{sr.memory.session_id}`")
                            st.markdown(f"**Time:** {sr.memory.timestamp}")
                            st.markdown(f"**Access Count:** {sr.memory.access_count}")
                            st.markdown(f"**Reason:** {sr.retrieval_reason}")
            else:
                st.info("No memories found for this query.")


def render_prompt_inspector():
    """Inspect saved RAG trace prompts."""
    st.markdown("### 📝 Prompt Inspector")
    st.markdown("View prompts from recent RAG traces.")

    trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
    if not os.path.exists(trace_dir):
        st.info("No traces found.")
        return

    # Find trace files
    trace_files = []
    for root, dirs, files in os.walk(trace_dir):
        for f in files:
            if f.endswith(".json"):
                trace_files.append(os.path.join(root, f))

    trace_files.sort(reverse=True)
    trace_files = trace_files[:20]  # Last 20

    if not trace_files:
        st.info("No trace files found.")
        return

    selected = st.selectbox(
        "Select a trace",
        trace_files,
        format_func=lambda x: os.path.basename(x),
    )

    if selected:
        try:
            with open(selected, "r", encoding="utf-8") as f:
                trace_data = json.load(f)

            st.markdown(f"**Trace ID:** `{trace_data.get('trace_id', 'N/A')}`")
            st.markdown(f"**Question:** {trace_data.get('question', 'N/A')}")
            st.markdown(f"**Timestamp:** {trace_data.get('timestamp', 'N/A')}")

            with st.expander("📋 Full Prompt Snapshot", expanded=True):
                st.code(trace_data.get("prompt_snapshot", "N/A"), language="text")

            with st.expander("📊 Execution Statistics"):
                st.json(trace_data.get("execution_statistics", {}))

            with st.expander("⚙️ Configuration Snapshot"):
                st.json(trace_data.get("configuration_snapshot", {}))

            with st.expander("🔗 Retrieved Chunk References"):
                for ref in trace_data.get("retrieved_chunk_references", []):
                    st.json(ref)

        except Exception as e:
            st.error(f"Error loading trace: {e}")


def render_exports():
    """Export functionality for sessions, memory DB, and traces."""
    st.markdown("### 📁 Export Data")

    export_type = st.selectbox(
        "What to export",
        ["Current Session (JSON)", "Current Session (Markdown)", "Current Session (CSV)",
         "All Memory (JSON)", "All Traces (JSON)"]
    )

    if st.button("Export"):
        try:
            if "Current Session" in export_type:
                sessions = mm.list_sessions()
                if not sessions:
                    st.warning("No sessions available.")
                    return

                session_id = st.session_state.get("current_session_id", sessions[0].session_id)

                if "JSON" in export_type:
                    data = mm.export_session(session_id, format="json")
                    st.download_button(
                        "📥 Download JSON",
                        json.dumps(data, indent=2, default=str),
                        f"session_{session_id}.json",
                        "application/json",
                    )
                elif "Markdown" in export_type:
                    data = mm.export_session(session_id, format="markdown")
                    st.download_button(
                        "📥 Download Markdown",
                        data.get("markdown", ""),
                        f"session_{session_id}.md",
                        "text/markdown",
                    )
                elif "CSV" in export_type:
                    data = mm.export_session(session_id, format="csv")
                    st.download_button(
                        "📥 Download CSV",
                        data.get("csv", ""),
                        f"session_{session_id}.csv",
                        "text/csv",
                    )

            elif "All Memory" in export_type:
                memories = mm.store.get_all_memories()
                data = [m.to_dict() for m in memories]
                st.download_button(
                    "📥 Download All Memories",
                    json.dumps(data, indent=2, default=str),
                    "all_memories.json",
                    "application/json",
                )

            elif "All Traces" in export_type:
                trace_dir = os.path.join(PROJECT_ROOT, "artifacts", "rag_traces")
                traces = []
                if os.path.exists(trace_dir):
                    for root, dirs, files in os.walk(trace_dir):
                        for f in files:
                            if f.endswith(".json"):
                                with open(os.path.join(root, f), "r") as fh:
                                    traces.append(json.load(fh))
                st.download_button(
                    "📥 Download All Traces",
                    json.dumps(traces, indent=2, default=str),
                    "all_traces.json",
                    "application/json",
                )

        except Exception as e:
            st.error(f"Export failed: {e}")

    # Import
    st.markdown("---")
    st.markdown("### 📤 Import Session")
    uploaded = st.file_uploader("Upload a session JSON file", type=["json"])
    if uploaded:
        try:
            data = json.load(uploaded)
            if st.button("Import Session"):
                session = mm.import_session(data)
                st.success(f"Imported session: {session.title} ({session.session_id})")
        except Exception as e:
            st.error(f"Import failed: {e}")


if __name__ == "__main__":
    main()
