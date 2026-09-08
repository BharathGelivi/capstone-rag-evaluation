"""
Reusable UI Components for X-RAG.

Shared rendering functions used across multiple pages.
"""

import streamlit as st
from typing import Any, Dict, List, Optional


def render_memory_card(memory, show_details: bool = True):
    """Render a single memory entry as an expandable card.

    Args:
        memory: MemoryEntry object
        show_details: Whether to show full details
    """
    with st.expander(
        f"💭 {memory.question[:60]}... — {memory.timestamp[:16]}",
        expanded=False,
    ):
        st.markdown(f"**Q:** {memory.question}")
        st.markdown(f"**A:** {memory.answer[:500]}{'...' if len(memory.answer) > 500 else ''}")

        if show_details:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Memory ID:** `{memory.memory_id}`")
                st.markdown(f"**Session:** `{memory.session_id}`")
            with col2:
                st.markdown(f"**Importance:** {memory.importance_score}")
                st.markdown(f"**Access Count:** {memory.access_count}")

            if memory.trace_id:
                st.markdown(f"**Trace:** `{memory.trace_id}`")
            if memory.retrieved_chunk_ids:
                st.markdown(f"**Chunks Used:** {len(memory.retrieved_chunk_ids)}")
            if memory.claim_ids:
                st.markdown(f"**Claims:** {len(memory.claim_ids)}")


def render_search_result(result, index: int):
    """Render a memory search result with scoring details.

    Args:
        result: MemorySearchResult object
        index: Display index (1-based)
    """
    with st.expander(
        f"#{index} — Score: {result.final_score:.3f} — {result.memory.question[:50]}...",
        expanded=index <= 3,
    ):
        st.markdown(f"**Q:** {result.memory.question}")
        st.markdown(f"**A:** {result.memory.answer[:300]}...")
        st.markdown(f"**Reason:** {result.retrieval_reason}")

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Semantic", f"{result.semantic_score:.3f}")
        with col2:
            st.metric("Recency", f"{result.recency_score:.3f}")
        with col3:
            st.metric("Frequency", f"{result.frequency_score:.3f}")
        with col4:
            st.metric("Importance", f"{result.importance_score:.3f}")


def render_verification_badge(status: str) -> str:
    """Return an icon for a verification status.

    Args:
        status: VerificationStatus string value

    Returns:
        Emoji string
    """
    icons = {
        "SUPPORTED": "✅",
        "PARTIALLY_SUPPORTED": "⚠️",
        "CONTRADICTED": "❌",
        "UNSUPPORTED": "❓",
        "NOT_VERIFIABLE": "🔍",
    }
    return icons.get(status, "❓")


def render_score_badge(score: float) -> str:
    """Return a colored HTML badge for a score value.

    Args:
        score: Numeric score (0-1)

    Returns:
        HTML string
    """
    if score >= 0.7:
        color = "#10b981"
    elif score >= 0.4:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    return (
        f'<span style="background: {color}20; color: {color}; '
        f'padding: 0.2rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem;">'
        f'{score:.3f}</span>'
    )


def render_pipeline_step(name: str, icon: str, time_seconds: float):
    """Render a pipeline timeline step.

    Args:
        name: Step name
        icon: Emoji icon
        time_seconds: Duration in seconds
    """
    if time_seconds < 1:
        time_str = f"{time_seconds * 1000:.0f}ms"
    else:
        time_str = f"{time_seconds:.1f}s"

    st.markdown(
        f"""
        <div style="text-align: center; padding: 0.5rem; background: rgba(124, 58, 237, 0.1);
             border-radius: 0.5rem; border: 1px solid rgba(124, 58, 237, 0.2);">
            <div style="font-size: 1.2rem;">{icon}</div>
            <div style="font-size: 0.7rem; color: #c4b5fd;">{name}</div>
            <div style="font-size: 0.65rem; color: #9ca3af;">{time_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
