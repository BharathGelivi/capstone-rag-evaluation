"""
Evaluation metric panel for the Streamlit UI.

Renders the thirteen scores in :data:`src.rag_eval.METRIC_PROVENANCE` for the
most recent turn, grouped by the family whose definition each name comes from.

Two design decisions worth stating, because they are what stop the panel from
being misleading:

**Nothing is computed until asked.** Five of these metrics need LLM judge calls
(roughly ten per evaluation). Recomputing them on every chat turn would triple
the cost of using the app and make the sidebar timings meaningless. The panel
computes on a button press and caches the result against the trace id.

**A missing measurement renders as "not computed", never as a number.**
Reference-based metrics are undefined without a reference answer, and citation
precision is undefined for an answer that cites nothing. Filling those with 0.0
would understate the system; filling them with 1.0 would reward silence, which
is the exact pathology E3 documented. They render as an em dash with the reason.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from src.rag_eval import METRIC_PROVENANCE

#: Display grouping. The same computation appears under two names in two
#: families (RAGAS "faithfulness" and ARES "answer faithfulness"); both are
#: shown, and the provenance table says they are the same number, rather than
#: implying two independent measurements agree.
FAMILIES = {
    "RAGAS": ["faithfulness", "answer_relevancy", "context_precision",
              "context_recall", "answer_correctness"],
    "RAGChecker-style": ["precision", "recall", "f1", "hallucination"],
    "ARES-style": ["context_relevance", "answer_relevance", "answer_faithfulness"],
    "Citation (this project)": ["citation_correctness"],
}

LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "answer_correctness": "Answer correctness",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "hallucination": "Hallucination",
    "context_relevance": "Context relevance",
    "answer_relevance": "Answer relevance",
    "answer_faithfulness": "Answer faithfulness",
    "citation_correctness": "Citation correctness",
}

#: Metrics where lower is better, so the colour scale must not be read backwards.
LOWER_IS_BETTER = {"hallucination"}


def _colour(name: str, value: Optional[float]) -> str:
    if value is None:
        return "#6b7280"
    score = 1.0 - value if name in LOWER_IS_BETTER else value
    if score >= 0.7:
        return "#10b981"
    if score >= 0.4:
        return "#f59e0b"
    return "#ef4444"


def _reason_missing(name: str, has_reference: bool, has_verification: bool,
                    cited_anything: bool) -> str:
    provenance = METRIC_PROVENANCE.get(name, {})
    if provenance.get("requires") == "reference answer" and not has_reference:
        return "needs a reference answer"
    if name in ("faithfulness", "precision", "hallucination", "answer_faithfulness",
                "f1") and not has_verification:
        return "needs Deep Analysis (claim verification)"
    if name == "citation_correctness" and not cited_anything:
        return "answer cited no authority"
    return "not computed"


def render_metric_panel(
    panel: Dict[str, Optional[float]],
    has_reference: bool = False,
    has_verification: bool = False,
    cited_anything: bool = False,
) -> None:
    """Render the computed panel."""
    for family, names in FAMILIES.items():
        st.markdown(f"##### {family}")
        columns = st.columns(len(names))
        for column, name in zip(columns, names):
            value = panel.get(name)
            with column:
                if value is None:
                    st.markdown(
                        f"<div style='font-size:0.7rem;color:#9ca3af;text-transform:uppercase;"
                        f"letter-spacing:.04em;'>{LABELS[name]}</div>"
                        f"<div style='font-size:1.25rem;font-weight:700;color:#6b7280;'>&mdash;</div>"
                        f"<div style='font-size:0.62rem;color:#6b7280;'>"
                        f"{_reason_missing(name, has_reference, has_verification, cited_anything)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='font-size:0.7rem;color:#9ca3af;text-transform:uppercase;"
                        f"letter-spacing:.04em;'>{LABELS[name]}</div>"
                        f"<div style='font-size:1.25rem;font-weight:700;"
                        f"color:{_colour(name, value)};'>{value:.3f}</div>"
                        f"<div style='font-size:0.62rem;color:#6b7280;'>"
                        f"{'lower is better' if name in LOWER_IS_BETTER else '&nbsp;'}</div>",
                        unsafe_allow_html=True,
                    )
        st.markdown("")

    with st.expander("How each metric is computed"):
        st.markdown(
            "These are computed by **this project's own judges** against the definitions "
            "below. They are comparable across retrieval arms measured here; they are not "
            "the numbers the RAGAS, RAGChecker or ARES packages would return, and are not "
            "presented as such."
        )
        rows = ["| Metric | Family | Definition | Judge |", "|---|---|---|---|"]
        for name, meta in METRIC_PROVENANCE.items():
            rows.append(
                f"| {LABELS.get(name, name)} | {meta['family']} | {meta['definition']} "
                f"| {meta['judge']} |"
            )
        st.markdown("\n".join(rows))


def render_citation_report(report) -> None:
    """Show which citations in the answer were backed by retrieved evidence."""
    if report is None:
        st.info("Citation validation has not run for this turn.")
        return

    if report.total_citations == 0:
        st.info(
            "The answer cited no case authority. Citation precision and correctness are "
            "undefined here rather than 1.0 -- an answer that cites nothing cannot be "
            "credited with citing correctly."
        )
        return

    columns = st.columns(4)
    columns[0].metric("Citations", report.total_citations)
    columns[1].metric("Grounded", report.grounded)
    columns[2].metric("Real, not retrieved", report.in_corpus_not_retrieved)
    columns[3].metric("Unverifiable", report.unverifiable)

    icons = {"GROUNDED": "✅", "IN_CORPUS_NOT_RETRIEVED": "⚠️", "UNVERIFIABLE": "❌"}
    explanations = {
        "GROUNDED": "appears in the retrieved evidence",
        "IN_CORPUS_NOT_RETRIEVED": "exists in the corpus but was not retrieved for this question",
        "UNVERIFIABLE": "not in the evidence and not in this corpus",
    }
    for verdict in report.verdicts:
        st.markdown(
            f"{icons.get(verdict.status, '❔')} **{verdict.raw}** — "
            f"{explanations.get(verdict.status, verdict.status)}"
            + (f"  \n<span style='font-size:0.75rem;color:#9ca3af;'>{verdict.source_url}</span>"
               if verdict.source_url else ""),
            unsafe_allow_html=True,
        )


def render_strategy_trace(metadata: Dict[str, Any]) -> None:
    """Show the IRCoT / agentic step log recorded in the retrieval metadata."""
    strategy = metadata.get("strategy", "plain")

    if strategy == "ircot":
        st.markdown(
            f"**IRCoT** — {metadata.get('ircot_retrieval_calls')} retrievals, "
            f"{metadata.get('ircot_llm_calls')} planner calls, stopped because "
            f"`{metadata.get('ircot_termination_reason')}`"
        )
        for step in metadata.get("ircot_steps", []):
            label = "initial query" if step["is_initial"] else f"hop {step['iteration']}"
            with st.expander(
                f"{label}: {step['query'][:70] or '(no further query)'} "
                f"— +{step['n_new_evidence']} new"
            ):
                st.markdown(f"**Looking for:** {step['reasoning_summary'] or '—'}")
                st.markdown(f"**Query:** `{step['query'] or '—'}`")
                st.markdown(f"**Retrieved:** {len(step['retrieved_chunk_ids'])} chunks, "
                            f"{step['n_new_evidence']} previously unseen")
                st.markdown(f"**Latency:** {step['latency_s']}s")
                if step["termination_reason"]:
                    st.markdown(f"**Stopped:** `{step['termination_reason']}`")

    elif strategy == "agentic":
        st.markdown(
            f"**Agentic controller** — actions: "
            f"`{' → '.join(metadata.get('agent_actions', []))}`, stopped because "
            f"`{metadata.get('agent_termination_reason')}`"
        )
        if metadata.get("agent_contradiction_searched"):
            st.success("Contradiction search ran for this question.")
        for step in metadata.get("agent_step_log", []):
            with st.expander(
                f"step {step['step']}: {step['action']} — +{step['n_new_evidence']} new"
            ):
                st.markdown(f"**Why:** {step['why'] or '—'}")
                st.markdown(f"**Argument:** `{step['arg'] or '—'}`")
                if step["graph_hops"]:
                    st.markdown(f"**Graph hops:** {step['graph_hops']}")
                    for path in step["relation_paths"][:5]:
                        st.markdown(f"- `{' → '.join(path)}`")
                if step["note"]:
                    st.caption(step["note"])
    else:
        st.caption(
            "Single-pass retrieval: no interleaved reasoning or controller steps to show."
        )
