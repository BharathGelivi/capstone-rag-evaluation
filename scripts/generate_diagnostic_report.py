"""
Generates a single, detailed, self-contained HTML diagnostic report for one
RAGTrace: the case (question/answer), the full retrieval trace, claim
decomposition + verification, chunk-level provenance, X-RAG's stage
attribution and root-cause reasoning chain, and -- if a matching row exists
in artifacts/benchmark_comparison/results.json -- a side-by-side comparison
against real RAGAS, RAGChecker, and ARES scores for the same example.

Unlike scripts/generate_comparison_report.py (aggregate, paper-facing,
markdown), this is a per-example, human-browsable HTML report meant to be
opened directly in a browser: heavy detail (full trace, every claim, every
chunk) is tucked into native <details>/<summary> disclosure widgets so the
page opens compact and expands on demand, with no JavaScript required.

Usage:
    python -m scripts.generate_diagnostic_report --trace-id <trace_id>

Output: artifacts/diagnostic_reports/<trace_id>.html
"""

import argparse
import glob
import html
import json
import os
from typing import Any, Dict, List, Optional

REPORTS_DIR = "artifacts/reports"
TRACES_DIR = "artifacts/rag_traces"
OUTPUT_DIR = "artifacts/diagnostic_reports"
BASELINE_RESULTS_PATH = "artifacts/benchmark_comparison/results.json"
CHUNK_REGISTRY_PATH = "artifacts/chunk_registry.json"


def load_report(trace_id: str) -> Dict[str, Any]:
    path = os.path.join(REPORTS_DIR, f"{trace_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_trace_path(trace_id: str) -> Optional[str]:
    matches = glob.glob(os.path.join(TRACES_DIR, "**", f"trace_{trace_id}.json"), recursive=True)
    return matches[0] if matches else None


def load_trace(trace_id: str) -> Dict[str, Any]:
    path = find_trace_path(trace_id)
    if path is None:
        raise FileNotFoundError(f"No trace file found for trace_id={trace_id} under {TRACES_DIR}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_baseline_row(trace_id: str, results_path: str = BASELINE_RESULTS_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(results_path):
        return None
    with open(results_path, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        if row.get("trace_id") == trace_id:
            return row
    return None


def load_chunk_records(chunk_ids: List[str], registry_path: str = CHUNK_REGISTRY_PATH) -> Dict[str, Dict[str, Any]]:
    """Loads only the chunk records referenced by this trace, keyed by chunk_id."""
    if not os.path.exists(registry_path):
        return {}
    with open(registry_path, encoding="utf-8") as f:
        data = json.load(f)
    all_records = data.get("records", data) if isinstance(data, dict) else data
    if isinstance(all_records, dict):
        candidates = all_records.values()
    else:
        candidates = all_records
    by_id = {r["chunk_id"]: r for r in candidates if r.get("chunk_id") in chunk_ids}
    return by_id


def esc(value: Any) -> str:
    return html.escape(str(value)) if value is not None else ""


def fmt_score(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_stage_strip(pipeline_stages: List[Dict[str, Any]]) -> str:
    cards = []
    for stage in pipeline_stages:
        status = stage.get("status", "UNKNOWN")
        css_class = "unknown" if status == "UNKNOWN" else ("fail" if status in ("FAIL", "CONTRADICTED") else "")

        efficiency_html = ""
        metadata = stage.get("metadata") or {}
        rate = metadata.get("chunk_utilization_rate")
        if stage.get("stage") == "RETRIEVER" and rate is not None:
            used = metadata.get("chunks_used")
            retrieved = metadata.get("chunks_retrieved")
            efficiency_html = f'<div class="stage-efficiency">Efficiency: {esc(used)}/{esc(retrieved)} chunks used ({rate * 100:.0f}%)</div>'

        cards.append(f"""
        <div class="stage {css_class}">
          <div class="stage-name">{esc(stage.get('stage'))}</div>
          <div class="stage-status">{esc(status)}</div>
          <div class="stage-obs">{esc(stage.get('observation'))}</div>
          <div class="stage-conf">confidence {esc(stage.get('confidence'))}</div>
          {efficiency_html}
        </div>""")
    return "".join(cards)


def load_answer_correctness(trace_id: str, base_dir: str = "artifacts/answer_correctness") -> Optional[Dict[str, Any]]:
    path = os.path.join(base_dir, f"TRACE_{trace_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render_answer_correctness(correctness: Optional[Dict[str, Any]]) -> str:
    """
    Renders src/answer_correctness_evaluator.py's AnswerCorrectnessSummary.
    Computed automatically by PipelineRunner.run() whenever a gold_answer is
    passed in (see src/runner.py); traces run without a gold answer (e.g. no
    gold_answer column, or an ad-hoc query.py run) won't have one.
    """
    if correctness is None:
        return (
            "<p class='muted'>Not computed for this trace -- no gold answer was available when this trace was "
            "run. Can be computed after the fact via <span class='mono'>scripts/evaluate_answer_correctness.py</span>.</p>"
        )

    recall = correctness.get("claim_recall")
    total = correctness.get("total_gold_claims")
    gold_answer = correctness.get("gold_answer", "")
    results = correctness.get("results", [])

    status_class = {"SUPPORTED": "good", "CONTRADICTED": "bad", "NOT_VERIFIABLE": "warn", "PARTIALLY_SUPPORTED": "warn"}
    rows = []
    for r in results:
        status = r.get("verification_status", "UNKNOWN")
        css = status_class.get(status, "")
        rows.append(f"""
        <details class="claim-detail">
          <summary>
            <span class="dot {css}"></span>
            <span class="claim-status {css}">{esc(status)}</span>
            <span class="claim-summary-text">{esc(r.get('claim_text'))}</span>
          </summary>
          <div class="claim-body">
            <div class="claim-field"><b>Confidence:</b> <span class="mono">{fmt_score(r.get('confidence'))}</span></div>
            <div class="claim-field"><b>Best-matching sentence in generated answer:</b></div>
            <pre class="evidence-text">{esc(r.get('best_matching_sentence') or '(no matching sentence found)')}</pre>
          </div>
        </details>""")

    return f"""
    <div class="summary-grid" style="grid-template-columns: repeat(2, 1fr); margin-bottom: 14px;">
      <div class="summary-tile"><div class="tile-label">Claim Recall</div><div class="tile-val mono">{fmt_score(recall)}</div></div>
      <div class="summary-tile"><div class="tile-label">Gold Claims Recalled</div><div class="tile-val mono">{sum(1 for r in results if r.get('verification_status') in ('SUPPORTED', 'PARTIALLY_SUPPORTED'))} / {esc(total)}</div></div>
    </div>
    <div class="claim-field"><b>Gold answer used:</b></div>
    <pre class="evidence-text">{esc(gold_answer)}</pre>
    {"".join(rows)}
    """

def render_native_ragas_metrics(ragas_metrics: Optional[Dict[str, Any]]) -> str:
    if ragas_metrics is None:
        return "<p class='muted'>Not computed for this trace.</p>"
    
    def metric(label: str, key: str, digits: int = 3) -> str:
        value = ragas_metrics.get(key)
        css = "hi" if (isinstance(value, (int, float)) and value >= 0.7) else ("mid" if isinstance(value, (int, float)) else "")
        # inline color logic similar to .m-val.hi / .m-val.mid
        color = "var(--good)" if css == "hi" else ("var(--warn)" if css == "mid" else "inherit")
        return f'<div class="summary-tile"><div class="tile-label">{esc(label)}</div><div class="tile-val mono" style="color: {color};">{fmt_score(value, digits)}</div></div>'
        
    return f"""
    <div class="summary-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 14px; row-gap: 12px;">
        {metric("Faithfulness", "faithfulness")}
        {metric("Answer Relevancy", "answer_relevancy")}
        {metric("Context Precision", "context_precision")}
        {metric("Context Relevancy", "context_relevancy")}
        {metric("Context Recall", "context_recall")}
        {metric("Answer Similarity", "answer_similarity")}
        {metric("Answer Correctness", "answer_correctness")}
    </div>
    """


def render_claim_rows(evidence_analysis: List[Dict[str, Any]]) -> str:
    rows = []
    status_class = {"SUPPORTED": "good", "CONTRADICTED": "bad", "NOT_VERIFIABLE": "warn", "PARTIALLY_SUPPORTED": "warn"}
    for claim in evidence_analysis:
        status = claim.get("verification_status", "UNKNOWN")
        css = status_class.get(status, "")
        rows.append(f"""
        <details class="claim-detail">
          <summary>
            <span class="dot {css}"></span>
            <span class="claim-status {css}">{esc(status)}</span>
            <span class="claim-summary-text">{esc(claim.get('claim_text'))}</span>
          </summary>
          <div class="claim-body">
            <div class="claim-field"><b>Claim ID:</b> <span class="mono">{esc(claim.get('claim_id'))}</span></div>
            <div class="claim-field"><b>Supporting chunk:</b> <span class="mono">{esc(claim.get('supporting_chunk_id') or 'none')}</span> (rank {esc(claim.get('supporting_chunk_rank'))})</div>
            <div class="claim-field"><b>Evidence text:</b></div>
            <pre class="evidence-text">{esc(claim.get('supporting_evidence') or '(no evidence found)')}</pre>
          </div>
        </details>""")
    return "".join(rows)


def render_chunk_mapping(chunk_refs: List[Dict[str, Any]], chunk_records: Dict[str, Dict[str, Any]]) -> str:
    rows = []
    for ref in chunk_refs:
        chunk_id = ref.get("chunk_id")
        record = chunk_records.get(chunk_id, {})
        rows.append(f"""
        <details class="chunk-detail">
          <summary>
            <span class="chunk-rank">#{esc(ref.get('rank'))}</span>
            <span class="mono chunk-id">{esc(chunk_id)}</span>
            <span class="chunk-src">{esc(ref.get('source_file'))} p.{esc(ref.get('page_number'))}</span>
          </summary>
          <div class="chunk-body">
            <div class="score-grid">
              <div><span class="m-name">Dense score</span><span class="mono">{fmt_score(ref.get('dense_score'))}</span></div>
              <div><span class="m-name">Sparse (BM25)</span><span class="mono">{fmt_score(ref.get('sparse_score'), 2)}</span></div>
              <div><span class="m-name">RRF score</span><span class="mono">{fmt_score(ref.get('rrf_score'), 4)}</span></div>
              <div><span class="m-name">Reranker score</span><span class="mono">{fmt_score(ref.get('reranker_score'))}</span></div>
              <div><span class="m-name">Dense rank</span><span class="mono">{esc(ref.get('dense_rank'))}</span></div>
              <div><span class="m-name">Sparse rank</span><span class="mono">{esc(ref.get('sparse_rank'))}</span></div>
            </div>
            <div class="claim-field"><b>Parent document:</b> <span class="mono">{esc(ref.get('parent_document_id'))}</span></div>
            <div class="claim-field"><b>Chunk index in document:</b> <span class="mono">{esc(ref.get('chunk_index'))}</span></div>
            {f'''<div class="claim-field"><b>Configured chunk size / overlap:</b> <span class="mono">{esc(record.get("configured_chunk_size"))} / {esc(record.get("configured_chunk_overlap"))}</span></div>
            <div class="claim-field"><b>Character span:</b> <span class="mono">{esc(record.get("character_start"))} - {esc(record.get("character_end"))} ({esc(record.get("text_length"))} chars)</span></div>
            <div class="claim-field"><b>Full chunk text:</b></div>
            <pre class="evidence-text">{esc(record.get("text"))}</pre>''' if record else '<div class="claim-field muted">Full chunk record not found in current chunk_registry.json (corpus may have been re-ingested since this trace ran).</div>'}
          </div>
        </details>""")
    return "".join(rows)


def render_reasoning_chain(chain: List[str]) -> str:
    items = "".join(f"<li>{esc(step)}</li>" for step in chain)
    return f"<ol class='reasoning-chain'>{items}</ol>"


def render_corrective_actions(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "<p class='muted'>None recommended -- pipeline is healthy for this example.</p>"
    items = []
    for action in actions:
        priority = action.get("priority", "")
        title = action.get("action_type", action.get("title", "Corrective Action"))
        priority_badge = f'<span class="claim-status" style="margin-right:8px;">[{esc(priority.upper())}]</span>' if priority else ""
        items.append(f"""
        <div class="action-card">
          <div class="action-title">{priority_badge}{esc(title)}</div>
          <div class="action-desc">{esc(action.get('description', ''))}</div>
          {f"<div class='claim-field muted' style='margin-top:6px;'>{esc(action.get('observed_evidence'))}</div>" if action.get('observed_evidence') else ""}
        </div>""")
    return "".join(items)


def render_framework_comparison(xrag_score: Optional[float], xrag_cause: str, baseline_row: Optional[Dict[str, Any]]) -> str:
    if baseline_row is None:
        return "<p class='muted'>No baseline comparison row found for this trace_id in artifacts/benchmark_comparison/results.json -- run scripts/run_baseline_comparison.py to populate it.</p>"

    def metric(label: str, key: str, digits: int = 3) -> str:
        value = baseline_row.get(key)
        css = "hi" if (isinstance(value, (int, float)) and value >= 0.7) else ("mid" if isinstance(value, (int, float)) else "")
        return f'<div class="metric-row"><span class="m-name">{esc(label)}</span><span class="m-val {css} mono">{fmt_score(value, digits)}</span></div>'

    return f"""
    <div class="fw-grid">
      <div class="fw-card xrag">
        <div class="fw-name">X-RAG<span class="tag">this project</span></div>
        <div class="fw-desc">Claim-level NLI verification + root-cause attribution to a specific pipeline stage.</div>
        <div class="metric-list">
          <div class="metric-row"><span class="m-name">Avg. entailment</span><span class="m-val hi mono">{fmt_score(xrag_score)}</span></div>
          <div class="metric-row"><span class="m-name">Primary cause</span><span class="m-val mono">{esc(xrag_cause)}</span></div>
        </div>
      </div>
      <div class="fw-card ragas">
        <div class="fw-name">RAGAS<span class="tag">real ragas==0.2.15</span></div>
        <div class="fw-desc">LLM-judged faithfulness, relevancy, and context quality. No stage attribution.</div>
        <div class="metric-list">
          {metric("Faithfulness", "ragas_faithfulness")}
          {metric("Answer relevancy", "ragas_answer_relevancy")}
          {metric("Context precision", "ragas_context_precision")}
          {metric("Context recall", "ragas_context_recall")}
          {metric("Answer correctness", "ragas_answer_correctness")}
        </div>
      </div>
      <div class="fw-card">
        <div class="fw-name">RAGChecker<span class="tag">retriever vs. generator</span></div>
        <div class="fw-desc">Claim-level precision/recall against the gold answer.</div>
        <div class="metric-list">
          {metric("Precision", "ragchecker_precision")}
          {metric("Recall", "ragchecker_recall")}
          {metric("F1", "ragchecker_f1")}
          {metric("Faithfulness", "ragchecker_faithfulness")}
          {metric("Hallucination", "ragchecker_hallucination")}
        </div>
      </div>
      <div class="fw-card">
        <div class="fw-name">ARES<span class="tag">ues_idp judge</span></div>
        <div class="fw-desc">Per-document LLM relevance and faithfulness judgments.</div>
        <div class="metric-list">
          {metric("Context relevance", "ares_context_relevance")}
          {metric("Answer relevance", "ares_answer_relevance")}
          {metric("Answer faithfulness", "ares_answer_faithfulness")}
        </div>
      </div>
    </div>"""


STYLE = """
:root {
  --bg: #F4F5F7; --surface: #FFFFFF; --ink: #1A1D24; --ink-soft: #565C68; --ink-faint: #8A8F9A;
  --accent: #2E3B6E; --good: #2F7D5D; --warn: #A6742D; --bad: #A6402D;
  --line: #E0E2E7; --line-strong: #C9CCD3;
}
:root[data-theme="dark"] {
  --bg: #12141A; --surface: #1B1E26; --ink: #E7E9EE; --ink-soft: #A6ACB8; --ink-faint: #6B7280;
  --accent: #8C9BE8; --good: #5FCB9B; --warn: #E3B15C; --bad: #E38C7A;
  --line: #2A2E38; --line-strong: #383D4A;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12141A; --surface: #1B1E26; --ink: #E7E9EE; --ink-soft: #A6ACB8; --ink-faint: #6B7280;
    --accent: #8C9BE8; --good: #5FCB9B; --warn: #E3B15C; --bad: #E38C7A;
    --line: #2A2E38; --line-strong: #383D4A;
  }
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.55; padding: 48px 24px 80px; }
.page { max-width: 940px; margin: 0 auto; }
.mono { font-family: ui-monospace, "SF Mono", "Cascadia Mono", Consolas, monospace; font-variant-numeric: tabular-nums; }
.muted { color: var(--ink-faint); }
header { border-bottom: 2px solid var(--ink); padding-bottom: 20px; margin-bottom: 32px; }
.eyebrow { font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); font-weight: 600; margin-bottom: 10px; }
h1 { font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif; font-size: 1.8rem; font-weight: 600; line-height: 1.2; text-wrap: balance; margin: 0 0 10px; }
.dek { color: var(--ink-soft); font-size: 1rem; max-width: 70ch; }
.meta-row { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 16px; font-size: 0.82rem; color: var(--ink-faint); }
.meta-row span b { color: var(--ink-soft); font-weight: 600; }
.section-title { font-size: 0.78rem; letter-spacing: 0.07em; text-transform: uppercase; color: var(--ink-faint); font-weight: 600; margin: 40px 0 14px; display: flex; align-items: center; gap: 10px; }
.section-title::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.summary-tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.summary-tile .tile-label { font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase; color: var(--ink-faint); font-weight: 700; margin-bottom: 6px; }
.summary-tile .tile-val { font-size: 1.3rem; font-weight: 700; }
.case { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 22px 24px; }
.case .q-label, .case .a-label { font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; font-weight: 700; color: var(--accent); margin-bottom: 6px; }
.case .q-text { font-size: 1.05rem; font-weight: 500; margin-bottom: 18px; }
.case .a-text { font-size: 0.95rem; color: var(--ink-soft); }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px; }
details > summary { cursor: pointer; padding: 14px 18px; font-weight: 600; list-style: none; display: flex; align-items: center; gap: 10px; }
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸"; color: var(--accent); font-size: 0.8em; transition: transform 0.15s; flex-shrink: 0; }
details[open] > summary::before { transform: rotate(90deg); }
details.top-level > summary { font-size: 0.95rem; }
.details-body-outer { padding: 0 18px 18px; }
.claim-detail, .chunk-detail { margin: 8px 18px; border-radius: 8px; }
.claim-detail summary, .chunk-detail summary { padding: 10px 12px; font-weight: 500; font-size: 0.86rem; }
.claim-body, .chunk-body { padding: 4px 12px 14px; font-size: 0.84rem; }
.claim-field { margin-bottom: 6px; }
.claim-field.muted { font-style: italic; }
.evidence-text { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; font-size: 0.78rem; white-space: pre-wrap; max-height: 260px; overflow-y: auto; margin-top: 4px; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; display: inline-block; background: var(--ink-faint); }
.dot.good { background: var(--good); } .dot.warn { background: var(--warn); } .dot.bad { background: var(--bad); }
.claim-status { font-size: 0.7rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.03em; flex-shrink: 0; }
.claim-status.good { color: var(--good); } .claim-status.warn { color: var(--warn); } .claim-status.bad { color: var(--bad); }
.claim-summary-text { color: var(--ink-soft); font-weight: 400; }
.chunk-rank { font-family: ui-monospace, monospace; background: var(--bg); border-radius: 5px; padding: 2px 7px; font-size: 0.75rem; flex-shrink: 0; }
.chunk-id { flex-shrink: 0; }
.chunk-src { color: var(--ink-faint); font-weight: 400; font-size: 0.8rem; }
.score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px 20px; margin-bottom: 10px; padding: 10px; background: var(--bg); border-radius: 8px; }
.score-grid > div { display: flex; justify-content: space-between; font-size: 0.78rem; }
.stage-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
.stage { background: var(--surface); border: 1px solid var(--line); border-top: 3px solid var(--good); border-radius: 8px; padding: 12px; }
.stage.unknown { border-top-color: var(--line-strong); }
.stage.fail { border-top-color: var(--bad); }
.stage-name { font-size: 0.68rem; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 700; color: var(--ink-faint); margin-bottom: 6px; }
.stage-status { font-size: 0.85rem; font-weight: 600; color: var(--good); margin-bottom: 4px; }
.stage.unknown .stage-status { color: var(--ink-faint); }
.stage.fail .stage-status { color: var(--bad); }
.stage-obs { font-size: 0.72rem; color: var(--ink-faint); line-height: 1.4; }
.stage-conf { font-size: 0.66rem; color: var(--ink-faint); margin-top: 6px; }
.stage-efficiency { font-size: 0.66rem; color: var(--accent); margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--line); }
.reasoning-chain { padding-left: 22px; font-size: 0.88rem; color: var(--ink-soft); }
.reasoning-chain li { margin-bottom: 6px; }
.action-card { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--warn); border-radius: 0 8px 8px 0; padding: 12px 16px; margin-bottom: 8px; }
.action-title { font-weight: 700; font-size: 0.86rem; margin-bottom: 4px; }
.action-desc { font-size: 0.82rem; color: var(--ink-soft); }
.fw-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.fw-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 18px 16px; display: flex; flex-direction: column; gap: 12px; }
.fw-card.xrag { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.fw-card.ragas { border-color: var(--good); }
.fw-name { font-weight: 700; font-size: 0.95rem; }
.fw-name .tag { display: block; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: var(--accent); margin-top: 2px; }
.fw-desc { font-size: 0.72rem; color: var(--ink-faint); line-height: 1.4; min-height: 3.5em; }
.metric-list { display: flex; flex-direction: column; gap: 7px; border-top: 1px solid var(--line); padding-top: 10px; }
.metric-row { display: flex; justify-content: space-between; align-items: baseline; font-size: 0.82rem; }
.metric-row .m-val { font-weight: 700; }
.metric-row .m-val.hi { color: var(--good); }
.metric-row .m-val.mid { color: var(--warn); }
footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--line); font-size: 0.76rem; color: var(--ink-faint); display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
.scroll-x { overflow-x: auto; }
@media (max-width: 720px) {
  .fw-grid, .summary-grid { grid-template-columns: repeat(2, 1fr); }
  .stage-strip { grid-template-columns: repeat(2, 1fr); }
  .score-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
  .fw-grid, .summary-grid, .stage-strip, .score-grid { grid-template-columns: 1fr; }
  h1 { font-size: 1.4rem; }
}
"""


def generate_report(
    trace: Dict[str, Any],
    report: Dict[str, Any],
    baseline_row: Optional[Dict[str, Any]],
    answer_correctness: Optional[Dict[str, Any]] = None,
) -> str:
    trace_id = trace["trace_id"]
    exec_summary = report["executive_summary"]
    metrics = report["evaluation_metrics"]
    rca = report["root_cause_analysis"]
    pipeline_stages = report["pipeline_overview"]["pipeline_stages"]
    evidence_analysis = report.get("evidence_analysis", [])
    corrective_actions = report.get("corrective_actions", [])
    chunk_refs = trace.get("retrieved_chunk_references", [])

    chunk_ids = [ref["chunk_id"] for ref in chunk_refs]
    chunk_records = load_chunk_records(chunk_ids)

    config = trace.get("configuration_snapshot", {})
    exec_stats = trace.get("execution_statistics", {})

    health = exec_summary.get("overall_health_score")
    health_pct = f"{health * 100:.0f}%" if isinstance(health, (int, float)) else "N/A"

    return f"""<title>X-RAG Diagnostic Report — {esc(trace_id)}</title>
<style>{STYLE}</style>
<div class="page">

  <header>
    <div class="eyebrow">X-RAG Diagnostic Framework — Full Report</div>
    <h1>{esc(exec_summary.get('question'))}</h1>
    <p class="dek">Trace <span class="mono">{esc(trace_id)}</span>, generated {esc(trace.get('timestamp'))}.</p>
    <div class="meta-row">
      <span><b>Generator:</b> {esc(config.get('llm_model'))}</span>
      <span><b>Embedding:</b> {esc(config.get('embedding_model'))}</span>
      <span><b>Chunk size / overlap:</b> {esc(config.get('chunk_size'))} / {esc(config.get('chunk_overlap'))}</span>
      <span><b>Retrieval top-k:</b> {esc(config.get('retrieval_top_k'))}</span>
    </div>
  </header>

  <section>
    <div class="section-title">Executive Summary</div>
    <div class="summary-grid">
      <div class="summary-tile"><div class="tile-label">Overall Health</div><div class="tile-val mono">{health_pct}</div></div>
      <div class="summary-tile"><div class="tile-label">Primary Issue</div><div class="tile-val mono">{esc(exec_summary.get('primary_issue'))}</div></div>
      <div class="summary-tile"><div class="tile-label">Claims Supported</div><div class="tile-val mono">{esc(exec_summary.get('supported_claims'))} / {esc(exec_summary.get('total_claims'))}</div></div>
      <div class="summary-tile"><div class="tile-label">Avg. Entailment</div><div class="tile-val mono">{fmt_score(metrics.get('average_entailment'))}</div></div>
    </div>
  </section>

  <section>
    <div class="section-title">The Case</div>
    <div class="case">
      <div class="q-label">Question</div>
      <div class="q-text">{esc(exec_summary.get('question'))}</div>
      <div class="a-label">X-RAG's Answer</div>
      <div class="a-text">{esc(exec_summary.get('generated_answer'))}</div>
    </div>
  </section>

  <section>
    <div class="section-title">Raw Retrieval &amp; Generation Trace</div>
    <details class="top-level">
      <summary>View full RAG trace ({len(chunk_refs)} chunks retrieved, {fmt_score(exec_stats.get('retrieval_time'), 2)}s retrieval + {fmt_score(exec_stats.get('generation_time'), 2)}s generation)</summary>
      <div class="details-body-outer">
        <div class="claim-field"><b>Total pipeline time:</b> <span class="mono">{fmt_score(exec_stats.get('total_pipeline_time'), 3)}s</span></div>
        <div class="claim-field"><b>Pre-rerank candidate pool:</b> <span class="mono">{esc(exec_stats.get('pre_rerank_candidate_pool_size'))}</span> chunks (BM25 + dense, before RRF fusion and reranking)</div>
        <div class="claim-field"><b>Full prompt sent to the generator:</b></div>
        <pre class="evidence-text">{esc(trace.get('prompt_snapshot'))}</pre>
      </div>
    </details>
  </section>

  <section>
    <div class="section-title">Claim Decomposition &amp; Verification ({len(evidence_analysis)} atomic claims)</div>
    {render_claim_rows(evidence_analysis)}
  </section>

  <section>
    <div class="section-title">Chunk Mapping &amp; Provenance</div>
    {render_chunk_mapping(chunk_refs, chunk_records)}
  </section>

  <section>
    <div class="section-title">X-RAG Stage Attribution</div>
    <div class="stage-strip">
      {render_stage_strip(pipeline_stages)}
    </div>
  </section>

  <section>
    <div class="section-title">Root Cause Reasoning Chain</div>
    {render_reasoning_chain(rca.get('reasoning_chain', []))}
    <div class="claim-field" style="margin-top: 10px;"><b>Diagnosis confidence:</b> <span class="mono">{fmt_score(rca.get('diagnosis_confidence'))}</span></div>
  </section>

  <section>
    <div class="section-title">Corrective Actions</div>
    {render_corrective_actions(corrective_actions)}
  </section>

  <section>
    <div class="section-title">Answer Correctness (Claim Recall)</div>
    {render_answer_correctness(answer_correctness)}
  </section>

  <section>
    <div class="section-title">Native RAGAS-style Metrics</div>
    {render_native_ragas_metrics(report.get("ragas_metrics"))}
  </section>

  <section>
    <div class="section-title">Cross-Framework Comparison</div>
    <div class="scroll-x">
      {render_framework_comparison(metrics.get('average_entailment'), rca.get('primary_cause', 'UNKNOWN'), baseline_row)}
    </div>
  </section>

  <footer>
    <span>X-RAG Diagnostic Framework</span>
    <span>Trace {esc(trace_id)} -- generated from a live run, not simulated data</span>
  </footer>

</div>
"""


def generate(trace_id: str, output_dir: str = OUTPUT_DIR) -> str:
    trace = load_trace(trace_id)
    report = load_report(trace_id)
    baseline_row = load_baseline_row(trace_id)
    answer_correctness = load_answer_correctness(trace_id)

    html_content = generate_report(trace, report, baseline_row, answer_correctness)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{trace_id}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a detailed HTML diagnostic report for one trace.")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    path = generate(args.trace_id, args.output_dir)
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
