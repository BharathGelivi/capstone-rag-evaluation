import json
from typing import List, Dict, Any, Optional
from src.report import DiagnosticEvaluationReport

class DiagnosticReportPresenter:
    """
    Renders the DiagnosticEvaluationReport into human-readable formats.
    Performs no reasoning or metric calculation.
    """
    def __init__(self, report: DiagnosticEvaluationReport):
        self.report = report

    @staticmethod
    def _format_metric(value: Optional[float]) -> str:
        return f"{value:.4f}" if value is not None else "N/A"

    @staticmethod
    def _retriever_efficiency_line(stage) -> Optional[str]:
        """
        Informational-only line surfacing RETRIEVER's chunk_utilization_rate
        (see PipelineStateAnalyzer / configs.thresholds.LOW_RETRIEVAL_EFFICIENCY_THRESHOLD).
        Returns None if the stage isn't RETRIEVER or the metadata isn't present
        (e.g. reports built before this field existed).
        """
        if stage.stage != "RETRIEVER":
            return None
        rate = stage.metadata.get("chunk_utilization_rate")
        if rate is None:
            return None
        used = stage.metadata.get("chunks_used")
        retrieved = stage.metadata.get("chunks_retrieved")
        return f"Efficiency: {used}/{retrieved} chunks used ({rate * 100:.0f}%)"

    def _get_traceability_data(self) -> Dict[str, Any]:
        """Extracts traceability paths from the existing report data."""
        claim_ids = [e.claim_id for e in self.report.evidence_analysis]
        chunk_ids = [e.supporting_chunk_id for e in self.report.evidence_analysis if e.supporting_chunk_id]
        
        return {
            "primary_cause": self.report.root_cause_analysis.primary_cause,
            "derived_from": "RootCauseAnalysis",
            "pipeline_stage": self.report.root_cause_analysis.primary_cause.replace("_FAILURE", ""),
            "supporting_claims": claim_ids,
            "supporting_chunks": chunk_ids,
            "supporting_verifications": [f"V_{cid}" for cid in claim_ids] # Derived for representation
        }

    def _get_appendix(self) -> str:
        t_id = self.report.framework_metadata.trace_id
        return f"""
Artifacts Generated

✓ RAGTrace
artifacts/rag_traces/{t_id}.json

✓ ClaimSet
artifacts/claim_sets/{t_id}.json

✓ VerificationResults
artifacts/verification/{t_id}.json

✓ PipelineStateMatrix
artifacts/pipeline_state_matrix/{t_id}.json

✓ RootCauseAnalysis
artifacts/root_cause_analysis/{t_id}.json

✓ CorrectiveActionPlan
artifacts/corrective_action_plan/{t_id}.json

✓ DiagnosticEvaluationReport
artifacts/reports/{t_id}.json
"""

    def render_console(self) -> str:
        r = self.report
        lines = []
        lines.append("==========================================================")
        lines.append(f"X-RAG DIAGNOSTIC EVALUATION REPORT")
        lines.append("==========================================================")
        lines.append(f"Framework Version : {r.framework_metadata.framework_version}")
        lines.append(f"Trace ID          : {r.framework_metadata.trace_id}")
        lines.append(f"Timestamp         : {r.framework_metadata.analysis_timestamp}")
        lines.append(f"Analysis Status   : {r.analysis_status}")
        
        lines.append("\n--- 2. EXECUTIVE SUMMARY ---")
        lines.append(f"Question         : {r.executive_summary.question}")
        lines.append(f"Generated Answer : {r.executive_summary.generated_answer}")
        lines.append(f"Health Score     : {r.executive_summary.overall_health_score:.2f}")
        lines.append(f"Primary Issue    : {r.executive_summary.primary_issue}")
        lines.append(f"Claims           : {r.executive_summary.supported_claims} / {r.executive_summary.total_claims} supported")
        lines.append(f"Summary          : {r.executive_summary.summary}")

        lines.append("\n--- 3. PIPELINE OVERVIEW ---")
        for stage in r.pipeline_overview.pipeline_stages:
            lines.append(f"[{stage.stage}] {stage.status} (Conf: {stage.confidence}) - {stage.observation}")
            efficiency_line = self._retriever_efficiency_line(stage)
            if efficiency_line:
                lines.append(f"  {efficiency_line}")

        lines.append("\n--- 4. EVALUATION METRICS ---")
        em = r.evaluation_metrics
        lines.append(f"Retrieved Chunks : {em.retrieved_chunks}")
        lines.append(f"Verified Claims  : {em.verified_claims}")
        lines.append(f"Supported        : {em.supported_claims}")
        lines.append(f"Partially Supp.  : {em.partially_supported_claims}")
        lines.append(f"Unsupported      : {em.unsupported_claims}")
        lines.append(f"Contradicted     : {em.contradicted_claims}")
        lines.append(f"Grounding Score  : {em.grounding_score:.2f}")
        lines.append(f"Evidence Coverage: {em.evidence_coverage:.2f}")
        lines.append(f"Avg Entailment   : {em.average_entailment:.2f}")
        lines.append(f"Retrieval Latency: {em.retrieval_latency_ms:.2f} ms")
        lines.append(f"Gen Latency      : {em.generation_latency_ms:.2f} ms")
        lines.append(f"Ver Latency      : {em.verification_latency_ms:.2f} ms")

        lines.append("\n--- 5. RAGAS METRICS ---")
        rm = r.ragas_metrics
        if rm is not None:
            lines.append(f"Faithfulness      : {self._format_metric(rm.faithfulness)}")
            lines.append(f"Answer Relevancy  : {self._format_metric(rm.answer_relevancy)}")
            lines.append(f"Context Precision : {self._format_metric(rm.context_precision)}")
            lines.append(f"Context Relevancy : {self._format_metric(rm.context_relevancy)}")
            lines.append(f"Context Recall    : {self._format_metric(rm.context_recall)}")
            lines.append(f"Answer Similarity : {self._format_metric(rm.answer_similarity)}")
            lines.append(f"Answer Correctness: {self._format_metric(rm.answer_correctness)}")
        else:
            lines.append("Not computed for this trace.")

        lines.append("\n--- ANSWER CORRECTNESS (CLAIM RECALL) ---")
        ac = r.answer_correctness
        if ac is not None:
            lines.append(f"Claim Recall     : {self._format_metric(ac.claim_recall)}")
            lines.append(f"Gold Claims      : {ac.recalled_gold_claims} / {ac.total_gold_claims} recalled")
        else:
            lines.append("Not computed for this trace.")

        lines.append("\n--- 6. EVIDENCE ANALYSIS ---")
        for ev in r.evidence_analysis:
            lines.append(f"- [{ev.claim_id}] {ev.verification_status}: {ev.claim_text}")
            lines.append(f"  Evidence (Chunk {ev.supporting_chunk_id}): {ev.supporting_evidence}")

        lines.append("\n--- 7. ROOT CAUSE ANALYSIS ---")
        rc = r.root_cause_analysis
        lines.append(f"Primary Cause: {rc.primary_cause}")
        lines.append(f"Confidence   : {rc.diagnosis_confidence}")
        lines.append("Reasoning:")
        for res in rc.reasoning_chain:
            lines.append(f"  - {res}")

        lines.append("\n--- 8. CORRECTIVE ACTION PLAN ---")
        for cap in r.corrective_actions:
            lines.append(f"[{cap.priority.upper()}] {cap.title}")
            lines.append(f"  Desc: {cap.description}")
            lines.append(f"  Expected: {cap.expected_improvement}")
            lines.append(f"  Metric: {cap.success_metric}")

        lines.append("\n--- 9. EVIDENCE TRACEABILITY ---")
        tr = self._get_traceability_data()
        lines.append(f"Primary Cause : {tr['primary_cause']}")
        lines.append(f"Derived From  : {tr['derived_from']}")
        lines.append(f"Pipeline Stage: {tr['pipeline_stage']}")
        lines.append(f"Supporting Claims: {', '.join(tr['supporting_claims']) if tr['supporting_claims'] else 'None'}")
        lines.append(f"Supporting Chunks: {', '.join(tr['supporting_chunks']) if tr['supporting_chunks'] else 'None'}")

        lines.append("\n--- 10. OVERALL ASSESSMENT ---")
        oa = r.overall_assessment
        lines.append(f"Strength      : {oa.major_strength}")
        lines.append(f"Weakness      : {oa.major_weakness}")
        lines.append(f"Next Priority : {oa.next_priority}")
        lines.append(f"Recommendation: {oa.overall_recommendation}")

        lines.append("\n--- 11. FRAMEWORK FOOTER ---")
        lines.append(f"Generated By: {r.framework_metadata.framework_name} v{r.framework_metadata.framework_version}")
        lines.append(f"Artifact Version: {r.artifact_version}")

        lines.append("\n==========================================================")
        lines.append("APPENDIX")
        lines.append("==========================================================")
        lines.append(self._get_appendix().strip())
        
        return "\n".join(lines)

    def render_markdown(self) -> str:
        r = self.report
        tr = self._get_traceability_data()
        
        md = f"""# Diagnostic Evaluation Report

## 1. Framework Information
- **Framework:** {r.framework_metadata.framework_name} (v{r.framework_metadata.framework_version})
- **Trace ID:** `{r.framework_metadata.trace_id}`
- **Timestamp:** {r.framework_metadata.analysis_timestamp}
- **Status:** {r.analysis_status}

## 2. Executive Summary
- **Question:** {r.executive_summary.question}
- **Answer:** {r.executive_summary.generated_answer}
- **Health Score:** {r.executive_summary.overall_health_score:.2f}
- **Primary Issue:** {r.executive_summary.primary_issue}
- **Claims Supported:** {r.executive_summary.supported_claims} / {r.executive_summary.total_claims}
- **Summary:** {r.executive_summary.summary}

## 3. Pipeline Overview
| Stage | Status | Confidence | Observation |
|---|---|---|---|
"""
        for stage in r.pipeline_overview.pipeline_stages:
            md += f"| {stage.stage} | {stage.status} | {stage.confidence} | {stage.observation} |\n"

        efficiency_lines = [
            self._retriever_efficiency_line(stage) for stage in r.pipeline_overview.pipeline_stages
        ]
        efficiency_lines = [line for line in efficiency_lines if line]
        if efficiency_lines:
            md += "\n" + "\n".join(f"*{line}*" for line in efficiency_lines) + "\n"

        em = r.evaluation_metrics
        md += f"""
## 4. Evaluation Metrics
- **Retrieved Chunks:** {em.retrieved_chunks}
- **Verified Claims:** {em.verified_claims} (Supported: {em.supported_claims}, Partial: {em.partially_supported_claims}, Unsupported: {em.unsupported_claims}, Contradicted: {em.contradicted_claims})
- **Grounding Score:** {em.grounding_score:.2f}
- **Evidence Coverage:** {em.evidence_coverage:.2f}
- **Average Entailment:** {em.average_entailment:.2f}
- **Latencies:** Retrieval ({em.retrieval_latency_ms:.1f}ms) | Generation ({em.generation_latency_ms:.1f}ms) | Verification ({em.verification_latency_ms:.1f}ms)

## 5. RAGAS Metrics
"""
        rm = r.ragas_metrics
        if rm is not None:
            md += f"""- **Faithfulness:** {self._format_metric(rm.faithfulness)}
- **Answer Relevancy:** {self._format_metric(rm.answer_relevancy)}
- **Context Precision:** {self._format_metric(rm.context_precision)}
- **Context Relevancy:** {self._format_metric(rm.context_relevancy)}
- **Context Recall:** {self._format_metric(rm.context_recall)}
- **Answer Similarity:** {self._format_metric(rm.answer_similarity)}
- **Answer Correctness:** {self._format_metric(rm.answer_correctness)}
"""
        else:
            md += "Not computed for this trace.\n"

        md += "\n## Answer Correctness (Claim Recall)\n"
        ac = r.answer_correctness
        if ac is not None:
            md += f"- **Claim Recall:** {self._format_metric(ac.claim_recall)}\n"
            md += f"- **Gold Claims:** {ac.recalled_gold_claims} / {ac.total_gold_claims} recalled\n"
        else:
            md += "Not computed for this trace.\n"

        md += "\n## 6. Evidence Analysis\n"
        for ev in r.evidence_analysis:
            md += f"- **[{ev.claim_id}]** ({ev.verification_status}): {ev.claim_text}\n"
            md += f"  - *Evidence (Chunk {ev.supporting_chunk_id}):* {ev.supporting_evidence}\n"

        rc = r.root_cause_analysis
        md += f"""
## 7. Root Cause Analysis
- **Primary Cause:** {rc.primary_cause}
- **Confidence:** {rc.diagnosis_confidence}
- **Reasoning:**
"""
        for res in rc.reasoning_chain:
            md += f"  - {res}\n"

        md += "\n## 8. Corrective Action Plan\n"
        for cap in r.corrective_actions:
            md += f"- **[{cap.priority.upper()}] {cap.title}**\n"
            md += f"  - Description: {cap.description}\n"
            md += f"  - Expected Improvement: {cap.expected_improvement} (Metric: {cap.success_metric})\n"

        claims_str = ", ".join(tr['supporting_claims']) if tr['supporting_claims'] else "None"
        chunks_str = ", ".join(tr['supporting_chunks']) if tr['supporting_chunks'] else "None"

        md += f"""
## 9. Evidence Traceability
- **Primary Cause:** {tr['primary_cause']}
- **Derived From:** {tr['derived_from']}
- **Pipeline Stage:** {tr['pipeline_stage']}
- **Supporting Claims:** {claims_str}
- **Supporting Chunks:** {chunks_str}

## 10. Overall Assessment
- **Major Strength:** {r.overall_assessment.major_strength}
- **Major Weakness:** {r.overall_assessment.major_weakness}
- **Next Priority:** {r.overall_assessment.next_priority}
- **Recommendation:** {r.overall_assessment.overall_recommendation}

## 11. Framework Footer
- **Generated By:** {r.framework_metadata.framework_name}
- **Artifact Version:** {r.artifact_version}

---

## APPENDIX

{self._get_appendix().strip()}
"""
        return md

    def render_html(self) -> str:
        # Convert markdown to simple semantic HTML manually to avoid dependencies
        md = self.render_markdown()
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Diagnostic Report - {self.report.framework_metadata.trace_id}</title>
<style>
    body {{ font-family: sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }}
    h1, h2 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background-color: #f4f4f4; }}
    .appendix {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #ddd; }}
</style>
</head>
<body>
    <pre>{md}</pre>
</body>
</html>"""
        return html

    def render_pdf(self) -> bytes:
        """Renders the report as PDF bytes by converting render_html()'s markup via xhtml2pdf."""
        from io import BytesIO
        from xhtml2pdf import pisa

        buffer = BytesIO()
        result = pisa.CreatePDF(src=self.render_html(), dest=buffer)
        if result.err:
            raise RuntimeError("Failed to render diagnostic report as PDF.")
        return buffer.getvalue()
