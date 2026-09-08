import json
import os
import tempfile
import unittest

from scripts.generate_diagnostic_report import (
    esc,
    fmt_score,
    render_stage_strip,
    render_claim_rows,
    render_chunk_mapping,
    render_reasoning_chain,
    render_corrective_actions,
    render_framework_comparison,
    render_answer_correctness,
    load_baseline_row,
    load_chunk_records,
    load_answer_correctness,
    generate_report,
    generate,
)


def make_trace(trace_id="T1"):
    return {
        "trace_id": trace_id,
        "timestamp": "2026-07-19T00:00:00Z",
        "generated_answer": "The answer.",
        "prompt_snapshot": "[SYSTEM]: ...\n[USER]: ...",
        "retrieved_chunk_references": [
            {
                "chunk_id": "c1", "rank": 1, "source_file": "bns.pdf", "page_number": "17",
                "dense_score": 0.68, "sparse_score": 12.79, "rrf_score": 0.0323,
                "reranker_score": 0.68, "dense_rank": 3, "sparse_rank": 1,
                "parent_document_id": "doc1", "chunk_index": 0,
            }
        ],
        "configuration_snapshot": {
            "embedding_model": "BAAI/bge-small-en-v1.5", "chunk_size": 512, "chunk_overlap": 50,
            "retrieval_top_k": 5, "llm_model": "llama-3.3-70b-versatile",
        },
        "execution_statistics": {
            "retrieval_time": 1.79, "generation_time": 1.76, "total_pipeline_time": 3.56,
            "pre_rerank_candidate_pool_size": 38,
        },
    }


def make_report():
    return {
        "executive_summary": {
            "question": "What is abetment?", "generated_answer": "The answer.",
            "overall_health_score": 0.83, "primary_issue": "UNKNOWN",
            "supported_claims": 5, "total_claims": 6,
        },
        "evaluation_metrics": {"average_entailment": 0.8367},
        "root_cause_analysis": {
            "primary_cause": "UNKNOWN", "diagnosis_confidence": 0.93,
            "reasoning_chain": ["Starting analysis.", "Assumption passed at CORPUS."],
        },
        "pipeline_overview": {
            "pipeline_stages": [
                {"stage": "CORPUS", "status": "PASS", "observation": "ok", "confidence": "0.9"},
                {"stage": "CHUNKING", "status": "UNKNOWN", "observation": "no evidence", "confidence": "1.0"},
            ]
        },
        "evidence_analysis": [
            {"claim_id": "cl1", "claim_text": "Claim one.", "verification_status": "SUPPORTED",
             "supporting_chunk_id": "c1", "supporting_chunk_rank": 1, "supporting_evidence": "Evidence text."},
            {"claim_id": "cl2", "claim_text": "Claim two.", "verification_status": "NOT_VERIFIABLE",
             "supporting_chunk_id": "c1", "supporting_chunk_rank": 1, "supporting_evidence": "Evidence text 2."},
        ],
        "corrective_actions": [],
    }


class TestEscAndFmt(unittest.TestCase):
    def test_esc_escapes_html(self):
        self.assertEqual(esc("<script>"), "&lt;script&gt;")

    def test_esc_none_returns_empty(self):
        self.assertEqual(esc(None), "")

    def test_fmt_score_formats_float(self):
        self.assertEqual(fmt_score(0.83671), "0.837")

    def test_fmt_score_none_returns_na(self):
        self.assertEqual(fmt_score(None), "N/A")

    def test_fmt_score_non_numeric_passthrough(self):
        self.assertEqual(fmt_score("UNKNOWN"), "UNKNOWN")


class TestRenderStageStrip(unittest.TestCase):
    def test_renders_all_stages(self):
        html_out = render_stage_strip([
            {"stage": "CORPUS", "status": "PASS", "observation": "ok", "confidence": "0.9"},
            {"stage": "CHUNKING", "status": "UNKNOWN", "observation": "no evidence", "confidence": "1.0"},
        ])
        self.assertIn("CORPUS", html_out)
        self.assertIn("PASS", html_out)
        self.assertIn("unknown", html_out)  # css class applied for UNKNOWN status

    def test_renders_retriever_efficiency_when_present(self):
        html_out = render_stage_strip([
            {"stage": "RETRIEVER", "status": "PASS", "observation": "ok", "confidence": "0.95",
             "metadata": {"chunk_utilization_rate": 0.25, "chunks_used": 1, "chunks_retrieved": 4}},
        ])
        self.assertIn("Efficiency: 1/4 chunks used (25%)", html_out)

    def test_no_efficiency_line_for_non_retriever_stage(self):
        html_out = render_stage_strip([
            {"stage": "GENERATOR", "status": "PASS", "observation": "ok", "confidence": "0.95",
             "metadata": {"chunk_utilization_rate": 0.25, "chunks_used": 1, "chunks_retrieved": 4}},
        ])
        self.assertNotIn("Efficiency:", html_out)

    def test_no_efficiency_line_when_metadata_missing(self):
        html_out = render_stage_strip([
            {"stage": "RETRIEVER", "status": "PASS", "observation": "ok", "confidence": "0.95"},
        ])
        self.assertNotIn("Efficiency:", html_out)


class TestRenderAnswerCorrectness(unittest.TestCase):
    def test_none_shows_not_computed_placeholder(self):
        html_out = render_answer_correctness(None)
        self.assertIn("Not computed for this trace", html_out)
        self.assertIn("scripts/evaluate_answer_correctness.py", html_out)

    def test_renders_recall_and_per_claim_breakdown(self):
        correctness = {
            "claim_recall": 0.667,
            "total_gold_claims": 3,
            "gold_answer": "The gold answer text.",
            "results": [
                {"claim_text": "Claim one.", "verification_status": "SUPPORTED", "confidence": 0.9, "best_matching_sentence": "Match one."},
                {"claim_text": "Claim two.", "verification_status": "PARTIALLY_SUPPORTED", "confidence": 0.5, "best_matching_sentence": "Match two."},
                {"claim_text": "Claim three.", "verification_status": "UNSUPPORTED", "confidence": 0.1, "best_matching_sentence": None},
            ],
        }
        html_out = render_answer_correctness(correctness)
        self.assertIn("0.667", html_out)
        self.assertIn("2 / 3", html_out)
        self.assertIn("Claim one.", html_out)
        self.assertIn("Match one.", html_out)
        self.assertIn("no matching sentence found", html_out)


class TestLoadAnswerCorrectness(unittest.TestCase):
    def test_loads_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "TRACE_T1.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"claim_recall": 1.0}, f)
            result = load_answer_correctness("T1", base_dir=tmpdir)
            self.assertEqual(result["claim_recall"], 1.0)

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_answer_correctness("T1", base_dir="/nonexistent"))


class TestRenderClaimRows(unittest.TestCase):
    def test_renders_supported_and_not_verifiable(self):
        html_out = render_claim_rows(make_report()["evidence_analysis"])
        self.assertIn("Claim one.", html_out)
        self.assertIn("SUPPORTED", html_out)
        self.assertIn("NOT_VERIFIABLE", html_out)
        self.assertIn("Evidence text.", html_out)

    def test_escapes_claim_text(self):
        html_out = render_claim_rows([{"claim_id": "x", "claim_text": "<b>bad</b>", "verification_status": "SUPPORTED"}])
        self.assertNotIn("<b>bad</b>", html_out)
        self.assertIn("&lt;b&gt;bad&lt;/b&gt;", html_out)


class TestRenderChunkMapping(unittest.TestCase):
    def test_renders_with_full_record(self):
        refs = make_trace()["retrieved_chunk_references"]
        records = {"c1": {
            "chunk_id": "c1", "configured_chunk_size": 512, "configured_chunk_overlap": 50,
            "character_start": 0, "character_end": 1110, "text_length": 1110, "text": "Full chunk text here.",
        }}
        html_out = render_chunk_mapping(refs, records)
        self.assertIn("Full chunk text here.", html_out)
        self.assertIn("bns.pdf", html_out)
        self.assertIn("0.68", html_out)

    def test_renders_gracefully_when_record_missing(self):
        refs = make_trace()["retrieved_chunk_references"]
        html_out = render_chunk_mapping(refs, {})
        self.assertIn("not found in current chunk_registry.json", html_out)


class TestRenderReasoningChain(unittest.TestCase):
    def test_renders_ordered_list(self):
        html_out = render_reasoning_chain(["Step one.", "Step two."])
        self.assertIn("<ol", html_out)
        self.assertIn("Step one.", html_out)
        self.assertIn("Step two.", html_out)


class TestRenderCorrectiveActions(unittest.TestCase):
    def test_empty_actions_shows_healthy_message(self):
        html_out = render_corrective_actions([])
        self.assertIn("None recommended", html_out)

    def test_renders_action_cards(self):
        html_out = render_corrective_actions([{"action_type": "RERANK_TUNE", "description": "Increase top_k."}])
        self.assertIn("RERANK_TUNE", html_out)
        self.assertIn("Increase top_k.", html_out)


class TestRenderFrameworkComparison(unittest.TestCase):
    def test_no_baseline_row_shows_placeholder(self):
        html_out = render_framework_comparison(0.83, "UNKNOWN", None)
        self.assertIn("No baseline comparison row found", html_out)

    def test_renders_all_four_frameworks(self):
        baseline_row = {
            "ragas_faithfulness": 1.0, "ragas_answer_relevancy": 0.84,
            "ragchecker_precision": 0.82, "ares_context_relevance": 0.33,
        }
        html_out = render_framework_comparison(0.8367, "UNKNOWN", baseline_row)
        self.assertIn("X-RAG", html_out)
        self.assertIn("RAGAS", html_out)
        self.assertIn("RAGChecker", html_out)
        self.assertIn("ARES", html_out)
        self.assertIn("1.000", html_out)


class TestLoadBaselineRow(unittest.TestCase):
    def test_finds_matching_row_by_trace_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"trace_id": "T1", "ragas_faithfulness": 1.0}, {"trace_id": "T2"}], f)
            row = load_baseline_row("T1", results_path=path)
            self.assertEqual(row["ragas_faithfulness"], 1.0)

    def test_missing_trace_id_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"trace_id": "T2"}], f)
            self.assertIsNone(load_baseline_row("T1", results_path=path))

    def test_missing_file_returns_none(self):
        self.assertIsNone(load_baseline_row("T1", results_path="/nonexistent/results.json"))


class TestLoadChunkRecords(unittest.TestCase):
    def test_loads_only_referenced_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chunk_registry.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "c1": {"chunk_id": "c1", "text": "chunk one"},
                    "c2": {"chunk_id": "c2", "text": "chunk two"},
                }, f)
            records = load_chunk_records(["c1"], registry_path=path)
            self.assertEqual(list(records.keys()), ["c1"])

    def test_missing_registry_returns_empty_dict(self):
        self.assertEqual(load_chunk_records(["c1"], registry_path="/nonexistent/registry.json"), {})


class TestGenerateReportIntegration(unittest.TestCase):
    def test_report_contains_all_major_sections(self):
        trace = make_trace()
        report = make_report()
        html_out = generate_report(trace, report, baseline_row=None)

        self.assertIn("<title>X-RAG Diagnostic Report", html_out)
        self.assertIn("Executive Summary", html_out)
        self.assertIn("The Case", html_out)
        self.assertIn("Raw Retrieval &amp; Generation Trace", html_out)
        self.assertIn("Claim Decomposition &amp; Verification", html_out)
        self.assertIn("Chunk Mapping &amp; Provenance", html_out)
        self.assertIn("X-RAG Stage Attribution", html_out)
        self.assertIn("Root Cause Reasoning Chain", html_out)
        self.assertIn("Corrective Actions", html_out)
        self.assertIn("Answer Correctness (Claim Recall)", html_out)
        self.assertIn("Cross-Framework Comparison", html_out)
        self.assertIn("What is abetment?", html_out)


class TestGenerateWritesFile(unittest.TestCase):
    def test_generate_writes_html_file_named_by_trace_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            traces_dir = os.path.join(tmpdir, "traces", "2026-07-19")
            reports_dir = os.path.join(tmpdir, "reports")
            output_dir = os.path.join(tmpdir, "diagnostic_reports")
            os.makedirs(traces_dir)
            os.makedirs(reports_dir)

            with open(os.path.join(traces_dir, "trace_T1.json"), "w", encoding="utf-8") as f:
                json.dump(make_trace(), f)
            with open(os.path.join(reports_dir, "T1.json"), "w", encoding="utf-8") as f:
                json.dump(make_report(), f)

            import scripts.generate_diagnostic_report as gdr
            old_traces_dir, old_reports_dir = gdr.TRACES_DIR, gdr.REPORTS_DIR
            gdr.TRACES_DIR = traces_dir
            gdr.REPORTS_DIR = reports_dir
            try:
                path = generate("T1", output_dir=output_dir)
            finally:
                gdr.TRACES_DIR = old_traces_dir
                gdr.REPORTS_DIR = old_reports_dir

            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith("T1.html"))
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("What is abetment?", content)


if __name__ == "__main__":
    unittest.main()
