import json
import os
import tempfile
import unittest

from scripts.generate_comparison_report import (
    safe_mean,
    render_summary_table,
    render_correlation_section,
    render_agreement_section,
    render_value_add_section,
    render_failures_section,
    generate_report,
    load_json,
    load_csv_rows,
    main,
)


def make_result(eval_id, xrag=0.9, ragas=0.9, ragchecker_faithfulness=0.9, ares=0.9):
    return {
        "eval_id": eval_id,
        "xrag_avg_entailment_score": xrag,
        "ragas_faithfulness": ragas,
        "ragchecker_faithfulness": ragchecker_faithfulness,
        "ares_answer_faithfulness": ares,
    }


class TestSafeMean(unittest.TestCase):
    def test_basic_mean(self):
        self.assertEqual(safe_mean([1, 2, 3]), 2.0)

    def test_ignores_none(self):
        self.assertEqual(safe_mean([1, None, 3]), 2.0)

    def test_empty_returns_none(self):
        self.assertIsNone(safe_mean([]))
        self.assertIsNone(safe_mean([None, None]))


class TestRenderSummaryTable(unittest.TestCase):
    def test_includes_all_frameworks_and_means(self):
        results = [make_result("1", 0.8, 0.9, 0.7, 0.6), make_result("2", 0.6, 0.7, 0.5, 0.4)]
        table = render_summary_table(results)
        self.assertIn("X-RAG", table)
        self.assertIn("RAGAS", table)
        self.assertIn("RAGChecker", table)
        self.assertIn("ARES (ues_idp)", table)
        self.assertIn("n=2", table)
        self.assertIn("0.7", table)  # xrag mean

    def test_handles_empty_results(self):
        table = render_summary_table([])
        self.assertIn("N/A", table)
        self.assertIn("n=0", table)


class TestRenderCorrelationSection(unittest.TestCase):
    def test_renders_known_keys(self):
        correlations = {
            "xrag_vs_ragas_faithfulness": 0.85,
            "xrag_vs_ragchecker_faithfulness": 0.7,
            "xrag_vs_ragchecker_precision": 0.6,
        }
        section = render_correlation_section(correlations)
        self.assertIn("0.85", section)
        self.assertIn("0.7", section)

    def test_empty_returns_placeholder(self):
        section = render_correlation_section({})
        self.assertIn("No correlation data", section)


class TestRenderAgreementSection(unittest.TestCase):
    def test_renders_confusion_matrix_and_kappa(self):
        agreement = {
            "xrag_vs_ragas": {
                "confusion_matrix": {
                    "both_flag_failure": 2,
                    "only_first_flags_failure": 1,
                    "only_second_flags_failure": 0,
                    "neither_flags_failure": 5,
                },
                "cohens_kappa": 0.65,
            },
            "xrag_vs_ragchecker": {
                "confusion_matrix": {
                    "both_flag_failure": 1,
                    "only_first_flags_failure": 0,
                    "only_second_flags_failure": 1,
                    "neither_flags_failure": 6,
                },
                "cohens_kappa": 0.4,
            },
        }
        section = render_agreement_section(agreement)
        self.assertIn("0.65", section)
        self.assertIn("0.4", section)
        self.assertIn("X-RAG vs. RAGAS", section)
        self.assertIn("X-RAG vs. RAGChecker", section)

    def test_empty_returns_placeholder(self):
        self.assertIn("No agreement data", render_agreement_section({}))


class TestRenderValueAddSection(unittest.TestCase):
    def test_includes_disagreement_examples(self):
        disagreements = [
            {
                "question": "What is the punishment for theft?",
                "xrag_primary_cause": "GROUNDING_FAILURE",
                "xrag_reasoning_chain": "claim unsupported by retrieved context",
                "ragas_faithfulness": 0.9,
                "ragchecker_hallucination": 0.1,
                "ares_context_relevance": 1.0,
            }
        ]
        section = render_value_add_section(disagreements)
        self.assertIn("localization", section)
        self.assertIn("GROUNDING_FAILURE", section)
        self.assertIn("punishment for theft", section)

    def test_empty_disagreements_still_explains_thesis(self):
        section = render_value_add_section([])
        self.assertIn("localization", section)
        self.assertIn("No disagreement examples", section)


class TestRenderFailuresSection(unittest.TestCase):
    def test_missing_file_returns_placeholder(self):
        section = render_failures_section("/nonexistent/failures.log")
        self.assertIn("No failures recorded", section)

    def test_parses_and_groups_by_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failures.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write("2026-01-01 eval_id=3 baseline=ragas: RuntimeError: boom\n")
                f.write("2026-01-01 eval_id=4 baseline=ares: 402 Payment Required\n")
            section = render_failures_section(path)
            self.assertIn("ragas", section)
            self.assertIn("ares", section)
            self.assertIn("boom", section)
            self.assertIn("402 Payment Required", section)

    def test_empty_file_returns_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "failures.log")
            open(path, "w", encoding="utf-8").close()
            self.assertIn("No failures recorded", render_failures_section(path))


class TestGenerateReportIntegration(unittest.TestCase):
    def test_report_is_self_contained_markdown(self):
        results = [make_result("1"), make_result("2")]
        correlations = {"xrag_vs_ragas_faithfulness": 0.8}
        agreement = {
            "xrag_vs_ragas": {
                "confusion_matrix": {"both_flag_failure": 1, "only_first_flags_failure": 0, "only_second_flags_failure": 0, "neither_flags_failure": 1},
                "cohens_kappa": 1.0,
            }
        }
        disagreements = []
        report = generate_report(results, correlations, agreement, disagreements, "/nonexistent/failures.log")

        self.assertIn("# X-RAG vs. RAGAS", report)
        self.assertIn("## 1. Summary", report)
        self.assertIn("## 2. Correlation", report)
        self.assertIn("## 3. Agreement", report)
        self.assertIn("## 4. Where X-RAG Adds Value", report)
        self.assertIn("## 5. Threats to Validity", report)
        self.assertIn("eval/eval_dataset.csv", report)


class TestIOHelpers(unittest.TestCase):
    def test_load_json_missing_returns_default(self):
        self.assertEqual(load_json("/nonexistent/x.json", default={}), {})

    def test_load_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "x.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"a": 1}, f)
            self.assertEqual(load_json(path), {"a": 1})

    def test_load_csv_rows_missing_returns_empty_list(self):
        self.assertEqual(load_csv_rows("/nonexistent/x.csv"), [])


class TestMainWritesReportFile(unittest.TestCase):
    def test_main_writes_report_to_results_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = os.path.join(tmpdir, "results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump([make_result("1")], f)

            import sys
            old_argv = sys.argv
            sys.argv = ["generate_comparison_report.py", "--results-dir", tmpdir]
            try:
                main()
            finally:
                sys.argv = old_argv

            report_path = os.path.join(tmpdir, "comparison_report.md")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("# X-RAG vs. RAGAS", content)


if __name__ == "__main__":
    unittest.main()
