import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from scripts import run_baseline_comparison as runner_module
from scripts.baseline_adapters.common import ResolvedExample


class TestManifestAndResultsIO(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self._patch_paths()

    def _patch_paths(self):
        base = self.tmpdir.name
        patcher1 = patch.object(runner_module, "RESULTS_DIR", base)
        patcher2 = patch.object(runner_module, "MANIFEST_PATH", os.path.join(base, "manifest.json"))
        patcher3 = patch.object(runner_module, "RESULTS_CSV_PATH", os.path.join(base, "results.csv"))
        patcher4 = patch.object(runner_module, "RESULTS_JSON_PATH", os.path.join(base, "results.json"))
        patcher5 = patch.object(runner_module, "FAILURES_LOG_PATH", os.path.join(base, "failures.log"))
        for p in (patcher1, patcher2, patcher3, patcher4, patcher5):
            p.start()
            self.addCleanup(p.stop)

    def test_manifest_roundtrip(self):
        manifest = runner_module.load_manifest()
        self.assertEqual(manifest["completed_eval_ids"], [])

        manifest["completed_eval_ids"].append("1")
        manifest["eval_id_to_trace_id"]["1"] = "TRACE_ABC"
        runner_module.save_manifest(manifest)

        reloaded = runner_module.load_manifest()
        self.assertEqual(reloaded["completed_eval_ids"], ["1"])
        self.assertEqual(reloaded["eval_id_to_trace_id"]["1"], "TRACE_ABC")

    def test_append_result_row_incremental(self):
        runner_module.append_result_row({"eval_id": "1", "question": "Q1?", "trace_id": "T1"})
        runner_module.append_result_row({"eval_id": "2", "question": "Q2?", "trace_id": "T2"})

        with open(runner_module.RESULTS_JSON_PATH, encoding="utf-8") as f:
            rows = json.load(f)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["eval_id"], "1")
        self.assertEqual(rows[1]["eval_id"], "2")

        with open(runner_module.RESULTS_CSV_PATH, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("eval_id", content.splitlines()[0])
        self.assertEqual(len(content.splitlines()), 3)  # header + 2 rows

    def test_log_failure_writes_to_log(self):
        runner_module.log_failure("1", "ragas", RuntimeError("boom"))
        with open(runner_module.FAILURES_LOG_PATH, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("eval_id=1", content)
        self.assertIn("baseline=ragas", content)
        self.assertIn("boom", content)


class TestRunSubprocessWorker(unittest.TestCase):
    def test_success_reads_scores_back(self):
        example = ResolvedExample(trace_id="T1", question="Q?", answer="A.", contexts=["c1"])

        def fake_run(cmd, capture_output, text, timeout):
            out_path = cmd[4]  # [python_exe, "-m", module, in_path, out_path]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"T1": {"precision": 0.9}}, f)
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            scores = runner_module.run_subprocess_worker("fake_python.exe", "fake.module", example)

        self.assertEqual(scores, {"precision": 0.9})

    def test_nonzero_exit_raises(self):
        example = ResolvedExample(trace_id="T1", question="Q?", answer="A.", contexts=["c1"])

        with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="worker crashed")):
            with self.assertRaises(RuntimeError):
                runner_module.run_subprocess_worker("fake_python.exe", "fake.module", example)

    def test_timeout_propagates(self):
        example = ResolvedExample(trace_id="T1", question="Q?", answer="A.", contexts=["c1"])

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner_module.run_subprocess_worker("fake_python.exe", "fake.module", example)


class TestBuildXragTraceAndReportSkipsFailedGeneration(unittest.TestCase):
    """Regression test for a real bug: Generator.generate() catches its own
    exceptions and returns a placeholder "did not respond in time" string
    with GenerationResult.error set (src/generator.py), rather than raising.
    build_xrag_trace_and_report must check .error and refuse to run that
    placeholder through claim decomposition/NLI as if it were a real answer
    -- otherwise every generation timeout gets scored as a hallucinating
    UNSUPPORTED_GENERATION diagnosis that has nothing to do with the
    pipeline's actual behavior on that question."""

    def test_generation_error_raises_instead_of_diagnosing_the_placeholder(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(retrieval_time=0.1)

        generator = MagicMock()
        failed_result = MagicMock()
        failed_result.error = "TimeoutError: request timed out"
        failed_result.generated_answer = (
            "The language model did not respond in time. This is an "
            "infrastructure problem, not a limitation of your documents — please retry."
        )
        generator.generate.return_value = failed_result

        runner = MagicMock()

        with self.assertRaises(runner_module.GenerationFailedError):
            runner_module.build_xrag_trace_and_report("Q?", retriever, generator, runner)

        # The whole point: a failed generation must never reach claim
        # decomposition / NLI / the reasoner.
        runner.run.assert_not_called()

    @patch("src.rag_trace.RAGTraceBuilder.save_to_json")
    @patch("src.rag_trace.RAGTraceBuilder.build")
    def test_successful_generation_proceeds_normally(self, mock_build, mock_save):
        retriever = MagicMock()
        retriever.retrieve.return_value = MagicMock(retrieval_time=0.1)

        generator = MagicMock()
        ok_result = MagicMock()
        ok_result.error = None
        ok_result.generated_answer = "Death or imprisonment for life."
        ok_result.generation_time = 0.2
        generator.generate.return_value = ok_result

        mock_build.return_value = MagicMock(trace_id="TRACE_OK")

        fake_report = MagicMock()
        runner = MagicMock()
        runner.run.return_value = fake_report

        trace, report = runner_module.build_xrag_trace_and_report("Q?", retriever, generator, runner)

        self.assertEqual(report, fake_report)
        runner.run.assert_called_once()
        fake_report.save.assert_called_once()


class TestProcessOneExample(unittest.TestCase):
    def setUp(self):
        self.eval_row = {"id": "1", "question": "What is the punishment for murder?", "gold_answer": "Death or life imprisonment.", "expected_failure_type": ""}

        self.fake_trace = MagicMock()
        self.fake_trace.trace_id = "TRACE_1"
        self.fake_trace.generated_answer = "Death or imprisonment for life."

        self.fake_report = MagicMock()
        self.fake_report.root_cause_analysis.primary_cause = "UNKNOWN"
        self.fake_report.evaluation_metrics.average_entailment = 0.9

        self.retriever = MagicMock()
        self.generator = MagicMock()
        self.runner = MagicMock()
        self.chunk_registry = MagicMock()

    @patch("scripts.run_baseline_comparison.compute_ares_scores")
    @patch("scripts.run_baseline_comparison.compute_ragchecker_scores")
    @patch("scripts.run_baseline_comparison.compute_ragas_scores")
    @patch("scripts.run_baseline_comparison.build_xrag_trace_and_report")
    @patch("scripts.baseline_adapters.common.resolve_examples")
    def test_assembles_row_with_xrag_signal(self, mock_resolve, mock_build_trace, mock_ragas, mock_ragchecker, mock_ares):
        mock_build_trace.return_value = (self.fake_trace, self.fake_report)
        mock_resolve.return_value = [ResolvedExample(trace_id="TRACE_1", question="Q?", answer="A.", contexts=["c1"], gold_answer="Gold.")]
        mock_ragas.return_value = {"ragas_faithfulness": 0.8}
        mock_ragchecker.return_value = {"ragchecker_precision": 0.7}
        mock_ares.return_value = {"ares_context_relevance": 1.0}

        row = runner_module.process_one_example(
            self.eval_row, self.retriever, self.generator, self.runner, self.chunk_registry,
            skip_ragchecker=False, skip_ares=False,
        )

        self.assertEqual(row["eval_id"], "1")
        self.assertEqual(row["trace_id"], "TRACE_1")
        self.assertEqual(row["xrag_primary_cause"], "UNKNOWN")
        self.assertEqual(row["xrag_avg_entailment_score"], 0.9)
        self.assertEqual(row["ragas_faithfulness"], 0.8)
        self.assertEqual(row["ragchecker_precision"], 0.7)
        self.assertEqual(row["ares_context_relevance"], 1.0)

    @patch("scripts.run_baseline_comparison.compute_ares_scores")
    @patch("scripts.run_baseline_comparison.compute_ragchecker_scores")
    @patch("scripts.run_baseline_comparison.compute_ragas_scores")
    @patch("scripts.run_baseline_comparison.build_xrag_trace_and_report")
    def test_one_baseline_failure_does_not_break_others(self, mock_build_trace, mock_ragas, mock_ragchecker, mock_ares):
        mock_build_trace.return_value = (self.fake_trace, self.fake_report)
        mock_ragas.side_effect = RuntimeError("ragas exploded")
        mock_ragchecker.return_value = {"ragchecker_precision": 0.7}
        mock_ares.return_value = {"ares_context_relevance": 1.0}

        with patch("scripts.baseline_adapters.common.resolve_examples",
                   return_value=[ResolvedExample(trace_id="TRACE_1", question="Q?", answer="A.", contexts=["c1"], gold_answer="Gold.")]):
            row = runner_module.process_one_example(
                self.eval_row, self.retriever, self.generator, self.runner, self.chunk_registry,
                skip_ragchecker=False, skip_ares=False,
            )

        # ragas columns simply absent/missing, but the row is still returned with everything else
        self.assertNotIn("ragas_faithfulness", row)
        self.assertEqual(row["ragchecker_precision"], 0.7)
        self.assertEqual(row["ares_context_relevance"], 1.0)
        self.assertEqual(row["xrag_primary_cause"], "UNKNOWN")

    @patch("scripts.run_baseline_comparison.compute_ares_scores")
    @patch("scripts.run_baseline_comparison.compute_ragchecker_scores")
    @patch("scripts.run_baseline_comparison.compute_ragas_scores")
    @patch("scripts.run_baseline_comparison.build_xrag_trace_and_report")
    def test_skip_flags_prevent_calls(self, mock_build_trace, mock_ragas, mock_ragchecker, mock_ares):
        mock_build_trace.return_value = (self.fake_trace, self.fake_report)
        mock_ragas.return_value = {}

        with patch("scripts.baseline_adapters.common.resolve_examples",
                   return_value=[ResolvedExample(trace_id="TRACE_1", question="Q?", answer="A.", contexts=["c1"], gold_answer="Gold.")]):
            runner_module.process_one_example(
                self.eval_row, self.retriever, self.generator, self.runner, self.chunk_registry,
                skip_ragchecker=True, skip_ares=True,
            )

        mock_ragchecker.assert_not_called()
        mock_ares.assert_not_called()


if __name__ == "__main__":
    unittest.main()
