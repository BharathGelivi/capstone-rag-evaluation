import json
import os
import tempfile
import unittest

from scripts.analyze_agreement import (
    _pearson_r,
    _cohens_kappa,
    _confusion_matrix,
    compute_correlations,
    compute_agreement,
    find_disagreements,
    save_disagreements_csv,
    load_results,
)


class TestPearsonR(unittest.TestCase):
    def test_perfect_positive_correlation(self):
        self.assertAlmostEqual(_pearson_r([1, 2, 3, 4], [2, 4, 6, 8]), 1.0, places=6)

    def test_perfect_negative_correlation(self):
        self.assertAlmostEqual(_pearson_r([1, 2, 3, 4], [8, 6, 4, 2]), -1.0, places=6)

    def test_ignores_none_pairs(self):
        # (1,2),(2,4),(3,6),(4,8) after dropping the None-containing pair -> still perfect r=1.0
        self.assertAlmostEqual(_pearson_r([1, 2, 3, 4, None], [2, 4, 6, 8, 99]), 1.0, places=6)

    def test_insufficient_data_returns_none(self):
        self.assertIsNone(_pearson_r([1], [2]))
        self.assertIsNone(_pearson_r([], []))

    def test_zero_variance_returns_none(self):
        self.assertIsNone(_pearson_r([1, 1, 1], [1, 2, 3]))


class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        self.assertAlmostEqual(_cohens_kappa([True, False, True, False], [True, False, True, False]), 1.0, places=6)

    def test_partial_agreement_hand_computed(self):
        # po=0.75, p_a_true=0.5, p_b_true=0.25, pe=0.5, kappa=(0.75-0.5)/(1-0.5)=0.5
        a = [True, True, False, False]
        b = [True, False, False, False]
        self.assertAlmostEqual(_cohens_kappa(a, b), 0.5, places=6)

    def test_empty_returns_none(self):
        self.assertIsNone(_cohens_kappa([], []))

    def test_none_values_are_excluded(self):
        a = [True, False, None]
        b = [True, False, True]
        self.assertAlmostEqual(_cohens_kappa(a, b), 1.0, places=6)


class TestConfusionMatrix(unittest.TestCase):
    def test_counts(self):
        a = [True, True, False, False]
        b = [True, False, False, False]
        matrix = _confusion_matrix(a, b)
        self.assertEqual(matrix, {
            "both_flag_failure": 1,
            "only_first_flags_failure": 1,
            "only_second_flags_failure": 0,
            "neither_flags_failure": 2,
        })


def make_result(eval_id, xrag_entailment, xrag_cause, ragas_faithfulness=None, ragchecker_hallucination=None, ragchecker_faithfulness=None, ragchecker_precision=None, trace_id=None):
    return {
        "eval_id": eval_id,
        "question": f"Question {eval_id}?",
        "expected_failure_type": "",
        "trace_id": trace_id or f"TRACE_{eval_id}",
        "xrag_primary_cause": xrag_cause,
        "xrag_avg_entailment_score": xrag_entailment,
        "ragas_faithfulness": ragas_faithfulness,
        "ragchecker_hallucination": ragchecker_hallucination,
        "ragchecker_faithfulness": ragchecker_faithfulness,
        "ragchecker_precision": ragchecker_precision,
    }


class TestComputeCorrelations(unittest.TestCase):
    def test_correlates_entailment_with_faithfulness(self):
        results = [
            make_result("1", 0.9, "UNKNOWN", ragas_faithfulness=0.9, ragchecker_faithfulness=0.9, ragchecker_precision=0.9),
            make_result("2", 0.5, "UNSUPPORTED_GENERATION", ragas_faithfulness=0.5, ragchecker_faithfulness=0.5, ragchecker_precision=0.5),
            make_result("3", 0.1, "GROUNDING_FAILURE", ragas_faithfulness=0.1, ragchecker_faithfulness=0.1, ragchecker_precision=0.1),
        ]
        correlations = compute_correlations(results)
        self.assertAlmostEqual(correlations["xrag_vs_ragas_faithfulness"], 1.0, places=6)
        self.assertAlmostEqual(correlations["xrag_vs_ragchecker_faithfulness"], 1.0, places=6)
        self.assertAlmostEqual(correlations["xrag_vs_ragchecker_precision"], 1.0, places=6)


class TestComputeAgreement(unittest.TestCase):
    def test_agreement_confusion_matrix_and_kappa(self):
        results = [
            make_result("1", 0.9, "UNKNOWN", ragas_faithfulness=0.9, ragchecker_hallucination=0.1),  # both pass
            make_result("2", 0.2, "GROUNDING_FAILURE", ragas_faithfulness=0.2, ragchecker_hallucination=0.8),  # both flag failure
            make_result("3", 0.2, "GROUNDING_FAILURE", ragas_faithfulness=0.9, ragchecker_hallucination=0.1),  # X-RAG flags, others don't
        ]
        agreement = compute_agreement(results, ragas_threshold=0.7, ragchecker_hallucination_threshold=0.5)

        self.assertEqual(agreement["xrag_vs_ragas"]["confusion_matrix"]["both_flag_failure"], 1)
        self.assertEqual(agreement["xrag_vs_ragas"]["confusion_matrix"]["only_first_flags_failure"], 1)
        self.assertEqual(agreement["xrag_vs_ragas"]["confusion_matrix"]["neither_flags_failure"], 1)
        self.assertIsNotNone(agreement["xrag_vs_ragas"]["cohens_kappa"])


class TestFindDisagreements(unittest.TestCase):
    def test_disagreement_surfaced_with_no_reasoning_chain_file(self):
        results = [
            make_result("1", 0.9, "UNKNOWN", ragas_faithfulness=0.9, ragchecker_hallucination=0.1),
            make_result("2", 0.2, "GROUNDING_FAILURE", ragas_faithfulness=0.9, ragchecker_hallucination=0.1, trace_id="NO_SUCH_TRACE"),
        ]
        disagreements = find_disagreements(results, ragas_threshold=0.7, ragchecker_hallucination_threshold=0.5)

        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0]["eval_id"], "2")
        self.assertEqual(disagreements[0]["xrag_primary_cause"], "GROUNDING_FAILURE")
        self.assertEqual(disagreements[0]["xrag_reasoning_chain"], "")  # missing RCA artifact -> empty, not a crash

    def test_no_disagreements_when_all_agree(self):
        results = [
            make_result("1", 0.9, "UNKNOWN", ragas_faithfulness=0.9, ragchecker_hallucination=0.1),
            make_result("2", 0.2, "GROUNDING_FAILURE", ragas_faithfulness=0.2, ragchecker_hallucination=0.8),
        ]
        disagreements = find_disagreements(results)
        self.assertEqual(disagreements, [])


class TestIOHelpers(unittest.TestCase):
    def test_load_results_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            data = [make_result("1", 0.9, "UNKNOWN")]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            loaded = load_results(path)
            self.assertEqual(loaded, data)

    def test_save_disagreements_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "disagreements.csv")
            save_disagreements_csv([{"eval_id": "1", "question": "Q?"}], path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("eval_id", content)
            self.assertIn("Q?", content)

    def test_save_disagreements_csv_empty_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "disagreements.csv")
            save_disagreements_csv([], path)
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
