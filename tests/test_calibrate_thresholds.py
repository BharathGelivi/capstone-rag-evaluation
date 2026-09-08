import os
import tempfile
import unittest

from scripts.calibrate_thresholds import load_labeled_dataset, compute_precision_recall_f1


class TestCalibrateThresholds(unittest.TestCase):
    def test_load_labeled_dataset_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write("claim_text,evidence_text,gold_label\n")
                f.write("Claim A,Evidence A,SUPPORTED\n")
                f.write("Claim B,Evidence B,NOT_SUPPORTED\n")

            rows = load_labeled_dataset(path)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["claim_text"], "Claim A")
            self.assertEqual(rows[1]["gold_label"], "NOT_SUPPORTED")

    def test_load_labeled_dataset_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write('[{"claim_text": "C", "evidence_text": "E", "gold_label": "SUPPORTED"}]')

            rows = load_labeled_dataset(path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["gold_label"], "SUPPORTED")

    def test_compute_precision_recall_f1_perfect_predictions(self):
        preds = ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED"]
        golds = ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED"]

        metrics = compute_precision_recall_f1(preds, golds, positive_label="SUPPORTED")

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_compute_precision_recall_f1_partial_predictions(self):
        # 1 true positive, 1 false positive, 1 false negative
        preds = ["SUPPORTED", "SUPPORTED", "NOT_SUPPORTED"]
        golds = ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED"]

        metrics = compute_precision_recall_f1(preds, golds, positive_label="SUPPORTED")

        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_compute_precision_recall_f1_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_precision_recall_f1(["SUPPORTED"], ["SUPPORTED", "NOT_SUPPORTED"], positive_label="SUPPORTED")


if __name__ == "__main__":
    unittest.main()
