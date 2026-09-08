"""
Run under venv_eval_ragchecker's interpreter, not the main project venv:
    venv_eval_ragchecker\\Scripts\\python.exe -m unittest discover tests_eval -p "test_ragchecker*.py"
"""

import unittest

from scripts.baseline_adapters.common import ResolvedExample
from scripts.baseline_adapters.ragchecker_adapter import to_ragchecker_results


class TestRagcheckerAdapter(unittest.TestCase):
    def test_schema_matches_ragchecker_containers(self):
        examples = [
            ResolvedExample(
                trace_id="T1",
                question="What is the punishment for murder?",
                answer="Death or life imprisonment.",
                contexts=["Whoever commits murder shall be punished with death or imprisonment for life."],
                gold_answer="Death or imprisonment for life, plus fine.",
            ),
            ResolvedExample(
                trace_id="T2",
                question="What is culpable homicide?",
                answer="See section 105.",
                contexts=["Section 105 covers culpable homicide not amounting to murder."],
                gold_answer="Imprisonment for life or up to ten years.",
            ),
        ]

        results = to_ragchecker_results(examples)

        self.assertEqual(len(results.results), 2)
        first = results.results[0]
        self.assertEqual(first.query_id, "T1")
        self.assertEqual(first.query, "What is the punishment for murder?")
        self.assertEqual(first.gt_answer, "Death or imprisonment for life, plus fine.")
        self.assertEqual(first.response, "Death or life imprisonment.")
        self.assertEqual(len(first.retrieved_context), 1)
        self.assertEqual(first.retrieved_context[0].text, "Whoever commits murder shall be punished with death or imprisonment for life.")

    def test_examples_without_gold_answer_are_skipped(self):
        examples = [
            ResolvedExample(trace_id="T1", question="Q1?", answer="A1.", contexts=["c1"], gold_answer="Gold."),
            ResolvedExample(trace_id="T2", question="Q2?", answer="A2.", contexts=["c2"], gold_answer=None),
        ]

        results = to_ragchecker_results(examples)

        self.assertEqual(len(results.results), 1)
        self.assertEqual(results.results[0].query_id, "T1")

    def test_empty_input_produces_empty_results(self):
        results = to_ragchecker_results([])
        self.assertEqual(len(results.results), 0)


if __name__ == "__main__":
    unittest.main()
