"""
Run under venv_eval_ares's interpreter, not the main project venv:
    venv_eval_ares\\Scripts\\python.exe -m unittest discover tests_eval -p "test_ares*.py"
"""

import os
import tempfile
import unittest

from scripts.baseline_adapters.common import ResolvedExample
from scripts.baseline_adapters.ares_adapter import to_ares_dataframe, save_ares_tsv, build_few_shot_prompts_dataframe


class TestAresAdapter(unittest.TestCase):
    def test_one_row_per_query_document_pair(self):
        examples = [
            ResolvedExample(trace_id="T1", question="Q1?", answer="A1.", contexts=["doc1", "doc2"]),
            ResolvedExample(trace_id="T2", question="Q2?", answer="A2.", contexts=["doc3"]),
        ]

        df = to_ares_dataframe(examples)

        self.assertEqual(list(df.columns), ["Query", "Document", "Answer"])
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[0].to_dict(), {"Query": "Q1?", "Document": "doc1", "Answer": "A1."})
        self.assertEqual(df.iloc[1].to_dict(), {"Query": "Q1?", "Document": "doc2", "Answer": "A1."})
        self.assertEqual(df.iloc[2].to_dict(), {"Query": "Q2?", "Document": "doc3", "Answer": "A2."})

    def test_example_with_no_contexts_produces_no_rows(self):
        examples = [ResolvedExample(trace_id="T1", question="Q?", answer="A.", contexts=[])]
        df = to_ares_dataframe(examples)
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), ["Query", "Document", "Answer"])

    def test_save_ares_tsv_matches_ues_idp_expected_schema(self):
        examples = [ResolvedExample(trace_id="T1", question="Q?", answer="A.", contexts=["doc1"])]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "eval.tsv")
            save_ares_tsv(examples, path)

            import pandas as pd
            reloaded = pd.read_csv(path, sep="\t")
            # ares/ues_idp.py reads row['Query'], row['Document'], row['Answer'] directly.
            self.assertEqual(reloaded.iloc[0]["Query"], "Q?")
            self.assertEqual(reloaded.iloc[0]["Document"], "doc1")
            self.assertEqual(reloaded.iloc[0]["Answer"], "A.")

    def test_few_shot_prompts_dataframe_has_required_columns(self):
        df = build_few_shot_prompts_dataframe()
        self.assertEqual(list(df.columns), [
            "Query", "Document", "Answer",
            "Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label",
        ])
        self.assertGreaterEqual(len(df), 1)
        self.assertIn(df.iloc[0]["Context_Relevance_Label"], ("[[Yes]]", "[[No]]"))


if __name__ == "__main__":
    unittest.main()
