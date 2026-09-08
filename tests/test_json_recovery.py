import unittest
from src.claim_decomposer import ClaimDecomposer, JSONRecoveryError

class TestJSONRecovery(unittest.TestCase):
    def setUp(self):
        self.decomposer = ClaimDecomposer(model_name="mock")

    def test_valid_json(self):
        text = '[{"claim_text": "Valid.", "sentence_id": "S1"}]'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim_text"], "Valid.")

    def test_markdown_wrapped_json(self):
        text = '```json\n[{"claim_text": "Markdown.", "sentence_id": "S1"}]\n```'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)

    def test_trailing_text_json(self):
        text = 'Here is the output:\n[{"claim_text": "Trailing.", "sentence_id": "S1"}]\nHope this helps!'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim_text"], "Trailing.")

    def test_unbalanced_brackets_json(self):
        # Missing closing bracket for array and brace for object
        text = '[{"claim_text": "Unbalanced.", "sentence_id": "S1"'
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim_text"], "Unbalanced.")

    def test_completely_invalid_json(self):
        text = 'This is not json at all.'
        with self.assertRaises(JSONRecoveryError):
            self.decomposer._robust_json_parse(text)

    def test_truncated_mid_object_with_dangling_comma(self):
        # Reproduces a response cut off by a token limit: the last element is
        # entirely incomplete and a trailing comma is left dangling, which naive
        # bracket-balancing (append missing '}'/']') cannot recover from.
        text = (
            '[\n'
            '  {"claim_text": "First claim.", "sentence_id": "S001"},\n'
            '  {"claim_text": "Second claim.", "sentence_id": "S002"},\n'
            '  {"claim_text": "Third, incomplete'
        )
        result = self.decomposer._robust_json_parse(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["claim_text"], "First claim.")
        self.assertEqual(result[1]["claim_text"], "Second claim.")

if __name__ == '__main__':
    unittest.main()
