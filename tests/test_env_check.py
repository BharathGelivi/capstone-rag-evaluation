import os
import unittest
from unittest.mock import patch

from src.env_check import ensure_hf_token, ensure_hf_token_or_exit


class TestEnsureHfToken(unittest.TestCase):
    def test_no_prompt_when_token_present(self):
        with patch.dict(os.environ, {"HF_TOKEN": "existing-token"}, clear=False):
            with patch("builtins.input") as mock_input:
                ensure_hf_token()
                mock_input.assert_not_called()

    def test_prompts_and_sets_env_when_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("builtins.input", side_effect=["new-token", "n"]) as mock_input:
                ensure_hf_token()
                self.assertEqual(mock_input.call_count, 2)
                self.assertEqual(os.environ.get("HF_TOKEN"), "new-token")

    def test_empty_input_leaves_env_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("builtins.input", return_value="") as mock_input:
                ensure_hf_token()
                mock_input.assert_called_once()
                self.assertIsNone(os.environ.get("HF_TOKEN"))


class TestEnsureHfTokenOrExit(unittest.TestCase):
    def test_no_exit_when_token_present(self):
        with patch.dict(os.environ, {"HF_TOKEN": "existing-token"}, clear=False):
            with patch("sys.exit") as mock_exit:
                ensure_hf_token_or_exit()
                mock_exit.assert_not_called()

    def test_exits_when_token_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with patch("sys.exit") as mock_exit:
                ensure_hf_token_or_exit()
                mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
