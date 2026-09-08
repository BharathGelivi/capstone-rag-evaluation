"""
Worker script meant to run under venv_eval_ares's interpreter (invoked as a
subprocess by scripts/run_baseline_comparison.py, which runs in the main
project venv where ares-ai itself isn't installed).

Usage:
    venv_eval_ares\\Scripts\\python.exe -m scripts.baseline_adapters.ares_worker <input_examples.json> <output_scores.json> [--model MODEL]

Reads a list of ResolvedExample dicts (see common.py) from <input_examples.json>,
scores EACH example individually with ARES's ues_idp mode (its own
`ues_idp_config` returns one aggregate score for a whole TSV, so scoring one
example at a time is what makes per-example scores possible at all), and
writes {trace_id: {"Context_Relevance_Score": ..., "Answer_Relevance_Score": ...,
"Answer_Faithfulness_Score": ...}} to <output_scores.json>.

Model note: ARES's ues_idp only has a real custom-endpoint (non-OpenAI/
non-Anthropic/non-TogetherAI) code path via vLLM. Its "gpt" branch calls the
bare `openai` module client, which respects OPENAI_BASE_URL/OPENAI_API_KEY --
so this worker points those at an OpenAI-compatible router (Groq's free tier
by default, or HF's router once HF credits are available again) and picks a
literal model whose name contains "gpt" (ARES routes to the "gpt" scoring
functions via a `"gpt" in model_choice` substring check, and then sends that
same string as the literal API model= parameter) -- `openai/gpt-oss-20b`
happens to be hosted under that exact name by both Groq and HF, so the same
default works for either provider; only the base_url/key differ.

This module runs under venv_eval_ares's own Python 3.10 interpreter (which
does NOT have configs/ installed), so provider selection is driven directly
by GROQ_API_KEY/HF_TOKEN env vars rather than importing configs.models.
"""

import argparse
import json
import os
import sys
import tempfile

from scripts.baseline_adapters.common import load_resolved_examples
from scripts.baseline_adapters.ares_adapter import (
    to_ares_dataframe,
    build_few_shot_prompts_dataframe,
)

DEFAULT_ARES_MODEL = "openai/gpt-oss-20b"


def score_one_example(example, few_shot_path: str, model: str) -> dict:
    from ares import ARES

    if not example.contexts:
        return {"Context_Relevance_Score": None, "Answer_Relevance_Score": None, "Answer_Faithfulness_Score": None}

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_path = os.path.join(tmpdir, "eval.tsv")
        to_ares_dataframe([example]).to_csv(eval_path, sep="\t", index=False)

        ares = ARES(ues_idp={
            "in_domain_prompts_dataset": few_shot_path,
            "unlabeled_evaluation_set": eval_path,
            "model_choice": model,
        })
        return ares.ues_idp()


def main():
    parser = argparse.ArgumentParser(description="Score ResolvedExamples with ARES (ues_idp mode).")
    parser.add_argument("input_path", help="Path to a JSON file of ResolvedExamples.")
    parser.add_argument("output_path", help="Path to write per-example scores JSON to.")
    parser.add_argument("--model", default=DEFAULT_ARES_MODEL)
    args = parser.parse_args()

    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if nvidia_key:
        os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
        os.environ["OPENAI_API_KEY"] = nvidia_key
    elif groq_key:
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
        os.environ["OPENAI_API_KEY"] = groq_key
    elif hf_token:
        os.environ["OPENAI_BASE_URL"] = "https://router.huggingface.co/v1"
        os.environ["OPENAI_API_KEY"] = hf_token
    else:
        raise ValueError("NVIDIA_API_KEY, GROQ_API_KEY, or HF_TOKEN must be set to use a judge LLM for ARES.")

    examples = load_resolved_examples(args.input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        few_shot_path = os.path.join(tmpdir, "few_shot.tsv")
        build_few_shot_prompts_dataframe().to_csv(few_shot_path, sep="\t", index=False)

        scores = {}
        for example in examples:
            try:
                scores[example.trace_id] = score_one_example(example, few_shot_path, args.model)
            except Exception as e:
                print(f"ARES scoring failed for trace {example.trace_id}: {e}", file=sys.stderr)
                scores[example.trace_id] = {"error": str(e)}

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ares_worker failed: {e}", file=sys.stderr)
        sys.exit(1)
