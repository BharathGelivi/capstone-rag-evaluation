"""
Converts the shared ResolvedExample intermediate format (see common.py) into
the TSV format ARES's ues_idp (lightweight LLM-judge) mode expects.

Schema note (verified against the installed package's actual source --
`ares/ues_idp.py` -- not the repo's docs, since the pip package doesn't bundle
the example TSVs the GitHub README references): both the
`unlabeled_evaluation_set` and `in_domain_prompts_dataset` TSVs need columns
`Query` (falls back to `Question` if absent), `Document`, `Answer`. Unlike
ragas/ragchecker (one row per query, contexts as a list), ARES scores
Context_Relevance per (query, document) PAIR -- a query with N retrieved
chunks becomes N rows sharing the same Query/Answer with a different Document
each.

Install note: `ares-ai` needs its own Python 3.10 venv (see
requirements-eval-ares.txt) -- this module is meant to be run under that
venv's interpreter, not this project's main venv, and therefore works off
ResolvedExample (plain strings) rather than RAGTrace/ChunkRegistry directly.
"""

from typing import List

from scripts.baseline_adapters.common import ResolvedExample


def to_ares_dataframe(examples: List[ResolvedExample]):
    """
    Builds a pandas DataFrame with one row per (query, document) pair, in the
    Query/Document/Answer schema ares/ues_idp.py reads via `row['Query']`,
    `row['Document']`, `row['Answer']`.
    """
    import pandas as pd

    rows = [
        {"Query": example.question, "Document": context, "Answer": example.answer}
        for example in examples
        for context in example.contexts
    ]
    return pd.DataFrame(rows, columns=["Query", "Document", "Answer"])


def save_ares_tsv(examples: List[ResolvedExample], path: str) -> str:
    """Writes the ARES-format TSV to disk and returns the path (ARES reads TSVs from a filepath, not a DataFrame)."""
    df = to_ares_dataframe(examples)
    df.to_csv(path, sep="\t", index=False)
    return path


def build_few_shot_prompts_dataframe():
    """
    ARES's ues_idp mode expects an `in_domain_prompts_dataset` TSV to prime its
    few-shot judge prompts. Verified against the installed package's actual
    source (ares/RAG_Automatic_Evaluation/Evaluation_Functions.py): each
    few-shot row needs Query/Document/Answer PLUS one label column per judged
    dimension -- `Context_Relevance_Label`, `Answer_Relevance_Label`,
    `Answer_Faithfulness_Label` -- each an explicit "[[Yes]]"/"[[No]]" string
    (passing bare 1/0 triggers a deprecation warning in that code path). The
    pip package doesn't bundle a ready-made file (unlike the GitHub repo's
    datasets/example_files/), so this project supplies a small hand-authored
    one covering clearly-relevant/faithful and clearly-irrelevant/unfaithful
    cases, in the same legal domain as eval/eval_dataset.csv.
    """
    import pandas as pd

    rows = [
        {
            "Query": "What is the punishment for murder under BNS section 103?",
            "Document": "103.(1) Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
            "Answer": "Murder is punished with death or imprisonment for life, and the offender is also liable to fine.",
            "Context_Relevance_Label": "[[Yes]]",
            "Answer_Relevance_Label": "[[Yes]]",
            "Answer_Faithfulness_Label": "[[Yes]]",
        },
        {
            "Query": "What is the punishment for murder under BNS section 103?",
            "Document": "9.Facts not otherwise relevant are relevant if they are inconsistent with any fact in issue or relevant fact.",
            "Answer": "Murder is punished with death or imprisonment for life, and the offender is also liable to fine.",
            "Context_Relevance_Label": "[[No]]",
            "Answer_Relevance_Label": "[[No]]",
            "Answer_Faithfulness_Label": "[[No]]",
        },
    ]
    return pd.DataFrame(rows, columns=[
        "Query", "Document", "Answer",
        "Context_Relevance_Label", "Answer_Relevance_Label", "Answer_Faithfulness_Label",
    ])


def save_few_shot_prompts_tsv(path: str) -> str:
    build_few_shot_prompts_dataframe().to_csv(path, sep="\t", index=False)
    return path
