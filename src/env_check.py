"""Centralized HF_TOKEN validation for all entry points.

Call load_dotenv() first, then one of these functions before any
component that needs a Hugging Face API token is constructed.
"""

import os
import sys


def ensure_hf_token() -> None:
    """CLI entry-point guard: prompt interactively if HF_TOKEN is missing.

    Sets the token into os.environ for the current process and, on request,
    persists it to .env. Never echoes the token back in logs.
    """
    if os.environ.get("HF_TOKEN"):
        return

    token = input("Enter your Hugging Face API token (HF_TOKEN): ").strip()
    if not token:
        print("No token provided. HF_TOKEN remains unset; LLM calls will fail.")
        return

    os.environ["HF_TOKEN"] = token

    answer = input("Save this token to .env for future runs? [y/N]: ").strip().lower()
    if answer == "y":
        _persist_to_env_file(token)


def ensure_hf_token_or_exit() -> None:
    """Server entry-point guard: fail fast with a clear message, no stdin prompt."""
    if os.environ.get("HF_TOKEN"):
        return

    print(
        "ERROR: HF_TOKEN is not set.\n"
        "The X-RAG API requires a Hugging Face API token to run generation and "
        "claim-decomposition models.\n"
        "Set HF_TOKEN in your .env file or environment before starting the server.",
        file=sys.stderr,
    )
    sys.exit(1)


def ensure_llm_credentials() -> None:
    """CLI entry-point guard, provider-aware: prompts interactively for
    whichever key configs.models.LLM_PROVIDER actually needs (GROQ_API_KEY for
    "groq", NVIDIA_API_KEY for "nvidia", HF_TOKEN for "huggingface")."""
    from configs.models import LLM_PROVIDER

    if LLM_PROVIDER == "huggingface":
        ensure_hf_token()
        return
    
    if LLM_PROVIDER == "nvidia":
        if os.environ.get("NVIDIA_API_KEY"):
            return

        token = input("Enter your NVIDIA API key (NVIDIA_API_KEY): ").strip()
        if not token:
            print("No token provided. NVIDIA_API_KEY remains unset; LLM calls will fail.")
            return

        os.environ["NVIDIA_API_KEY"] = token

        answer = input("Save this token to .env for future runs? [y/N]: ").strip().lower()
        if answer == "y":
            _persist_to_env_file(token, key_name="NVIDIA_API_KEY")
        return

    if os.environ.get("GROQ_API_KEY"):
        return

    token = input("Enter your Groq API key (GROQ_API_KEY): ").strip()
    if not token:
        print("No token provided. GROQ_API_KEY remains unset; LLM calls will fail.")
        return

    os.environ["GROQ_API_KEY"] = token

    answer = input("Save this token to .env for future runs? [y/N]: ").strip().lower()
    if answer == "y":
        _persist_to_env_file(token, key_name="GROQ_API_KEY")


def ensure_llm_credentials_or_exit() -> None:
    """Server entry-point guard, provider-aware: fail fast with a clear
    message, no stdin prompt."""
    from configs.models import LLM_PROVIDER

    if LLM_PROVIDER == "huggingface":
        ensure_hf_token_or_exit()
        return

    if LLM_PROVIDER == "nvidia":
        if os.environ.get("NVIDIA_API_KEY"):
            return
        
        print(
            "ERROR: NVIDIA_API_KEY is not set.\n"
            "The X-RAG API requires an NVIDIA API key to run generation and "
            "claim-decomposition models (configs/models.py LLM_PROVIDER = \"nvidia\").\n"
            "Set NVIDIA_API_KEY in your .env file or environment before starting the server.",
            file=sys.stderr,
        )
        sys.exit(1)

    if os.environ.get("GROQ_API_KEY"):
        return

    print(
        "ERROR: GROQ_API_KEY is not set.\n"
        "The X-RAG API requires a Groq API key to run generation and "
        "claim-decomposition models (configs/models.py LLM_PROVIDER = \"groq\").\n"
        "Set GROQ_API_KEY in your .env file or environment before starting the server.",
        file=sys.stderr,
    )
    sys.exit(1)


def _persist_to_env_file(token: str, env_path: str = ".env", key_name: str = "HF_TOKEN") -> None:
    lines = []
    found = False
    prefix = f"{key_name}="
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                lines[i] = f"{prefix}{token}\n"
                found = True
                break
    if not found:
        lines.append(f"{prefix}{token}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    print(f"Saved {key_name} to {env_path}.")
