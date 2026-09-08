# Contributing to X-RAG Diagnostic Framework

Welcome! This document outlines how to contribute.

## Coding Standards
- Follow standard Python PEP-8 guidelines.
- Add type hints to all public methods.
- Document all classes and modules.
- Ensure diagnostic modules do NOT recalculate metrics or modify previous pipeline states.

## Folder Structure
- `src/`: Core framework and API logic.
- `configs/`: Modular configuration settings (models, pipelines, API).
- `tests/`: Unit tests (one per module).
- `artifacts/`: Generated outputs from pipeline runs (traces, reports, etc.).
- `data/`: Sample raw documents.
- `docs/`: Additional documentation and research logs.

## How to Add Diagnostic Modules
1. Create your module in `src/`.
2. Ensure it implements a distinct pipeline phase (e.g., `MyNewAnalyzer`).
3. Have it consume deterministic objects from the previous phase.
4. Add the execution call into `src/runner.py`.

## How to Run Tests
Run the entire suite using standard unittest discovery:
```bash
python -m unittest discover tests
```

## How to Submit Pull Requests
- Create a feature branch off `main`.
- Write unit tests for new functionality.
- Ensure `python -m unittest discover tests` passes with 0 failures.
- Open a PR explaining the changes, the diagnostic impact, and linking relevant issues.
