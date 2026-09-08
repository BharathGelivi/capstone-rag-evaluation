"""
Ingestion Module.

Responsible for loading source documents from disk into LlamaIndex Document objects.
Supports PDF, plain-text, Markdown, and HTML files. Each file is loaded independently
so that a single corrupt file never aborts the entire ingestion run.
"""

import logging
import os
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PyMuPDFReader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format support
# ---------------------------------------------------------------------------
# PyMuPDFReader is used for PDFs to avoid the binary garbage that the default
# PDFReader produces on complex layouts (e.g., Indian legal codes).
# Plain-text, Markdown, and HTML fall back to SimpleDirectoryReader's built-in
# readers which handle them correctly out of the box.
_SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md", ".html"]
_FILE_EXTRACTOR = {".pdf": PyMuPDFReader()}


def load_documents_from_directory(data_dir: str) -> List[Document]:
    """Load all supported documents from *data_dir* (recursive).

    Walks the directory tree and ingests every file whose extension is in
    ``_SUPPORTED_EXTENSIONS``. Each file is loaded in isolation: if one file
    raises an exception it is skipped and a full error is logged, but the rest
    of the run continues unaffected.

    Args:
        data_dir: Path to the directory containing source files.

    Returns:
        A list of LlamaIndex ``Document`` objects. One object per page for PDFs;
        one object per file for plain-text / Markdown / HTML.

    Raises:
        FileNotFoundError: If *data_dir* does not exist.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Collect all candidate files up front so we can log a meaningful count
    # before handing them to SimpleDirectoryReader.
    candidate_files = [
        str(f)
        for f in data_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]

    if not candidate_files:
        logger.warning(
            "No supported files found in '%s'. "
            "Supported extensions: %s",
            data_dir,
            ", ".join(_SUPPORTED_EXTENSIONS),
        )
        return []

    logger.info(
        "Found %d supported file(s) in '%s'. Loading...",
        len(candidate_files),
        data_dir,
    )

    all_documents: List[Document] = []
    failed_files: List[str] = []

    for file_path in candidate_files:
        try:
            reader = SimpleDirectoryReader(
                input_files=[file_path],
                file_extractor=_FILE_EXTRACTOR,
                exclude_hidden=True,
            )
            docs = reader.load_data()
            all_documents.extend(docs)
            logger.debug("Loaded %d page(s) from '%s'.", len(docs), file_path)
        except Exception:
            # Log the full traceback at ERROR level but keep going — one bad
            # file must not kill the entire ingestion run.
            logger.error(
                "Failed to load '%s' — skipping. Full traceback:",
                file_path,
                exc_info=True,
            )
            failed_files.append(file_path)

    logger.info(
        "Ingestion complete. Loaded %d document object(s) from %d file(s). "
        "%d file(s) skipped due to errors.",
        len(all_documents),
        len(candidate_files) - len(failed_files),
        len(failed_files),
    )

    if failed_files:
        logger.warning("Skipped files:\n  %s", "\n  ".join(failed_files))

    return all_documents
