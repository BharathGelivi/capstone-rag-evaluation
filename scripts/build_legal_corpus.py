"""
Build the persistent legal corpus: parse -> select -> chunk -> embed -> store.

One pass over the downloaded judgments produces, for each chunking strategy,
a ``ChunkRegistry`` and a ChromaDB collection, plus three durable artifacts
that everything downstream reads:

    data/legal_corpus/metadata/manifest.csv    provenance, one row per document
    data/legal_corpus/metadata/failures.csv    documents that did not make it in
    data/legal_corpus/processed/citations.jsonl  extracted citation edges

Checkpointed at the document level: a document already in the manifest with the
same content hash is not re-parsed, and the embedding cache means re-running
after adding documents only embeds the new chunks. Killing this script and
re-running it is always safe.

Selection, not exhaustion: documents are ranked by
``legal_corpus.research_value_score`` (citation density and structural
completeness) and the top ``--limit`` are ingested. A corpus of 300 heavily
cross-citing judgments is worth more to a multi-hop retrieval study than 3000
arbitrary ones, and costs an order of magnitude less to embed.

Usage:
    python -m scripts.build_legal_corpus --limit 400
    python -m scripts.build_legal_corpus --limit 400 --strategies legal fixed
    python -m scripts.build_legal_corpus --stats-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

from src.legal_corpus import (
    CORPUS_ROOT,
    JUDGMENTS_ROOT,
    ParsedJudgment,
    append_manifest_rows,
    chunk_document,
    load_corpus_manifest,
    manifest_row,
    parse_judgment,
    read_fetch_manifests,
    record_failure,
    research_value_score,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(CORPUS_ROOT, "processed")
CITATIONS_PATH = os.path.join(PROCESSED_DIR, "citations.jsonl")
REGISTRY_DIR = os.path.join("artifacts", "legal")

#: Chroma collection and registry file per strategy. Two collections rather than
#: one tagged collection, because the chunking-strategy comparison must not let
#: one arm's chunks be retrievable during the other arm's run.
COLLECTIONS = {
    "legal": "legal_corpus_legal",
    "fixed": "legal_corpus_fixed",
}


def registry_path(strategy: str) -> str:
    return os.path.join(REGISTRY_DIR, f"chunk_registry_{strategy}.json")


def iter_pdfs(root: str = JUDGMENTS_ROOT) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".pdf"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def parse_all(paths: List[str], fetch_meta: Dict[str, Dict[str, str]]) -> List[ParsedJudgment]:
    """Parse every PDF, recording failures rather than aborting the run."""
    parsed: List[ParsedJudgment] = []
    for i, path in enumerate(paths, start=1):
        if i % 100 == 0:
            logger.info("parsed %d/%d", i, len(paths))
        try:
            doc = parse_judgment(path, fetch_meta.get(os.path.basename(path), {}))
        except Exception as exc:                      # corrupt PDF, encrypted, etc.
            record_failure(path, f"{type(exc).__name__}: {exc}")
            continue
        if len(doc.text.strip()) < 500:
            record_failure(path, "extracted text under 500 chars (scanned or empty PDF)")
            continue
        parsed.append(doc)
    return parsed


def write_citations(docs: List[ParsedJudgment]) -> None:
    """Persist the citation edges so the graph can be rebuilt without re-parsing."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(CITATIONS_PATH, "w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps({
                "document_id": doc.document_id,
                "case_name": doc.case_name,
                "court": doc.court,
                "date": doc.date,
                "citation": doc.citation,
                "neutral_citation": doc.neutral_citation,
                "source_url": doc.source_url,
                "local_path": doc.local_path,
                "sections_present": sorted(doc.sections.keys()),
                "citations": [c.to_dict() for c in doc.citations],
            }) + "\n")


def build_strategy(docs: List[ParsedJudgment], strategy: str, rebuild: bool) -> Dict[str, int]:
    """Chunk, register, embed and store one strategy's view of the corpus."""
    from src.chunk_registry import ChunkRegistry
    from src.embedding_engine import generate_embeddings
    from src.vector_store import ChromaVectorStore

    logger.info("=== strategy=%s ===", strategy)
    nodes = []
    for doc in docs:
        nodes.extend(chunk_document(doc, strategy))
    logger.info("%s: %d chunks from %d documents", strategy, len(nodes), len(docs))

    registry = ChunkRegistry()
    registry.register(nodes)
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    registry.save_to_json(registry_path(strategy))

    embeddings = generate_embeddings(registry)

    store = ChromaVectorStore(collection_name=COLLECTIONS[strategy])
    store.initialize_collection()
    if rebuild:
        try:
            store.delete_collection()
        except Exception:
            pass
        store.initialize_collection()
    store.add_embeddings(embeddings, registry)

    stats = registry.get_statistics()
    logger.info("%s: stored %d vectors (avg chunk %.0f chars)",
                strategy, store.count(), stats["avg_chunk_length"])
    return {"chunks": len(nodes), "vectors": store.count(),
            "avg_chunk_chars": stats["avg_chunk_length"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=400,
                        help="max documents to ingest, highest research value first (0 = all)")
    parser.add_argument("--strategies", nargs="+", default=["legal", "fixed"],
                        choices=["legal", "fixed"])
    parser.add_argument("--judgments-root", default=JUDGMENTS_ROOT)
    parser.add_argument("--rebuild", action="store_true",
                        help="drop and recreate the Chroma collections first")
    parser.add_argument("--stats-only", action="store_true",
                        help="parse and report corpus composition without embedding")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    paths = iter_pdfs(args.judgments_root)
    if not paths:
        logger.error("no PDFs under %s -- run scripts/fetch_judgments.py first", args.judgments_root)
        return 1
    logger.info("found %d judgment PDFs", len(paths))

    fetch_meta = read_fetch_manifests(args.judgments_root)
    logger.info("joined %d fetch-manifest rows", len(fetch_meta))

    docs = parse_all(paths, fetch_meta)
    logger.info("parsed %d documents (%d failed)", len(docs), len(paths) - len(docs))

    docs.sort(key=research_value_score, reverse=True)
    if args.limit:
        docs = docs[: args.limit]

    structured = sum(1 for d in docs if "headnote" in d.sections)
    with_cites = sum(1 for d in docs if d.case_citations)
    total_case_edges = sum(len(d.case_citations) for d in docs)
    total_statute_edges = sum(len(d.statute_citations) for d in docs)

    logger.info(
        "selected %d documents | %d with headnotes | %d citing at least one case | "
        "%d case edges | %d statute edges",
        len(docs), structured, with_cites, total_case_edges, total_statute_edges,
    )

    write_citations(docs)
    logger.info("citation edges written to %s", CITATIONS_PATH)

    already = load_corpus_manifest()
    new_rows = [manifest_row(d, fetch_meta.get(os.path.basename(d.local_path), {}))
                for d in docs if d.document_id not in already]
    append_manifest_rows(new_rows)
    logger.info("manifest: %d new rows (%d already recorded)", len(new_rows), len(already))

    if args.stats_only:
        print(json.dumps({
            "pdfs_found": len(paths),
            "documents_parsed": len(docs),
            "documents_with_headnotes": structured,
            "documents_citing_cases": with_cites,
            "case_citation_edges": total_case_edges,
            "statute_citation_edges": total_statute_edges,
        }, indent=2))
        return 0

    summary = {}
    for strategy in args.strategies:
        summary[strategy] = build_strategy(docs, strategy, args.rebuild)

    print(json.dumps({"documents": len(docs), "strategies": summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
