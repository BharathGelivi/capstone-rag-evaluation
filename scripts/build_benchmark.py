"""
Build the multi-hop legal retrieval benchmark from the ingested corpus.

Questions are **derived from the corpus**, not written from memory. That is the
only way to get two properties this study needs at once:

* **Gold evidence that is actually gold.** Each question names the exact
  document and chunk ids that answer it, because the question was generated
  *from* those documents. Hand-written questions would need a separate
  annotation pass to get the same thing, and would risk asking about text the
  corpus does not contain.
* **Provenance for every item.** Each row records how it was constructed, which
  graph edge or citation it came from, and the source URL of the underlying
  judgment -- so any question can be audited back to the record that produced it.

Nothing here invents a case, a citation or a holding. Every string that names a
legal authority is copied from the parsed corpus.

Ten question types, spanning what the ablation is supposed to separate:

    exact_citation        verbatim identifier lookup           (lexical)
    lexical_mismatch      issue-language query, body answer    (semantic)
    single_hop            one document answers it              (baseline)
    case_to_case          which judgment cites this authority  (graph)
    case_to_statute       which provision does this construe   (graph)
    statute_to_cases      which judgments construe this section(graph)
    citation_chain        A -> B -> C, two hops                (graph/IRCoT)
    contradictory         later adverse treatment              (contradiction)
    temporal              later cases citing an authority      (graph/temporal)
    entity_resolution     two citation formats, one decision   (normalisation)

Usage:
    python -m scripts.build_benchmark --per-type 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

BENCHMARK_PATH = os.path.join("eval", "legal_benchmark.json")
BENCHMARK_VERSION = "1.0"
SEED = 20260820


def _year(date_or_id: str) -> Optional[int]:
    match = re.search(r"(19|20)\d{2}", date_or_id or "")
    return int(match.group(0)) if match else None


def _doc_chunks(registry, document_id: str, section: Optional[str] = None) -> List[str]:
    out = []
    for record in registry._records.values():
        if record.metadata.get("document_id") != document_id:
            continue
        if section and record.metadata.get("section") != section:
            continue
        out.append(record.chunk_id)
    return out


def _best_chunks(registry, document_id: str) -> List[str]:
    """The chunks that state what the case decided.

    All of the headnote-family sections count, not just the first one present:
    a question about a judgment is answered by its issue, its headnote *or* its
    holding, and marking only one of them gold would score a correct retrieval
    as a miss. Falls back to the opening body chunks for documents whose
    structure did not parse.
    """
    found: List[str] = []
    for section in ("headnote", "held", "issue"):
        found.extend(_doc_chunks(registry, document_id, section))
    return found[:6] or _doc_chunks(registry, document_id)[:3]


def _clean(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    return text[:limit].rstrip(" ,;:-")


def _row(
    question_id: str, qtype: str, question: str, hops: int,
    gold_documents: Sequence[str], gold_chunks: Sequence[str],
    provenance: Dict[str, Any], docs_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "id": question_id,
        "type": qtype,
        "question": question,
        "hops": hops,
        "gold_document_ids": list(dict.fromkeys(gold_documents)),
        "gold_chunk_ids": list(dict.fromkeys(gold_chunks)),
        "provenance": {
            **provenance,
            "source_urls": [docs_by_id[d]["source_url"] for d in dict.fromkeys(gold_documents)
                            if d in docs_by_id],
            "construction": "auto-generated from parsed corpus structure",
            "benchmark_version": BENCHMARK_VERSION,
        },
    }


def build(registry, graph, docs: List[Dict[str, Any]], per_type: int) -> List[Dict[str, Any]]:
    from src.legal_graph import CASE, case_node

    rng = random.Random(SEED)
    docs_by_id = {d["document_id"]: d for d in docs}
    node_by_document = {
        data.get("document_id"): node
        for node, data in graph.nodes(data=True)
        if data.get("type") == CASE and data.get("in_corpus")
    }
    rows: List[Dict[str, Any]] = []

    # Only documents that actually produced chunks can be gold.
    chunked = {d for d in docs_by_id if _doc_chunks(registry, d)}
    logger.info("%d/%d documents have chunks in this registry", len(chunked), len(docs_by_id))

    # --- exact_citation ----------------------------------------------------
    cited_docs = [d for d in docs if d["document_id"] in chunked and d.get("citation")]
    rng.shuffle(cited_docs)
    for doc in cited_docs[:per_type]:
        rows.append(_row(
            f"exact_citation_{len(rows):03d}", "exact_citation",
            f"What was decided in {doc['citation']}?", 1,
            [doc["document_id"]], _best_chunks(registry, doc["document_id"]),
            {"basis": "document's own reporter citation", "citation": doc["citation"]},
            docs_by_id,
        ))

    # --- lexical_mismatch / single_hop -------------------------------------
    # The question is built from the *issue* section; the answer lives in the
    # judgment body, which uses different wording. That gap is the point.
    issue_docs = []
    for doc in docs:
        if doc["document_id"] not in chunked:
            continue
        issue_chunks = _doc_chunks(registry, doc["document_id"], "issue")
        if issue_chunks:
            record = registry.get_chunk(issue_chunks[0])
            if record and len(record.text) > 80:
                issue_docs.append((doc, record.text))
    rng.shuffle(issue_docs)

    for doc, issue_text in issue_docs[:per_type]:
        rows.append(_row(
            f"lexical_mismatch_{len(rows):03d}", "lexical_mismatch",
            f"{_clean(issue_text)} -- how did the Supreme Court resolve this?", 1,
            [doc["document_id"]],
            _doc_chunks(registry, doc["document_id"], "judgment")[:3]
            or _best_chunks(registry, doc["document_id"]),
            {"basis": "issue-for-consideration text; answer lies in the judgment body"},
            docs_by_id,
        ))

    headnote_docs = [d for d in docs if d["document_id"] in chunked
                     and _doc_chunks(registry, d["document_id"], "headnote")]
    rng.shuffle(headnote_docs)
    for doc in headnote_docs[:per_type]:
        chunks = _doc_chunks(registry, doc["document_id"], "headnote")
        record = registry.get_chunk(chunks[0])
        rows.append(_row(
            f"single_hop_{len(rows):03d}", "single_hop",
            f"In {doc.get('citation') or doc['case_name']}, {_clean(record.text, 180)} "
            f"-- what did the Court hold?", 1,
            [doc["document_id"]], chunks,
            {"basis": "headnote of a single judgment"},
            docs_by_id,
        ))

    # --- case_to_case ------------------------------------------------------
    # Pick authorities cited by a judgment we also hold, so the gold answer is
    # retrievable rather than merely referenced.
    in_corpus_edges: List[tuple] = []
    for source, target, data in graph.edges(data=True):
        if data.get("relation") not in ("CASE_CITES_CASE", "CASE_FOLLOWS_CASE",
                                        "CASE_APPROVES_CASE"):
            continue
        source_doc = graph.nodes[source].get("document_id")
        target_doc = graph.nodes[target].get("document_id")
        if source_doc in chunked and target_doc in chunked and source_doc != target_doc:
            in_corpus_edges.append((source, target, source_doc, target_doc, data))
    rng.shuffle(in_corpus_edges)

    for source, target, source_doc, target_doc, data in in_corpus_edges[:per_type]:
        target_citation = graph.nodes[target].get("citation") or docs_by_id[target_doc].get("citation")
        rows.append(_row(
            f"case_to_case_{len(rows):03d}", "case_to_case",
            f"Which judgment in this corpus relies on {target_citation}, and for what proposition?",
            2, [source_doc, target_doc],
            _best_chunks(registry, source_doc) + _best_chunks(registry, target_doc),
            {"basis": "citation edge asserted by the reporter",
             "relation": data.get("relation"), "edge_source_text": data.get("source_text", "")[:200]},
            docs_by_id,
        ))

    # --- case_to_statute ---------------------------------------------------
    interprets = [(s, t, d) for s, t, d in graph.edges(data=True)
                  if d.get("relation") == "CASE_INTERPRETS_SECTION"
                  and graph.nodes[s].get("document_id") in chunked]
    rng.shuffle(interprets)
    seen_docs = set()
    for source, target, data in interprets:
        document_id = graph.nodes[source]["document_id"]
        if document_id in seen_docs:
            continue
        seen_docs.add(document_id)
        label = graph.nodes[target].get("label") or target
        rows.append(_row(
            f"case_to_statute_{len(rows):03d}", "case_to_statute",
            f"Which statutory provision did the Court construe in "
            f"{docs_by_id[document_id].get('citation') or docs_by_id[document_id]['case_name']}, "
            f"and with what effect?", 1,
            [document_id], _best_chunks(registry, document_id),
            {"basis": "interpretation edge from headnote/held section",
             "statute_node": target, "statute_label": label},
            docs_by_id,
        ))
        if len(seen_docs) >= per_type:
            break

    # --- statute_to_cases --------------------------------------------------
    by_section: Dict[str, List[str]] = defaultdict(list)
    for source, target, data in graph.edges(data=True):
        if data.get("relation") not in ("CASE_INTERPRETS_SECTION", "CASE_CITES_STATUTE"):
            continue
        document_id = graph.nodes[source].get("document_id")
        if document_id in chunked:
            by_section[target].append(document_id)
    multi = [(section, sorted(set(ds))) for section, ds in by_section.items() if len(set(ds)) >= 3]
    rng.shuffle(multi)
    for section, document_ids in multi[:per_type]:
        label = graph.nodes[section].get("label") or section
        gold = document_ids[:4]
        rows.append(_row(
            f"statute_to_cases_{len(rows):03d}", "statute_to_cases",
            f"Which judgments in this corpus apply or construe {label}?", 2,
            gold, [c for d in gold for c in _best_chunks(registry, d)[:1]],
            {"basis": "reverse statute edge", "statute_node": section,
             "n_citing_documents": len(document_ids)},
            docs_by_id,
        ))

    # --- citation_chain (two hops) -----------------------------------------
    chains = []
    for source, middle, _source_doc, _middle_doc, _data in in_corpus_edges[:400]:
        middle_doc = graph.nodes[middle].get("document_id")
        for _, third, data2 in graph.edges(middle, data=True):
            if data2.get("relation") not in ("CASE_CITES_CASE", "CASE_FOLLOWS_CASE"):
                continue
            third_doc = graph.nodes[third].get("document_id")
            if third_doc in chunked and third_doc not in (
                    graph.nodes[source].get("document_id"), middle_doc):
                chains.append((source, middle, third))
                break
    rng.shuffle(chains)
    for source, middle, third in chains[:per_type]:
        source_doc = graph.nodes[source]["document_id"]
        middle_doc = graph.nodes[middle]["document_id"]
        third_doc = graph.nodes[third]["document_id"]
        source_citation = docs_by_id[source_doc].get("citation") or source_doc
        rows.append(_row(
            f"citation_chain_{len(rows):03d}", "citation_chain",
            f"Starting from {source_citation}: which authority does the case it relies on "
            f"itself rely on, and what does that authority establish?", 3,
            [source_doc, middle_doc, third_doc],
            [c for d in (source_doc, middle_doc, third_doc) for c in _best_chunks(registry, d)[:1]],
            {"basis": "two-hop citation path", "path": [source, middle, third]},
            docs_by_id,
        ))

    # --- contradictory authority -------------------------------------------
    adverse = [(s, t, d) for s, t, d in graph.edges(data=True)
               if d.get("relation") in ("CASE_OVERRULES_CASE", "CASE_DISTINGUISHES_CASE",
                                        "CASE_DOUBTS_CASE", "CASE_DOES_NOT_FOLLOW_CASE")]
    rng.shuffle(adverse)
    for source, target, data in adverse[:per_type]:
        source_doc = graph.nodes[source].get("document_id")
        target_doc = graph.nodes[target].get("document_id")
        gold = [d for d in (source_doc, target_doc) if d in chunked]
        if not gold:
            continue
        target_citation = graph.nodes[target].get("citation") or target
        rows.append(_row(
            f"contradictory_{len(rows):03d}", "contradictory",
            f"Has the view taken in {target_citation} been doubted, distinguished or "
            f"overruled by any later judgment in this corpus?", 2,
            gold, [c for d in gold for c in _best_chunks(registry, d)[:2]],
            {"basis": "adverse treatment recorded in a Case Law Reference table",
             "relation": data.get("relation"),
             "edge_source_text": data.get("source_text", "")[:200]},
            docs_by_id,
        ))

    # --- temporal ----------------------------------------------------------
    temporal = []
    for source, target, data in graph.edges(data=True):
        if data.get("relation", "").startswith("CASE_") and "CASE" in data.get("relation", ""):
            source_doc = graph.nodes[source].get("document_id")
            target_doc = graph.nodes[target].get("document_id")
            if source_doc in chunked and target_doc in chunked:
                y1, y2 = _year(source_doc), _year(target_doc)
                if y1 and y2 and y1 > y2:
                    temporal.append((source_doc, target_doc, y1, y2))
    rng.shuffle(temporal)
    for source_doc, target_doc, y1, y2 in temporal[:per_type]:
        target_citation = docs_by_id[target_doc].get("citation") or target_doc
        rows.append(_row(
            f"temporal_{len(rows):03d}", "temporal",
            f"How has {target_citation} been treated by the Supreme Court after {y2}?", 2,
            [target_doc, source_doc],
            _best_chunks(registry, target_doc)[:2] + _best_chunks(registry, source_doc)[:2],
            {"basis": "citation edge with a later citing year",
             "cited_year": y2, "citing_year": y1},
            docs_by_id,
        ))

    # --- entity_resolution --------------------------------------------------
    # Documents carrying two independent identifiers for one decision. Asking
    # about both formats tests whether retrieval resolves them to one entity.
    dual = [d for d in docs if d["document_id"] in chunked
            and d.get("citation") and d.get("neutral_citation")]
    rng.shuffle(dual)
    for doc in dual[:per_type]:
        rows.append(_row(
            f"entity_resolution_{len(rows):03d}", "entity_resolution",
            f"Are {doc['citation']} and {doc['neutral_citation']} the same decision? "
            f"State what it decided.", 1,
            [doc["document_id"]], _best_chunks(registry, doc["document_id"]),
            {"basis": "two reporter identifiers on one document",
             "citation": doc["citation"], "neutral_citation": doc["neutral_citation"]},
            docs_by_id,
        ))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-type", type=int, default=6)
    parser.add_argument("--strategy", default="legal", choices=["legal", "fixed"],
                        help="registry whose chunk ids become gold evidence")
    parser.add_argument("--out", default=BENCHMARK_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    from src.chunk_registry import ChunkRegistry
    from src.legal_graph import build_graph, enrich_with_treatments, graph_statistics, save_graph
    from scripts.build_legal_corpus import CITATIONS_PATH, registry_path

    registry = ChunkRegistry.load_from_json(registry_path(args.strategy))
    docs = [json.loads(line) for line in open(CITATIONS_PATH, encoding="utf-8") if line.strip()]

    graph = build_graph(registry)
    enrich_with_treatments(graph, registry)
    save_graph(graph)
    logger.info("graph stats: %s", json.dumps(graph_statistics(graph)))

    rows = build(registry, graph, docs, args.per_type)

    by_type: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_type[row["type"]] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({
            "benchmark_version": BENCHMARK_VERSION,
            "seed": SEED,
            "chunking_strategy_for_gold": args.strategy,
            "n_questions": len(rows),
            "by_type": dict(by_type),
            "questions": rows,
        }, handle, indent=2)

    print(json.dumps({"n_questions": len(rows), "by_type": dict(by_type),
                      "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
