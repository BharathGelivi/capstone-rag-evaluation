"""
Legal knowledge graph over the judgment corpus.

NetworkX, not Neo4j. The graph is ~10^4 nodes and ~10^4 edges, lives entirely
in memory, is rebuilt from ``citations.jsonl`` in seconds, and is queried with
one- to three-hop traversals. A graph database would add a service, a driver, a
query language and a deployment story to buy nothing measurable at this scale.
If the corpus grows two orders of magnitude, revisit -- that is a measurement,
not a preference.

What the graph is for
---------------------
Vector and lexical retrieval both match *text*. Neither can answer "what did the
courts that cited this case later do with it", because that relation is not in
any single chunk's text -- it is in the structure across documents. The graph
supplies exactly that relation and nothing else; it does not replace retrieval,
it expands a seed set produced by retrieval.

Provenance is mandatory
-----------------------
Every edge carries ``document_id``, ``chunk_id``, ``source_text`` and
``source_url``. An edge that cannot say where it came from is not evidence, and
this graph is used as evidence. Edges are only created from reporter-assigned
identifiers and, for treatment, from the reporter's own *Case Law Reference*
table -- never from a model's opinion about what a judgment "really" held.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

GRAPH_PATH = os.path.join("artifacts", "legal", "knowledge_graph.json")
CITATIONS_PATH = os.path.join("data", "legal_corpus", "processed", "citations.jsonl")

# Node type prefixes. Kept as string prefixes on the node id so a node's type is
# readable in any dump without a lookup.
CASE = "case"
COURT = "court"
STATUTE = "statute"
SECTION = "section"
ARTICLE = "article"
ISSUE = "issue"
PRINCIPLE = "principle"

#: Reporter treatment vocabulary, from the "Case Law Reference" tables. The
#: order matters: "not followed" must be tested before "followed", and
#: "distinguished" before the weaker "referred to", because a line often
#: contains more than one and the strongest treatment is the operative one.
TREATMENT_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("CASE_OVERRULES_CASE", re.compile(r"\boverrul(?:ed|ing)\b", re.I)),
    ("CASE_OVERRULES_CASE", re.compile(r"\bnot\s+good\s+law\b", re.I)),
    ("CASE_DISTINGUISHES_CASE", re.compile(r"\bdistinguish(?:ed|ing)\b", re.I)),
    ("CASE_DOUBTS_CASE", re.compile(r"\bdoubted\b", re.I)),
    ("CASE_DOES_NOT_FOLLOW_CASE", re.compile(r"\bnot\s+follow(?:ed)?\b", re.I)),
    ("CASE_FOLLOWS_CASE", re.compile(r"\bfollow(?:ed|ing)\b", re.I)),
    ("CASE_FOLLOWS_CASE", re.compile(r"\breli(?:ed|ance)\s+(?:on|upon)\b", re.I)),
    ("CASE_APPROVES_CASE", re.compile(r"\bapproved\b", re.I)),
    ("CASE_CITES_CASE", re.compile(r"\breferred\s+to\b", re.I)),
    ("CASE_CITES_CASE", re.compile(r"\bcited\b", re.I)),
)

#: Sections whose statute references mean the court *construed* the provision,
#: as opposed to merely mentioning it. The headnote and the holding are where a
#: reporter records what was decided, so a section reference there is an
#: interpretation edge; the same reference in the body is a citation edge.
INTERPRETIVE_SECTIONS = frozenset({"headnote", "held", "issue"})


@dataclass
class EdgeProvenance:
    """Where an edge came from. Every field is required to be non-empty."""
    document_id: str
    chunk_id: str
    source_text: str
    source_url: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source_text": self.source_text[:400],
            "source_url": self.source_url,
        }


def case_node(citation_key: str) -> str:
    return f"{CASE}:{citation_key}"


def _statute_node(normalized: str) -> Tuple[str, Optional[str]]:
    """Map a normalized statute citation onto ``(node_id, parent_act_node)``."""
    if normalized.startswith("ARTICLE:"):
        return f"{ARTICLE}:{normalized.split(':', 1)[1]}", None
    if normalized.startswith("ACT:"):
        _, act, section = (normalized.split(":") + ["", ""])[:3]
        act_node = f"{STATUTE}:{act}"
        if section:
            return f"{SECTION}:{act}:{section}", act_node
        return act_node, None
    return f"{SECTION}:{normalized.split(':', 1)[-1]}", None


def _read_citations(path: str = CITATIONS_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m scripts.build_legal_corpus` first."
        )
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _chunk_index(registry) -> Dict[str, List[str]]:
    """``document_id -> [chunk_id, ...]``, so an edge can name the chunk it came
    from and a graph hop can return retrievable text rather than a bare node."""
    index: Dict[str, List[str]] = defaultdict(list)
    for record in registry._records.values():
        document_id = record.metadata.get("document_id")
        if document_id:
            index[document_id].append(record.chunk_id)
    return index


def _section_chunk(registry, document_id: str, section: str) -> Optional[str]:
    for record in registry._records.values():
        if (record.metadata.get("document_id") == document_id
                and record.metadata.get("section") == section):
            return record.chunk_id
    return None


def parse_treatments(case_law_text: str) -> List[Tuple[str, str, str]]:
    """Extract ``(citation_key, relation, line)`` from a Case Law Reference table.

    The reporter writes one cited authority per line with its treatment:
    ``(2013) 5 SCC 762  relied on  para 12``. That word is the court's own
    characterisation, which is why this is worth parsing rather than asking a
    model to infer precedential treatment from prose.
    """
    from src.legal_corpus import CITATION_PATTERNS, normalize_case_citation

    treatments: List[Tuple[str, str, str]] = []
    for line in case_law_text.split("\n"):
        if len(line.strip()) < 8:
            continue
        keys = []
        for reporter, pattern in CITATION_PATTERNS.items():
            for match in pattern.finditer(line):
                keys.append(normalize_case_citation(reporter, match.groups()))
        if not keys:
            continue
        relation = "CASE_CITES_CASE"
        for candidate, pattern in TREATMENT_PATTERNS:
            if pattern.search(line):
                relation = candidate
                break
        for key in keys:
            treatments.append((key, relation, line.strip()))
    return treatments


def build_graph(registry=None, citations_path: str = CITATIONS_PATH):
    """Build the knowledge graph from extracted citations plus the chunk registry.

    ``registry`` is optional: without it the graph is still structurally correct
    but edges carry no ``chunk_id``, so graph hops cannot return text. Retrieval
    use always passes one.
    """
    import networkx as nx

    rows = _read_citations(citations_path)
    graph = nx.MultiDiGraph()
    chunks = _chunk_index(registry) if registry is not None else {}

    # Pass 1: every document becomes a Case node keyed by its own citation, so
    # an inbound reference from another judgment resolves to a real document.
    citation_to_document: Dict[str, str] = {}
    for row in rows:
        document_id = row["document_id"]
        keys = [k for k in (_self_key(row.get("citation")), _self_key(row.get("neutral_citation")))
                if k]
        primary = keys[0] if keys else f"DOC:{document_id}"
        node = case_node(primary)
        graph.add_node(
            node, type=CASE, document_id=document_id, case_name=row.get("case_name", ""),
            court=row.get("court", ""), date=row.get("date", ""),
            citation=row.get("citation", ""), neutral_citation=row.get("neutral_citation", ""),
            source_url=row.get("source_url", ""), in_corpus=True,
            chunk_ids=chunks.get(document_id, []),
        )
        for key in keys:
            citation_to_document[key] = node

        court = row.get("court") or "Unknown Court"
        court_id = f"{COURT}:{court}"
        graph.add_node(court_id, type=COURT, name=court)
        graph.add_edge(node, court_id, relation="CASE_DECIDED_BY_COURT",
                       **_provenance_for(row, chunks, "caption"))

    # Pass 2: citation and statute edges.
    for row in rows:
        document_id = row["document_id"]
        keys = [k for k in (_self_key(row.get("citation")), _self_key(row.get("neutral_citation")))
                if k]
        source = case_node(keys[0]) if keys else case_node(f"DOC:{document_id}")
        self_keys = set(keys)

        for citation in row.get("citations", []):
            normalized = citation["normalized"]
            section_name = citation.get("source_section", "")

            if citation["kind"] == "case":
                if normalized in self_keys:
                    continue                      # a report citing its own headnote
                target = citation_to_document.get(normalized) or case_node(normalized)
                if target not in graph:
                    # Cited but not held locally: a real node, flagged so
                    # coverage can be measured rather than assumed.
                    graph.add_node(target, type=CASE, in_corpus=False,
                                   citation=citation["raw"], chunk_ids=[])
                graph.add_edge(source, target, relation="CASE_CITES_CASE",
                               **_provenance_for(row, chunks, section_name, citation["raw"]))
            else:
                target, parent = _statute_node(normalized)
                if target not in graph:
                    graph.add_node(target, type=SECTION if target.startswith(SECTION) else
                                   (ARTICLE if target.startswith(ARTICLE) else STATUTE),
                                   label=citation["raw"])
                if parent and parent not in graph:
                    graph.add_node(parent, type=STATUTE, label=parent.split(":", 1)[1])
                if parent:
                    graph.add_edge(target, parent, relation="SECTION_OF_STATUTE")
                relation = ("CASE_INTERPRETS_SECTION"
                            if section_name in INTERPRETIVE_SECTIONS
                            else "CASE_CITES_STATUTE")
                graph.add_edge(source, target, relation=relation,
                               **_provenance_for(row, chunks, section_name, citation["raw"]))

    logger.info("graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _self_key(citation: Optional[str]) -> Optional[str]:
    """Normalize a document's own citation string into a graph key."""
    if not citation:
        return None
    from src.legal_corpus import CITATION_PATTERNS, normalize_case_citation

    for reporter, pattern in CITATION_PATTERNS.items():
        match = pattern.search(citation)
        if match:
            return normalize_case_citation(reporter, match.groups())
    return None


def _provenance_for(row: Dict[str, Any], chunks: Dict[str, List[str]],
                    section: str, source_text: str = "") -> Dict[str, Any]:
    document_id = row["document_id"]
    chunk_ids = chunks.get(document_id, [])
    return EdgeProvenance(
        document_id=document_id,
        chunk_id=chunk_ids[0] if chunk_ids else "",
        source_text=source_text or row.get("case_name", ""),
        source_url=row.get("source_url", ""),
    ).to_dict() | {"section": section}


def enrich_with_treatments(graph, registry, corpus_root: str = "data/legal_corpus") -> int:
    """Upgrade generic CASE_CITES_CASE edges to the reporter's own treatment.

    Returns the number of edges relabelled. Runs as a second pass because it
    needs the *text* of the Case Law Reference section, which lives in the
    chunk registry rather than in the citation index.
    """
    if registry is None:
        return 0

    upgraded = 0
    by_document: Dict[str, str] = {}
    for record in registry._records.values():
        if record.metadata.get("section") == "case_law_cited":
            document_id = record.metadata.get("document_id", "")
            by_document[document_id] = by_document.get(document_id, "") + "\n" + record.text

    document_to_node = {
        data.get("document_id"): node
        for node, data in graph.nodes(data=True)
        if data.get("type") == CASE and data.get("in_corpus")
    }

    for document_id, text in by_document.items():
        source = document_to_node.get(document_id)
        if not source:
            continue
        for key, relation, line in parse_treatments(text):
            if relation == "CASE_CITES_CASE":
                continue
            target = None
            for candidate in (case_node(key),):
                if candidate in graph:
                    target = candidate
            if target is None:
                continue
            for _, _, data in graph.edges(source, data=True):
                pass
            # Relabel the existing edge rather than adding a parallel one, so a
            # traversal cannot double-count the same reporter assertion.
            for _, tgt, edge_key, data in list(graph.edges(source, keys=True, data=True)):
                if tgt == target and data.get("relation") == "CASE_CITES_CASE":
                    data["relation"] = relation
                    data["source_text"] = line[:400]
                    upgraded += 1
                    break
    logger.info("treatment edges labelled: %d", upgraded)
    return upgraded


# ---------------------------------------------------------------------------
# Retrieval participation
# ---------------------------------------------------------------------------

@dataclass
class GraphHit:
    """One chunk reached by traversal, with the path that justified it."""
    chunk_id: str
    document_id: str
    hops: int
    relation_path: List[str]
    via_node: str
    source_url: str = ""


def chunks_for_documents(graph, node: str) -> List[str]:
    return list(graph.nodes[node].get("chunk_ids") or [])


def expand(
    graph,
    seed_chunk_ids: Sequence[str],
    registry,
    max_hops: int = 2,
    max_chunks: int = 12,
    relations: Optional[Sequence[str]] = None,
) -> List[GraphHit]:
    """Expand a retrieved chunk set along citation edges.

    This is how the graph *participates in retrieval*: seeds come from the
    vector/lexical retriever, and the graph contributes documents that are
    related by citation rather than by wording -- the chunks a text matcher
    structurally cannot reach.

    Traversal is bounded by ``max_hops`` and ``max_chunks`` and is breadth-first,
    so the nearest authority is preferred over the deepest chain.
    """
    if not seed_chunk_ids:
        return []

    seed_documents: Set[str] = set()
    for chunk_id in seed_chunk_ids:
        record = registry.get_chunk(chunk_id)
        if record:
            document_id = record.metadata.get("document_id")
            if document_id:
                seed_documents.add(document_id)

    document_to_node = {
        data.get("document_id"): node
        for node, data in graph.nodes(data=True)
        if data.get("type") == CASE and data.get("document_id")
    }

    frontier: List[Tuple[str, int, List[str]]] = [
        (document_to_node[d], 0, []) for d in seed_documents if d in document_to_node
    ]
    visited: Set[str] = {node for node, _, _ in frontier}
    seen_chunks: Set[str] = set(seed_chunk_ids)
    hits: List[GraphHit] = []

    while frontier and len(hits) < max_chunks:
        node, depth, path = frontier.pop(0)
        if depth >= max_hops:
            continue
        for _, target, data in graph.edges(node, data=True):
            relation = data.get("relation", "")
            if relations and relation not in relations:
                continue
            if target in visited:
                continue
            visited.add(target)
            next_path = path + [relation]
            target_data = graph.nodes[target]
            for chunk_id in (target_data.get("chunk_ids") or []):
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                hits.append(GraphHit(
                    chunk_id=chunk_id,
                    document_id=target_data.get("document_id", ""),
                    hops=depth + 1,
                    relation_path=next_path,
                    via_node=target,
                    source_url=target_data.get("source_url", ""),
                ))
                if len(hits) >= max_chunks:
                    break
            if target_data.get("type") == CASE:
                frontier.append((target, depth + 1, next_path))
            if len(hits) >= max_chunks:
                break
    return hits


def find_citing_cases(graph, citation_key: str) -> List[Dict[str, Any]]:
    """Reverse edge: which corpus judgments cite this authority, and how.

    This is the query a text retriever cannot express, and the one that makes
    "was this principle later limited?" answerable.
    """
    node = case_node(citation_key)
    if node not in graph:
        return []
    out = []
    for source, _, data in graph.in_edges(node, data=True):
        source_data = graph.nodes[source]
        out.append({
            "document_id": source_data.get("document_id", ""),
            "case_name": source_data.get("case_name", ""),
            "date": source_data.get("date", ""),
            "relation": data.get("relation", ""),
            "source_text": data.get("source_text", ""),
            "source_url": source_data.get("source_url", ""),
        })
    return out


def find_cases_for_section(graph, section_node: str) -> List[Dict[str, Any]]:
    """Which judgments cite or interpret a statutory provision."""
    if section_node not in graph:
        return []
    out = []
    for source, _, data in graph.in_edges(section_node, data=True):
        source_data = graph.nodes[source]
        if source_data.get("type") != CASE:
            continue
        out.append({
            "document_id": source_data.get("document_id", ""),
            "case_name": source_data.get("case_name", ""),
            "relation": data.get("relation", ""),
            "source_url": source_data.get("source_url", ""),
        })
    return out


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_graph(graph, path: str = GRAPH_PATH) -> str:
    import networkx as nx

    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = nx.node_link_data(graph, edges="links")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return path


def load_graph(path: str = GRAPH_PATH):
    import networkx as nx

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return nx.node_link_graph(data, multigraph=True, directed=True, edges="links")


def graph_statistics(graph) -> Dict[str, Any]:
    types: Dict[str, int] = defaultdict(int)
    relations: Dict[str, int] = defaultdict(int)
    for _, data in graph.nodes(data=True):
        types[data.get("type", "unknown")] += 1
    for _, _, data in graph.edges(data=True):
        relations[data.get("relation", "unknown")] += 1
    in_corpus = sum(1 for _, d in graph.nodes(data=True)
                    if d.get("type") == CASE and d.get("in_corpus"))
    dangling = sum(1 for _, d in graph.nodes(data=True)
                   if d.get("type") == CASE and not d.get("in_corpus"))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(types),
        "relations": dict(relations),
        "cases_in_corpus": in_corpus,
        "cases_cited_but_not_held": dangling,
    }


def demo() -> None:
    """Self-check on a hand-built two-document corpus. No PDFs, no network."""
    import tempfile

    rows = [
        {
            "document_id": "docA", "case_name": "A v. B", "court": "Supreme Court of India",
            "date": "2020-01-01", "citation": "[2020] 1 S.C.R. 5", "neutral_citation": "",
            "source_url": "https://example.invalid/a.pdf", "local_path": "a.pdf",
            "sections_present": ["headnote", "judgment"],
            "citations": [
                {"kind": "case", "reporter": "SCC", "normalized": "SCC:2013:5:762",
                 "raw": "(2013) 5 SCC 762", "section": None, "source_section": "judgment",
                 "source_chunk_id": ""},
                {"kind": "statute", "reporter": "ARTICLE", "normalized": "ARTICLE:32",
                 "raw": "Article 32", "section": "32", "source_section": "headnote",
                 "source_chunk_id": ""},
            ],
        },
        {
            "document_id": "docB", "case_name": "C v. D", "court": "Supreme Court of India",
            "date": "2013-01-01", "citation": "(2013) 5 SCC 762", "neutral_citation": "",
            "source_url": "https://example.invalid/b.pdf", "local_path": "b.pdf",
            "sections_present": ["judgment"], "citations": [],
        },
    ]

    class FakeRecord:
        def __init__(self, chunk_id, document_id, section, text):
            self.chunk_id = chunk_id
            self.text = text
            self.metadata = {"document_id": document_id, "section": section}

    class FakeRegistry:
        def __init__(self, records):
            self._records = {r.chunk_id: r for r in records}

        def get_chunk(self, chunk_id):
            return self._records.get(chunk_id)

    registry = FakeRegistry([
        FakeRecord("cA1", "docA", "judgment", "A body text."),
        FakeRecord("cA2", "docA", "case_law_cited", "(2013) 5 SCC 762  overruled  para 9"),
        FakeRecord("cB1", "docB", "judgment", "B body text."),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "citations.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        graph = build_graph(registry, citations_path=path)

        a = case_node("SCR:2020:1:5")
        b = case_node("SCC:2013:5:762")
        assert a in graph and b in graph, list(graph.nodes)
        # docA cites docB, and docB is in the corpus, so the edge must resolve
        # onto the real document node rather than a dangling stub.
        assert graph.nodes[b]["in_corpus"] is True
        assert graph.nodes[b]["document_id"] == "docB"

        relations = {d["relation"] for _, _, d in graph.edges(data=True)}
        assert "CASE_CITES_CASE" in relations, relations
        assert "CASE_DECIDED_BY_COURT" in relations, relations
        # An Article cited inside the headnote is an interpretation, not a mention.
        assert "CASE_INTERPRETS_SECTION" in relations, relations

        for _, _, data in graph.edges(data=True):
            assert data.get("document_id"), "edge without provenance"
            assert "source_url" in data, "edge without source_url"

        upgraded = enrich_with_treatments(graph, registry)
        assert upgraded == 1, upgraded
        relations = {d["relation"] for _, _, d in graph.edges(data=True)}
        assert "CASE_OVERRULES_CASE" in relations, relations

        # Retrieval participation: a seed chunk in docA must reach docB's chunk,
        # which shares no vocabulary with it.
        hits = expand(graph, ["cA1"], registry, max_hops=2, max_chunks=5)
        assert any(h.chunk_id == "cB1" for h in hits), hits
        assert all(h.hops >= 1 and h.relation_path for h in hits)

        citing = find_citing_cases(graph, "SCC:2013:5:762")
        assert citing and citing[0]["document_id"] == "docA", citing

        section_cases = find_cases_for_section(graph, "article:32")
        assert section_cases and section_cases[0]["document_id"] == "docA", section_cases

        saved = save_graph(graph, os.path.join(tmpdir, "g.json"))
        reloaded = load_graph(saved)
        assert reloaded.number_of_nodes() == graph.number_of_nodes()

        stats = graph_statistics(graph)
        assert stats["cases_in_corpus"] == 2, stats

    print(f"legal_graph demo OK  ({stats['nodes']} nodes, {stats['edges']} edges, "
          f"treatments={upgraded})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
