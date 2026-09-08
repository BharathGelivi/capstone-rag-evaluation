"""
Citation validation: does every legal citation in the answer exist, and was it
actually retrieved?

This is separate from claim verification and does not duplicate it. The existing
``ClaimDecomposer`` -> ``ClaimVerifier`` path answers *"is this proposition
supported by the evidence?"*. This module answers a different question that NLI
cannot: *"does the authority the answer names exist, and did the system actually
see it?"* A model can state a perfectly entailed proposition and attach a
citation it invented; entailment scoring will pass it, because the claim text is
supported by the retrieved passage. Only an identifier check catches it.

Three verdicts, ordered by how much trust they justify:

``GROUNDED``
    The citation appears in the retrieved evidence -- in a chunk's own citation
    metadata, or verbatim in a retrieved passage. The answer is pointing at
    something it was actually shown.
``IN_CORPUS_NOT_RETRIEVED``
    The citation resolves to a document in the corpus manifest, but that
    document was not retrieved for this question. The authority is real; the
    answer is drawing on parametric memory rather than on evidence. Reported
    separately because it is a *different* failure from invention.
``UNVERIFIABLE``
    Neither in the evidence nor in the corpus. From the system's own point of
    view this citation is unsupported. It may exist in the world -- the corpus
    is 400 judgments, not Indian law -- so the label deliberately says
    "unverifiable", not "hallucinated".

The distinction matters for honest reporting: calling every out-of-corpus
citation a hallucination would overstate the result on a partial corpus.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.legal_corpus import CITATION_PATTERNS, extract_citations, normalize_case_citation

logger = logging.getLogger(__name__)

GROUNDED = "GROUNDED"
IN_CORPUS_NOT_RETRIEVED = "IN_CORPUS_NOT_RETRIEVED"
UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class CitationVerdict:
    """One citation found in a generated answer, and what backs it."""
    raw: str
    normalized: str
    reporter: str
    status: str
    evidence_chunk_id: Optional[str] = None
    evidence_document_id: Optional[str] = None
    source_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CitationReport:
    """Aggregate citation behaviour for one answer."""
    total_citations: int = 0
    grounded: int = 0
    in_corpus_not_retrieved: int = 0
    unverifiable: int = 0
    verdicts: List[CitationVerdict] = field(default_factory=list)
    gold_citations_expected: int = 0
    gold_citations_produced: int = 0

    @property
    def citation_precision(self) -> Optional[float]:
        """Of the citations the answer made, how many were grounded in evidence.

        ``None`` when the answer cited nothing -- an answer with no citations
        has undefined precision, and scoring it 1.0 would reward silence
        exactly the way the faithfulness metric does (see E3).
        """
        if self.total_citations == 0:
            return None
        return round(self.grounded / self.total_citations, 4)

    @property
    def citation_recall(self) -> Optional[float]:
        """Of the citations the gold evidence supports, how many were produced."""
        if self.gold_citations_expected == 0:
            return None
        return round(self.gold_citations_produced / self.gold_citations_expected, 4)

    @property
    def citation_correctness(self) -> Optional[float]:
        """Fraction of citations that are not unverifiable inventions."""
        if self.total_citations == 0:
            return None
        return round(1.0 - (self.unverifiable / self.total_citations), 4)

    @property
    def fabrication_rate(self) -> Optional[float]:
        if self.total_citations == 0:
            return None
        return round(self.unverifiable / self.total_citations, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "grounded": self.grounded,
            "in_corpus_not_retrieved": self.in_corpus_not_retrieved,
            "unverifiable": self.unverifiable,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_correctness": self.citation_correctness,
            "fabrication_rate": self.fabrication_rate,
            "gold_citations_expected": self.gold_citations_expected,
            "gold_citations_produced": self.gold_citations_produced,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def _normalize_text(text: str) -> str:
    """Collapse punctuation and whitespace so ``[2024] 10 S.C.R. 1`` and
    ``[2024] 10 SCR 1`` compare equal as raw strings."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def corpus_citation_index(manifest_rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """``normalized citation -> manifest row`` for everything in the corpus.

    Built from the manifest rather than the vector store: whether a citation is
    *real* is a property of the corpus, independent of what any one retrieval
    happened to return.
    """
    index: Dict[str, Dict[str, str]] = {}
    for row in manifest_rows:
        for value in (row.get("citation"), row.get("neutral_citation")):
            if not value:
                continue
            for reporter, pattern in CITATION_PATTERNS.items():
                match = pattern.search(value)
                if match:
                    index[normalize_case_citation(reporter, match.groups())] = row
    return index


def evidence_citation_index(chunks: Sequence[Any], registry=None) -> Dict[str, Dict[str, str]]:
    """``normalized citation -> {chunk_id, document_id, source_url}`` for the
    evidence actually shown to the generator.

    Two sources are merged: the chunk's own provenance metadata (the document it
    came from) and any citation appearing in the chunk *text* (an authority the
    passage itself quotes). Both count as "the model was shown this".
    """
    index: Dict[str, Dict[str, str]] = {}

    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", "")
        metadata: Dict[str, Any] = {}
        record = registry.get_chunk(chunk_id) if registry is not None else None
        if record is not None:
            metadata = record.metadata or {}

        entry = {
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id", getattr(chunk, "parent_document_id", "")),
            "source_url": metadata.get("source_url", ""),
        }

        for value in (metadata.get("citation"), metadata.get("neutral_citation")):
            if not value:
                continue
            for reporter, pattern in CITATION_PATTERNS.items():
                match = pattern.search(str(value))
                if match:
                    index.setdefault(normalize_case_citation(reporter, match.groups()), entry)

        for citation in extract_citations(getattr(chunk, "chunk_text", "") or ""):
            if citation.kind == "case":
                index.setdefault(citation.normalized, entry)

    return index


def validate_answer_citations(
    answer: str,
    retrieved_chunks: Sequence[Any],
    registry=None,
    corpus_index: Optional[Dict[str, Dict[str, str]]] = None,
    gold_chunk_ids: Optional[Sequence[str]] = None,
) -> CitationReport:
    """Check every case citation in ``answer`` against evidence, then corpus.

    Statute and Article references are intentionally not scored: a section
    number is not a document identifier, cannot be resolved to a corpus object,
    and treating "Section 138" as a fabricable citation would produce a metric
    dominated by true negatives.
    """
    corpus_index = corpus_index or {}
    evidence_index = evidence_citation_index(retrieved_chunks, registry)

    report = CitationReport()
    seen: set = set()

    for citation in extract_citations(answer or ""):
        if citation.kind != "case":
            continue
        if citation.normalized in seen:
            continue
        seen.add(citation.normalized)
        report.total_citations += 1

        hit = evidence_index.get(citation.normalized)
        if hit:
            report.grounded += 1
            report.verdicts.append(CitationVerdict(
                raw=citation.raw, normalized=citation.normalized, reporter=citation.reporter,
                status=GROUNDED, evidence_chunk_id=hit["chunk_id"],
                evidence_document_id=hit["document_id"], source_url=hit["source_url"],
            ))
            continue

        corpus_hit = corpus_index.get(citation.normalized)
        if corpus_hit:
            report.in_corpus_not_retrieved += 1
            report.verdicts.append(CitationVerdict(
                raw=citation.raw, normalized=citation.normalized, reporter=citation.reporter,
                status=IN_CORPUS_NOT_RETRIEVED,
                evidence_document_id=corpus_hit.get("document_id", ""),
                source_url=corpus_hit.get("source_url", ""),
            ))
            continue

        report.unverifiable += 1
        report.verdicts.append(CitationVerdict(
            raw=citation.raw, normalized=citation.normalized, reporter=citation.reporter,
            status=UNVERIFIABLE,
        ))

    if gold_chunk_ids and registry is not None:
        # Counted per gold *document*, not per identifier. One decision carries
        # several equivalent citations ("[2024] 10 S.C.R. 1" and "2024 INSC
        # 746"); counting each would inflate the denominator and mark an answer
        # that cited the case correctly, once, as half-recalled.
        gold_by_document: Dict[str, set] = {}
        for chunk_id in gold_chunk_ids:
            record = registry.get_chunk(chunk_id)
            if record is None:
                continue
            document_id = record.metadata.get("document_id") or chunk_id
            keys = gold_by_document.setdefault(document_id, set())
            for value in (record.metadata.get("citation"), record.metadata.get("neutral_citation")):
                if not value:
                    continue
                for reporter, pattern in CITATION_PATTERNS.items():
                    match = pattern.search(str(value))
                    if match:
                        keys.add(normalize_case_citation(reporter, match.groups()))

        cited_documents = {document_id for document_id, keys in gold_by_document.items()
                           if keys and (keys & seen)}
        report.gold_citations_expected = sum(1 for keys in gold_by_document.values() if keys)
        report.gold_citations_produced = len(cited_documents)

    return report


def load_corpus_index(manifest_path: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    from src.legal_corpus import MANIFEST_PATH, load_corpus_manifest

    rows = load_corpus_manifest(manifest_path or MANIFEST_PATH)
    return corpus_citation_index(rows.values())


def demo() -> None:
    """Self-check. No corpus, no models."""

    class FakeChunk:
        def __init__(self, chunk_id, text):
            self.chunk_id = chunk_id
            self.chunk_text = text
            self.parent_document_id = "docA"

    class FakeRecord:
        def __init__(self, chunk_id, metadata, text):
            self.chunk_id = chunk_id
            self.metadata = metadata
            self.text = text

    class FakeRegistry:
        def __init__(self, records):
            self._records = {r.chunk_id: r for r in records}

        def get_chunk(self, chunk_id):
            return self._records.get(chunk_id)

    registry = FakeRegistry([
        FakeRecord("c1", {"document_id": "docA", "citation": "[2024] 10 S.C.R. 1",
                          "neutral_citation": "2024 INSC 746",
                          "source_url": "https://example.invalid/a.pdf"},
                   "The Court considered (2013) 5 SCC 762 at length."),
    ])
    chunks = [FakeChunk("c1", "The Court considered (2013) 5 SCC 762 at length.")]

    corpus_index = corpus_citation_index([
        {"document_id": "docA", "citation": "[2024] 10 S.C.R. 1",
         "neutral_citation": "2024 INSC 746", "source_url": "https://example.invalid/a.pdf"},
        {"document_id": "docZ", "citation": "[2019] 4 S.C.R. 88", "neutral_citation": "",
         "source_url": "https://example.invalid/z.pdf"},
    ])

    answer = (
        "The position is settled by [2024] 10 S.C.R. 1, which followed (2013) 5 SCC 762. "
        "See also [2019] 4 S.C.R. 88 and the decision in [2031] 99 S.C.R. 12345."
    )

    report = validate_answer_citations(answer, chunks, registry, corpus_index,
                                       gold_chunk_ids=["c1"])

    assert report.total_citations == 4, report.to_dict()
    # Retrieved document's own citation and a citation quoted inside the passage
    # both count as grounded -- the model was shown each of them.
    assert report.grounded == 2, [v.to_dict() for v in report.verdicts]
    # Real, in the corpus, but not retrieved for this question: a distinct failure.
    assert report.in_corpus_not_retrieved == 1, report.to_dict()
    # Invented identifier: nowhere in evidence, nowhere in the corpus.
    assert report.unverifiable == 1, report.to_dict()

    assert report.citation_precision == 0.5, report.citation_precision
    assert report.citation_correctness == 0.75, report.citation_correctness
    assert report.fabrication_rate == 0.25
    # One gold document, cited in one of its two equivalent formats -> full recall.
    assert report.gold_citations_expected == 1, report.to_dict()
    assert report.gold_citations_produced == 1, report.to_dict()
    assert report.citation_recall == 1.0

    statuses = {v.raw: v.status for v in report.verdicts}
    assert statuses["[2024] 10 S.C.R. 1"] == GROUNDED
    assert statuses["[2019] 4 S.C.R. 88"] == IN_CORPUS_NOT_RETRIEVED
    assert statuses["[2031] 99 S.C.R. 12345"] == UNVERIFIABLE

    grounded = [v for v in report.verdicts if v.status == GROUNDED]
    assert all(v.evidence_chunk_id == "c1" for v in grounded), "grounded verdict lost provenance"
    assert all(v.source_url for v in grounded), "grounded verdict lost source_url"

    # An answer that cites nothing has undefined precision, not perfect precision.
    empty = validate_answer_citations("No citation here at all.", chunks, registry, corpus_index)
    assert empty.total_citations == 0
    assert empty.citation_precision is None and empty.citation_correctness is None

    # Statutory references are not scored as citations.
    statutes = validate_answer_citations(
        "Section 138 of the Negotiable Instruments Act, 1881 and Article 32 apply.",
        chunks, registry, corpus_index)
    assert statutes.total_citations == 0, statutes.to_dict()

    print(f"citation_check demo OK  (precision={report.citation_precision} "
          f"correctness={report.citation_correctness} fabricated={report.unverifiable})")


if __name__ == "__main__":
    demo()
