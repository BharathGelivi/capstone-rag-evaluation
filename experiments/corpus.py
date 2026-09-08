"""
Corpus instrumentation shared by E2 and E4.

Two things live here:

``classify_chunk``
    A rule-based classifier that labels a chunk by what kind of text it is --
    an operative provision, a table-of-contents entry list, a bare heading,
    enacting boilerplate, or an extraction fragment. Only the first carries
    answers; the rest are retrievable text that can occupy a context slot
    without contributing anything.

``build_gold_index``
    Weak-supervision gold labels derived from the eval set's
    ``source_document`` / ``source_section`` columns, so both experiments score
    against the same notion of a correct chunk.

Why a rule and not a model: the classifier's decisions have to be auditable
line by line for a paper, and the distinctions are typographic (line shape,
section-number density, verb presence) rather than semantic. Its features are
returned alongside every label so a disputed classification can be checked.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Chunk classification
# ---------------------------------------------------------------------------

#: A table-of-contents line: a section number followed by a short noun-phrase
#: title, with no sentence body. "45. Abetment of a thing." matches;
#: "45. A person abets the doing of a thing who ..." does not, because the
#: length bound rejects it.
_TOC_ENTRY = re.compile(r"^\s*\d+[A-Z]?\.\s*[A-Z][^.]{2,70}\.?\s*$")

#: A structural heading with no body text.
_HEADING = re.compile(
    r"^\s*(?:CHAPTER\s+[IVXLC\d]+|PART\s+[IVXLC\d]+|SCHEDULE|ARRANGEMENT OF SECTIONS"
    r"|OF\s+[A-Z][A-Z\s,]+|[A-Z][A-Z\s,'-]{6,})\s*$"
)

#: Running headers and footers the PDF repeats on every page. These are *page
#: furniture*: they are not a kind of chunk, they are noise that bleeds into
#: otherwise-good chunks during extraction. Measured per line and reported as a
#: contamination rate, deliberately NOT used to classify a chunk -- an early
#: version of this classifier did, and labelled 173 perfectly good provision
#: chunks "boilerplate" purely because the extractor had stapled
#: "THE GAZETTE OF INDIA EXTRAORDINARY" to the top of each one.
_PAGE_FURNITURE = re.compile(
    r"the gazette of india|extraordinary|registered no\.|printed by the manager"
    r"|government of india press|published by the controller",
    re.IGNORECASE,
)

#: Front-matter formulae. A chunk made mostly of these is genuinely
#: unanswerable content, not merely a contaminated provision.
_BOILERPLATE_MARKERS = (
    "be it enacted by parliament",
    "ministry of law and justice",
    "legislative department",
    "received the assent of the president",
    "the following act of parliament",
)

#: Verbs that mark operative statutory language.
_OPERATIVE_TERMS = (
    "shall", "means", "whoever", "may", "is said to", "punished", "liable",
    "provided that", "shall be", "is guilty", "extend to", "deemed",
)

_WORD = re.compile(r"[A-Za-z']+")
_SECTION_REF = re.compile(r"\b\d+[A-Z]?\s*[.(]")


class ChunkClass:
    PROVISION = "PROVISION"
    #: An arrangement-of-sections page, or a column of marginal section
    #: headings extracted as its own chunk. Both are lists of section titles
    #: with no operative text, and both are dangerous for the same reason:
    #: maximum keyword density, zero content.
    TOC_LIKE = "TOC_LIKE"
    HEADING_ONLY = "HEADING_ONLY"
    BOILERPLATE = "BOILERPLATE"
    FRAGMENT = "FRAGMENT"


#: Every class except PROVISION is retrievable text that cannot answer a
#: question about the law. Occupying a context slot with one is a wasted slot.
NON_ANSWERING_CLASSES = (
    ChunkClass.TOC_LIKE,
    ChunkClass.HEADING_ONLY,
    ChunkClass.BOILERPLATE,
    ChunkClass.FRAGMENT,
)


def chunk_features(text: str) -> Dict[str, Any]:
    """Typographic features behind a classification, exposed for auditing."""
    raw = text or ""
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    words = _WORD.findall(raw)
    lowered = raw.lower()

    n_lines = len(lines)
    toc_lines = sum(1 for line in lines if _TOC_ENTRY.match(line))
    heading_lines = sum(1 for line in lines if _HEADING.match(line))
    furniture_lines = sum(1 for line in lines if _PAGE_FURNITURE.search(line))

    return {
        "n_words": len(words),
        "n_lines": n_lines,
        "toc_entry_line_fraction": (toc_lines / n_lines) if n_lines else 0.0,
        "heading_line_fraction": (heading_lines / n_lines) if n_lines else 0.0,
        # A column of marginal section headings extracts as many one- and
        # two-word lines. This is the signal that catches it.
        "mean_line_words": (len(words) / n_lines) if n_lines else 0.0,
        "section_refs_per_100_words": (
            100.0 * len(_SECTION_REF.findall(raw)) / len(words) if words else 0.0
        ),
        "operative_term_hits": sum(1 for term in _OPERATIVE_TERMS if term in lowered),
        "boilerplate_hits": sum(1 for marker in _BOILERPLATE_MARKERS if marker in lowered),
        "page_furniture_lines": furniture_lines,
        "page_furniture_line_fraction": (furniture_lines / n_lines) if n_lines else 0.0,
    }


def classify_chunk(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Label a chunk and return ``(label, features)``.

    Order matters: the cheap unambiguous checks run first, and PROVISION is the
    default. Defaulting *toward* PROVISION is deliberate -- E4's claim is that
    non-answering chunks displace real ones, so a classifier biased toward
    calling things provisions understates the effect rather than inventing it.
    """
    features = chunk_features(text)

    if features["n_words"] < 20:
        return ChunkClass.FRAGMENT, features

    if features["boilerplate_hits"] >= 1 and features["operative_term_hits"] == 0:
        return ChunkClass.BOILERPLATE, features

    # Two shapes of the same problem, a list of section titles with no
    # operative text:
    #   (a) an arrangement-of-sections page -- numbered short titles;
    #   (b) a marginal-heading column -- the side-margin notes of a statute
    #       PDF, extracted as their own chunk, which come out as a long run of
    #       one- and two-word lines carrying no verb.
    is_toc_page = features["toc_entry_line_fraction"] >= 0.5 and features["mean_line_words"] <= 12
    is_heading_column = (
        features["mean_line_words"] <= 4.5
        and features["n_lines"] >= 8
        and features["operative_term_hits"] <= 1
    )
    if is_toc_page or is_heading_column:
        return ChunkClass.TOC_LIKE, features

    if features["heading_line_fraction"] >= 0.6:
        return ChunkClass.HEADING_ONLY, features

    return ChunkClass.PROVISION, features


def synthesize_toc_chunk(section_titles: Sequence[str], start_number: int) -> str:
    """
    Build a table-of-contents chunk from real section titles.

    Used by E4's injection arm. Synthesising from the corpus's own headings
    (rather than inventing text) keeps the injected chunk lexically close to
    genuine provisions -- which is exactly why TOC pages are dangerous for
    retrieval: they are a dense concentration of the same keywords, with none
    of the content.
    """
    lines = [
        f"{start_number + i}. {title.strip().rstrip('.')}."
        for i, title in enumerate(section_titles)
    ]
    return "ARRANGEMENT OF SECTIONS\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Weak-supervision gold labels
# ---------------------------------------------------------------------------

_SECTION_NUM = re.compile(r"(\d+)")


def build_gold_index(registry, eval_rows: Sequence[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    ``eval_id -> [chunk_id, ...]`` for chunks that come from the row's source
    document and carry its section number as a line-initial marker.

    This is weak supervision, not adjudicated relevance. It can miss (a
    provision restated elsewhere) and over-fire (a cross-reference to the same
    number). It is sound for *relative* comparisons -- the same labels are
    applied to every condition being compared, so label noise cannot
    manufacture a difference between conditions -- and unsound as an absolute
    recall figure. Both experiments that use it say so where they report it.
    """
    index: Dict[str, List[str]] = {}
    records = list(registry._records.values())

    for row in eval_rows:
        match = _SECTION_NUM.search(row.get("source_section", "") or "")
        if not match:
            index[row["id"]] = []
            continue
        number = match.group(1)
        doc = (row.get("source_document") or "").lower()
        marker = re.compile(rf"(?m)^\s*{number}\s*[.\(]")
        index[row["id"]] = [
            record.chunk_id
            for record in records
            if doc in (record.source_file or "").lower() and marker.search(record.text or "")
        ]
    return index


def classify_registry(registry) -> Dict[str, Any]:
    """Corpus-level composition: how much of the ingested corpus can answer a
    question at all. This is the upstream number every retrieval metric is
    silently conditioned on."""
    counts: Dict[str, int] = {}
    per_document: Dict[str, Dict[str, int]] = {}
    furniture_contaminated = 0
    furniture_fractions: List[float] = []

    for record in registry._records.values():
        label, features = classify_chunk(record.text)
        counts[label] = counts.get(label, 0) + 1
        doc = record.source_file or "unknown"
        per_document.setdefault(doc, {})
        per_document[doc][label] = per_document[doc].get(label, 0) + 1

        if features["page_furniture_lines"] > 0:
            furniture_contaminated += 1
        furniture_fractions.append(features["page_furniture_line_fraction"])

    total = sum(counts.values())
    return {
        "total_chunks": total,
        "counts": counts,
        "answerable_fraction": counts.get(ChunkClass.PROVISION, 0) / total if total else None,
        "non_answering_fraction": (
            sum(counts.get(c, 0) for c in NON_ANSWERING_CLASSES) / total if total else None
        ),
        # Separate axis from the class breakdown: a provision chunk with a
        # running header stapled to it is still a provision, but the header is
        # indexed text that every query can match on.
        "page_furniture": {
            "chunks_containing_furniture": furniture_contaminated,
            "fraction_of_chunks_contaminated": (
                furniture_contaminated / total if total else None
            ),
            "mean_furniture_line_fraction": (
                sum(furniture_fractions) / len(furniture_fractions)
                if furniture_fractions else None
            ),
        },
        "per_document": per_document,
    }
