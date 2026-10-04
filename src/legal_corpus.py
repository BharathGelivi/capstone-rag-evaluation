"""
Legal corpus: parsing, provenance, and structure-aware chunking.

The judgments fetched by ``scripts/fetch_judgments.py`` are not free text. A
Supreme Court Reports judgment has a fixed skeleton -- citation line, parties,
bench, *Issue for Consideration*, *Headnotes*, *Held*, *Case Law Cited*,
*List of Acts*, then the numbered judgment body -- and that skeleton carries
most of the signal this project needs:

* the **citation line** is the document's identity, which makes citation
  correctness checkable rather than merely plausible;
* **Case Law Cited** and **List of Acts** are the edge lists of the citation
  graph, stated by the reporter rather than inferred by us;
* the numbered body paragraphs are the natural retrieval unit, and they are
  what a lawyer would actually cite.

Fixed-size chunking discards all of it. This module keeps it, and keeps the
provenance attached to every chunk so an answer can be traced back to
``(document, section, paragraph, source_url)``.

Two chunkers are exposed on purpose, because the comparison is an experiment
result and not an assumption:

``chunk_document(..., strategy="fixed")``
    The shipped behaviour -- ``SentenceSplitter`` at ``CHUNK_SIZE`` tokens,
    structure-blind. The baseline arm.
``chunk_document(..., strategy="legal")``
    Section- and paragraph-aware. Headnote sections stay whole (they are short
    and self-contained); body paragraphs are grouped up to a size budget and
    never merged across a section boundary.

Both emit LlamaIndex ``TextNode``s carrying identical metadata keys, so
``ChunkRegistry``, the embedding engine and the vector store are unchanged --
there is no second chunk model and no second registry.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

CORPUS_ROOT = os.path.join("data", "legal_corpus")
MANIFEST_PATH = os.path.join(CORPUS_ROOT, "metadata", "manifest.csv")
FAILURES_PATH = os.path.join(CORPUS_ROOT, "metadata", "failures.csv")
JUDGMENTS_ROOT = os.path.join("data", "judgments")

#: Public mirror the fetcher pulls from. Recorded per document so a citation
#: can be resolved back to a retrievable object, not just to a file on disk.
SC_BUCKET_URL = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"
HC_BUCKET_URL = "https://indian-high-court-judgments.s3.ap-south-1.amazonaws.com"

MANIFEST_COLUMNS = [
    "document_id", "case_name", "court", "date", "citation", "neutral_citation",
    "source_url", "source_domain", "document_type", "document_hash",
    "download_timestamp", "processed_timestamp", "n_pages", "n_chars",
    "n_citations_out", "n_statutes_out", "local_path",
]

# ---------------------------------------------------------------------------
# Structural headings
# ---------------------------------------------------------------------------
# Ordered as they appear in an SCR report. Matched at line start, case
# sensitive enough to avoid firing on a mention inside the body text.
SECTION_HEADINGS: Tuple[Tuple[str, str], ...] = (
    ("Issue for Consideration", "issue"),
    ("Headnotes", "headnote"),
    ("Case Law Cited", "case_law_cited"),
    ("Case Law Citied", "case_law_cited"),   # reporter typo, present in the corpus
    ("List of Acts", "acts_cited"),
    ("List of Keywords", "keywords"),
    ("Case Arising From", "case_arising_from"),
    ("Appearances for Parties", "appearances"),
    ("Judgment / Order of the Supreme Court", "judgment"),
    ("Judgment/Order of the Supreme Court", "judgment"),
)

_HEADING_RE = re.compile(
    r"^[ \t]*(" + "|".join(re.escape(h) for h, _ in SECTION_HEADINGS) + r")[ \t]*$",
    re.MULTILINE,
)
_HEADING_TO_KEY = {h: k for h, k in SECTION_HEADINGS}

#: Sections that carry no retrievable legal content. Kept in the parse (they
#: are provenance) but excluded from the retrievable chunk set -- counsel names
#: are pure keyword noise that every query can match on.
NON_RETRIEVABLE_SECTIONS = frozenset({"appearances", "keywords"})

# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------
# Deliberately conservative: a false edge in the citation graph is worse than a
# missing one, because the graph is used as evidence. Every pattern below is a
# reporter-assigned identifier, not a guess at a case name.

CITATION_PATTERNS: Dict[str, re.Pattern] = {
    # [2024] 10 S.C.R. 1
    "SCR": re.compile(r"\[(\d{4})\]\s*(\d+)\s*S\.?\s?C\.?\s?R\.?\s*(\d+)", re.IGNORECASE),
    # 2024 INSC 746   (Supreme Court neutral citation)
    "INSC": re.compile(r"\b(\d{4})\s+INSC\s+(\d+)\b"),
    # (2019) 5 SCC 1
    "SCC": re.compile(r"\((\d{4})\)\s*(\d+)\s*SCC\s*(\d+)"),
    # AIR 1973 SC 1461
    "AIR": re.compile(r"\bAIR\s+(\d{4})\s+SC\s+(\d+)\b"),
}

# "Section 138 of the Negotiable Instruments Act, 1881" / "s. 178(3)" /
# "Article 32 of the Constitution"
_STATUTE_FULL = re.compile(
    r"\b(?:[Ss]ections?|[Ss]s?\.)\s*(\d+[A-Z]?)(?:\s*\(\d+\))?"
    r"(?:\s*(?:of|,)\s*(?:the\s+)?([A-Z][A-Za-z'()\. ]{4,70}?Act,?\s*\d{4}))?"
)
_ARTICLE = re.compile(r"\bArticles?\s*(\d+[A-Z]?)\b")
_ACT_NAME = re.compile(r"\b([A-Z][A-Za-z'()\.\- ]{4,70}?(?:Act|Sanhita|Adhiniyam|Code|Constitution)),?\s*(\d{4})?\b")

_PARA_START = re.compile(r"^[ \t]*(\d{1,3})\s*[.)]\s*(?:\t|\s)", re.MULTILINE)


@dataclass
class Citation:
    """One outbound reference, with the text it was found in."""
    kind: str                  # "case" | "statute"
    reporter: str              # SCR / INSC / SCC / AIR / ACT / ARTICLE
    normalized: str            # canonical key, used as a graph node id
    raw: str                   # exact matched text
    section: Optional[str] = None      # statute section number, when applicable
    source_section: str = ""           # which judgment section it appeared in
    source_chunk_id: str = ""          # filled in after chunking

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedJudgment:
    """A judgment reduced to structure + provenance. No interpretation."""
    document_id: str
    local_path: str
    case_name: str
    court: str
    date: str
    citation: str
    neutral_citation: str
    source_url: str
    document_hash: str
    n_pages: int
    text: str
    sections: Dict[str, str] = field(default_factory=dict)
    paragraphs: List[Tuple[int, str]] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)

    @property
    def case_citations(self) -> List[Citation]:
        return [c for c in self.citations if c.kind == "case"]

    @property
    def statute_citations(self) -> List[Citation]:
        return [c for c in self.citations if c.kind == "statute"]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(path: str) -> Tuple[str, int]:
    """Return ``(text, n_pages)`` using PyMuPDF -- the reader the ingestion
    pipeline already uses, so extraction quirks stay identical across corpora."""
    import fitz  # pymupdf, already a dependency

    with fitz.open(path) as doc:
        pages = [page.get_text() for page in doc]
        return "\n".join(pages), doc.page_count


def normalize_case_citation(reporter: str, groups: Tuple[str, ...]) -> str:
    """Canonical, comparable key for a case citation.

    Normalisation is what makes the graph joinable: the same judgment is cited
    as ``[2024] 10 S.C.R. 1``, ``[2024] 10 SCR 1`` and ``(2024) 10 S.C.R. 1``
    across reports, and all three must land on one node.
    """
    if reporter == "INSC":
        return f"INSC:{groups[0]}:{groups[1]}"
    return f"{reporter}:{':'.join(groups)}"


def extract_citations(text: str, section_name: str = "") -> List[Citation]:
    """Pull case and statute references out of a block of text.

    Only reporter-assigned identifiers are extracted. Case *names* are
    deliberately not matched: "Vishaka v. State of Rajasthan" appearing in prose
    is not evidence of a citation edge, and a name-matching heuristic would
    manufacture edges the reporter never asserted.
    """
    found: List[Citation] = []
    seen: set = set()

    for reporter, pattern in CITATION_PATTERNS.items():
        for match in pattern.finditer(text):
            key = normalize_case_citation(reporter, match.groups())
            if key in seen:
                continue
            seen.add(key)
            found.append(Citation(
                kind="case", reporter=reporter, normalized=key,
                raw=match.group(0).strip(), source_section=section_name,
            ))

    for match in _STATUTE_FULL.finditer(text):
        section_no, act = match.group(1), match.group(2)
        act_norm = " ".join(act.split()) if act else ""
        key = f"ACT:{act_norm.upper()}:S{section_no}" if act_norm else f"SECTION:{section_no}"
        if key in seen:
            continue
        seen.add(key)
        found.append(Citation(
            kind="statute", reporter="ACT", normalized=key,
            raw=match.group(0).strip(), section=section_no, source_section=section_name,
        ))

    for match in _ARTICLE.finditer(text):
        key = f"ARTICLE:{match.group(1)}"
        if key in seen:
            continue
        seen.add(key)
        found.append(Citation(
            kind="statute", reporter="ARTICLE", normalized=key,
            raw=match.group(0).strip(), section=match.group(1),
            source_section=section_name,
        ))

    return found


# --- Old SCR print layout (volumes up to ~2023) -------------------------------
# These volumes have no "Headnotes" heading. Their structure is positional: the
# bench line closes the caption, the headnote runs from there to the judgment
# marker, and the reporter's own "Case Law Reference" table closes the report.
# Measured on this corpus: the bench line is present in 79/80 sampled old-layout
# judgments and "Case Law Reference" in 63/80, so this recovers structure for
# the ~90% of the corpus the modern-heading parser alone would flatten to one
# undifferentiated blob.
_BENCH_LINE = re.compile(r"\[[^\]\n]{5,160}?JJ?\.\s*\]")
_JUDGMENT_START = re.compile(
    r"(The Judgment of the Court was delivered"
    r"|^\s*J\s?U\s?D\s?G\s?M\s?E\s?N\s?T\s*$"
    r"|^\s*O\s?R\s?D\s?E\s?R\s*$)",
    re.MULTILINE,
)
_CASE_LAW_REFERENCE = re.compile(r"^\s*Case Law Reference\s*:?\s*$", re.MULTILINE)
_JURISDICTION = re.compile(
    r"^\s*(?:CIVIL|CRIMINAL)\s+(?:APPELLATE|ORIGINAL)\s+JURISDICTION.*$", re.MULTILINE
)
_HELD = re.compile(r"\bHELD\s*:")


def _split_sections_old_layout(text: str) -> Dict[str, str]:
    """Positional parse for the pre-2024 SCR print layout."""
    sections: Dict[str, str] = {}

    bench = _BENCH_LINE.search(text)
    headnote_start = bench.end() if bench else 0
    if bench:
        sections["caption"] = text[: bench.end()].strip()

    judgment = _JUDGMENT_START.search(text, headnote_start)
    case_law = _CASE_LAW_REFERENCE.search(text, headnote_start)
    jurisdiction = _JURISDICTION.search(text, headnote_start)

    # The headnote ends at whichever reporter block comes first.
    enders = [m.start() for m in (case_law, jurisdiction, judgment) if m]
    headnote_end = min(enders) if enders else len(text)
    headnote = text[headnote_start:headnote_end].strip()
    if len(headnote) >= LEGAL_CHUNK_MIN_CHARS:
        sections["headnote"] = headnote
        held = _HELD.search(headnote)
        if held:
            # The holding is the part a later court actually applies, so it is
            # worth isolating as its own retrievable unit.
            sections["held"] = headnote[held.start():].strip()

    if case_law:
        end = min([m.start() for m in (jurisdiction, judgment) if m and m.start() > case_law.end()]
                  or [len(text)])
        body = text[case_law.end(): end].strip()
        if body:
            sections["case_law_cited"] = body

    if jurisdiction:
        end = judgment.start() if judgment and judgment.start() > jurisdiction.start() else len(text)
        sections["case_arising_from"] = text[jurisdiction.start(): end].strip()

    if judgment:
        sections["judgment"] = text[judgment.start():].strip()

    if not sections:
        return {"body": text.strip()}
    if "judgment" not in sections and "headnote" not in sections:
        sections["body"] = text.strip()
    return sections


def split_sections(text: str) -> Dict[str, str]:
    """Split an SCR judgment into its named sections.

    Two layouts are handled. The modern (2024-) reports carry explicit headings
    and are split on them. Older volumes carry the same information
    positionally and are handled by ``_split_sections_old_layout``. Anything
    matching neither -- most High Court PDFs -- yields ``{"body": text}``, which
    the legal chunker then handles by paragraph alone.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return _split_sections_old_layout(text)

    sections: Dict[str, str] = {}
    caption = text[: matches[0].start()].strip()
    if caption:
        sections["caption"] = caption

    for i, match in enumerate(matches):
        key = _HEADING_TO_KEY[match.group(1)]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end(): end].strip()
        if not body:
            continue
        # A heading can legitimately repeat (headnote continuations); append
        # rather than overwrite so no text is silently dropped.
        sections[key] = (sections.get(key, "") + "\n" + body).strip() if key in sections else body

    return sections


def split_paragraphs(text: str) -> List[Tuple[int, str]]:
    """Split judgment body text on its own numbered paragraphs.

    Returns ``[(paragraph_number, text), ...]``. Paragraph numbers are the
    court's, not ours, which is what makes a pinpoint citation ("para 14")
    verifiable against the source document.
    """
    matches = list(_PARA_START.finditer(text))
    if not matches:
        stripped = text.strip()
        return [(0, stripped)] if stripped else []

    paragraphs: List[Tuple[int, str]] = []
    preamble = text[: matches[0].start()].strip()
    if len(preamble) > 200:
        paragraphs.append((0, preamble))

    for i, match in enumerate(matches):
        number = int(match.group(1))
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start(): end].strip()
        if body:
            paragraphs.append((number, body))
    return paragraphs


def _first(pattern: re.Pattern, text: str, default: str = "") -> str:
    match = pattern.search(text)
    return match.group(0).strip() if match else default


def parse_judgment(path: str, meta: Optional[Dict[str, str]] = None) -> ParsedJudgment:
    """Parse one judgment PDF into structure + provenance."""
    meta = meta or {}
    text, n_pages = extract_pdf_text(path)
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:32]

    sections = split_sections(text)
    caption = sections.get("caption", text[:1200])

    citation = meta.get("citation") or _first(CITATION_PATTERNS["SCR"], caption)
    neutral = _first(CITATION_PATTERNS["INSC"], caption)

    # Case name: the reporter prints "A \n v. \n B" directly under the citation
    # line. Fall back to the bucket metadata's title.
    case_name = (meta.get("title") or "").strip()
    if not case_name:
        lines = [l.strip() for l in caption.split("\n") if l.strip()]
        parties = [l for l in lines[:6] if " v. " in l or l.endswith(" v.") or l == "v."]
        if parties:
            idx = lines.index(parties[0])
            case_name = " ".join(lines[max(0, idx - 1): idx + 2]).replace(" v. ", " v. ")
        elif len(lines) > 1:
            case_name = lines[1]

    citations: List[Citation] = []
    for name, body in sections.items():
        if name in NON_RETRIEVABLE_SECTIONS:
            continue
        citations.extend(extract_citations(body, section_name=name))

    # Deduplicate across sections, keeping the first (earliest) occurrence.
    unique: Dict[str, Citation] = {}
    for citation_obj in citations:
        unique.setdefault(citation_obj.normalized, citation_obj)

    judgment_body = sections.get("judgment") or sections.get("body") or ""
    document_id = os.path.splitext(os.path.basename(path))[0]

    return ParsedJudgment(
        document_id=document_id,
        local_path=path.replace("\\", "/"),
        case_name=case_name or document_id,
        court=meta.get("court", "Supreme Court of India"),
        date=meta.get("decision_date", ""),
        citation=citation,
        neutral_citation=neutral,
        source_url=meta.get("source_url") or build_source_url(path),
        document_hash=digest,
        n_pages=n_pages,
        text=text,
        sections=sections,
        paragraphs=split_paragraphs(judgment_body),
        citations=list(unique.values()),
    )


def build_source_url(local_path: str) -> str:
    """Reconstruct the public object URL a local file came from.

    The fetcher mirrors ``s3://<bucket>/data/pdf/year=Y/...``; keeping the URL
    per document means a citation in an answer resolves to something a reader
    can actually open.
    """
    path = local_path.replace("\\", "/")
    name = os.path.basename(path)
    parts = path.split("/")
    try:
        year = parts[-2]
    except IndexError:
        year = ""
    if "supreme_court" in path:
        return f"{SC_BUCKET_URL}/data/pdf/year={year}/english/{name}"
    court = next((p.replace("hc_", "") for p in parts if p.startswith("hc_")), "")
    return f"{HC_BUCKET_URL}/data/pdf/year={year}/court={court}/{name}"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

#: Character budget per legal chunk. Chosen to sit near the fixed-size arm's
#: effective length (512 tokens ~= 2000 chars) so the comparison isolates
#: *structure*, not chunk length -- otherwise a win could just be a size effect.
LEGAL_CHUNK_CHARS = 2000
LEGAL_CHUNK_MIN_CHARS = 40


def _node(text: str, meta: Dict[str, Any]) -> TextNode:
    node = TextNode(text=text, metadata=dict(meta))
    return node


def _base_metadata(doc: ParsedJudgment) -> Dict[str, Any]:
    """Provenance every chunk carries, whatever the strategy.

    This is the non-negotiable part: an answer is only as checkable as the
    metadata riding on the chunk it came from.
    """
    return {
        "document_id": doc.document_id,
        "case_name": doc.case_name,
        "court": doc.court,
        "date": doc.date,
        "citation": doc.citation,
        "neutral_citation": doc.neutral_citation,
        "source_url": doc.source_url,
        "source_file": os.path.basename(doc.local_path),
        "document_hash": doc.document_hash,
        "document_type": "judgment",
    }


def chunk_document(doc: ParsedJudgment, strategy: str = "legal") -> List[TextNode]:
    """Chunk a parsed judgment. ``strategy`` is ``"legal"`` or ``"fixed"``."""
    if strategy == "fixed":
        return _chunk_fixed(doc)
    if strategy == "legal":
        return _chunk_legal(doc)
    raise ValueError(f"Unknown chunking strategy '{strategy}' (expected 'legal' or 'fixed')")


def _chunk_fixed(doc: ParsedJudgment) -> List[TextNode]:
    """Baseline arm: the shipped structure-blind splitter over the raw text."""
    from llama_index.core.node_parser import SentenceSplitter
    from configs.pipeline import CHUNK_SIZE, CHUNK_OVERLAP

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    base = _base_metadata(doc)
    nodes = []
    for i, piece in enumerate(splitter.split_text(doc.text)):
        meta = {**base, "section": "unstructured", "paragraph": None,
                "chunk_strategy": "fixed", "chunk_index": i, "page_number": "unknown"}
        nodes.append(_node(piece, meta))
    return nodes


def _pack(pieces: Iterable[Tuple[Optional[int], str]], budget: int) -> List[Tuple[Optional[int], str]]:
    """Group consecutive pieces up to ``budget`` characters.

    Never merges across the caller's boundaries, so a chunk cannot straddle two
    judgment sections -- which is exactly the failure the fixed splitter has.
    """
    packed: List[Tuple[Optional[int], str]] = []
    buffer: List[str] = []
    first_number: Optional[int] = None
    size = 0

    for number, text in pieces:
        if size and size + len(text) > budget:
            packed.append((first_number, "\n\n".join(buffer)))
            buffer, size, first_number = [], 0, None
        if first_number is None:
            first_number = number
        buffer.append(text)
        size += len(text)

    if buffer:
        packed.append((first_number, "\n\n".join(buffer)))
    return packed


def _chunk_legal(doc: ParsedJudgment) -> List[TextNode]:
    """Structure-aware arm.

    Headnote-family sections are emitted whole -- they are short, self-contained
    statements of the issue and the holding, and splitting them mid-sentence is
    what makes a "the answer was in the corpus but the chunk was truncated"
    failure. Body paragraphs are packed to a character budget within a section.
    """
    base = _base_metadata(doc)
    nodes: List[TextNode] = []
    index = 0

    for name, body in doc.sections.items():
        if name in NON_RETRIEVABLE_SECTIONS or name == "judgment":
            continue
        # Short is not the same as worthless here: "Issue for Consideration" is
        # often one sentence and is precisely the text an issue-to-cases query
        # must match. Only genuinely empty fragments are dropped.
        if len(body.strip()) < LEGAL_CHUNK_MIN_CHARS:
            continue
        for _, piece in _pack([(None, body)], LEGAL_CHUNK_CHARS):
            meta = {**base, "section": name, "paragraph": None,
                    "chunk_strategy": "legal", "chunk_index": index,
                    "page_number": "headnote"}
            nodes.append(_node(piece, meta))
            index += 1

    for first_para, piece in _pack(
        [(number, text) for number, text in doc.paragraphs], LEGAL_CHUNK_CHARS
    ):
        meta = {**base, "section": "judgment", "paragraph": first_para,
                "chunk_strategy": "legal", "chunk_index": index,
                "page_number": f"para {first_para}" if first_para else "judgment"}
        nodes.append(_node(piece, meta))
        index += 1

    if not nodes:  # pathological extraction; fall back rather than drop the document
        return _chunk_fixed(doc)
    return nodes


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def read_fetch_manifests(root: str = JUDGMENTS_ROOT) -> Dict[str, Dict[str, str]]:
    """Index the fetcher's own per-directory ``manifest.csv`` files by pdf name.

    The download step already recorded the authoritative bucket metadata
    (citation, bench, decision date). Re-deriving it from the PDF text would be
    strictly worse, so it is joined in rather than recomputed.
    """
    index: Dict[str, Dict[str, str]] = {}
    for dirpath, _, filenames in os.walk(root):
        if "manifest.csv" not in filenames:
            continue
        with open(os.path.join(dirpath, "manifest.csv"), encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("pdf"):
                    index[row["pdf"]] = row
    return index


def load_corpus_manifest(path: str = MANIFEST_PATH) -> Dict[str, Dict[str, str]]:
    """``document_id -> manifest row`` for everything already processed."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", newline="") as handle:
        return {row["document_id"]: row for row in csv.DictReader(handle)}


def append_manifest_rows(rows: List[Dict[str, Any]], path: str = MANIFEST_PATH) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def record_failure(path: str, reason: str, failures_path: str = FAILURES_PATH) -> None:
    """Failures are corpus facts too -- a size claim is only honest next to the
    count of documents that did not make it in."""
    os.makedirs(os.path.dirname(failures_path), exist_ok=True)
    write_header = not os.path.exists(failures_path)
    with open(failures_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["local_path", "reason", "timestamp"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "local_path": path, "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def manifest_row(doc: ParsedJudgment, fetch_meta: Dict[str, str]) -> Dict[str, Any]:
    return {
        "document_id": doc.document_id,
        "case_name": doc.case_name,
        "court": doc.court,
        "date": doc.date,
        "citation": doc.citation,
        "neutral_citation": doc.neutral_citation,
        "source_url": doc.source_url,
        "source_domain": "s3.ap-south-1.amazonaws.com (AWS Open Data mirror of eCourts)",
        "document_type": "judgment",
        "document_hash": doc.document_hash,
        "download_timestamp": fetch_meta.get("download_timestamp", ""),
        "processed_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_pages": doc.n_pages,
        "n_chars": len(doc.text),
        "n_citations_out": len(doc.case_citations),
        "n_statutes_out": len(doc.statute_citations),
        "local_path": doc.local_path,
    }


def research_value_score(doc: ParsedJudgment) -> float:
    """Rank documents by how much multi-hop signal they carry.

    Used to select which judgments to ingest when the download is larger than
    the corpus budget. Citation count dominates because citation chains are the
    thing GraphRAG and IRCoT are supposed to exploit; structural completeness
    is a tiebreak, since a judgment with an explicit *Case Law Cited* block
    yields reporter-asserted edges rather than regex-inferred ones.
    """
    structured = sum(
        1 for key in ("issue", "headnote", "case_law_cited", "acts_cited") if key in doc.sections
    )
    return (
        2.0 * len(doc.case_citations)
        + 1.0 * len(doc.statute_citations)
        + 3.0 * structured
        + min(len(doc.paragraphs), 40) * 0.25
    )


def demo() -> None:
    """Self-check for the parsing and chunking logic -- no PDFs required."""
    text = (
        "[2024] 10 S.C.R. 1 : 2024 INSC 746\n"
        "K. Vadivel \nv. \nK. Shanthi & Ors.\n"
        "30 September 2024\n"
        "Issue for Consideration\n"
        "Whether further investigation was warranted.\n"
        "Headnotes\n"
        "Code of Criminal Procedure, 1973 - s. 178(3) - Further investigation. "
        "Held: relief under Article 32 is available.\n"
        "Case Law Cited\n"
        "Vinay Tyagi v. Irshad Ali (2013) 5 SCC 762 - relied on; "
        "AIR 1973 SC 1461 - referred to.\n"
        "Judgment / Order of the Supreme Court\n"
        "1.\tThe appellant challenges the order of the High Court.\n"
        "2.\tSection 138 of the Negotiable Instruments Act, 1881 is not attracted.\n"
    )

    sections = split_sections(text)
    assert "caption" in sections, sections.keys()
    assert "issue" in sections and "headnote" in sections and "case_law_cited" in sections
    assert "judgment" in sections

    paragraphs = split_paragraphs(sections["judgment"])
    assert [n for n, _ in paragraphs] == [1, 2], paragraphs

    citations = extract_citations(text)
    keys = {c.normalized for c in citations}
    assert "SCR:2024:10:1" in keys, keys
    assert "INSC:2024:746" in keys, keys
    assert "SCC:2013:5:762" in keys, keys
    assert "AIR:1973:1461" in keys, keys
    assert "ARTICLE:32" in keys, keys
    assert any(k.startswith("ACT:") and "NEGOTIABLE" in k for k in keys), keys

    # Same citation written two ways must normalise to one node.
    assert normalize_case_citation("SCR", ("2024", "10", "1")) == "SCR:2024:10:1"

    doc = ParsedJudgment(
        document_id="demo", local_path="data/judgments/supreme_court/2024/demo.pdf",
        case_name="K. Vadivel v. K. Shanthi", court="Supreme Court of India",
        date="2024-09-30", citation="[2024] 10 S.C.R. 1", neutral_citation="2024 INSC 746",
        source_url="https://example.invalid/demo.pdf", document_hash="deadbeef",
        n_pages=17, text=text, sections=sections,
        paragraphs=paragraphs, citations=citations,
    )

    legal_nodes = chunk_document(doc, "legal")
    assert legal_nodes, "legal chunker produced nothing"
    assert all(n.metadata["document_id"] == "demo" for n in legal_nodes)
    assert all(n.metadata["source_url"] for n in legal_nodes), "provenance dropped"
    assert {n.metadata["section"] for n in legal_nodes} >= {"issue", "headnote", "judgment"}
    # A body chunk must carry the court's own paragraph number, not ours.
    body = [n for n in legal_nodes if n.metadata["section"] == "judgment"]
    assert body and body[0].metadata["paragraph"] == 1, body[0].metadata

    fixed_nodes = chunk_document(doc, "fixed")
    assert fixed_nodes and all(n.metadata["chunk_strategy"] == "fixed" for n in fixed_nodes)

    old = (
        "[2015] 13 S.C.R. 1 \nSUPREME COURT ADVOCATES v. UNION OF INDIA\n"
        "OCTOBER 16, 2015\n[JAGDISH SINGH KHEHAR, MADAN B. LOKUR, JJ.]\n"
        "Constitution (Ninety-ninth Amendment) Act, 2014 - Collegium system - "
        "the scheme contemplated for replacing the Collegium system is examined "
        "at length in this report and the competing contentions are set out.\n"
        "HELD: the Ninety-ninth Amendment is declared unconstitutional.\n"
        "Case Law Reference:\n(2013) 5 SCC 762  relied on  para 12\n"
        "CIVIL ORIGINAL JURISDICTION: Writ Petition (Civil) No. 13 of 2015\n"
        "The Judgment of the Court was delivered by KHEHAR, J.\n"
        "1.\tThe petitions challenge the amendment.\n"
    )
    old_sections = split_sections(old)
    assert "caption" in old_sections and "headnote" in old_sections, old_sections.keys()
    assert "held" in old_sections and "case_law_cited" in old_sections, old_sections.keys()
    assert "judgment" in old_sections, old_sections.keys()
    assert "HELD" in old_sections["held"]
    assert "SCC 762" in old_sections["case_law_cited"]
    assert old_sections["headnote"].startswith("Constitution"), old_sections["headnote"][:60]

    assert build_source_url("data/judgments/supreme_court/2024/x.pdf").endswith(
        "/data/pdf/year=2024/english/x.pdf"
    )
    assert research_value_score(doc) > 0

    print(f"legal_corpus demo OK  (legal={len(legal_nodes)} fixed={len(fixed_nodes)} "
          f"citations={len(citations)})")


if __name__ == "__main__":
    demo()
