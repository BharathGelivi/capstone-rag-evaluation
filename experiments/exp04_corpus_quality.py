"""
E4 -- Corpus quality as an upstream determinant of retrieval failure.

The argument
-----------
Retrieval metrics are reported as though they measure a retriever. They do not.
They measure a retriever *over a particular corpus*, and a corpus carries text
that is retrievable but cannot answer anything: tables of contents, chapter
headings, enacting formulae, extraction fragments. A table-of-contents page is
the worst of these, because it is a dense concentration of exactly the
vocabulary a question uses -- every section title, no content. Lexical
retrieval loves it. It wins a context slot and contributes nothing.

The consequence for diagnosis is the interesting part. When a TOC chunk
displaces the provision that held the answer, every downstream signal reports a
*retrieval* failure: low entailment, unsupported claims, a RETRIEVAL_MISS
verdict. The retriever did its job on the corpus it was given. The fault is
upstream of every stage the diagnosis models, which means a stage-attribution
framework that stops at CORPUS-as-a-distance-threshold (as this one does) will
systematically mislabel an ingestion defect as a retrieval defect.

Three arms
----------
**Composition.** Classify the whole ingested corpus. What fraction of it can
answer a question at all? This is the number every retrieval metric is silently
conditioned on and that no RAG paper reports.

**Observation.** For each query, how many of the final context slots go to
non-answering chunks, and how deep is the first real provision? Plus the
counterfactual: recompute gold recall with non-answering chunks removed from
the candidate pool. The gap is the cost the corpus is imposing.

**Injection (causal).** Observation alone is confounded -- a clean corpus shows
nothing, and a dirty one cannot prove the direction of the effect. So TOC
chunks are *synthesised from the corpus's own section headings* and injected
into the candidate pool at a controlled contamination level, and the
displacement is measured as the level rises. Same logic as E1: manufacture the
cause, then check the effect.

Note on this repository's corpus
--------------------------------
The currently ingested BNS/BNSS/BSA extraction contains essentially no
arrangement-of-sections pages -- the composition arm reports what is actually
there, whatever that is, and the experiment does not assume contamination it
cannot find. This is why the injection arm exists: it establishes the mechanism
on a clean corpus rather than relying on a dirty one being available. A reader
with a dirty corpus gets the observation arm for free.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from configs.pipeline import RERANKER_TOP_N
from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    load_eval_dataset,
    mean,
)
from experiments.corpus import (
    NON_ANSWERING_CLASSES,
    ChunkClass,
    build_gold_index,
    classify_chunk,
    classify_registry,
    synthesize_toc_chunk,
)

logger = logging.getLogger(__name__)

#: Fraction of the candidate pool replaced by synthesised TOC chunks.
CONTAMINATION_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)

#: Candidate pool depth for the live arms.
DEFAULT_POOL_SIZE = 30

#: Section titles per synthesised TOC chunk.
TOC_ENTRIES_PER_CHUNK = 14

_TITLE_LINE = re.compile(r"^\s*\d+[A-Z]?\.\s*([A-Z][^.\n]{5,70})")


class CorpusQualityExperiment(Experiment):
    key = "exp04_corpus_quality"
    number = 4
    title = "Corpus quality as an upstream determinant"
    claim = (
        "Non-answering chunks (tables of contents above all) win context slots from real "
        "provisions, and the resulting failure is misattributed to retrieval."
    )

    def __init__(self) -> None:
        self._registry = None
        self._retriever = None
        self._gold_index: Dict[str, List[str]] = {}
        self._composition: Dict[str, Any] = {}
        self._section_titles: List[str] = []

    # -- planning --------------------------------------------------------

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        if ctx.is_live:
            rows = load_eval_dataset()
            if not rows:
                raise RuntimeError("live mode needs eval/eval_dataset.csv.")
            specs = []
            for row in rows:
                for level in CONTAMINATION_GRID:
                    specs.append(ExampleSpec(
                        example_id=f"live/{row['id']}/contam{level:.2f}",
                        payload={
                            "eval_id": row["id"],
                            "question": row["question"],
                            "contamination": level,
                        },
                    ))
            return specs

        n_queries = int(ctx.extra.get("n_simulated_queries", 60))
        return [
            ExampleSpec(
                example_id=f"sim/q{i:03d}",
                payload={"query_index": i},
            )
            for i in range(n_queries)
        ]

    # -- resources -------------------------------------------------------

    def setup(self, ctx: ExperimentContext) -> None:
        import os

        from src.chunk_registry import ChunkRegistry

        registry_path = "artifacts/chunk_registry.json"
        if os.path.exists(registry_path):
            self._registry = ChunkRegistry.load_from_json(registry_path)
            self._composition = classify_registry(self._registry)
            self._section_titles = self._harvest_section_titles()

        if not ctx.is_live:
            return

        from src.retriever import get_retriever
        from src.vector_store import ChromaVectorStore

        if self._registry is None:
            raise RuntimeError(
                "live mode needs artifacts/chunk_registry.json. Run `python run_pipeline.py`."
            )
        vector_store = ChromaVectorStore()
        vector_store.initialize_collection()
        self._retriever = get_retriever(vector_store, self._registry)
        self._gold_index = build_gold_index(self._registry, load_eval_dataset())

    def _harvest_section_titles(self) -> List[str]:
        """
        Pull the opening phrase of each numbered provision out of the corpus.
        These become the entries of the synthesised TOC chunks, which is what
        makes the injection realistic: a real TOC is built from exactly these
        strings, so it inherits their vocabulary without their content.
        """
        titles: List[str] = []
        for record in self._registry._records.values():
            for line in (record.text or "").split("\n"):
                match = _TITLE_LINE.match(line)
                if match:
                    title = " ".join(match.group(1).split()[:8])
                    if len(title) > 12:
                        titles.append(title)
        # Deduplicated and ordered so the injected chunks are reproducible.
        return sorted(set(titles))

    # -- execution -------------------------------------------------------

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        if ctx.is_live:
            return self._run_live(spec, ctx)
        return self._run_simulated(spec, ctx)

    def _run_live(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        from src.retriever import RetrievedChunk

        payload = spec.payload
        question = payload["question"]
        level = payload["contamination"]
        pool_size = int(ctx.extra.get("pool_size", DEFAULT_POOL_SIZE))

        candidates, _ = self._retriever.rank_candidates(question, pool_size)
        if not candidates:
            return {"skipped": True, "reason": "empty candidate pool"}

        gold_ids = set(self._gold_index.get(payload["eval_id"], []))

        # Injected TOC chunks are scored by the *real* cross-encoder alongside
        # the real candidates -- they compete on the pipeline's own terms, not
        # on an assumed score.
        injected = self._build_injected_chunks(spec, level, len(candidates), RetrievedChunk)
        pool = list(candidates) + injected

        reranked = self._retriever.rerank(question, pool)
        final = reranked[:RERANKER_TOP_N]

        classes = {}
        for chunk in pool:
            if chunk.chunk_id.startswith("INJECTED_TOC_"):
                classes[chunk.chunk_id] = ChunkClass.TOC_LIKE
            else:
                classes[chunk.chunk_id] = classify_chunk(chunk.chunk_text)[0]

        clean_final = [c for c in reranked if classes[c.chunk_id] == ChunkClass.PROVISION][
            :RERANKER_TOP_N
        ]

        return {
            "skipped": False,
            "contamination": level,
            "n_injected": len(injected),
            "pool_size": len(pool),
            **self._slot_metrics(final, clean_final, classes, gold_ids),
            "gold_labelled": bool(gold_ids),
        }

    def _build_injected_chunks(self, spec: ExampleSpec, level: float, n_real: int, chunk_cls):
        if level <= 0 or not self._section_titles:
            return []
        n_inject = max(1, int(round(n_real * level)))
        rng = self.__dict__.setdefault("_rng_cache", {}).setdefault(
            spec.example_id, None
        )
        # Deterministic slice of titles per injected chunk: no randomness needed,
        # and a fixed slice keeps the injected text identical across resumes.
        chunks = []
        for i in range(n_inject):
            start = (i * TOC_ENTRIES_PER_CHUNK) % max(1, len(self._section_titles) - TOC_ENTRIES_PER_CHUNK)
            titles = self._section_titles[start : start + TOC_ENTRIES_PER_CHUNK]
            chunks.append(chunk_cls(
                chunk_id=f"INJECTED_TOC_{i}",
                similarity_score=0.0,
                rank=n_real + i + 1,
                page_number="0",
                source_file="synthetic_toc",
                chunk_index=-1,
                chunk_text=synthesize_toc_chunk(titles, start_number=1 + start),
                parent_document_id="synthetic_toc",
            ))
        return chunks

    def _run_simulated(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        """
        Simulated arm: a candidate pool of provisions plus TOC chunks, where a
        TOC chunk's retrieval score is drawn from a *higher* distribution than
        an average provision's.

        That premise is the whole model, and it is not arbitrary: a TOC page
        concentrates every section title in the document, so its lexical
        overlap with any section-shaped query is unusually high while its
        semantic content is null. The simulation asks what that scoring
        advantage costs in context slots -- it does not assume the cost.
        """
        results = []
        for level in CONTAMINATION_GRID:
            rng = ctx.rng_for(f"e4:{spec.example_id}:{level}")
            pool_size = DEFAULT_POOL_SIZE
            n_toc = int(round(pool_size * level))
            n_prov = pool_size - n_toc

            provisions = [(f"prov{i}", ChunkClass.PROVISION, rng.gauss(0.45, 0.18)) for i in range(n_prov)]
            # The gold provision is the strongest genuine match.
            if provisions:
                provisions[0] = ("gold", ChunkClass.PROVISION, rng.gauss(0.78, 0.10))
            tocs = [(f"toc{i}", ChunkClass.TOC_LIKE, rng.gauss(0.62, 0.14)) for i in range(n_toc)]

            pool = provisions + tocs
            ranked = sorted(pool, key=lambda x: -x[2])
            final = ranked[:RERANKER_TOP_N]
            clean = [c for c in ranked if c[1] == ChunkClass.PROVISION][:RERANKER_TOP_N]

            classes = {cid: label for cid, label, _ in pool}
            results.append({
                "contamination": level,
                "n_injected": n_toc,
                "pool_size": pool_size,
                **self._slot_metrics_simple(
                    [c[0] for c in final], [c[0] for c in clean], classes, {"gold"}
                ),
            })

        return {"skipped": False, "levels": results}

    # -- metrics ---------------------------------------------------------

    @staticmethod
    def _slot_metrics_simple(
        final_ids: Sequence[str],
        clean_ids: Sequence[str],
        classes: Dict[str, str],
        gold_ids: set,
    ) -> Dict[str, Any]:
        non_answering = [cid for cid in final_ids if classes[cid] in NON_ANSWERING_CLASSES]
        first_provision_rank = next(
            (i + 1 for i, cid in enumerate(final_ids) if classes[cid] == ChunkClass.PROVISION),
            None,
        )
        return {
            "slots_total": len(final_ids),
            "slots_to_non_answering": len(non_answering),
            "slot_waste_rate": len(non_answering) / len(final_ids) if final_ids else None,
            "first_provision_rank": first_provision_rank,
            "gold_in_final": bool(gold_ids & set(final_ids)),
            # Counterfactual: the same ranking with non-answering chunks removed
            # from contention. The difference is the corpus's cost.
            "gold_in_clean_final": bool(gold_ids & set(clean_ids)),
            "gold_displaced_by_corpus": bool(gold_ids & set(clean_ids))
            and not bool(gold_ids & set(final_ids)),
            "final_class_mix": _mix([classes[cid] for cid in final_ids]),
        }

    def _slot_metrics(self, final, clean_final, classes, gold_ids) -> Dict[str, Any]:
        return self._slot_metrics_simple(
            [c.chunk_id for c in final],
            [c.chunk_id for c in clean_final],
            classes,
            set(gold_ids),
        )

    # -- aggregation -----------------------------------------------------

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        usable = [r for r in records if not r.get("skipped")]
        if not usable:
            return {"error": "no usable records"}

        # Flatten simulated (nested per level) and live (one level per record)
        # into a single per-(record, level) view.
        flat: List[Dict[str, Any]] = []
        for record in usable:
            if "levels" in record:
                flat.extend(record["levels"])
            else:
                flat.append(record)

        by_level = []
        for level in sorted({row["contamination"] for row in flat}):
            rows = [row for row in flat if row["contamination"] == level]
            displaced = sum(1 for row in rows if row["gold_displaced_by_corpus"])
            by_level.append({
                "contamination": level,
                "n": len(rows),
                "mean_slot_waste_rate": mean([row["slot_waste_rate"] for row in rows]),
                "mean_slots_to_non_answering": mean([row["slots_to_non_answering"] for row in rows]),
                "gold_recall": (
                    sum(1 for row in rows if row["gold_in_final"]) / len(rows) if rows else None
                ),
                "gold_recall_without_non_answering_chunks": (
                    sum(1 for row in rows if row["gold_in_clean_final"]) / len(rows) if rows else None
                ),
                "gold_displacement_rate": displaced / len(rows) if rows else None,
                "mean_first_provision_rank": mean(
                    [row["first_provision_rank"] for row in rows]
                ),
            })

        baseline = next((row for row in by_level if row["contamination"] == 0.0), None)
        worst = max(by_level, key=lambda row: row["contamination"]) if by_level else None

        return {
            "headline": {
                "corpus_composition": self._composition or "registry unavailable",
                "slot_waste_at_zero_injection": baseline["mean_slot_waste_rate"] if baseline else None,
                "gold_recall_clean_vs_observed_at_zero_injection": (
                    {
                        "observed": baseline["gold_recall"],
                        "counterfactual_without_non_answering": baseline[
                            "gold_recall_without_non_answering_chunks"
                        ],
                    }
                    if baseline else None
                ),
                "recall_lost_to_injected_toc": (
                    (baseline["gold_recall"] - worst["gold_recall"])
                    if baseline and worst and baseline is not worst else None
                ),
                "slot_waste_lost_to_injected_toc": (
                    (worst["mean_slot_waste_rate"] - baseline["mean_slot_waste_rate"])
                    if baseline and worst and baseline is not worst else None
                ),
            },
            "by_contamination_level": by_level,
            "n_records": len(usable),
            "interpretation_notes": [
                "The observation arm reports this corpus as it is. If its non-answering "
                "fraction is near zero the observation arm will show no effect -- that is a "
                "finding about the corpus, not a null result about the mechanism, which the "
                "injection arm establishes separately.",
                "gold_displacement_rate counts queries where the gold provision would have "
                "reached the context window but for a non-answering chunk taking its slot. "
                "Every one of those looks like a retrieval failure to every downstream "
                "metric and to this framework's own RETRIEVAL_MISS verdict.",
                "Gold labels are weak supervision (see experiments/corpus.py). They are "
                "applied identically across contamination levels, so they can bias the "
                "absolute recall figures but not the trend across levels.",
            ],
        }


def _mix(labels: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


EXPERIMENT = CorpusQualityExperiment()
