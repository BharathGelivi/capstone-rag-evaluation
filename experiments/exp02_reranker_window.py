"""
E2 -- The reranker-window pathology.

This repository shipped the bug the experiment is named after: when the
cross-encoder's input window (``RERANK_INPUT_SIZE``) is close to the number of
chunks finally handed to the generator (``RERANKER_TOP_N``), the reranker
cannot change which chunks reach the generator. It can only permute a set that
Reciprocal Rank Fusion has already fixed. The pipeline still pays the full
cross-encoder latency, the logs still say "Reranking...", and every offline
metric that looks at the final context set is unchanged. It is a silent no-op.

At ratio exactly 1.0 this is a theorem, not a measurement: the reranker is
handed N candidates and must return N, so the output set equals the input set
for every query, every corpus and every reranker. The experiment's job is to
measure how fast the pathology decays as the ratio grows, and to turn that
curve into a design rule someone can check against their own pipeline.

Method
------
One retrieval pass per query captures a wide candidate pool (RRF order) and the
cross-encoder score for every candidate in it. Every (window W, top_n N)
setting is then derived from that single pass -- the reranker is never re-run
per setting, which is what makes a 2-D sweep affordable. This is the same
"score once, derive many" tactic ``scripts/ablate_aggregation_strategy.py``
uses for aggregation strategies.

Per (query, W, N) we record:

* ``is_noop``      -- does the final set equal RRF's own top-N? (Set identity,
                      not order: reordering chunks inside the context window is
                      not what reranking is for.)
* ``n_rescued``    -- chunks in the final set that RRF ranked below N. These
                      are exactly the chunks reranking exists to recover, and
                      at W == N there are none by construction.
* ``gold_recall``  -- did a gold chunk reach the final set? The outcome that
                      matters; everything else is mechanism.
* ``displacement`` -- mean |RRF rank - final rank| over the final set.
* ``cost_units``   -- W, since cross-encoder cost is linear in candidates.

Gold labels
-----------
Live mode uses weak supervision: a chunk is gold if it comes from the eval
row's ``source_document`` and its text carries that row's ``source_section``
number as a section marker. That is a heuristic, not adjudicated relevance, and
it can both miss (a provision restated elsewhere) and over-fire (a
cross-reference). It is honest for a *relative* comparison across window
settings -- the same labels are used for every setting, so label noise cannot
manufacture the trend -- and should not be read as absolute recall.

Modes
-----
``offline`` (default) simulates RRF and cross-encoder orderings as two noisy
permutations of a latent relevance order, with the reranker the less noisy of
the two. It reproduces the curve's shape and the exact ratio-1.0 invariant
without needing an ingested corpus, so the experiment is runnable and testable
anywhere. ``live`` runs the real retriever over the real corpus.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from configs.pipeline import RERANKER_TOP_N, RERANK_INPUT_SIZE
from experiments.corpus import build_gold_index
from experiments.common import (
    ExampleSpec,
    Experiment,
    ExperimentContext,
    load_eval_dataset,
    mean,
    wilson_interval,
)

logger = logging.getLogger(__name__)

#: Widest candidate pool scored per query. Every narrower window is a prefix of
#: it, so this also bounds the cost of the whole sweep.
DEFAULT_WINDOW_MAX = 40

#: Final context sizes swept. Two values, so the summary can show that the
#: governing variable is the *ratio* W/N and not the absolute window size.
TOP_N_GRID: Tuple[int, ...] = (3, 6)

#: Window/top_n ratios swept. Starts at exactly 1.0 -- the degenerate setting.
RATIO_GRID: Tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.5, 5.0, 6.5)

#: Marginal gold-recall gain per extra scored candidate, below which the design
#: rule declares the extra window not worth its latency.
DEFAULT_KNEE_EPSILON = 0.005


# ---------------------------------------------------------------------------
# Window arithmetic -- pure, so it can be unit-tested without a corpus
# ---------------------------------------------------------------------------


def evaluate_window(
    rrf_order: Sequence[str],
    reranker_scores: Dict[str, float],
    window: int,
    top_n: int,
    gold_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Derive one (window, top_n) setting from an already-scored candidate pool.

    ``rrf_order`` is the pre-rerank ranking; ``reranker_scores`` maps chunk id
    to cross-encoder score. No model is invoked here -- this is the arithmetic
    that turns one scoring pass into a whole sweep.
    """
    if window < top_n:
        raise ValueError(f"window ({window}) must be >= top_n ({top_n}).")

    candidates = list(rrf_order[:window])
    rrf_rank = {cid: i + 1 for i, cid in enumerate(rrf_order)}

    # Ties broken by RRF rank so the ordering is total and reproducible.
    ranked = sorted(candidates, key=lambda c: (-reranker_scores[c], rrf_rank[c]))
    final = ranked[:top_n]

    rrf_baseline = list(rrf_order[:top_n])
    gold = set(gold_ids)

    rescued = [c for c in final if rrf_rank[c] > top_n]
    displacement = mean(
        [abs(rrf_rank[c] - (i + 1)) for i, c in enumerate(final)]
    )

    return {
        "window": window,
        "top_n": top_n,
        "ratio": round(window / top_n, 4),
        # Set identity, not order: whether the generator sees a different set
        # of chunks is the question. Reordering within the window is not.
        "is_noop": set(final) == set(rrf_baseline),
        "n_rescued": len(rescued),
        "rescued_chunk_ids": rescued,
        "gold_in_final": bool(gold & set(final)),
        "gold_in_rrf_baseline": bool(gold & set(rrf_baseline)),
        "gold_rescued_by_reranker": bool(gold & set(final)) and not bool(gold & set(rrf_baseline)),
        "gold_lost_by_reranker": bool(gold & set(rrf_baseline)) and not bool(gold & set(final)),
        "mean_rank_displacement": displacement,
        "cost_units": window,
    }


def _kendall_tau(order_a: Sequence[str], order_b: Sequence[str]) -> Optional[float]:
    """
    Kendall's tau-a between two orderings of the same items: how much the
    reranker disagrees with RRF over the window. High disagreement with a
    ratio near 1.0 is the worst case -- the reranker has strong opinions and no
    room to act on them.
    """
    common = [c for c in order_a if c in set(order_b)]
    n = len(common)
    if n < 2:
        return None
    pos_b = {c: i for i, c in enumerate(order_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b[common[i]] < pos_b[common[j]]:
                concordant += 1
            else:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total else None


# ---------------------------------------------------------------------------
# Candidate pool sources
# ---------------------------------------------------------------------------


def _simulate_pool(rng, window_max: int) -> Tuple[List[str], Dict[str, float], List[str]]:
    """
    Simulated candidate pool.

    A latent relevance score per chunk defines truth. RRF observes it through
    heavy noise, the cross-encoder through light noise -- which is the premise
    that makes reranking worth doing at all. The gold chunk is the most
    relevant one, and its RRF rank is drawn from a heavy-tailed distribution so
    that it sometimes sits deep in the pool. Those deep cases are precisely the
    ones a wide window is supposed to rescue and a ratio-1.0 window cannot.
    """
    n = window_max
    ids = [f"sim_c{i:03d}" for i in range(n)]
    latent = {cid: rng.uniform(0.0, 1.0) for cid in ids}

    gold_id = max(ids, key=lambda c: latent[c])

    rrf_noise, ce_noise = 0.45, 0.12
    rrf_view = {c: latent[c] + rng.gauss(0, rrf_noise) for c in ids}
    ce_view = {c: latent[c] + rng.gauss(0, ce_noise) for c in ids}

    # Heavy tail: most of the time the gold chunk is near the top of the RRF
    # ranking, but a meaningful minority of queries bury it.
    target_rank = min(n - 1, int(rng.paretovariate(1.6)) - 1)
    rrf_order = sorted(ids, key=lambda c: -rrf_view[c])
    rrf_order.remove(gold_id)
    rrf_order.insert(target_rank, gold_id)

    return rrf_order, ce_view, [gold_id]


def _keyword_variant(question: str) -> str:
    """
    A deterministic keyword-only rewrite of a question: drop function words,
    keep the content terms and any section numbers.

    Included as a second query per eval row because it stresses the two
    retrieval arms differently -- BM25 gets sharper, dense retrieval loses the
    sentence structure it was trained on -- which changes how deep the gold
    chunk sits in the RRF ranking, and therefore how much window is needed.
    """
    stop = {
        "what", "does", "do", "is", "a", "an", "the", "in", "on", "at", "to", "for",
        "of", "and", "or", "with", "by", "as", "it", "this", "that", "are", "was",
        "were", "be", "has", "have", "had", "not", "how", "why", "who", "when",
        "under", "according", "circumstances", "which", "can", "person", "must",
    }
    tokens = re.findall(r"[\w']+", question.lower())
    kept = [t for t in tokens if t not in stop]
    return " ".join(kept) if kept else question


class RerankerWindowExperiment(Experiment):
    key = "exp02_reranker_window"
    number = 2
    title = "The reranker-window pathology"
    claim = (
        "When rerank_input approaches top_n the cross-encoder cannot change which "
        "chunks reach the generator; sweeping the ratio yields a checkable design rule."
    )

    def __init__(self) -> None:
        self._retriever = None
        self._registry = None
        self._gold_index: Dict[str, List[str]] = {}

    # -- planning --------------------------------------------------------

    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        if ctx.is_live:
            rows = load_eval_dataset()
            if not rows:
                raise RuntimeError(
                    "live mode needs eval/eval_dataset.csv; none found."
                )
            specs = []
            for row in rows:
                specs.append(
                    ExampleSpec(
                        example_id=f"live/{row['id']}/verbatim",
                        payload={"eval_id": row["id"], "question": row["question"], "variant": "verbatim"},
                    )
                )
                specs.append(
                    ExampleSpec(
                        example_id=f"live/{row['id']}/keywords",
                        payload={
                            "eval_id": row["id"],
                            "question": _keyword_variant(row["question"]),
                            "variant": "keywords",
                        },
                    )
                )
            return specs

        n_queries = int(ctx.extra.get("n_simulated_queries", 60))
        return [
            ExampleSpec(example_id=f"sim/q{i:03d}", payload={"query_index": i})
            for i in range(n_queries)
        ]

    # -- live resources --------------------------------------------------

    def setup(self, ctx: ExperimentContext) -> None:
        if not ctx.is_live:
            return

        import os

        from src.chunk_registry import ChunkRegistry
        from src.retriever import get_retriever
        from src.vector_store import ChromaVectorStore

        registry_path = "artifacts/chunk_registry.json"
        if not os.path.exists(registry_path):
            raise RuntimeError(
                f"live mode needs {registry_path}. Run `python run_pipeline.py` first."
            )
        self._registry = ChunkRegistry.load_from_json(registry_path)
        vector_store = ChromaVectorStore()
        vector_store.initialize_collection()
        self._retriever = get_retriever(vector_store, self._registry)
        # Shared with E4 so both experiments score against the same gold notion.
        self._gold_index = build_gold_index(self._registry, load_eval_dataset())

    # -- execution -------------------------------------------------------

    def _window_max(self, ctx: ExperimentContext) -> int:
        return int(ctx.extra.get("window_max", DEFAULT_WINDOW_MAX))

    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        window_max = self._window_max(ctx)

        if ctx.is_live:
            question = spec.payload["question"]
            eval_id = spec.payload["eval_id"]
            t0 = time.time()
            candidates, fusion_meta = self._retriever.rank_candidates(question, window_max)
            rrf_order = [c.chunk_id for c in candidates]
            reranked = self._retriever.rerank(question, candidates)
            rerank_seconds = time.time() - t0
            reranker_scores = {c.chunk_id: c.reranker_score for c in reranked}
            gold_ids = [g for g in self._gold_index.get(eval_id, []) if g in reranker_scores]
            pool_meta = {
                "candidate_pool_size": fusion_meta.get("pre_rerank_candidate_pool_size"),
                "min_dense_distance": fusion_meta.get("pre_rerank_min_dense_distance"),
                "rerank_seconds_for_full_window": round(rerank_seconds, 3),
                "gold_labelled": bool(gold_ids),
            }
        else:
            rng = ctx.rng_for(f"e2:{spec.example_id}")
            rrf_order, reranker_scores, gold_ids = _simulate_pool(rng, window_max)
            pool_meta = {"candidate_pool_size": len(rrf_order), "gold_labelled": True}

        if len(rrf_order) < max(TOP_N_GRID):
            # A pool shallower than the largest top_n cannot express the sweep.
            return {"skipped": True, "reason": "candidate pool too small", **pool_meta}

        ce_order = sorted(rrf_order, key=lambda c: (-reranker_scores[c], rrf_order.index(c)))
        gold_rrf_rank = min(
            (rrf_order.index(g) + 1 for g in gold_ids if g in rrf_order), default=None
        )

        settings = []
        for top_n in TOP_N_GRID:
            for ratio in RATIO_GRID:
                window = int(round(top_n * ratio))
                if window < top_n or window > len(rrf_order):
                    continue
                settings.append(
                    evaluate_window(rrf_order, reranker_scores, window, top_n, gold_ids)
                )

        return {
            "skipped": False,
            "variant": spec.payload.get("variant", "simulated"),
            "gold_rrf_rank": gold_rrf_rank,
            "n_gold_chunks": len(gold_ids),
            "kendall_tau_rrf_vs_reranker": _kendall_tau(rrf_order, ce_order),
            "settings": settings,
            **pool_meta,
        }

    # -- aggregation -----------------------------------------------------

    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        usable = [r for r in records if not r.get("skipped")]
        if not usable:
            return {"error": "no usable records", "n_skipped": len(records)}

        # (top_n, ratio) -> list of per-query setting dicts
        grid: Dict[Tuple[int, float], List[Dict[str, Any]]] = {}
        for record in usable:
            for setting in record["settings"]:
                grid.setdefault((setting["top_n"], setting["ratio"]), []).append(setting)

        sweep = []
        for (top_n, ratio), settings in sorted(grid.items()):
            n = len(settings)
            noops = sum(1 for s in settings if s["is_noop"])
            gold_hits = sum(1 for s in settings if s["gold_in_final"])
            lo, hi = wilson_interval(noops, n)
            sweep.append({
                "top_n": top_n,
                "ratio": ratio,
                "window": settings[0]["window"],
                "n_queries": n,
                "noop_rate": noops / n,
                "noop_rate_ci95": [lo, hi],
                "mean_rescued_chunks": mean([s["n_rescued"] for s in settings]),
                "gold_recall": gold_hits / n,
                "gold_rescued_rate": sum(1 for s in settings if s["gold_rescued_by_reranker"]) / n,
                "gold_lost_rate": sum(1 for s in settings if s["gold_lost_by_reranker"]) / n,
                "mean_rank_displacement": mean([s["mean_rank_displacement"] for s in settings]),
                "cost_units": settings[0]["cost_units"],
            })

        design_rule = self._derive_design_rule(sweep, ctx)

        ratio_one = [row for row in sweep if row["ratio"] == 1.0]
        current = {
            "rerank_input_size": RERANK_INPUT_SIZE,
            "reranker_top_n": RERANKER_TOP_N,
            "ratio": round(RERANK_INPUT_SIZE / RERANKER_TOP_N, 3),
        }

        return {
            "headline": {
                "noop_rate_at_ratio_1": (
                    mean([row["noop_rate"] for row in ratio_one]) if ratio_one else None
                ),
                "noop_rate_is_exactly_1_at_ratio_1": all(
                    row["noop_rate"] == 1.0 for row in ratio_one
                ) if ratio_one else None,
                "mean_kendall_tau_rrf_vs_reranker": mean(
                    [r["kendall_tau_rrf_vs_reranker"] for r in usable]
                ),
                "design_rule": design_rule,
                "repository_current_setting": current,
            },
            "sweep": sweep,
            "gold_depth_distribution": self._gold_depth_histogram(usable),
            "n_skipped": len(records) - len(usable),
            "interpretation_notes": [
                "noop_rate at ratio 1.0 is 1.0 by construction, not by measurement: a "
                "reranker handed exactly top_n candidates must return all of them. It is "
                "reported because it is the invariant a pipeline can be checked against, "
                "and because the failure is silent -- latency is paid, logs look normal, "
                "and no metric over the final context set moves.",
                "Kendall tau measures how much the reranker disagrees with RRF over the "
                "window. Strong disagreement combined with a ratio near 1.0 is the worst "
                "case: the reranker has opinions and no room to act on them.",
                "gold_lost_rate is the honest counterweight to gold_rescued_rate -- a wider "
                "window also gives the reranker more opportunities to demote a chunk RRF "
                "had correctly placed.",
            ],
        }

    def _derive_design_rule(self, sweep: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        """
        The checkable rule: the smallest ratio past which each additional
        scored candidate buys less than ``epsilon`` gold recall. Reported per
        top_n, plus the recall actually available at that knee against the
        widest window measured.
        """
        epsilon = float(ctx.extra.get("knee_epsilon", DEFAULT_KNEE_EPSILON))
        rules = {}
        for top_n in sorted({row["top_n"] for row in sweep}):
            rows = sorted([r for r in sweep if r["top_n"] == top_n], key=lambda r: r["ratio"])
            knee = rows[-1]
            for prev, nxt in zip(rows, rows[1:]):
                extra_candidates = nxt["cost_units"] - prev["cost_units"]
                if extra_candidates <= 0:
                    continue
                gain_per_candidate = (nxt["gold_recall"] - prev["gold_recall"]) / extra_candidates
                if gain_per_candidate < epsilon:
                    knee = prev
                    break
            best = max(rows, key=lambda r: r["gold_recall"])
            rules[f"top_n={top_n}"] = {
                "recommended_min_ratio": knee["ratio"],
                "recommended_window": knee["window"],
                "gold_recall_at_knee": knee["gold_recall"],
                "best_gold_recall_measured": best["gold_recall"],
                "recall_left_on_the_table": round(best["gold_recall"] - knee["gold_recall"], 4),
                "cost_units_at_knee": knee["cost_units"],
            }
        return {
            "knee_epsilon_recall_per_candidate": epsilon,
            "per_top_n": rules,
            "statement": (
                "Set RERANK_INPUT_SIZE to at least the recommended ratio times "
                "RERANKER_TOP_N. Below it the cross-encoder is partly or wholly a no-op; "
                "above it, added candidates buy less recall than they cost in latency."
            ),
        }

    @staticmethod
    def _gold_depth_histogram(records: List[Dict[str, Any]]) -> Dict[str, int]:
        """How deep in the RRF ranking the gold chunk sits. This distribution is
        what determines how much window a corpus actually needs -- the design
        rule is a property of the corpus, not of the reranker."""
        buckets = {"1-3": 0, "4-6": 0, "7-12": 0, "13-20": 0, "21+": 0, "not_in_pool": 0}
        for record in records:
            rank = record.get("gold_rrf_rank")
            if rank is None:
                buckets["not_in_pool"] += 1
            elif rank <= 3:
                buckets["1-3"] += 1
            elif rank <= 6:
                buckets["4-6"] += 1
            elif rank <= 12:
                buckets["7-12"] += 1
            elif rank <= 20:
                buckets["13-20"] += 1
            else:
                buckets["21+"] += 1
        return buckets


EXPERIMENT = RerankerWindowExperiment()
