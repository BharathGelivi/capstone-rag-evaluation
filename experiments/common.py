"""
Shared infrastructure for the X-RAG paper experiments (E1-E5).

Provides three things every experiment needs:

1. **Resumable execution.** ``Checkpoint`` appends one JSON record per example
   to ``records.jsonl`` and tracks run state in ``state.json``. Killing a run
   at example 27 and re-invoking the same command resumes at 27 -- completed
   example ids are never re-executed. The same mechanism at the suite level
   (see ``experiments/run_all.py``) means stopping after experiment 5 resumes
   at 5, not at 1.

2. **Determinism.** Every experiment derives its randomness from
   ``ExperimentContext.rng_for(salt)``, a seeded generator keyed by the run
   seed plus a per-purpose salt. Two runs with the same seed produce
   byte-identical records, which is what makes the resume semantics safe: a
   resumed run and an uninterrupted run yield the same result set.

3. **Statistics.** Correlation and chance-corrected agreement helpers. The
   binary-kappa and Pearson implementations are imported from
   ``scripts/analyze_agreement.py`` rather than reimplemented, so the paper's
   agreement numbers come from exactly one code path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Reuse -- do not reimplement. scripts/analyze_agreement.py is the canonical
# home for the binary agreement statistics reported in the paper.
from scripts.analyze_agreement import _cohens_kappa as binary_cohens_kappa
from scripts.analyze_agreement import _pearson_r as _pearson_r_raw

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = os.path.join("artifacts", "experiments")
DEFAULT_SEED = 20260801

#: Minimum examples per experiment. The paper's claims are per-experiment, so
#: each study has to stand on its own sample; a run that plans fewer than this
#: is a configuration mistake and is refused rather than silently reported.
MIN_EXAMPLES = 50


# ---------------------------------------------------------------------------
# Time / identity helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(payload: Any) -> str:
    """Order-insensitive SHA-256 of a JSON-serialisable payload (first 16 hex chars)."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def finite(x: Any) -> Optional[float]:
    """
    Coerce to a float or None. NaN and infinity become None.

    Baseline result rows carry NaN wherever a baseline failed for an example
    (``scripts/run_baseline_comparison.py`` records NaN rather than aborting).
    NaN propagates silently through a correlation and turns the whole
    coefficient into NaN, so it is filtered at the boundary instead.
    """
    if x is None or isinstance(x, bool):
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value


def clean_series(xs: Sequence[Any]) -> List[Optional[float]]:
    return [finite(x) for x in xs]


def pearson_r(xs: Sequence[Any], ys: Sequence[Any]) -> Optional[float]:
    """
    ``scripts.analyze_agreement._pearson_r`` with NaN filtered at the boundary.

    The underlying implementation is the canonical one and is left untouched;
    it treats NaN as a real value, which turns any correlation involving a
    failed baseline into NaN rather than into a correlation over the rows that
    do have data.
    """
    return _pearson_r_raw(clean_series(xs), clean_series(ys))


def mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in clean_series(xs) if v is not None]
    return sum(vals) / len(vals) if vals else None


def stdev(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [v for v in clean_series(xs) if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _rank(xs: Sequence[float]) -> List[float]:
    """Fractional (average) ranks, so ties do not bias Spearman."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]]) -> Optional[float]:
    """Spearman rank correlation: Pearson over fractional ranks."""
    pairs = [
        (x, y)
        for x, y in zip(clean_series(xs), clean_series(ys))
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    xs2, ys2 = zip(*pairs)
    return pearson_r(_rank(list(xs2)), _rank(list(ys2)))


def multiclass_cohens_kappa(
    a: Sequence[Optional[str]], b: Sequence[Optional[str]]
) -> Optional[float]:
    """
    Chance-corrected agreement over categorical labels (more than two classes).

    ``scripts.analyze_agreement._cohens_kappa`` handles the binary
    failure/no-failure case. The paper's *cause-level* claim needs the
    multiclass form: two diagnosers can agree perfectly on "this run failed"
    (high binary kappa) while disagreeing completely on *which stage* failed
    (low multiclass kappa). That gap is the finding.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for x, y in pairs if x == y) / n
    labels = {x for x, _ in pairs} | {y for _, y in pairs}
    pe = 0.0
    for label in labels:
        p_a = sum(1 for x, _ in pairs if x == label) / n
        p_b = sum(1 for _, y in pairs if y == label) / n
        pe += p_a * p_b
    if abs(pe - 1.0) < 1e-12:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def confusion_matrix(
    truth: Sequence[str], predicted: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    """Nested ``{true_label: {predicted_label: count}}``."""
    labels = sorted(set(truth) | set(predicted))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truth, predicted):
        matrix[t][p] += 1
    return matrix


def per_label_prf(truth: Sequence[str], predicted: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-label precision / recall / F1 / support."""
    labels = sorted(set(truth) | set(predicted))
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(truth, predicted) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, predicted) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, predicted) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    return out


def macro_f1(truth: Sequence[str], predicted: Sequence[str]) -> float:
    scores = per_label_prf(truth, predicted)
    supported = [v["f1"] for v in scores.values() if v["support"] > 0]
    return sum(supported) / len(supported) if supported else 0.0


def accuracy(truth: Sequence[str], predicted: Sequence[str]) -> Optional[float]:
    if not truth:
        return None
    return sum(1 for t, p in zip(truth, predicted) if t == p) / len(truth)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """
    Wilson score interval for a proportion. Preferred over the normal
    approximation here because several of these experiments report rates near
    0.0 or 1.0 (e.g. a no-op rate of exactly 1.0), where the normal interval
    produces impossible bounds outside [0, 1].
    """
    if total == 0:
        return (None, None)
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class ConfigDriftError(RuntimeError):
    """Raised when a resumed run's plan no longer matches the checkpointed one."""


class Checkpoint:
    """
    Append-only per-example checkpoint for one experiment.

    Layout under ``artifacts/experiments/<key>/``::

        state.json      run status, seed, mode, plan fingerprint
        records.jsonl   one JSON object per completed example
        summary.json    aggregate result, written once the run completes

    Records are appended and flushed one at a time, so a process killed
    mid-experiment loses at most the example currently in flight. A truncated
    final line (killed during the write itself) is detected and dropped on the
    next load rather than corrupting the resume.
    """

    def __init__(self, key: str, base_dir: str = EXPERIMENTS_DIR):
        self.key = key
        self.dir = os.path.join(base_dir, key)
        self.records_path = os.path.join(self.dir, "records.jsonl")
        self.state_path = os.path.join(self.dir, "state.json")
        self.summary_path = os.path.join(self.dir, "summary.json")
        self._handle = None
        self._checked_records_signature = None

    # -- state -----------------------------------------------------------

    def load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Unreadable state.json for %s; treating as a fresh run.", self.key)
        return {}

    def save_state(self, **fields: Any) -> None:
        os.makedirs(self.dir, exist_ok=True)
        state = self.load_state()
        state.update(fields)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def is_complete(self) -> bool:
        return self.load_state().get("status") == "complete"

    # -- records ---------------------------------------------------------

    def load_records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.records_path):
            return []
        records: List[Dict[str, Any]] = []
        with open(self.records_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    if line_no != len(lines):
                        raise ValueError(f"Corrupt interior checkpoint record: {self.records_path}:{line_no}")
                    # Only ever expected on the final line, from a process
                    # killed mid-write. Dropping it is correct: the example is
                    # simply not marked complete and will be re-run.
                    logger.warning(
                        "Dropping malformed record at %s:%d (interrupted write).",
                        self.records_path, line_no,
                    )
        return records

    def completed_ids(self) -> List[str]:
        return [r["example_id"] for r in self.load_records() if "example_id" in r]

    def append(self, record: Dict[str, Any]) -> None:
        if "example_id" not in record:
            raise ValueError("Checkpoint records must carry an 'example_id'.")
        os.makedirs(self.dir, exist_ok=True)
        signature = None
        if os.path.exists(self.records_path):
            stat = os.stat(self.records_path)
            signature = (stat.st_size, stat.st_mtime_ns)
        if signature is not None and signature != self._checked_records_signature:
            # Repair only an interrupted final fragment; never hide interior corruption.
            self.load_records()
            with open(self.records_path, "rb+") as handle:
                content = handle.read()
                lines = content.splitlines(keepends=True)
                if lines:
                    try:
                        json.loads(lines[-1])
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        handle.truncate(sum(map(len, lines[:-1])))
                    else:
                        if not content.endswith(b"\n"):
                            handle.write(b"\n")
        with open(self.records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        stat = os.stat(self.records_path)
        self._checked_records_signature = (stat.st_size, stat.st_mtime_ns)

    def save_summary(self, summary: Dict[str, Any]) -> str:
        os.makedirs(self.dir, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return self.summary_path

    def load_summary(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.summary_path):
            return None
        with open(self.summary_path, encoding="utf-8") as f:
            return json.load(f)

    def reset(self) -> None:
        for path in (self.records_path, self.state_path, self.summary_path):
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Experiment scaffolding
# ---------------------------------------------------------------------------

@dataclass
class ExampleSpec:
    """One unit of work. ``example_id`` must be stable across runs -- it is the
    resume key, so a non-deterministic id (a uuid, a timestamp) would silently
    disable resumption by making every planned example look uncompleted."""
    example_id: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentContext:
    """Runtime configuration handed to every experiment."""
    mode: str = "offline"          # "offline" | "live"
    seed: int = DEFAULT_SEED
    limit: Optional[int] = None    # cap planned examples (testing / smoke runs)
    force: bool = False            # ignore existing checkpoint and start over
    base_dir: str = EXPERIMENTS_DIR
    allow_small_sample: bool = False  # bypass the MIN_EXAMPLES floor (tests only)
    extra: Dict[str, Any] = field(default_factory=dict)

    def rng_for(self, salt: str) -> random.Random:
        """A generator seeded by (run seed, salt) -- independent per purpose,
        reproducible across runs, and unaffected by how many examples were
        drawn before it. That last property is what keeps a resumed run
        identical to an uninterrupted one."""
        digest = hashlib.sha256(f"{self.seed}:{salt}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


class Experiment(ABC):
    """
    Base class: plan a list of examples, run each one, summarise the records.

    Subclasses implement ``plan``, ``run_example`` and ``summarize``. The base
    ``run`` supplies resumption, determinism guards, incremental persistence,
    and timing -- so an experiment body never has to think about any of it.
    """

    #: stable directory / manifest key, e.g. "exp01_fault_injection"
    key: str = ""
    #: 1-based ordinal used by ``run_all --from N``
    number: int = 0
    title: str = ""
    #: one-line statement of what the experiment establishes
    claim: str = ""
    #: modes this experiment supports; "live" experiments degrade to offline
    supported_modes: Tuple[str, ...] = ("offline", "live")

    @abstractmethod
    def plan(self, ctx: ExperimentContext) -> List[ExampleSpec]:
        """Enumerate every example. Must be deterministic given ``ctx``."""

    @abstractmethod
    def run_example(self, spec: ExampleSpec, ctx: ExperimentContext) -> Dict[str, Any]:
        """Execute one example and return its record (without ``example_id``)."""

    @abstractmethod
    def summarize(self, records: List[Dict[str, Any]], ctx: ExperimentContext) -> Dict[str, Any]:
        """Aggregate completed records into the experiment's reported result."""

    # -- optional hooks --------------------------------------------------

    def setup(self, ctx: ExperimentContext) -> None:
        """Load shared heavy resources once, after the resume check but before
        the first example. Skipped entirely when nothing is left to run."""

    def plan_fingerprint(self, specs: List[ExampleSpec], ctx: ExperimentContext) -> str:
        return stable_hash({
            "ids": [s.example_id for s in specs],
            "mode": ctx.mode,
            "seed": ctx.seed,
        })

    # -- driver ----------------------------------------------------------

    def run(self, ctx: ExperimentContext) -> Dict[str, Any]:
        checkpoint = Checkpoint(self.key, base_dir=ctx.base_dir)
        if ctx.force:
            checkpoint.reset()

        specs = self.plan(ctx)
        if ctx.limit is not None:
            specs = specs[: ctx.limit]

        if not specs:
            raise RuntimeError(f"{self.key}: plan() produced no examples.")

        duplicates = len(specs) - len({s.example_id for s in specs})
        if duplicates:
            raise RuntimeError(
                f"{self.key}: plan() produced {duplicates} duplicate example_id(s); "
                "ids are the resume key and must be unique."
            )

        if len(specs) < MIN_EXAMPLES and not ctx.allow_small_sample:
            raise RuntimeError(
                f"{self.key}: planned {len(specs)} examples but the suite floor is "
                f"{MIN_EXAMPLES}. Fix the plan, or pass allow_small_sample for a smoke run."
            )

        fingerprint = self.plan_fingerprint(specs, ctx)
        state = checkpoint.load_state()
        prior_fingerprint = state.get("plan_fingerprint")
        if prior_fingerprint and prior_fingerprint != fingerprint:
            raise ConfigDriftError(
                f"{self.key}: the checkpoint at {checkpoint.dir} was written with a "
                f"different plan (seed/mode/example set changed). Re-run with --force "
                f"to discard it, or restore the original settings to resume."
            )

        done = set(checkpoint.completed_ids())
        remaining = [s for s in specs if s.example_id not in done]

        logger.info(
            "%s: %d/%d examples already recorded; %d remaining.",
            self.key, len(specs) - len(remaining), len(specs), len(remaining),
        )

        checkpoint.save_state(
            key=self.key,
            number=self.number,
            title=self.title,
            status="running",
            mode=ctx.mode,
            seed=ctx.seed,
            n_planned=len(specs),
            plan_fingerprint=fingerprint,
            started_at=state.get("started_at") or utc_now(),
        )

        if remaining:
            self.setup(ctx)

        t0 = time.time()
        for i, spec in enumerate(remaining, start=1):
            logger.info("%s: example %d/%d (%s)", self.key, i, len(remaining), spec.example_id)
            record = self.run_example(spec, ctx)
            record["example_id"] = spec.example_id
            record.setdefault("recorded_at", utc_now())
            checkpoint.append(record)

        records = checkpoint.load_records()
        # Records land in completion order across resumed runs; sort by the
        # planned order so the summary is identical whether or not the run was
        # interrupted.
        order = {s.example_id: i for i, s in enumerate(specs)}
        records.sort(key=lambda r: order.get(r.get("example_id"), 1 << 30))

        summary = self.summarize(records, ctx)
        summary = {
            "experiment": self.key,
            "number": self.number,
            "title": self.title,
            "claim": self.claim,
            "mode": ctx.mode,
            "seed": ctx.seed,
            "n_examples": len(records),
            "generated_at": utc_now(),
            **summary,
        }
        checkpoint.save_summary(summary)
        checkpoint.save_state(
            status="complete",
            completed_at=utc_now(),
            n_records=len(records),
            wall_clock_s_last_segment=round(time.time() - t0, 2),
        )
        return summary


# ---------------------------------------------------------------------------
# Small shared utilities used by more than one experiment
# ---------------------------------------------------------------------------

def load_eval_dataset(path: str = "eval/eval_dataset.csv") -> List[Dict[str, str]]:
    """The labeled 40-question benchmark set. Shared by E2/E3/E4."""
    import csv
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_registry_path(default: str = "artifacts/chunk_registry.json") -> Optional[str]:
    return default if os.path.exists(default) else None


def iter_saved_traces(base_dir: str = "artifacts/rag_traces") -> Iterable[Dict[str, Any]]:
    """Yields every saved RAGTrace dict, oldest date partition first.

    Used by E3, which measures refusal behaviour over answers the pipeline
    actually produced rather than over answers written for the experiment.
    """
    import glob
    paths = sorted(glob.glob(os.path.join(base_dir, "*", "*.json")))
    paths += sorted(glob.glob(os.path.join(base_dir, "*.json")))
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "generated_answer" in data:
            data["_source_path"] = path
            yield data
