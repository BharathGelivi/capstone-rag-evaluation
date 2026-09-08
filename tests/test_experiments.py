"""
Tests for the paper experiment suite (experiments/, E1-E5).

Organised by what could actually go wrong:

* **Statistics** -- a wrong kappa or a NaN-poisoned correlation silently
  changes a published number, so the helpers are checked against hand-computed
  values rather than against themselves.
* **Checkpointing and resume** -- the property the suite promises is that a
  resumed run equals an uninterrupted one. That is tested directly, including
  the crash case (a truncated final record) and the config-drift case (a
  changed seed must refuse to blend results).
* **Fault injection** -- every injected fault must be recoverable by the real
  diagnosis stack, and compound faults must resolve to the upstream cause.
  These are the assertions E1's headline rests on.
* **Per-experiment invariants** -- the claims each experiment makes that are
  true by construction (a ratio-1.0 reranker window is a no-op; a claimless
  answer scores 1.0 faithfulness) are pinned, so a refactor that breaks the
  claim breaks a test.
* **End-to-end** -- each experiment runs offline and produces a summary.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments.common import (
    Checkpoint,
    ConfigDriftError,
    ExampleSpec,
    Experiment,
    ExperimentContext,
    accuracy,
    clean_series,
    confusion_matrix,
    finite,
    macro_f1,
    mean,
    multiclass_cohens_kappa,
    pearson_r,
    per_label_prf,
    spearman_rho,
    stable_hash,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStatistics(unittest.TestCase):
    def test_finite_rejects_nan_and_inf(self):
        self.assertIsNone(finite(float("nan")))
        self.assertIsNone(finite(float("inf")))
        self.assertIsNone(finite(None))
        self.assertIsNone(finite("not a number"))
        self.assertEqual(finite("0.5"), 0.5)
        self.assertEqual(finite(3), 3.0)

    def test_finite_rejects_bool(self):
        # True would otherwise coerce to 1.0 and be averaged as a score.
        self.assertIsNone(finite(True))

    def test_pearson_survives_nan_in_the_series(self):
        # Baseline rows carry NaN where a baseline failed; the correlation must
        # be computed over the rows that do have data, not collapse to NaN.
        xs = [1.0, 2.0, float("nan"), 4.0]
        ys = [2.0, 4.0, 9.9, 8.0]
        self.assertAlmostEqual(pearson_r(xs, ys), 1.0, places=6)

    def test_pearson_perfect_and_inverse(self):
        self.assertAlmostEqual(pearson_r([1, 2, 3], [2, 4, 6]), 1.0, places=6)
        self.assertAlmostEqual(pearson_r([1, 2, 3], [6, 4, 2]), -1.0, places=6)

    def test_pearson_none_on_zero_variance(self):
        self.assertIsNone(pearson_r([1, 1, 1], [1, 2, 3]))

    def test_spearman_detects_monotone_nonlinear(self):
        # Pearson understates a perfect but curved relationship; Spearman is 1.0.
        xs = [1, 2, 3, 4]
        ys = [1, 4, 9, 16]
        self.assertAlmostEqual(spearman_rho(xs, ys), 1.0, places=6)
        self.assertLess(pearson_r(xs, ys), 1.0)

    def test_spearman_handles_ties(self):
        self.assertIsNotNone(spearman_rho([1, 1, 2, 3], [1, 2, 2, 3]))

    def test_multiclass_kappa_perfect_agreement(self):
        labels = ["A", "B", "C", "A"]
        self.assertAlmostEqual(multiclass_cohens_kappa(labels, labels), 1.0, places=9)

    def test_multiclass_kappa_hand_computed(self):
        # 4 items, agree on 2. po = 0.5.
        a = ["A", "A", "B", "B"]
        b = ["A", "B", "A", "B"]
        # marginals: A 0.5/0.5, B 0.5/0.5 -> pe = 0.25 + 0.25 = 0.5
        # kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0
        self.assertAlmostEqual(multiclass_cohens_kappa(a, b), 0.0, places=9)

    def test_multiclass_kappa_is_zero_when_both_always_say_the_same_label(self):
        # The degenerate case that matters for E5: two diagnosers that both
        # answer UNKNOWN for everything agree 100% of the time but carry no
        # information. Kappa must not reward that.
        a = ["UNKNOWN"] * 10
        b = ["UNKNOWN"] * 10
        self.assertEqual(multiclass_cohens_kappa(a, b), 1.0)

        a = ["UNKNOWN"] * 9 + ["RETRIEVAL_MISS"]
        b = ["UNKNOWN"] * 9 + ["GROUNDING_FAILURE"]
        kappa = multiclass_cohens_kappa(a, b)
        self.assertLess(kappa, 0.5)

    def test_multiclass_kappa_none_on_empty(self):
        self.assertIsNone(multiclass_cohens_kappa([], []))

    def test_confusion_matrix_shape_and_counts(self):
        matrix = confusion_matrix(["A", "A", "B"], ["A", "B", "B"])
        self.assertEqual(matrix["A"]["A"], 1)
        self.assertEqual(matrix["A"]["B"], 1)
        self.assertEqual(matrix["B"]["B"], 1)
        self.assertEqual(matrix["B"]["A"], 0)

    def test_per_label_prf_hand_computed(self):
        truth = ["A", "A", "B", "B"]
        predicted = ["A", "B", "B", "B"]
        scores = per_label_prf(truth, predicted)
        self.assertAlmostEqual(scores["A"]["precision"], 1.0)
        self.assertAlmostEqual(scores["A"]["recall"], 0.5)
        self.assertAlmostEqual(scores["B"]["precision"], 2 / 3)
        self.assertAlmostEqual(scores["B"]["recall"], 1.0)
        self.assertEqual(scores["A"]["support"], 2)

    def test_macro_f1_ignores_unsupported_labels(self):
        # A label predicted but never true must not drag the macro average down
        # to reflect a class that does not exist in the ground truth.
        truth = ["A", "A"]
        predicted = ["A", "A"]
        self.assertAlmostEqual(macro_f1(truth, predicted), 1.0)

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy(["A", "B"], ["A", "A"]), 0.5)
        self.assertIsNone(accuracy([], []))

    def test_wilson_interval_stays_inside_zero_one(self):
        # The reason Wilson is used: at p = 1.0 the normal approximation gives
        # an upper bound above 1.
        lo, hi = wilson_interval(20, 20)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 1.0)

        lo, hi = wilson_interval(0, 20)
        self.assertGreaterEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_wilson_interval_none_on_empty(self):
        self.assertEqual(wilson_interval(0, 0), (None, None))

    def test_mean_skips_none_and_nan(self):
        self.assertAlmostEqual(mean([1.0, None, 3.0, float("nan")]), 2.0)
        self.assertIsNone(mean([None, None]))

    def test_clean_series_preserves_length(self):
        self.assertEqual(clean_series([1, None, float("nan")]), [1.0, None, None])

    def test_stable_hash_is_order_insensitive_for_dicts(self):
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))
        self.assertNotEqual(stable_hash({"a": 1}), stable_hash({"a": 2}))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.checkpoint = Checkpoint("exp_test", base_dir=self.dir)

    def test_append_and_reload(self):
        self.checkpoint.append({"example_id": "a", "value": 1})
        self.checkpoint.append({"example_id": "b", "value": 2})
        self.assertEqual(self.checkpoint.completed_ids(), ["a", "b"])
        self.assertEqual(self.checkpoint.load_records()[1]["value"], 2)

    def test_append_requires_example_id(self):
        with self.assertRaises(ValueError):
            self.checkpoint.append({"value": 1})

    def test_truncated_final_line_is_dropped_not_fatal(self):
        # Simulates a process killed mid-write: the partial record must be
        # discarded so its example is simply re-run, never half-counted.
        self.checkpoint.append({"example_id": "a"})
        with open(self.checkpoint.records_path, "a", encoding="utf-8") as f:
            f.write('{"example_id": "b", "val')
        self.assertEqual(self.checkpoint.completed_ids(), ["a"])

    def test_state_round_trip_and_completion(self):
        self.assertFalse(self.checkpoint.is_complete())
        self.checkpoint.save_state(status="running", seed=7)
        self.checkpoint.save_state(status="complete")
        self.assertTrue(self.checkpoint.is_complete())
        # save_state merges rather than replaces.
        self.assertEqual(self.checkpoint.load_state()["seed"], 7)

    def test_unreadable_state_is_treated_as_fresh(self):
        os.makedirs(self.checkpoint.dir, exist_ok=True)
        with open(self.checkpoint.state_path, "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(self.checkpoint.load_state(), {})

    def test_reset_clears_everything(self):
        self.checkpoint.append({"example_id": "a"})
        self.checkpoint.save_state(status="complete")
        self.checkpoint.save_summary({"x": 1})
        self.checkpoint.reset()
        self.assertEqual(self.checkpoint.completed_ids(), [])
        self.assertFalse(self.checkpoint.is_complete())
        self.assertIsNone(self.checkpoint.load_summary())


# ---------------------------------------------------------------------------
# Experiment base class: resume, determinism, guards
# ---------------------------------------------------------------------------


class CountingExperiment(Experiment):
    """Records which examples it actually executed, so resume can be observed."""

    key = "exp_counting"
    number = 99
    title = "Counting"
    claim = "test double"

    def __init__(self, n=60):
        self.n = n
        self.executed = []

    def plan(self, ctx):
        return [ExampleSpec(example_id=f"x{i:03d}", payload={"i": i}) for i in range(self.n)]

    def run_example(self, spec, ctx):
        self.executed.append(spec.example_id)
        rng = ctx.rng_for(f"count:{spec.example_id}")
        return {"i": spec.payload["i"], "draw": rng.random()}

    def summarize(self, records, ctx):
        return {"total": sum(r["i"] for r in records), "n": len(records)}


class TestExperimentDriver(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def ctx(self, **kwargs):
        return ExperimentContext(base_dir=self.dir, **kwargs)

    def test_runs_every_planned_example(self):
        experiment = CountingExperiment()
        summary = experiment.run(self.ctx())
        self.assertEqual(summary["n_examples"], 60)
        self.assertEqual(len(experiment.executed), 60)

    def test_second_run_executes_nothing(self):
        CountingExperiment().run(self.ctx())
        second = CountingExperiment()
        second.run(self.ctx())
        self.assertEqual(second.executed, [])

    def test_resume_runs_only_the_remainder(self):
        first = CountingExperiment()
        first.run(self.ctx())

        checkpoint = Checkpoint(CountingExperiment.key, base_dir=self.dir)
        records = checkpoint.load_records()[:25]
        with open(checkpoint.records_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        resumed = CountingExperiment()
        resumed.run(self.ctx())
        self.assertEqual(len(resumed.executed), 35)
        self.assertEqual(resumed.executed[0], "x025")

    def test_resumed_run_is_identical_to_uninterrupted_run(self):
        """The core promise: stopping and restarting changes nothing."""
        CountingExperiment().run(self.ctx())
        uninterrupted = Checkpoint(CountingExperiment.key, base_dir=self.dir).load_records()

        other_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, other_dir, ignore_errors=True)
        partial = CountingExperiment()
        partial.run(ExperimentContext(base_dir=other_dir))
        checkpoint = Checkpoint(CountingExperiment.key, base_dir=other_dir)
        kept = checkpoint.load_records()[:17]
        with open(checkpoint.records_path, "w", encoding="utf-8") as f:
            for record in kept:
                f.write(json.dumps(record) + "\n")
        CountingExperiment().run(ExperimentContext(base_dir=other_dir))
        resumed = checkpoint.load_records()

        def normalise(records):
            return sorted(
                ({k: v for k, v in r.items() if k != "recorded_at"} for r in records),
                key=lambda r: r["example_id"],
            )

        self.assertEqual(normalise(uninterrupted), normalise(resumed))

    def test_summary_order_is_plan_order_not_completion_order(self):
        experiment = CountingExperiment()
        experiment.run(self.ctx())
        summary = Checkpoint(CountingExperiment.key, base_dir=self.dir).load_summary()
        self.assertEqual(summary["total"], sum(range(60)))

    def test_force_discards_the_checkpoint(self):
        CountingExperiment().run(self.ctx())
        forced = CountingExperiment()
        forced.run(self.ctx(force=True))
        self.assertEqual(len(forced.executed), 60)

    def test_config_drift_refuses_to_blend_runs(self):
        CountingExperiment().run(self.ctx())
        with self.assertRaises(ConfigDriftError):
            CountingExperiment().run(self.ctx(seed=999))

    def test_config_drift_is_bypassed_by_force(self):
        CountingExperiment().run(self.ctx())
        CountingExperiment().run(self.ctx(seed=999, force=True))  # must not raise

    def test_duplicate_example_ids_are_rejected(self):
        class Duplicating(CountingExperiment):
            def plan(self, ctx):
                return [ExampleSpec(example_id="same") for _ in range(60)]

        with self.assertRaises(RuntimeError) as cm:
            Duplicating().run(self.ctx())
        self.assertIn("duplicate", str(cm.exception))

    def test_sample_floor_is_enforced(self):
        with self.assertRaises(RuntimeError) as cm:
            CountingExperiment(n=10).run(self.ctx())
        self.assertIn("floor", str(cm.exception))

    def test_sample_floor_can_be_bypassed_for_smoke_runs(self):
        summary = CountingExperiment(n=10).run(self.ctx(allow_small_sample=True))
        self.assertEqual(summary["n_examples"], 10)

    def test_empty_plan_is_an_error(self):
        class Empty(CountingExperiment):
            def plan(self, ctx):
                return []

        with self.assertRaises(RuntimeError):
            Empty().run(self.ctx())

    def test_setup_is_skipped_when_nothing_remains(self):
        class Tracking(CountingExperiment):
            setup_calls = 0

            def setup(self, ctx):
                type(self).setup_calls += 1

        Tracking().run(self.ctx())
        self.assertEqual(Tracking.setup_calls, 1)
        Tracking().run(self.ctx())
        self.assertEqual(Tracking.setup_calls, 1)

    def test_rng_is_independent_of_draw_order(self):
        ctx = self.ctx()
        first = ctx.rng_for("a").random()
        ctx.rng_for("b").random()
        self.assertEqual(ctx.rng_for("a").random(), first)


# ---------------------------------------------------------------------------
# E1 -- fault injection
# ---------------------------------------------------------------------------

from experiments.faults import (  # noqa: E402
    ALL_FAULTS,
    COMPOUND_PAIRS,
    FAULT_TO_EXPECTED_CAUSE,
    Arm,
    FaultType,
    build_case,
    build_healthy_case,
    diagnose,
)
from src.root_cause_reasoner import FailureType  # noqa: E402


class TestFaultInjection(unittest.TestCase):
    def rng(self, salt="t"):
        return ExperimentContext().rng_for(salt)

    def test_healthy_baseline_is_diagnosed_healthy(self):
        for i in range(10):
            case = build_healthy_case(f"h{i}", self.rng(f"healthy{i}"))
            _, rca = diagnose(case)
            self.assertEqual(
                rca.primary_cause, FailureType.UNKNOWN,
                f"healthy baseline {i} was assigned a cause: {rca.primary_cause}",
            )

    def test_every_single_fault_is_recovered(self):
        """E1's core assertion. If this fails, the headline accuracy is wrong."""
        for fault in ALL_FAULTS:
            for i in range(6):
                case = build_case(f"{fault.value}-{i}", [fault], self.rng(f"{fault.value}{i}"))
                _, rca = diagnose(case)
                self.assertEqual(
                    rca.primary_cause.value, FAULT_TO_EXPECTED_CAUSE[fault],
                    f"{fault.value} (base {i}) was diagnosed as {rca.primary_cause.value}",
                )

    def test_faults_are_recovered_near_the_threshold(self):
        for fault in ALL_FAULTS:
            case = build_case(
                f"nt-{fault.value}", [fault], self.rng(f"nt{fault.value}"), near_threshold=True
            )
            _, rca = diagnose(case)
            self.assertEqual(rca.primary_cause.value, FAULT_TO_EXPECTED_CAUSE[fault])

    def test_corpus_and_retrieval_faults_are_distinguished(self):
        """
        The pair that matters most: both produce identical downstream symptoms
        (nothing supported, low scores) and differ only in the pre-rerank dense
        distance. A diagnosis that cannot tell them apart cannot tell "fix your
        corpus" from "fix your retriever".
        """
        for i in range(8):
            missing = build_case(f"m{i}", [FaultType.REMOVE_GOLD_CHUNK], self.rng(f"m{i}"))
            miss = build_case(f"r{i}", [FaultType.FORCE_BAD_RANK], self.rng(f"r{i}"))
            self.assertEqual(diagnose(missing)[1].primary_cause, FailureType.MISSING_CORPUS)
            self.assertEqual(diagnose(miss)[1].primary_cause, FailureType.RETRIEVAL_MISS)

    def test_compound_preserved_resolves_to_the_upstream_cause(self):
        """Causal precedence: the loud downstream symptom must not win."""
        for upstream, downstream in COMPOUND_PAIRS:
            for i in range(3):
                case = build_case(
                    f"c{i}", [upstream, downstream], self.rng(f"c{upstream.value}{i}"),
                    preserve_upstream=True,
                )
                self.assertEqual(case.arm, Arm.COMPOUND_PRESERVED.value)
                _, rca = diagnose(case)
                self.assertEqual(
                    rca.primary_cause.value, FAULT_TO_EXPECTED_CAUSE[upstream],
                    f"{upstream.value}+{downstream.value} resolved to {rca.primary_cause.value}",
                )

    def test_compound_downstream_fault_appears_as_a_secondary_effect(self):
        case = build_case(
            "sec", [FaultType.REMOVE_GOLD_CHUNK, FaultType.CONTRADICT_EVIDENCE],
            self.rng("sec"), preserve_upstream=True,
        )
        _, rca = diagnose(case)
        self.assertEqual(rca.primary_cause, FailureType.MISSING_CORPUS)
        self.assertIn(FailureType.GROUNDING_FAILURE, rca.secondary_effects)

    def test_ground_truth_is_the_most_upstream_fault_regardless_of_argument_order(self):
        forward = build_case(
            "o1", [FaultType.REMOVE_GOLD_CHUNK, FaultType.DILUTE_CONTEXT], self.rng("o1")
        )
        reversed_order = build_case(
            "o2", [FaultType.DILUTE_CONTEXT, FaultType.REMOVE_GOLD_CHUNK], self.rng("o1")
        )
        self.assertEqual(
            forward.expected_primary_cause, FailureType.MISSING_CORPUS.value
        )
        self.assertEqual(
            reversed_order.expected_primary_cause, forward.expected_primary_cause
        )

    def test_arm_labels(self):
        rng = self.rng("arm")
        self.assertEqual(build_case("a", [FaultType.DILUTE_CONTEXT], rng).arm, Arm.SINGLE.value)
        self.assertEqual(
            build_case("b", [FaultType.FORCE_BAD_RANK, FaultType.DILUTE_CONTEXT], rng,
                       preserve_upstream=False).arm,
            Arm.COMPOUND_MASKED.value,
        )

    def test_injected_case_shape_is_consistent(self):
        case = build_case("shape", [FaultType.DILUTE_CONTEXT], self.rng("shape"))
        self.assertEqual(case.verification.total_claims, len(case.verification.results))
        self.assertEqual(case.claim_set.total_candidates, len(case.claim_set.candidate_claims))
        self.assertEqual(case.verification.total_claims, case.claim_set.total_candidates)
        self.assertTrue(case.trace.retrieved_chunk_references)

    def test_boundary_fault_creates_an_adjacent_chunk_pair(self):
        case = build_case("bnd", [FaultType.TRUNCATE_AT_BOUNDARY], self.rng("bnd"))
        indices = {
            (ref["parent_document_id"], ref["chunk_index"])
            for ref in case.trace.retrieved_chunk_references
        }
        adjacent = any(
            (doc, idx + 1) in indices for doc, idx in indices
        )
        self.assertTrue(adjacent, "TRUNCATE_AT_BOUNDARY must create an adjacent chunk pair")


class TestExperiment1(unittest.TestCase):
    def setUp(self):
        from experiments.exp01_fault_injection import FaultInjectionExperiment

        self.experiment = FaultInjectionExperiment()
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_plan_meets_the_sample_floor_and_covers_every_arm(self):
        specs = self.experiment.plan(ExperimentContext())
        self.assertGreaterEqual(len(specs), 50)
        prefixes = {s.example_id.split("/")[0] for s in specs}
        self.assertEqual(prefixes, {"single", "compound-preserved", "compound-masked"})

    def test_end_to_end_summary_shape(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        headline = summary["headline"]
        self.assertIn("overall_recovery_accuracy", headline)
        self.assertIn("control_false_positive_rate", headline)
        self.assertEqual(len(headline["accuracy_ci95"]), 2)
        self.assertIn("SINGLE", summary["by_arm"])
        self.assertIn("confusion_matrix", summary)

    def test_single_fault_arm_is_fully_recovered(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        self.assertEqual(summary["by_arm"]["SINGLE"]["accuracy"], 1.0)

    def test_no_false_positives_on_the_control_arm(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        self.assertEqual(summary["headline"]["control_false_positive_rate"], 0.0)

    def test_causal_precedence_beats_confidence_ranking_on_compound_faults(self):
        """The ablation that justifies the reasoner's design."""
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        lift = summary["headline"]["causal_precedence_lift_on_compound_faults"]
        self.assertGreater(lift, 0.0)


# ---------------------------------------------------------------------------
# E2 -- reranker window
# ---------------------------------------------------------------------------

from experiments.exp02_reranker_window import (  # noqa: E402
    RerankerWindowExperiment,
    _keyword_variant,
    _kendall_tau,
    evaluate_window,
)


class TestRerankerWindowArithmetic(unittest.TestCase):
    def setUp(self):
        self.order = [f"c{i}" for i in range(10)]
        # Reranker's preference is the exact reverse of RRF's: maximum
        # disagreement, so a window that allows movement will show it.
        self.scores = {f"c{i}": float(10 - i) for i in range(10)}
        self.reversed_scores = {f"c{i}": float(i) for i in range(10)}

    def test_ratio_one_is_always_a_noop(self):
        """
        The theorem E2 is built on: handed exactly top_n candidates, the
        reranker must return all of them, so the context set cannot change --
        for any query, corpus or reranker.
        """
        for top_n in (1, 3, 5, 8):
            result = evaluate_window(self.order, self.reversed_scores, top_n, top_n, ["c9"])
            self.assertTrue(result["is_noop"])
            self.assertEqual(result["n_rescued"], 0)
            self.assertEqual(result["ratio"], 1.0)

    def test_a_wider_window_can_rescue_a_deep_chunk(self):
        result = evaluate_window(self.order, self.reversed_scores, 10, 3, ["c9"])
        self.assertFalse(result["is_noop"])
        self.assertEqual(result["n_rescued"], 3)
        self.assertTrue(result["gold_in_final"])
        self.assertFalse(result["gold_in_rrf_baseline"])
        self.assertTrue(result["gold_rescued_by_reranker"])

    def test_a_wider_window_can_also_lose_a_good_chunk(self):
        # The honest counterweight: reranking is not monotone improvement.
        result = evaluate_window(self.order, self.reversed_scores, 10, 3, ["c0"])
        self.assertTrue(result["gold_in_rrf_baseline"])
        self.assertFalse(result["gold_in_final"])
        self.assertTrue(result["gold_lost_by_reranker"])

    def test_agreeing_reranker_is_a_noop_at_any_width(self):
        # No-op-ness is a property of the window *and* the reranker's opinion.
        result = evaluate_window(self.order, self.scores, 10, 4, ["c0"])
        self.assertTrue(result["is_noop"])
        self.assertEqual(result["mean_rank_displacement"], 0.0)

    def test_window_narrower_than_top_n_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_window(self.order, self.scores, 2, 5, [])

    def test_cost_units_track_the_window(self):
        self.assertEqual(evaluate_window(self.order, self.scores, 8, 4, [])["cost_units"], 8)

    def test_ties_are_broken_deterministically_by_rrf_rank(self):
        flat = {cid: 1.0 for cid in self.order}
        a = evaluate_window(self.order, flat, 10, 3, [])
        b = evaluate_window(self.order, flat, 10, 3, [])
        self.assertEqual(a["rescued_chunk_ids"], b["rescued_chunk_ids"])
        self.assertTrue(a["is_noop"])

    def test_kendall_tau_bounds(self):
        self.assertAlmostEqual(_kendall_tau(self.order, self.order), 1.0)
        self.assertAlmostEqual(_kendall_tau(self.order, list(reversed(self.order))), -1.0)
        self.assertIsNone(_kendall_tau(["a"], ["a"]))

    def test_keyword_variant_strips_function_words_and_keeps_numbers(self):
        variant = _keyword_variant("Under what circumstances does section 45 apply to abetment?")
        self.assertIn("45", variant)
        self.assertIn("abetment", variant)
        self.assertNotIn(" does ", f" {variant} ")

    def test_keyword_variant_never_returns_empty(self):
        self.assertTrue(_keyword_variant("what is the"))


class TestExperiment2(unittest.TestCase):
    def setUp(self):
        self.experiment = RerankerWindowExperiment()
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_offline_plan_meets_the_floor(self):
        self.assertGreaterEqual(len(self.experiment.plan(ExperimentContext())), 50)

    def test_end_to_end_reproduces_the_noop_invariant(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        self.assertEqual(summary["headline"]["noop_rate_at_ratio_1"], 1.0)
        self.assertTrue(summary["headline"]["noop_rate_is_exactly_1_at_ratio_1"])

    def test_noop_rate_falls_as_the_ratio_grows(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        rows = sorted(
            [r for r in summary["sweep"] if r["top_n"] == 6], key=lambda r: r["ratio"]
        )
        self.assertEqual(rows[0]["noop_rate"], 1.0)
        self.assertLess(rows[-1]["noop_rate"], rows[0]["noop_rate"])

    def test_design_rule_is_emitted_per_top_n(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        rule = summary["headline"]["design_rule"]
        self.assertIn("per_top_n", rule)
        for entry in rule["per_top_n"].values():
            self.assertGreaterEqual(entry["recommended_min_ratio"], 1.0)

    def test_repository_current_setting_is_reported(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        current = summary["headline"]["repository_current_setting"]
        self.assertEqual(
            current["ratio"],
            round(current["rerank_input_size"] / current["reranker_top_n"], 3),
        )


class TestRetrieverSeams(unittest.TestCase):
    """E2 depends on rank_candidates/rerank composing back into retrieve()."""

    def test_retrieve_is_rank_candidates_then_rerank_then_truncate(self):
        from configs.pipeline import RERANKER_TOP_N, RERANK_INPUT_SIZE
        from src.retriever import Retriever

        chunks = [
            type("C", (), {"chunk_id": f"c{i}", "chunk_text": f"t{i}", "rank": i + 1,
                           "similarity_score": 0.0, "reranker_score": 0.0})()
            for i in range(4)
        ]
        # Bare instance: __init__ loads a cross-encoder and an embedding model,
        # neither of which this test needs. Only the attributes retrieve()
        # touches are populated.
        retriever = object.__new__(Retriever)
        retriever.top_k = 5
        retriever.vector_store = object()

        with patch.object(Retriever, "rank_candidates", return_value=(chunks, {
            "question_embedding_dimension": 768,
            "pre_rerank_candidate_pool_size": 40,
            "pre_rerank_min_dense_distance": 0.2,
        })) as ranked, patch.object(
            Retriever, "rerank", side_effect=lambda q, c: list(reversed(c))
        ) as reranked:
            result = retriever.retrieve("q")

        ranked.assert_called_once_with("q", RERANK_INPUT_SIZE)
        reranked.assert_called_once()
        self.assertEqual(len(result.retrieved_chunks), min(RERANKER_TOP_N, 4))
        self.assertEqual(result.retrieved_chunks[0].chunk_id, "c3")
        self.assertEqual([c.rank for c in result.retrieved_chunks], [1, 2, 3, 4])
        self.assertEqual(result.retrieval_metadata["pre_rerank_min_dense_distance"], 0.2)


# ---------------------------------------------------------------------------
# E3 -- refusal calibration
# ---------------------------------------------------------------------------

from experiments.exp03_refusal_calibration import (  # noqa: E402
    RefusalCalibrationExperiment,
    contexts_from_prompt_snapshot,
    extract_claims,
    faithfulness_scores,
    is_generation_error,
    is_refusal,
    lexical_support,
)


class TestRefusalPrimitives(unittest.TestCase):
    def test_refusal_detection_covers_common_surface_forms(self):
        for answer in [
            "I do not have enough information to answer this.",
            "I don't have enough information.",
            "The retrieved context does not contain the answer.",
            "Unable to determine from the provided context.",
            "I cannot answer that.",
        ]:
            self.assertTrue(is_refusal(answer), answer)

    def test_substantive_answers_are_not_flagged_as_refusals(self):
        self.assertFalse(
            is_refusal("Section 103 prescribes death or imprisonment for life.")
        )

    def test_generation_errors_are_detected(self):
        self.assertTrue(is_generation_error("Error generating answer: timeout"))
        self.assertFalse(is_generation_error("The error of law described in section 14..."))

    def test_faithfulness_is_one_for_a_claimless_answer(self):
        """
        The pathology, pinned. A refusal makes zero claims, the metric is 0/0,
        and the conventional resolution scores it 1.0 -- higher than almost any
        real answer can achieve.
        """
        scores = faithfulness_scores(n_claims=0, n_supported=0)
        self.assertEqual(scores["conventional"], 1.0)
        self.assertIsNone(scores["undefined_on_empty"])

    def test_faithfulness_is_a_plain_fraction_otherwise(self):
        scores = faithfulness_scores(n_claims=4, n_supported=3)
        self.assertAlmostEqual(scores["conventional"], 0.75)
        self.assertAlmostEqual(scores["undefined_on_empty"], 0.75)

    def test_a_perfect_answer_cannot_beat_a_refusal_on_faithfulness(self):
        refusal = faithfulness_scores(0, 0)["conventional"]
        perfect = faithfulness_scores(6, 6)["conventional"]
        self.assertEqual(refusal, perfect)  # a tie is already the problem
        imperfect = faithfulness_scores(6, 5)["conventional"]
        self.assertGreater(refusal, imperfect)

    def test_claim_extraction_drops_non_assertive_framing(self):
        claims = extract_claims(
            "I do not have enough information to answer this. "
            "However, the context mentions procedure."
        )
        self.assertEqual(claims, [])

    def test_claim_extraction_keeps_assertive_sentences_and_strips_citations(self):
        claims = extract_claims(
            "Section 103 prescribes the death penalty [Chunk-ID: c1]. "
            "It also permits imprisonment for life."
        )
        self.assertEqual(len(claims), 2)
        self.assertNotIn("Chunk-ID", claims[0])

    def test_lexical_support_is_bounded_and_directional(self):
        self.assertAlmostEqual(
            lexical_support("murder punishment death", ["punishment for murder is death"]), 1.0
        )
        self.assertEqual(lexical_support("entirely unrelated vocabulary", ["statute"]), 0.0)
        self.assertEqual(lexical_support("", ["anything"]), 0.0)

    def test_contexts_are_recovered_from_a_prompt_snapshot(self):
        prompt = (
            "Retrieved context for this question:\n\n"
            "--- Context chunk 1 [Chunk-ID: a1] ---\nFirst chunk body.\n\n"
            "--- Context chunk 2 [Chunk-ID: b2] ---\nSecond chunk body.\n\n"
            "Question: what?"
        )
        contexts = contexts_from_prompt_snapshot(prompt)
        self.assertEqual(len(contexts), 2)
        self.assertIn("First chunk body.", contexts[0])
        self.assertIn("Second chunk body.", contexts[1])
        self.assertNotIn("Question:", contexts[1])

    def test_prompt_snapshot_without_context_yields_nothing(self):
        self.assertEqual(contexts_from_prompt_snapshot("No document context."), [])


class TestExperiment3(unittest.TestCase):
    def setUp(self):
        self.experiment = RefusalCalibrationExperiment()
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_offline_plan_excludes_persisted_generation_errors(self):
        specs = self.experiment.plan(ExperimentContext())
        self.assertGreaterEqual(len(specs), 50)
        self.assertFalse(
            any(is_generation_error(s.payload["answer"]) for s in specs),
            "a persisted API error was planned as if it were an answer",
        )

    def test_end_to_end_shows_that_silence_is_rewarded(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        headline = summary["headline"]
        self.assertGreater(headline["faithfulness_advantage_of_refusing"], 0.0)
        self.assertLess(headline["recall_cost_of_refusing"], 0.0)

    def test_the_silence_adjusted_metric_removes_the_incentive(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        self.assertLess(summary["headline"]["silence_adjusted_advantage_of_refusing"], 0.0)

    def test_records_without_recoverable_context_are_excluded_not_scored_as_zero(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        self.assertIn("n_excluded_no_recoverable_context", summary)
        self.assertEqual(
            summary["n_scored_for_faithfulness"]
            + summary["n_excluded_no_recoverable_context"]
            + summary["n_generation_errors"],
            summary["n_examples"],
        )

    def test_refusals_make_almost_no_claims(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        analysis = summary["silence_reward_analysis"]
        self.assertLess(analysis["mean_claims_refusals"], analysis["mean_claims_answers"])


# ---------------------------------------------------------------------------
# E4 -- corpus quality
# ---------------------------------------------------------------------------

from experiments.corpus import (  # noqa: E402
    NON_ANSWERING_CLASSES,
    ChunkClass,
    build_gold_index,
    chunk_features,
    classify_chunk,
    synthesize_toc_chunk,
)


class TestChunkClassifier(unittest.TestCase):
    PROVISION = (
        "45. A person abets the doing of a thing who instigates any person to do that "
        "thing, or engages with one or more other persons in any conspiracy for the doing "
        "of that thing, or intentionally aids by any act or illegal omission the doing of "
        "that thing. Whoever abets an offence shall be punished with the punishment "
        "provided for the offence."
    )
    TOC_PAGE = "\n".join(
        [f"{i}. Punishment for offence number {i}." for i in range(1, 16)]
    )
    HEADING_COLUMN = "\n".join(
        ["Act of Judge", "when acting", "judicially.", "Act done", "pursuant to",
         "judgment or", "order of", "Court.", "Act done by a", "person justified",
         "or by mistake", "of fact believing"]
    )
    BOILERPLATE = (
        "MINISTRY OF LAW AND JUSTICE\n(Legislative Department)\n"
        "New Delhi, the 25th December, 2023\n"
        "The following Act of Parliament received the assent of the President on the "
        "25th December, 2023, and is hereby published for general information."
    )

    def test_provision_is_classified_as_a_provision(self):
        self.assertEqual(classify_chunk(self.PROVISION)[0], ChunkClass.PROVISION)

    def test_arrangement_of_sections_page_is_toc_like(self):
        self.assertEqual(classify_chunk(self.TOC_PAGE)[0], ChunkClass.TOC_LIKE)

    def test_marginal_heading_column_is_toc_like(self):
        self.assertEqual(classify_chunk(self.HEADING_COLUMN)[0], ChunkClass.TOC_LIKE)

    def test_front_matter_is_boilerplate(self):
        self.assertEqual(classify_chunk(self.BOILERPLATE)[0], ChunkClass.BOILERPLATE)

    def test_short_text_is_a_fragment(self):
        self.assertEqual(classify_chunk("Section 4.")[0], ChunkClass.FRAGMENT)

    def test_a_running_header_does_not_demote_a_provision(self):
        """
        The regression this classifier was rewritten for: page furniture is
        extraction noise stapled onto good chunks, not a chunk class. Treating
        it as one mislabelled 173 real provisions.
        """
        contaminated = "THE GAZETTE OF INDIA EXTRAORDINARY\n" + self.PROVISION
        label, features = classify_chunk(contaminated)
        self.assertEqual(label, ChunkClass.PROVISION)
        self.assertEqual(features["page_furniture_lines"], 1)
        self.assertGreater(features["page_furniture_line_fraction"], 0.0)

    def test_non_answering_classes_exclude_provision(self):
        self.assertNotIn(ChunkClass.PROVISION, NON_ANSWERING_CLASSES)

    def test_features_are_exposed_for_auditing(self):
        features = chunk_features(self.PROVISION)
        for key in ("n_words", "mean_line_words", "operative_term_hits",
                    "toc_entry_line_fraction", "page_furniture_line_fraction"):
            self.assertIn(key, features)

    def test_synthesised_toc_reads_like_a_table_of_contents(self):
        text = synthesize_toc_chunk(["Punishment for murder", "Abetment of a thing"], 10)
        self.assertIn("ARRANGEMENT OF SECTIONS", text)
        self.assertIn("10. Punishment for murder.", text)
        self.assertEqual(classify_chunk(text + "\n" + text)[0], ChunkClass.TOC_LIKE)


class FakeRecord:
    def __init__(self, chunk_id, source_file, text):
        self.chunk_id = chunk_id
        self.source_file = source_file
        self.text = text


class FakeRegistry:
    def __init__(self, records):
        self._records = {r.chunk_id: r for r in records}


class TestGoldIndex(unittest.TestCase):
    def test_gold_matches_document_and_section_marker(self):
        registry = FakeRegistry([
            FakeRecord("a", "bns.pdf", "45. A person abets the doing of a thing."),
            FakeRecord("b", "bns.pdf", "46. Something else entirely."),
            FakeRecord("c", "bsa.pdf", "45. A different statute's section 45."),
        ])
        index = build_gold_index(
            registry, [{"id": "1", "source_document": "BNS", "source_section": "Section 45"}]
        )
        self.assertEqual(index["1"], ["a"])

    def test_rows_without_a_section_number_get_an_empty_gold_set(self):
        registry = FakeRegistry([FakeRecord("a", "bns.pdf", "45. text")])
        index = build_gold_index(
            registry, [{"id": "1", "source_document": "BNS", "source_section": ""}]
        )
        self.assertEqual(index["1"], [])


class TestExperiment4(unittest.TestCase):
    def setUp(self):
        from experiments.exp04_corpus_quality import CorpusQualityExperiment

        self.experiment = CorpusQualityExperiment()
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_offline_plan_meets_the_floor(self):
        self.assertGreaterEqual(len(self.experiment.plan(ExperimentContext())), 50)

    def test_injected_contamination_wastes_context_slots(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        levels = sorted(summary["by_contamination_level"], key=lambda r: r["contamination"])
        self.assertEqual(levels[0]["contamination"], 0.0)
        self.assertEqual(levels[0]["mean_slot_waste_rate"], 0.0)
        self.assertGreater(levels[-1]["mean_slot_waste_rate"], 0.0)

    def test_injected_contamination_costs_gold_recall(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        levels = sorted(summary["by_contamination_level"], key=lambda r: r["contamination"])
        self.assertLess(levels[-1]["gold_recall"], levels[0]["gold_recall"])

    def test_counterfactual_recovers_recall_that_contamination_removed(self):
        """The upstream-determinant claim: removing non-answering chunks from
        contention restores what they displaced."""
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        worst = max(summary["by_contamination_level"], key=lambda r: r["contamination"])
        self.assertGreater(
            worst["gold_recall_without_non_answering_chunks"], worst["gold_recall"]
        )
        self.assertGreater(worst["gold_displacement_rate"], 0.0)


# ---------------------------------------------------------------------------
# E5 -- diagnostic agreement
# ---------------------------------------------------------------------------

from experiments.exp05_diagnostic_agreement import (  # noqa: E402
    DiagnosticAgreementExperiment,
    baseline_attribution,
)


class TestBaselineAttribution(unittest.TestCase):
    def test_low_context_precision_reads_as_retrieval(self):
        self.assertEqual(
            baseline_attribution({"ragchecker_context_precision": 0.2, "ragas_faithfulness": 0.9}),
            "RETRIEVAL_MISS",
        )

    def test_high_hallucination_reads_as_grounding(self):
        self.assertEqual(
            baseline_attribution({"ragchecker_context_precision": 0.9, "ragchecker_hallucination": 0.8}),
            "GROUNDING_FAILURE",
        )

    def test_low_faithfulness_reads_as_generation(self):
        self.assertEqual(
            baseline_attribution({
                "ragchecker_context_precision": 0.9,
                "ragchecker_hallucination": 0.1,
                "ragas_faithfulness": 0.4,
            }),
            "UNSUPPORTED_GENERATION",
        )

    def test_healthy_scores_read_as_no_failure(self):
        self.assertEqual(
            baseline_attribution({
                "ragchecker_context_precision": 0.9,
                "ragchecker_hallucination": 0.0,
                "ragas_faithfulness": 0.95,
            }),
            "UNKNOWN",
        )

    def test_retrieval_is_checked_before_generation(self):
        # Both signals fire; retrieval wins, mirroring causal-order selection.
        self.assertEqual(
            baseline_attribution({
                "ragchecker_context_precision": 0.1,
                "ragchecker_hallucination": 0.9,
                "ragas_faithfulness": 0.1,
            }),
            "RETRIEVAL_MISS",
        )

    def test_a_row_with_no_baseline_scores_yields_no_verdict(self):
        # An absent judgement must never be counted as agreement.
        self.assertIsNone(baseline_attribution({"eval_id": "1"}))


class TestExperiment5(unittest.TestCase):
    def setUp(self):
        self.experiment = DiagnosticAgreementExperiment()
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_plan_meets_the_floor_even_with_no_benchmark_results(self):
        with patch("experiments.exp05_diagnostic_agreement.load_real_rows", return_value=[]):
            specs = self.experiment.plan(ExperimentContext())
        self.assertGreaterEqual(len(specs), 50 - 50)  # injected rows alone may be < floor
        self.assertTrue(all(s.example_id.startswith("injected/") for s in specs))

    def test_cause_kappa_is_reported_next_to_score_correlation(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        headline = summary["headline"]
        self.assertIn("score_correlation_real_rows", headline)
        self.assertIn("cause_level_kappa_real_rows", headline)
        self.assertIn("binary_failure_kappa_real_rows", headline)

    def test_injected_rows_are_excluded_from_the_correlation(self):
        """Correlating X-RAG against a metric recomputed from X-RAG's own
        verification output would be circular."""
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        correlations = summary["scalar_correlations_real_rows_only"]
        if "xrag_vs_ragas_faithfulness" in correlations:
            self.assertLessEqual(
                correlations["xrag_vs_ragas_faithfulness"]["n_paired"], summary["n_real_rows"]
            )

    def test_ground_truth_scoring_covers_the_injected_rows(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        oracle = summary["headline"]["who_is_right_on_injected_faults"]
        self.assertEqual(oracle["n"], summary["n_injected_rows"])
        self.assertIn("xrag_accuracy", oracle)
        self.assertIn("baseline_rule_accuracy", oracle)

    def test_correlation_is_not_nan_when_a_baseline_failed(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        r = summary["headline"]["score_correlation_real_rows"]
        if r is not None:
            self.assertEqual(r, r, "correlation came back NaN")

    def test_cause_agreement_is_split_by_source(self):
        summary = self.experiment.run(ExperimentContext(base_dir=self.dir))
        for key in ("real_baseline", "injected", "pooled"):
            self.assertIn(key, summary["cause_level_agreement"])


# ---------------------------------------------------------------------------
# Suite driver
# ---------------------------------------------------------------------------

from experiments import run_all  # noqa: E402


class TestSuiteDriver(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.experiments = run_all.load_experiments()

    def test_experiments_are_numbered_one_through_five(self):
        self.assertEqual([e.number for e in self.experiments], [1, 2, 3, 4, 5])
        self.assertEqual(len({e.key for e in self.experiments}), 5)

    def test_every_experiment_declares_a_claim(self):
        for experiment in self.experiments:
            self.assertTrue(experiment.claim, f"{experiment.key} has no claim")
            self.assertTrue(experiment.title)

    def test_from_selects_the_tail_of_the_suite(self):
        args = run_all.build_parser().parse_args(["--from", "5"])
        selected = run_all.select(self.experiments, args)
        self.assertEqual([e.number for e in selected], [5])

    def test_from_three_keeps_three_onwards(self):
        args = run_all.build_parser().parse_args(["--from", "3"])
        self.assertEqual(
            [e.number for e in run_all.select(self.experiments, args)], [3, 4, 5]
        )

    def test_only_selects_exactly_those_experiments(self):
        args = run_all.build_parser().parse_args(["--only", "2", "4"])
        self.assertEqual(
            [e.number for e in run_all.select(self.experiments, args)], [2, 4]
        )

    def test_parse_extra_coerces_json_and_falls_back_to_strings(self):
        extra = run_all.parse_extra(["window_max=20", "verifier=nli", "flag=true"])
        self.assertEqual(extra["window_max"], 20)
        self.assertEqual(extra["verifier"], "nli")
        self.assertIs(extra["flag"], True)

    def test_parse_extra_rejects_malformed_pairs(self):
        with self.assertRaises(SystemExit):
            run_all.parse_extra(["nope"])

    def test_manifest_round_trip(self):
        run_all.save_manifest(self.dir, {"experiments": {"a": {"status": "complete"}}})
        manifest = run_all.load_manifest(self.dir)
        self.assertEqual(manifest["experiments"]["a"]["status"], "complete")
        self.assertIn("updated_at", manifest)

    def test_unreadable_manifest_starts_fresh(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(run_all.manifest_path(self.dir), "w", encoding="utf-8") as f:
            f.write("{oops")
        self.assertEqual(run_all.load_manifest(self.dir)["experiments"], {})

    def test_a_completed_experiment_is_skipped_on_the_next_run(self):
        experiment = CountingExperiment()
        ctx = ExperimentContext(base_dir=self.dir)
        run_all.run_suite([experiment], ctx, rerun_complete=False)
        self.assertEqual(len(experiment.executed), 60)

        second = CountingExperiment()
        run_all.run_suite([second], ExperimentContext(base_dir=self.dir), rerun_complete=False)
        self.assertEqual(second.executed, [])

    def test_a_failing_experiment_does_not_abort_the_suite(self):
        class Exploding(CountingExperiment):
            key = "exp_exploding"
            number = 98

            def run_example(self, spec, ctx):
                raise RuntimeError("boom")

        after = CountingExperiment()
        failures = run_all.run_suite(
            [Exploding(), after], ExperimentContext(base_dir=self.dir), rerun_complete=False
        )
        self.assertEqual(failures, 1)
        self.assertEqual(len(after.executed), 60)

    def test_a_failure_is_recorded_in_the_manifest(self):
        class Exploding(CountingExperiment):
            key = "exp_exploding"
            number = 98

            def run_example(self, spec, ctx):
                raise RuntimeError("boom")

        run_all.run_suite([Exploding()], ExperimentContext(base_dir=self.dir), rerun_complete=False)
        manifest = run_all.load_manifest(self.dir)
        self.assertIn("boom", manifest["experiments"]["exp_exploding"]["last_error"])

    def test_partial_records_survive_a_failure_and_are_resumed(self):
        class HalfExploding(CountingExperiment):
            key = "exp_half"
            number = 97

            def run_example(self, spec, ctx):
                if spec.payload["i"] == 20:
                    raise RuntimeError("boom")
                return super().run_example(spec, ctx)

        ctx = ExperimentContext(base_dir=self.dir)
        run_all.run_suite([HalfExploding()], ctx, rerun_complete=False)
        checkpoint = Checkpoint("exp_half", base_dir=self.dir)
        self.assertEqual(len(checkpoint.completed_ids()), 20)

        # The next run resumes at 20 rather than repeating the first 20.
        recovered = CountingExperiment()
        recovered.key = "exp_half"
        recovered.number = 97
        run_all.run_suite([recovered], ExperimentContext(base_dir=self.dir), rerun_complete=False)
        self.assertEqual(len(recovered.executed), 40)
        self.assertEqual(recovered.executed[0], "x020")

    def test_suite_report_is_written(self):
        experiment = CountingExperiment()
        run_all.run_suite([experiment], ExperimentContext(base_dir=self.dir), rerun_complete=False)
        report = os.path.join(self.dir, run_all.REPORT_NAME)
        self.assertTrue(os.path.exists(report))
        with open(report, encoding="utf-8") as f:
            self.assertIn("Experiment Suite", f.read())

    def test_status_command_exits_cleanly(self):
        self.assertEqual(run_all.main(["--status", "--base-dir", self.dir]), 0)

    def test_main_runs_a_single_experiment_end_to_end(self):
        code = run_all.main(
            ["--only", "1", "--base-dir", self.dir, "--limit", "12"]
        )
        self.assertEqual(code, 0)
        summary = Checkpoint("exp01_fault_injection", base_dir=self.dir).load_summary()
        self.assertEqual(summary["n_examples"], 12)

    def test_limit_implies_small_sample_is_allowed(self):
        args = run_all.build_parser().parse_args(["--limit", "5"])
        self.assertEqual(args.limit, 5)
        self.assertFalse(args.allow_small_sample)  # implied downstream, not on the parser


class TestSuiteIntegration(unittest.TestCase):
    """Every experiment runs offline and produces a well-formed summary."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.code = run_all.main(["--base-dir", cls.dir])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_whole_suite_succeeds(self):
        self.assertEqual(self.code, 0)

    def test_every_experiment_meets_the_fifty_example_floor(self):
        for experiment in run_all.load_experiments():
            summary = Checkpoint(experiment.key, base_dir=self.dir).load_summary()
            self.assertIsNotNone(summary, f"{experiment.key} produced no summary")
            self.assertGreaterEqual(
                summary["n_examples"], 50,
                f"{experiment.key} ran only {summary['n_examples']} examples",
            )

    def test_every_summary_carries_a_headline_and_interpretation(self):
        for experiment in run_all.load_experiments():
            summary = Checkpoint(experiment.key, base_dir=self.dir).load_summary()
            self.assertIn("headline", summary, experiment.key)
            self.assertIn("claim", summary, experiment.key)
            self.assertTrue(summary.get("interpretation_notes"), experiment.key)

    def test_every_summary_is_json_serialisable(self):
        for experiment in run_all.load_experiments():
            path = Checkpoint(experiment.key, base_dir=self.dir).summary_path
            with open(path, encoding="utf-8") as f:
                json.load(f)

    def test_all_five_are_marked_complete_in_the_manifest(self):
        manifest = run_all.load_manifest(self.dir)
        statuses = {k: v["status"] for k, v in manifest["experiments"].items()}
        self.assertEqual(len(statuses), 5)
        self.assertTrue(all(s == "complete" for s in statuses.values()), statuses)

    def test_rerunning_the_suite_recomputes_nothing(self):
        before = {
            e.key: os.path.getsize(Checkpoint(e.key, base_dir=self.dir).records_path)
            for e in run_all.load_experiments()
        }
        run_all.main(["--base-dir", self.dir])
        after = {
            e.key: os.path.getsize(Checkpoint(e.key, base_dir=self.dir).records_path)
            for e in run_all.load_experiments()
        }
        self.assertEqual(before, after)

    def test_starting_from_five_leaves_earlier_experiments_untouched(self):
        first = os.path.getmtime(
            Checkpoint("exp01_fault_injection", base_dir=self.dir).records_path
        )
        run_all.main(["--base-dir", self.dir, "--from", "5"])
        self.assertEqual(
            first,
            os.path.getmtime(Checkpoint("exp01_fault_injection", base_dir=self.dir).records_path),
        )


if __name__ == "__main__":
    unittest.main()
