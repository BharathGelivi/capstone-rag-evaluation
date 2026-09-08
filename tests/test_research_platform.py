"""
Tests for the research-platform modules added for the retrieval-strategy study.

Each module ships a ``demo()`` self-check that runs against hand-built fixtures
-- no corpus, no GPU, no API key -- and asserts the properties that module's
claims rest on. This file runs them under unittest so they are part of
``python -m unittest discover tests`` rather than something a developer has to
remember to invoke, and adds the cross-module invariants no single demo can
check on its own.
"""

import unittest


class TestModuleSelfChecks(unittest.TestCase):
    """Each demo asserts its own module's invariants; failures name the module."""

    def test_legal_corpus(self):
        from src.legal_corpus import demo

        demo()

    def test_legal_graph(self):
        from src.legal_graph import demo

        demo()

    def test_ircot(self):
        from src.ircot import demo

        demo()

    def test_agentic(self):
        from src.agentic import demo

        demo()

    def test_citation_check(self):
        from src.citation_check import demo

        demo()

    def test_rag_eval(self):
        from src.rag_eval import demo

        demo()


class TestCrossModuleInvariants(unittest.TestCase):
    """Properties that span modules, where drift would be silent."""

    def test_metric_panel_matches_provenance_table(self):
        """Every panel metric must declare where its definition comes from.

        A metric shown in the UI without provenance would read as if it were
        RAGAS's or ARES's own output, which it is not.
        """
        from src.rag_eval import METRIC_PROVENANCE, compute_metric_panel

        panel = compute_metric_panel("q", "a", [])
        self.assertEqual(set(panel), set(METRIC_PROVENANCE))
        for name, meta in METRIC_PROVENANCE.items():
            self.assertIn("family", meta, name)
            self.assertIn("definition", meta, name)
            self.assertIn("judge", meta, name)

    def test_ui_panel_covers_every_metric_exactly_once(self):
        from src.rag_eval import METRIC_PROVENANCE
        from ui.components.metrics import FAMILIES, LABELS

        shown = [name for names in FAMILIES.values() for name in names]
        self.assertEqual(len(shown), len(set(shown)), "a metric is rendered twice")
        self.assertEqual(set(shown), set(METRIC_PROVENANCE))
        self.assertEqual(set(shown), set(LABELS))

    def test_every_ablation_arm_is_executable(self):
        """Each arm's config must name a strategy the executor implements.

        A typo here would silently drop an arm from the study, or raise only
        after an hour of the run had already completed.
        """
        from experiments.exp06_strategy_ablation import ARMS

        for arm, config in ARMS.items():
            self.assertIn(config["strategy"], ("plain", "ircot", "agentic"), arm)
            self.assertIn(config["mode"], ("vector", "bm25", "hybrid"), arm)
            self.assertIn(config["chunking"], ("legal", "fixed"), arm)
            self.assertIsInstance(config["rerank"], bool, arm)

    def test_generation_arms_are_a_subset_of_ablation_arms(self):
        from experiments.exp06_strategy_ablation import ARMS
        from experiments.exp07_generation_ablation import GENERATION_ARMS

        self.assertTrue(set(GENERATION_ARMS) <= set(ARMS),
                        set(GENERATION_ARMS) - set(ARMS))

    def test_retrieval_modes_agree_between_config_and_retriever(self):
        from configs.pipeline import RETRIEVAL_MODE, VALID_RETRIEVAL_MODES

        self.assertIn(RETRIEVAL_MODE, VALID_RETRIEVAL_MODES)

    def test_agent_action_set_is_closed(self):
        """The controller must not be able to name an action it cannot run."""
        from src.agentic import ACTIONS

        self.assertIn("stop", ACTIONS)
        self.assertEqual(len(ACTIONS), len(set(ACTIONS)))

    def test_graph_relations_used_by_the_agent_exist_in_the_builder(self):
        """Adverse-treatment relations the contradiction search filters on must
        be relations the graph builder can actually produce, or the filter
        silently returns nothing."""
        from src.agentic import ADVERSE_RELATIONS
        from src.legal_graph import TREATMENT_PATTERNS

        producible = {relation for relation, _ in TREATMENT_PATTERNS}
        for relation in ADVERSE_RELATIONS:
            self.assertIn(relation, producible, relation)


if __name__ == "__main__":
    unittest.main()
