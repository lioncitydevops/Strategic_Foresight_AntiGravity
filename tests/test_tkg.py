"""
Unit tests for Sparse Temporal Knowledge Graph (TKG) and State transitions.
"""
import unittest
from src.core.tkg import SparseTKG, Fact
from src.core.state import WorldState
from src.core.actions import Action
from src.core.simulator import TransitionSimulator


class TestTKG(unittest.TestCase):

    def setUp(self):
        self.g = SparseTKG(time_step=0)
        self.g.add_fact("USA", "Tariff", "Semiconductors", 25.0)
        self.g.add_fact("CHN", "Sanctions", "TechEntity_Alpha", 0.7)

    def test_facts_storage(self):
        self.assertEqual(self.g.num_facts(), 2)
        self.assertTrue(self.g.has_triple("USA", "Tariff", "Semiconductors"))
        self.assertEqual(self.g.get_weight("USA", "Tariff", "Semiconductors"), 25.0)

    def test_graph_edits(self):
        state = WorldState(time_step=0, graph=self.g)
        act = Action("USA", "ImposeTariff", "Semiconductors", delta_weight=10.0)
        act.apply(state)
        self.assertEqual(state.graph.get_weight("USA", "Tariff", "Semiconductors"), 35.0)

    def test_shock_propagation(self):
        sim = TransitionSimulator()
        state = WorldState(time_step=0, graph=self.g, exogenous_indicators={"gdp_growth_USA": 2.2})
        next_s = sim.step(state, [Action("USA", "ImposeTariff", "Semiconductors", delta_weight=20.0)])
        self.assertLess(next_s.get_indicator("gdp_growth_USA"), 2.2)


if __name__ == "__main__":
    unittest.main()
