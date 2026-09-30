"""
Unit tests for Scenario Compiler and 5 Scenario Tests.
"""
import unittest
from src.core.tkg import SparseTKG
from src.core.state import WorldState
from src.compiler.compiler import ScenarioCompiler
from src.compiler.tester import ScenarioTester


class TestCompilerAndTester(unittest.TestCase):

    def setUp(self):
        self.compiler = ScenarioCompiler()
        self.tester = ScenarioTester()
        g = SparseTKG()
        g.add_fact("USA", "Tariff", "Semiconductors", 10.0)
        self.state = WorldState(time_step=0, graph=g)

    def test_compilation(self):
        raw = "A blockade begins in mid-2027. Beijing puts self-sufficiency ahead of growth. Oil stays above $95 a barrel through 2029."
        spec = self.compiler.compile("Test_1", "Test Narrative", raw)
        self.assertEqual(len(spec.interventions), 1)
        self.assertEqual(len(spec.driver_constraints), 1)
        self.assertEqual(len(spec.actor_dispositions), 1)

    def test_consistency(self):
        raw = "Oil stays above $95 a barrel through 2029."
        spec = self.compiler.compile("Test_2", "Consistency Check", raw)
        finding = self.tester.test_consistency(spec)
        self.assertEqual(finding.status, "PASS")


if __name__ == "__main__":
    unittest.main()
