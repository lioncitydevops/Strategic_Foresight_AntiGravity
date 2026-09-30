"""
Diagnostic Signpost Identifier (Section 6.3).
Finds observable variables with maximum divergence across scenario distributions
over a forward horizon (6-12 months) to serve as decision triggers.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.core.state import WorldState
from src.core.simulator import TransitionSimulator
from src.compiler.compiler import ScenarioSpec


@dataclass
class DiagnosticSignpost:
    variable_name: str
    variable_type: str   # 'ExogenousIndicator' or 'GraphFact'
    divergence_score: float  # Variance across scenarios
    scenario_predictions: Dict[str, float]  # scenario_id -> expected value
    recommended_trigger_threshold: float
    trigger_direction: str  # 'above' or 'below'


class SignpostIdentifier:
    """Computes forward diagnostic signposts separating candidate scenarios."""

    def __init__(self, simulator: Optional[TransitionSimulator] = None):
        self.simulator = simulator or TransitionSimulator()

    def identify_signposts(
        self,
        scenarios: List[ScenarioSpec],
        base_state: WorldState,
        forward_horizon: int = 4
    ) -> List[DiagnosticSignpost]:
        """
        Rolls out scenarios forward and identifies indicators with highest variance across scenario end-points.
        """
        # Variables to evaluate
        test_indicators = ["gdp_growth_USA", "gdp_growth_CHN", "brent_crude", "chip_stress_index"]
        scenario_projections: Dict[str, Dict[str, float]] = {}

        for sc in scenarios:
            s_rollout = base_state.clone()
            intervention_map = {}
            for t, act in sc.interventions:
                intervention_map.setdefault(t, []).append(act)

            for step in range(forward_horizon):
                t_cur = base_state.time_step + step + 1
                acts = intervention_map.get(t_cur, [])
                s_rollout = self.simulator.step(s_rollout, acts)

            scenario_projections[sc.scenario_id] = {
                ind: s_rollout.get_indicator(ind, default=0.0) for ind in test_indicators
            }
            # Also capture graph fact
            tension = s_rollout.graph.get_weight("USA", "DiplomaticTension", "CHN", default=0.2)
            scenario_projections[sc.scenario_id]["USA_CHN_Tension"] = tension

        signposts = []
        all_vars = test_indicators + ["USA_CHN_Tension"]

        for var_name in all_vars:
            values = [scenario_projections[sc.scenario_id][var_name] for sc in scenarios]
            var_div = float(np.var(values))
            mean_val = float(np.mean(values))
            min_sc = min(scenario_projections.keys(), key=lambda sid: scenario_projections[sid][var_name])
            max_sc = max(scenario_projections.keys(), key=lambda sid: scenario_projections[sid][var_name])

            # Trigger threshold sits at midpoint between extremes
            threshold = (scenario_projections[min_sc][var_name] + scenario_projections[max_sc][var_name]) / 2.0
            direction = "above" if scenario_projections[max_sc][var_name] > threshold else "below"

            signposts.append(DiagnosticSignpost(
                variable_name=var_name,
                variable_type="GraphFact" if "Tension" in var_name else "ExogenousIndicator",
                divergence_score=round(var_div, 4),
                scenario_predictions={sc.scenario_id: round(scenario_projections[sc.scenario_id][var_name], 2) for sc in scenarios},
                recommended_trigger_threshold=round(threshold, 2),
                trigger_direction=direction
            ))

        # Sort descending by divergence score
        signposts.sort(key=lambda s: s.divergence_score, reverse=True)
        return signposts
