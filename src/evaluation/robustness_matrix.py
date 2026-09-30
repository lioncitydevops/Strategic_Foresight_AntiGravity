"""
Strategy x Scenario Robustness Matrix (Sections 8.1 - 8.3).
Compares candidate strategies across assigned scenarios without imposing scenario probabilities.
Computes median outcome, 10th percentile downside, and minimax regret.
Supports adaptive signpost triggers.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.core.state import WorldState
from src.core.actions import Action
from src.core.simulator import TransitionSimulator
from src.compiler.compiler import ScenarioSpec
from src.games.posg import BayesianPOSG


@dataclass
class RobustnessCell:
    strategy_name: str
    scenario_id: str
    median_value: float
    downside_p10: float
    regret: float
    trajectory_trace: List[Dict[str, Any]]


@dataclass
class RobustnessMatrixResult:
    focal_actor: str
    candidate_strategies: List[str]
    scenarios: List[str]
    matrix: Dict[Tuple[str, str], RobustnessCell]  # (strat, scenario) -> Cell
    worst_case_regrets: Dict[str, float]          # strat -> max_n Reg_n(strat)
    dominated_strategies: List[str]
    recommended_strategy: str


class RobustnessEvaluator:
    """Evaluates candidate strategies across assigned scenario futures."""

    def __init__(
        self,
        focal_actor: str = "USA",
        simulator: Optional[TransitionSimulator] = None,
        game: Optional[BayesianPOSG] = None
    ):
        self.focal_actor = focal_actor
        self.simulator = simulator or TransitionSimulator()
        self.game = game or BayesianPOSG()

    def simulate_strategy_in_scenario(
        self,
        strategy_actions: List[Action],
        scenario: ScenarioSpec,
        base_state: WorldState,
        num_replications: int = 15,
        horizon: int = 6
    ) -> Tuple[float, float, List[Dict[str, Any]], List[float]]:
        """
        Runs replications of strategy inside scenario world distribution p_n.
        Returns (median, p10, trace, all_payoffs).
        """
        payoffs = []
        best_trace = []

        # Interventions in scenario
        interventions_map = {}
        for t, act in scenario.interventions:
            interventions_map.setdefault(t, []).append(act)

        for rep in range(num_replications):
            s = base_state.clone()
            rep_payoff = 0.0
            discount = 1.0
            trace = []

            for step in range(horizon):
                t_cur = base_state.time_step + step + 1
                # Interventions from scenario stipulation
                sc_acts = interventions_map.get(t_cur, [])

                # Focal actor's strategy action
                strat_act = strategy_actions[min(step, len(strategy_actions) - 1)]

                # Rival reaction (moderate or aggressive)
                rival_act = Action("CHN", "Hold", "Global")
                if sc_acts:
                    rival_act = Action("CHN", "ImposeTariff", "CriticalMinerals", delta_weight=15.0)

                joint_acts = [strat_act, rival_act] + sc_acts
                s = self.simulator.step(s, joint_acts)

                # Payoff
                growth = s.get_indicator(f"gdp_growth_{self.focal_actor}", 2.0)
                tension = s.graph.get_weight(self.focal_actor, "DiplomaticTension", "CHN", default=0.2)
                step_u = growth * 0.5 + (1.0 - tension) * 0.5
                rep_payoff += discount * step_u
                discount *= 0.95

                if rep == 0:
                    trace.append({
                        "step": t_cur,
                        "actions": [a.name() for a in joint_acts],
                        "gdp_growth": round(growth, 2),
                        "tension": round(tension, 2),
                        "payoff": round(step_u, 2)
                    })

            payoffs.append(rep_payoff)
            if rep == 0:
                best_trace = trace

        p_arr = np.array(payoffs)
        median_val = float(np.median(p_arr))
        p10_val = float(np.percentile(p_arr, 10))
        return median_val, p10_val, best_trace, payoffs

    def evaluate(
        self,
        candidate_strategies: Dict[str, List[Action]],
        scenarios: List[ScenarioSpec],
        base_state: WorldState
    ) -> RobustnessMatrixResult:
        """Constructs strategy x scenario robustness matrix and calculates regret."""
        matrix: Dict[Tuple[str, str], RobustnessCell] = {}
        strat_names = list(candidate_strategies.keys())
        scen_ids = [s.scenario_id for s in scenarios]

        expected_values = {sid: {} for sid in scen_ids}

        # Step 1: Simulate all cells
        for sid, sc in zip(scen_ids, scenarios):
            for sname, s_acts in candidate_strategies.items():
                med, p10, trace, all_p = self.simulate_strategy_in_scenario(s_acts, sc, base_state)
                expected_values[sid][sname] = med
                matrix[(sname, sid)] = RobustnessCell(
                    strategy_name=sname,
                    scenario_id=sid,
                    median_value=round(med, 3),
                    downside_p10=round(p10, 3),
                    regret=0.0,  # will update below
                    trajectory_trace=trace
                )

        # Step 2: Compute regret Reg_n(d) = max_{d'} E[V(d')] - E[V(d)]
        worst_case_regrets = {sname: 0.0 for sname in strat_names}

        for sid in scen_ids:
            best_in_scenario = max(expected_values[sid].values())
            for sname in strat_names:
                regret_val = best_in_scenario - expected_values[sid][sname]
                matrix[(sname, sid)].regret = round(regret_val, 3)
                if regret_val > worst_case_regrets[sname]:
                    worst_case_regrets[sname] = round(regret_val, 3)

        # Step 3: Flag dominated strategies
        dominated = []
        for s1 in strat_names:
            is_dom = False
            for s2 in strat_names:
                if s1 == s2:
                    continue
                # If s2 is strictly better than s1 in every scenario
                if all(expected_values[sid][s2] > expected_values[sid][s1] for sid in scen_ids):
                    is_dom = True
                    break
            if is_dom:
                dominated.append(s1)

        # Recommended strategy: minimizes worst-case regret (minimax regret)
        recommended = min(worst_case_regrets.keys(), key=lambda s: worst_case_regrets[s])

        return RobustnessMatrixResult(
            focal_actor=self.focal_actor,
            candidate_strategies=strat_names,
            scenarios=scen_ids,
            matrix=matrix,
            worst_case_regrets=worst_case_regrets,
            dominated_strategies=dominated,
            recommended_strategy=recommended
        )
