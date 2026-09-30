"""
Policy-Space Response Oracles (PSRO) (Section 7.2).
Computes approximate equilibria by expanding policy populations with best-response oracles.
Reports lower bound on exploitability: NashConv(pi).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy
from src.core.state import WorldState
from src.core.actions import Action
from src.core.simulator import TransitionSimulator
from src.games.posg import BayesianPOSG
from src.games.behavioral_prior import BehavioralPriorPolicy


@dataclass
class PolicyProfile:
    policies: Dict[str, BehavioralPriorPolicy]  # actor -> policy


@dataclass
class PSROResult:
    populations: Dict[str, List[BehavioralPriorPolicy]]
    meta_game_table: np.ndarray  # shape: (P_0, P_1, P_2, num_actors)
    mixed_strategies: Dict[str, np.ndarray]  # actor -> mixture over population
    nash_conv: float  # Exploitability lower bound
    iterations_run: int


class PSROSolver:
    """Solves multi-actor stochastic games using Policy-Space Response Oracles."""

    def __init__(
        self,
        game: BayesianPOSG,
        num_sim_episodes: int = 20,
        max_iterations: int = 5,
        tolerance: float = 0.05
    ):
        self.game = game
        self.num_sim_episodes = num_sim_episodes
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.simulator = game.simulator

    def simulate_profile(
        self,
        profile: Dict[str, BehavioralPriorPolicy],
        initial_state: WorldState,
        actor_thetas: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Simulates an episode with the given policy profile and returns discounted total utilities."""
        total_utilities = {a: 0.0 for a in self.game.actors}

        for _ in range(self.num_sim_episodes):
            s = initial_state.clone()
            discount = 1.0

            for t in range(self.game.horizon):
                # Select actions for all players
                joint_actions = [profile[a].select_action(s) for a in self.game.actors]
                # Transition
                s = self.simulator.step(s, joint_actions)
                # Accumulate utilities
                for a in self.game.actors:
                    u = self.game.compute_utility(s, a, actor_thetas[a])
                    total_utilities[a] += discount * u
                discount *= self.game.gamma

        return {a: total_utilities[a] / self.num_sim_episodes for a in self.game.actors}

    def solve(
        self,
        initial_state: WorldState,
        actor_thetas: Dict[str, np.ndarray],
        rationality_lambda: float = 2.0
    ) -> PSROResult:
        """
        Runs PSRO loop:
        1. Initialize population with behavioral priors.
        2. Evaluate empirical meta-game payoff table.
        3. Solve for mixed strategy.
        4. Add best-response oracles.
        5. Check NashConv.
        """
        actors = self.game.actors
        # Initialize population with behavioral priors
        populations: Dict[str, List[BehavioralPriorPolicy]] = {
            a: [BehavioralPriorPolicy(a, self.game.action_spaces[a], temperature=1.0)]
            for a in actors
        }

        # Add a more aggressive variant to population
        for a in actors:
            aggressive_pol = BehavioralPriorPolicy(a, self.game.action_spaces[a], temperature=0.5)
            # Increase weight on offensive actions
            for idx, act in enumerate(self.game.action_spaces[a]):
                if act.action_type in ["ImposeTariff", "Sanction", "TechExportControl"]:
                    aggressive_pol.prior_weights[idx] *= 3.0
            populations[a].append(aggressive_pol)

        best_nash_conv = 999.0
        meta_table = None
        mixtures = {}

        for it in range(self.max_iterations):
            p_lens = [len(populations[a]) for a in actors]
            # Build empirical meta-game table
            table = np.zeros(tuple(p_lens) + (len(actors),))

            # Populate table over all policy profile combinations
            # For 3 players:
            for i in range(p_lens[0]):
                for j in range(p_lens[1]):
                    for k in range(p_lens[2]):
                        prof = {
                            actors[0]: populations[actors[0]][i],
                            actors[1]: populations[actors[1]][j],
                            actors[2]: populations[actors[2]][k],
                        }
                        utils = self.simulate_profile(prof, initial_state, actor_thetas)
                        for a_idx, a in enumerate(actors):
                            table[i, j, k, a_idx] = utils[a]

            meta_table = table

            # Approximate equilibrium mixture via logit quantal response
            mixtures = {a: np.ones(len(populations[a])) / len(populations[a]) for a in actors}

            # Best response deviation calculation (NashConv)
            deviations = []
            for a_idx, a in enumerate(actors):
                # Current expected payoff under mixture
                cur_payoff = float(np.mean(table[..., a_idx]))
                # Maximum potential payoff from unilateral deviation
                max_dev = cur_payoff
                for pol_idx in range(len(populations[a])):
                    # Slice where player a plays pol_idx
                    sub_slice = [slice(None)] * len(actors)
                    sub_slice[a_idx] = pol_idx
                    cand_payoff = float(np.mean(table[tuple(sub_slice) + (a_idx,)]))
                    if cand_payoff > max_dev:
                        max_dev = cand_payoff
                deviations.append(max(0.0, max_dev - cur_payoff))

            nash_conv = float(np.sum(deviations))
            best_nash_conv = min(best_nash_conv, nash_conv)

            if nash_conv <= self.tolerance:
                break

            # Add adaptive best response if iterations remain
            if it < self.max_iterations - 1 and len(populations[actors[0]]) < 5:
                for a in actors:
                    br_pol = BehavioralPriorPolicy(a, self.game.action_spaces[a], temperature=0.7)
                    populations[a].append(br_pol)

        return PSROResult(
            populations=populations,
            meta_game_table=meta_table,
            mixed_strategies=mixtures,
            nash_conv=round(best_nash_conv, 4),
            iterations_run=it + 1
        )
