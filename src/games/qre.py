"""
Logit Quantal Response Equilibrium (QRE) and Interventional Actor Relevance (Sections 7.3 & 7.4).
sigma_i(pi_i) proportional to exp(lambda * U_i(pi_i, sigma_{-i})).
Cross-actor relevance: Delta_ij = U_i(pi) - U_i(pi_i, pi_j^{base}, pi_{-ij}).
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class QREResult:
    rationality_lambda: float
    actor_mixtures: Dict[str, np.ndarray]  # actor -> probability distribution over policy population
    expected_payoffs: Dict[str, float]
    interventional_relevance: Dict[str, Dict[str, float]]  # Delta_ij: actor_i -> actor_j -> payoff change


class LogitQRESolver:
    """Computes logit quantal response equilibria on the empirical meta-game table."""

    def __init__(self, max_iter: int = 50, tol: float = 1e-5):
        self.max_iter = max_iter
        self.tol = tol

    def solve_qre(
        self,
        meta_table: np.ndarray,
        actors: List[str],
        rationality_lambda: float = 2.5
    ) -> QREResult:
        """
        Solves for QRE fixed point by damped Picard iteration along the principal branch.
        """
        num_actors = len(actors)
        p_lens = [meta_table.shape[i] for i in range(num_actors)]

        # Start from uniform distribution (lambda = 0)
        sigma = [np.ones(p_lens[i]) / p_lens[i] for i in range(num_actors)]

        if rationality_lambda <= 1e-4:
            # Exactly uniform
            exp_payoffs = {actors[i]: float(np.mean(meta_table[..., i])) for i in range(num_actors)}
            return QREResult(
                rationality_lambda=0.0,
                actor_mixtures={actors[i]: sigma[i] for i in range(num_actors)},
                expected_payoffs=exp_payoffs,
                interventional_relevance={a: {b: 0.0 for b in actors if b != a} for a in actors}
            )

        # Tracing principal branch via damped updates
        alpha = 0.25  # damping factor
        for it in range(self.max_iter):
            new_sigma = []
            max_diff = 0.0

            for i in range(num_actors):
                # Expected utility for each pure policy of player i given others' current mixtures
                expected_u = np.zeros(p_lens[i])
                for pol_idx in range(p_lens[i]):
                    sub_slice = [slice(None)] * num_actors
                    sub_slice[i] = pol_idx
                    # Slice of meta-table for this pure action
                    sub_table = meta_table[tuple(sub_slice) + (i,)]
                    # Average or expectation under opponent mixtures
                    expected_u[pol_idx] = float(np.mean(sub_table))

                scaled = rationality_lambda * expected_u
                exp_scaled = np.exp(scaled - np.max(scaled))
                target_prob = exp_scaled / np.sum(exp_scaled)

                damped_prob = (1.0 - alpha) * sigma[i] + alpha * target_prob
                max_diff = max(max_diff, float(np.max(np.abs(damped_prob - sigma[i]))))
                new_sigma.append(damped_prob)

            sigma = new_sigma
            if max_diff < self.tol:
                break

        # Compute equilibrium expected payoffs
        exp_payoffs = {}
        for i in range(num_actors):
            exp_payoffs[actors[i]] = float(np.mean(meta_table[..., i]))

        # Section 7.4: Interventional Cross-Actor Relevance Delta_ij
        relevance: Dict[str, Dict[str, float]] = {a: {} for a in actors}
        for i, a_i in enumerate(actors):
            u_equil = exp_payoffs[a_i]
            for j, a_j in enumerate(actors):
                if i == j:
                    continue
                # Replace player j's policy with baseline (pure action 0: status quo / hold)
                sub_slice = [slice(None)] * num_actors
                sub_slice[j] = 0  # baseline policy
                u_counterfactual = float(np.mean(meta_table[tuple(sub_slice) + (i,)]))
                delta_ij = u_equil - u_counterfactual
                relevance[a_i][a_j] = round(delta_ij, 4)

        return QREResult(
            rationality_lambda=rationality_lambda,
            actor_mixtures={actors[i]: sigma[i] for i in range(num_actors)},
            expected_payoffs=exp_payoffs,
            interventional_relevance=relevance
        )
