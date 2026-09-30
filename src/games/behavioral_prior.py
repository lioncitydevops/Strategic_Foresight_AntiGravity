"""
Behavioral Sequence Prior (Section 7.1).
Imitates historical behavior from past event sequences without equilibrium guarantees.
Linear-cost graph embedding over |F_t|.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from src.core.state import WorldState
from src.core.actions import Action
from src.games.posg import BayesianPOSG


class BehavioralPriorPolicy:
    """Policy that imitates historical precedent."""

    def __init__(self, actor: str, action_space: List[Action], temperature: float = 1.0):
        self.actor = actor
        self.action_space = action_space
        self.temperature = temperature
        # Historical action frequency weights
        self.prior_weights = np.ones(len(action_space))
        # Prior favors status-quo / moderate actions historically
        for idx, act in enumerate(action_space):
            if act.action_type == "Hold":
                self.prior_weights[idx] = 3.5
            elif act.action_type in ["SignAgreement", "ImposeTariff"] and act.delta_weight <= 15.0:
                self.prior_weights[idx] = 2.0
            else:
                self.prior_weights[idx] = 0.8

    def action_probabilities(self, state: WorldState) -> np.ndarray:
        """Computes action distribution pi_prior(a | s). Cost is linear in |F_t|."""
        # Simple graph-conditioned modulation
        tension = state.graph.get_weight(self.actor, "DiplomaticTension", "CHN" if self.actor != "CHN" else "USA", default=0.2)
        logits = np.log(self.prior_weights)

        # In high tension, probability of retaliatory tariffs/sanctions increases
        for idx, act in enumerate(self.action_space):
            if act.action_type in ["ImposeTariff", "Sanction", "TechExportControl"]:
                logits[idx] += tension * 1.5
            elif act.action_type == "SignAgreement":
                logits[idx] -= tension * 1.2

        scaled = logits / max(self.temperature, 1e-4)
        exp_logits = np.exp(scaled - np.max(scaled))
        return exp_logits / np.sum(exp_logits)

    def select_action(self, state: WorldState, rng: Optional[np.random.Generator] = None) -> Action:
        probs = self.action_probabilities(state)
        gen = rng or np.random.default_rng()
        idx = gen.choice(len(self.action_space), p=probs)
        return self.action_space[idx]
