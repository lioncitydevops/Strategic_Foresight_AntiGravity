"""
Bayesian Partially Observable Stochastic Game (POSG) formulation (Section 2.2).
Players I = {USA, CHN, EU}.
State s_t = (G_t, x_t).
Payoffs: u_i(s; theta_i) = theta_i^T phi_i(s).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.core.state import WorldState
from src.core.actions import Action
from src.core.simulator import TransitionSimulator


@dataclass
class ActorPayoffFeatures:
    """Interpretable feature vector phi_i(s)."""
    growth: float
    security: float
    domestic_approval: float
    alliance_cohesion: float

    def to_array(self) -> np.ndarray:
        return np.array([self.growth, self.security, self.domestic_approval, self.alliance_cohesion])


class BayesianPOSG:
    """Bayesian POSG environment for strategic actors."""

    def __init__(
        self,
        actors: Optional[List[str]] = None,
        simulator: Optional[TransitionSimulator] = None,
        discount_factor: float = 0.95,
        horizon: int = 8
    ):
        self.actors = actors or ["USA", "CHN", "EU"]
        self.simulator = simulator or TransitionSimulator()
        self.gamma = discount_factor
        self.horizon = horizon

        # Action spaces for each player
        self.action_spaces: Dict[str, List[Action]] = {
            "USA": [
                Action("USA", "Hold", "Global"),
                Action("USA", "ImposeTariff", "Semiconductors", delta_weight=15.0),
                Action("USA", "ImposeTariff", "Semiconductors", delta_weight=30.0),
                Action("USA", "TechExportControl", "TechEntity_Alpha", delta_weight=0.5),
                Action("USA", "SignAgreement", "EU", delta_weight=0.8),
            ],
            "CHN": [
                Action("CHN", "Hold", "Global"),
                Action("CHN", "ImposeTariff", "CriticalMinerals", delta_weight=15.0),
                Action("CHN", "ImposeTariff", "CriticalMinerals", delta_weight=30.0),
                Action("CHN", "Sanction", "TechEntity_Alpha", delta_weight=0.4),
                Action("CHN", "SignAgreement", "EU", delta_weight=0.6),
            ],
            "EU": [
                Action("EU", "Hold", "Global"),
                Action("EU", "SignAgreement", "USA", delta_weight=0.8),
                Action("EU", "SignAgreement", "CHN", delta_weight=0.6),
                Action("EU", "ImposeTariff", "Automotive", delta_weight=10.0),
            ],
        }

    def compute_features(self, state: WorldState, actor: str) -> ActorPayoffFeatures:
        """Extracts interpretable linear utility features phi_i(s)."""
        growth = state.get_indicator(f"gdp_growth_{actor}", default=2.0)
        # Security is inverse of tension with main rivals
        rival = "CHN" if actor == "USA" else ("USA" if actor == "CHN" else "CHN")
        tension = state.graph.get_weight(actor, "DiplomaticTension", rival, default=0.2)
        security = 1.0 - tension
        # Approval proxy
        approval = growth * 0.4 + security * 0.6
        # Alliance cohesion
        cohesion = 0.85 if actor in ["USA", "EU"] else 0.70

        return ActorPayoffFeatures(
            growth=growth,
            security=security,
            domestic_approval=approval,
            alliance_cohesion=cohesion
        )

    def compute_utility(self, state: WorldState, actor: str, theta: np.ndarray) -> float:
        """Payoff u_i(s; theta_i) = theta_i^T phi_i(s)."""
        feats = self.compute_features(state, actor).to_array()
        return float(np.dot(theta, feats))
