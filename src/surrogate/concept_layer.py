"""
Concept Layer for Interpretable Policy Surrogate (Section 9.1).
Maps high-dimensional state s_t to p expert-defined auditable concepts g(s) in R^p.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from src.core.state import WorldState


@dataclass
class StrategicConcepts:
    """Auditable concept vector g(s) in R^p."""
    trade_exposure_rival: float       # Bilateral trade exposure to rival
    rival_avg_tariff: float           # Average tariff rate imposed by rival
    diplomatic_tension: float         # Bilateral diplomatic tension
    domestic_growth: float            # Domestic GDP growth rate
    brent_crude_stress: float         # Exogenous energy stress
    time_to_horizon: float            # tau = H - t (treated as standard input)

    def to_array(self) -> np.ndarray:
        return np.array([
            self.trade_exposure_rival,
            self.rival_avg_tariff,
            self.diplomatic_tension,
            self.domestic_growth,
            self.brent_crude_stress,
            self.time_to_horizon,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "trade_exposure",
            "rival_tariff",
            "tension",
            "gdp_growth",
            "energy_stress",
            "tau_horizon",
        ]


class ConceptLayer:
    """Extracts interpretable concept vector g(s) from WorldState."""

    def __init__(self, focal_actor: str = "USA", rival_actor: str = "CHN", max_horizon: int = 8):
        self.focal_actor = focal_actor
        self.rival_actor = rival_actor
        self.max_horizon = max_horizon

    def extract(self, state: WorldState) -> StrategicConcepts:
        g = state.graph
        rival_tariff = g.get_weight(self.rival_actor, "Tariff", "CriticalMinerals", default=0.0) / 100.0
        tension = g.get_weight(self.focal_actor, "DiplomaticTension", self.rival_actor, default=0.2)
        growth = state.get_indicator(f"gdp_growth_{self.focal_actor}", default=2.0)
        brent = state.get_indicator("brent_crude", default=80.0)
        energy_stress = (brent - 70.0) / 30.0

        # Trade exposure proxy
        exposure = 0.65 - 0.3 * rival_tariff
        tau = max(0.0, float(self.max_horizon - state.time_step))

        return StrategicConcepts(
            trade_exposure_rival=float(exposure),
            rival_avg_tariff=float(rival_tariff),
            diplomatic_tension=float(tension),
            domestic_growth=float(growth),
            brent_crude_stress=float(energy_stress),
            time_to_horizon=float(tau)
        )
