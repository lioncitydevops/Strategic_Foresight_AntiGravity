"""
World state representation: s_t = (G_t, x_t).
Combines sparse temporal knowledge graph G_t with exogenous continuous indicators x_t.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import copy
from src.core.tkg import SparseTKG


@dataclass
class WorldState:
    """Full world state s_t = (G_t, x_t) at time step t."""
    time_step: int
    graph: SparseTKG
    exogenous_indicators: Dict[str, float] = field(default_factory=dict)

    def clone(self, new_time_step: Optional[int] = None) -> "WorldState":
        t = self.time_step if new_time_step is None else new_time_step
        return WorldState(
            time_step=t,
            graph=self.graph.clone(new_time_step=t),
            exogenous_indicators=dict(self.exogenous_indicators),
        )

    def get_indicator(self, key: str, default: float = 0.0) -> float:
        return self.exogenous_indicators.get(key, default)

    def set_indicator(self, key: str, value: float) -> None:
        self.exogenous_indicators[key] = float(value)

    def to_dict(self) -> Dict:
        return {
            "time_step": self.time_step,
            "graph": self.graph.to_dict(),
            "exogenous_indicators": {k: round(v, 4) for k, v in self.exogenous_indicators.items()},
        }
