"""
Typed graph edits representing actions A_i for strategic players.
Actions: impose, raise, lower, remove tariff; join, exit agreement; sanction; or hold.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.core.state import WorldState


@dataclass(frozen=True)
class Action:
    actor: str
    action_type: str  # 'ImposeTariff', 'RemoveTariff', 'Sanction', 'LiftSanctions', 'SignAgreement', 'ExitAgreement', 'Hold'
    target: str
    parameter: Optional[str] = None  # e.g., sector name or agreement name
    delta_weight: float = 0.0        # intensity value or delta

    def name(self) -> str:
        param_str = f"_{self.parameter}" if self.parameter else ""
        return f"{self.actor}:{self.action_type}{param_str}:{self.target}({self.delta_weight:.1f})"

    def apply(self, state: WorldState) -> None:
        """Applies the typed graph edit directly to the world state's graph."""
        g = state.graph
        if self.action_type == "Hold":
            return

        elif self.action_type == "ImposeTariff":
            # Target is sector, or target entity with parameter=sector
            current = g.get_weight(self.actor, "Tariff", self.target, default=0.0)
            new_w = min(100.0, max(0.0, current + self.delta_weight))
            g.add_fact(self.actor, "Tariff", self.target, new_w)

        elif self.action_type == "RemoveTariff":
            g.remove_fact(self.actor, "Tariff", self.target)

        elif self.action_type == "Sanction":
            current = g.get_weight(self.actor, "Sanctions", self.target, default=0.0)
            new_w = min(1.0, max(0.0, current + self.delta_weight))
            g.add_fact(self.actor, "Sanctions", self.target, new_w)

        elif self.action_type == "LiftSanctions":
            g.remove_fact(self.actor, "Sanctions", self.target)

        elif self.action_type == "SignAgreement":
            g.add_fact(self.actor, "TradeAgreement", self.target, self.delta_weight)
            g.add_fact(self.target, "TradeAgreement", self.actor, self.delta_weight)

        elif self.action_type == "ExitAgreement":
            g.remove_fact(self.actor, "TradeAgreement", self.target)
            g.remove_fact(self.target, "TradeAgreement", self.actor)

        elif self.action_type == "TechExportControl":
            current = g.get_weight(self.actor, "TechExportControl", self.target, default=0.0)
            new_w = min(1.0, max(0.0, current + self.delta_weight))
            g.add_fact(self.actor, "TechExportControl", self.target, new_w)

        elif self.action_type == "DiplomaticTension":
            current = g.get_weight(self.actor, "DiplomaticTension", self.target, default=0.0)
            new_w = min(1.0, max(0.0, current + self.delta_weight))
            g.add_fact(self.actor, "DiplomaticTension", self.target, new_w)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor,
            "action_type": self.action_type,
            "target": self.target,
            "parameter": self.parameter,
            "delta_weight": self.delta_weight,
            "display": self.name(),
        }
