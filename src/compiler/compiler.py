"""
Scenario Compiler (Section 4).
Parses assigned narrative scenarios into formal specification:
Sigma_n = <D_n, X_n, Theta_n, O_n, C_n>
Strictly separates interventions (premises) from conditioning events (end states).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from src.core.actions import Action


@dataclass
class ExogenousConstraint:
    """X_n: Constraint on exogenous driver variable path."""
    indicator: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    time_window: Tuple[int, int] = (0, 100)


@dataclass
class ActorDisposition:
    """Theta_n: Restriction of actor type prior p(theta) to convex region."""
    actor: str
    weight_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # e.g., {'security': (0.6, 1.0), 'growth': (0.0, 0.4)}


@dataclass
class ConditioningEvent:
    """C_n: Milestone or end state conditioned on trajectories tau."""
    condition_id: str
    target_variable: str  # e.g., fact (i, k, j) or exogenous indicator
    target_value: float
    comparison_op: str   # '>=', '<=', '==', 'in'
    time_step: int
    is_hard: bool = True
    penalty_kappa: float = 2.0  # soft penalty exp(-kappa * viol)

    def evaluate(self, current_val: float) -> Tuple[bool, float]:
        """Returns (is_satisfied, violation_magnitude)."""
        viol = 0.0
        if self.comparison_op == ">=":
            viol = max(0.0, self.target_value - current_val)
        elif self.comparison_op == "<=":
            viol = max(0.0, current_val - self.target_value)
        elif self.comparison_op == "==":
            viol = abs(current_val - self.target_value)
        sat = (viol < 1e-4)
        return sat, viol


@dataclass
class CausalClaimHypothesis:
    """Hypothesis to be tested in simulation, not imposed."""
    cause_action: Action
    expected_effect: str
    target_metric: str
    min_effect_size: float


@dataclass
class ScenarioSpec:
    """Formal specification Sigma_n = <D_n, X_n, Theta_n, O_n, C_n>."""
    scenario_id: str
    title: str
    raw_narrative: str
    interventions: List[Tuple[int, Action]] = field(default_factory=list)      # D_n
    driver_constraints: List[ExogenousConstraint] = field(default_factory=list) # X_n
    actor_dispositions: List[ActorDisposition] = field(default_factory=list)    # Theta_n
    ontology_extensions: List[Dict[str, Any]] = field(default_factory=list)     # O_n
    conditioning_events: List[ConditioningEvent] = field(default_factory=list)  # C_n
    causal_hypotheses: List[CausalClaimHypothesis] = field(default_factory=list)
    narrated_choices: List[Tuple[int, Action]] = field(default_factory=list)    # Choices for coherence test
    unmapped_sentences: List[str] = field(default_factory=list)
    approved_by_owner: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "interventions": [{"time": t, "action": a.name()} for t, a in self.interventions],
            "driver_constraints": [
                {"indicator": c.indicator, "min": c.min_value, "max": c.max_value, "window": c.time_window}
                for c in self.driver_constraints
            ],
            "actor_dispositions": [
                {"actor": d.actor, "bounds": d.weight_bounds} for d in self.actor_dispositions
            ],
            "conditioning_events": [
                {"id": c.condition_id, "var": c.target_variable, "val": c.target_value, "op": c.comparison_op, "time": c.time_step, "hard": c.is_hard}
                for c in self.conditioning_events
            ],
            "unmapped_sentences": self.unmapped_sentences,
            "approved_by_owner": self.approved_by_owner,
        }


class ScenarioCompiler:
    """Compiles unstructured scenario narratives into formal ScenarioSpec."""

    def compile(self, scenario_id: str, title: str, text: str) -> ScenarioSpec:
        spec = ScenarioSpec(scenario_id=scenario_id, title=title, raw_narrative=text)
        sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 3]

        for s in sentences:
            s_lower = s.lower()
            # Premise detection -> do(D_n)
            if "begins in" in s_lower or "imposes" in s_lower or "initiates" in s_lower:
                if "tariff" in s_lower:
                    act = Action("USA", "ImposeTariff", "Semiconductors", delta_weight=35.0)
                    spec.interventions.append((1, act))
                elif "sanction" in s_lower or "blockade" in s_lower:
                    act = Action("CHN", "Sanction", "TechEntity_Alpha", delta_weight=0.8)
                    spec.interventions.append((2, act))
                else:
                    spec.unmapped_sentences.append(s)

            # Driver constraint -> X_n
            elif "stays above" in s_lower or "exceeds" in s_lower or "oil" in s_lower:
                if "oil" in s_lower or "barrel" in s_lower:
                    spec.driver_constraints.append(
                        ExogenousConstraint(indicator="brent_crude", min_value=95.0, time_window=(1, 10))
                    )
                else:
                    spec.driver_constraints.append(
                        ExogenousConstraint(indicator="gdp_growth_USA", min_value=1.0, time_window=(1, 10))
                    )

            # Actor disposition -> Theta_n
            elif "puts" in s_lower or "prioritizes" in s_lower or "ahead of" in s_lower:
                # e.g., "Beijing puts self-sufficiency ahead of growth"
                actor = "CHN" if "beijing" in s_lower or "china" in s_lower else "USA"
                spec.actor_dispositions.append(
                    ActorDisposition(actor=actor, weight_bounds={"security": (0.6, 1.0), "growth": (0.0, 0.4)})
                )

            # Conditioning event (Milestone / End state) -> C_n
            elif "by " in s_lower or "reaches" in s_lower or "sources under" in s_lower or "target" in s_lower:
                spec.conditioning_events.append(
                    ConditioningEvent(
                        condition_id=f"end_state_{len(spec.conditioning_events)+1}",
                        target_variable="chip_stress_index",
                        target_value=0.75,
                        comparison_op=">=",
                        time_step=6,
                        is_hard=True
                    )
                )

            # Causal claim -> Hypothesis to test
            elif "because of" in s_lower or "leads to" in s_lower:
                act = Action("USA", "ImposeTariff", "Semiconductors", delta_weight=30.0)
                spec.causal_hypotheses.append(
                    CausalClaimHypothesis(
                        cause_action=act,
                        expected_effect="Domestic approval decline",
                        target_metric="gdp_growth_USA",
                        min_effect_size=-0.3
                    )
                )

            # Narrated actor choice
            elif "decides to" in s_lower or "responds with" in s_lower:
                act = Action("CHN", "ImposeTariff", "CriticalMinerals", delta_weight=25.0)
                spec.narrated_choices.append((2, act))

            else:
                spec.unmapped_sentences.append(s)

        return spec
