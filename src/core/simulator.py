"""
Stochastic simulator P_psi.
Applies graph edits, propagates shocks along graph edges,
and updates exogenous indicators x_t.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.core.state import WorldState
from src.core.actions import Action


@dataclass
class SimulatorParams:
    """Calibrated parameters psi for graph shock propagation and exogenous dynamics."""
    tariff_growth_drag: float = 0.04        # Growth reduction per 10% tariff
    tariff_inflation_pass: float = 0.03     # Inflation increase per 10% tariff
    sanctions_impact: float = 0.08          # Impact on target sector and GDP
    agreement_boost: float = 0.05           # Growth boost from trade agreements
    tension_decay: float = 0.90             # Baseline autoregression for tensions
    shock_sigma: float = 0.02               # Volatility of exogenous shocks xi


class TransitionSimulator:
    """Stochastic simulator P_psi for state transitions."""

    def __init__(self, params: Optional[SimulatorParams] = None, seed: Optional[int] = None):
        self.params = params or SimulatorParams()
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        current_state: WorldState,
        actions: List[Action],
        exogenous_shock: Optional[Dict[str, float]] = None
    ) -> WorldState:
        """
        Executes one transition step s_{t+1} ~ P_psi(s_t, {a_i}, xi_t).
        1. Clones state and advances time step.
        2. Applies typed graph actions.
        3. Propagates shocks along graph edges.
        4. Updates exogenous indicators.
        """
        next_state = current_state.clone(new_time_step=current_state.time_step + 1)
        g = next_state.graph

        # Step 2: Apply actions
        for act in actions:
            act.apply(next_state)

        # Step 3: Edge-level shock propagation
        # Collect tariffs
        total_tariffs = 0.0
        us_chn_tariff = g.get_weight("USA", "Tariff", "Semiconductors", default=0.0)
        chn_us_tariff = g.get_weight("CHN", "Tariff", "CriticalMinerals", default=0.0)
        total_tariffs = (us_chn_tariff + chn_us_tariff) / 100.0

        # Sanctions and controls
        sanctions_level = g.get_weight("USA", "Sanctions", "TechEntity_Alpha", default=0.0)
        tech_ctrl = g.get_weight("USA", "TechExportControl", "TechEntity_Alpha", default=0.0)

        # Agreements
        agreements = len(g.get_facts_by_relation("TradeAgreement"))

        # Tension propagation
        current_tension = g.get_weight("USA", "DiplomaticTension", "CHN", default=0.2)
        induced_tension = 0.15 * total_tariffs + 0.25 * sanctions_level + 0.20 * tech_ctrl
        new_tension = min(1.0, max(0.0, current_tension * self.params.tension_decay + induced_tension * 0.1))
        g.add_fact("USA", "DiplomaticTension", "CHN", new_tension)
        g.add_fact("CHN", "DiplomaticTension", "USA", new_tension)

        # Step 4: Update exogenous indicators x_{t+1}
        cur_growth_us = next_state.get_indicator("gdp_growth_USA", default=2.2)
        cur_growth_chn = next_state.get_indicator("gdp_growth_CHN", default=4.5)
        cur_growth_eu = next_state.get_indicator("gdp_growth_EU", default=1.2)
        cur_brent = next_state.get_indicator("brent_crude", default=80.0)
        cur_chip_idx = next_state.get_indicator("chip_stress_index", default=0.2)

        # Stochastic shock xi
        shock = exogenous_shock or {}
        xi_us = shock.get("gdp_growth_USA", self.rng.normal(0, self.params.shock_sigma))
        xi_chn = shock.get("gdp_growth_CHN", self.rng.normal(0, self.params.shock_sigma))
        xi_oil = shock.get("brent_crude", self.rng.normal(0, 1.5))

        # Dynamic updates
        drag_us = self.params.tariff_growth_drag * (us_chn_tariff / 10.0)
        drag_chn = self.params.tariff_growth_drag * (chn_us_tariff / 10.0) + self.params.sanctions_impact * sanctions_level
        boost_trade = self.params.agreement_boost * (agreements / 4.0)

        next_state.set_indicator("gdp_growth_USA", cur_growth_us - drag_us + boost_trade + xi_us)
        next_state.set_indicator("gdp_growth_CHN", cur_growth_chn - drag_chn + boost_trade + xi_chn)
        next_state.set_indicator("gdp_growth_EU", cur_growth_eu - 0.5 * drag_us + xi_us * 0.5)
        next_state.set_indicator("brent_crude", max(40.0, cur_brent + xi_oil + (new_tension * 5.0)))
        next_state.set_indicator("chip_stress_index", min(1.0, max(0.0, cur_chip_idx + 0.3 * tech_ctrl + 0.2 * sanctions_level)))

        return next_state
