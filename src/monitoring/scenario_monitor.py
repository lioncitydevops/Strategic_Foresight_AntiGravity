"""
Scenario Monitor & Bayes Factors (Section 6.3).
Maintains parallel particle filters for assigned scenarios Sigma_n to calculate
marginal likelihoods p_hat(o_{1:t} | Sigma_n) and Bayes factors.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from src.core.state import WorldState
from src.compiler.compiler import ScenarioSpec
from src.monitoring.particle_filter import ParticleFilter
from src.extraction.observation_model import Claim


@dataclass
class ScenarioMonitoringStatus:
    scenario_id: str
    title: str
    marginal_likelihood: float
    bayes_factor_vs_unconstrained: float
    posterior_probability: float
    trajectory_sample: Dict[str, float]


class ScenarioMonitor:
    """Monitors multiple assigned scenarios simultaneously against unfolding evidence."""

    def __init__(
        self,
        scenarios: List[ScenarioSpec],
        base_state: WorldState,
        actors: List[str],
        num_particles: int = 100
    ):
        self.scenarios = scenarios
        self.actors = actors
        self.filters: Dict[str, ParticleFilter] = {}
        # Unconstrained baseline filter
        self.unconstrained_filter = ParticleFilter(num_particles=num_particles)
        self.unconstrained_filter.initialize(base_state, actors)

        for s in scenarios:
            pf = ParticleFilter(num_particles=num_particles)
            pf.initialize(base_state, actors)
            self.filters[s.scenario_id] = pf

        self.cum_log_marginals: Dict[str, float] = {s.scenario_id: 0.0 for s in scenarios}
        self.unconstrained_log_marginal: float = 0.0

    def step(self, observed_claims: List[Claim], observed_actions: list) -> Dict[str, ScenarioMonitoringStatus]:
        """Updates each filter with current observations and re-evaluates scenario odds."""
        # Update unconstrained baseline
        avg_unnorm_base = self.unconstrained_filter.update(observed_claims, observed_actions)
        self.unconstrained_log_marginal += np.log(max(avg_unnorm_base, 1e-12))

        statuses = {}
        # Update scenario-specific filters
        log_likes = {}
        for s in self.scenarios:
            pf = self.filters[s.scenario_id]
            # Interventions at current time step
            avg_unnorm = pf.update(observed_claims, observed_actions)
            self.cum_log_marginals[s.scenario_id] += np.log(max(avg_unnorm, 1e-12))
            log_likes[s.scenario_id] = self.cum_log_marginals[s.scenario_id]

        # Calculate normalized posterior scenario probabilities (assuming equal scenario priors)
        max_log = max(log_likes.values()) if log_likes else 0.0
        exp_likes = {sid: np.exp(np.clip(val - max_log, -50, 50)) for sid, val in log_likes.items()}
        sum_exp = sum(exp_likes.values()) or 1.0

        for s in self.scenarios:
            sid = s.scenario_id
            post_prob = float(exp_likes[sid] / sum_exp)
            # Bayes factor vs unconstrained model
            log_bf = self.cum_log_marginals[sid] - self.unconstrained_log_marginal
            bf = float(np.exp(np.clip(log_bf, -20.0, 20.0)))

            pf = self.filters[sid]
            summary = pf.get_latent_summary()

            statuses[sid] = ScenarioMonitoringStatus(
                scenario_id=sid,
                title=s.title,
                marginal_likelihood=float(np.exp(np.clip(self.cum_log_marginals[sid], -50, 50))),
                bayes_factor_vs_unconstrained=bf,
                posterior_probability=round(post_prob, 4),
                trajectory_sample=summary
            )

        return statuses
