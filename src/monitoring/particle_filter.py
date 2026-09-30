"""
Belief Tracking via Particle Filter (Section 6).
Represents beliefs over joint latent variable z_t = (s_t, theta, psi) with K particles.
Includes Bayesian IRL action update, ESS monitoring, and Gilks-Berzuini MCMC rejuvenation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import copy
from src.core.state import WorldState
from src.core.actions import Action
from src.core.simulator import TransitionSimulator, SimulatorParams
from src.extraction.observation_model import Claim, ObservationModel


@dataclass
class ActorPreferences:
    """Hidden actor type weights theta_j."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        "growth": 0.4,
        "security": 0.3,
        "approval": 0.2,
        "cohesion": 0.1,
    })

    def clone(self) -> "ActorPreferences":
        return ActorPreferences(weights=dict(self.weights))

    def evaluate_utility(self, state: WorldState, actor: str) -> float:
        """Payoff u_i(s; theta_i) = theta_i^T phi_i(s)."""
        growth = state.get_indicator(f"gdp_growth_{actor}", default=2.0)
        tension = state.graph.get_weight(actor, "DiplomaticTension", "CHN" if actor != "CHN" else "USA", default=0.3)
        security = 1.0 - tension
        approval = growth * 0.5 + security * 0.5
        cohesion = 0.8

        phi = np.array([growth, security, approval, cohesion])
        th = np.array([
            self.weights.get("growth", 0.4),
            self.weights.get("security", 0.3),
            self.weights.get("approval", 0.2),
            self.weights.get("cohesion", 0.1),
        ])
        return float(np.dot(th, phi))


@dataclass
class Particle:
    """Hypothesis z_t = (s_t, theta, psi)."""
    state: WorldState
    actor_types: Dict[str, ActorPreferences]
    params: SimulatorParams
    weight: float = 1.0

    def clone(self) -> "Particle":
        return Particle(
            state=self.state.clone(),
            actor_types={k: v.clone() for k, v in self.actor_types.items()},
            params=copy.deepcopy(self.params),
            weight=self.weight
        )


class ParticleFilter:
    """Sequential Monte Carlo filter over joint latent space."""

    def __init__(
        self,
        num_particles: int = 150,
        obs_model: Optional[ObservationModel] = None,
        rationality_lambda: float = 2.0,
        seed: Optional[int] = 42
    ):
        self.num_particles = num_particles
        self.obs_model = obs_model or ObservationModel()
        self.rationality_lambda = rationality_lambda
        self.rng = np.random.default_rng(seed)
        self.particles: List[Particle] = []
        self.simulator = TransitionSimulator()
        self.marginal_likelihood_history: List[float] = []

    def initialize(self, base_state: WorldState, actors: List[str]) -> None:
        """Initializes particles from the prior p(theta, psi)."""
        self.particles = []
        init_weight = 1.0 / self.num_particles
        for _ in range(self.num_particles):
            # Sample preferences theta
            actor_types = {}
            for a in actors:
                sec_w = float(self.rng.uniform(0.1, 0.7))
                actor_types[a] = ActorPreferences(weights={
                    "growth": float(self.rng.uniform(0.2, 0.6)),
                    "security": sec_w,
                    "approval": float(self.rng.uniform(0.1, 0.3)),
                    "cohesion": float(self.rng.uniform(0.05, 0.2)),
                })
            # Sample simulator parameters psi
            sim_params = SimulatorParams(
                tariff_growth_drag=float(self.rng.normal(0.04, 0.008)),
                tariff_inflation_pass=float(self.rng.normal(0.03, 0.005)),
                shock_sigma=0.02
            )
            self.particles.append(Particle(
                state=base_state.clone(),
                actor_types=actor_types,
                params=sim_params,
                weight=init_weight
            ))

    def _myopic_action_log_prob(
        self,
        particle: Particle,
        observed_action: Action,
        actor: str
    ) -> float:
        """
        Bayesian IRL step: pi_j(a | s, theta_j) proportional to exp(lambda * Q_hat_j(s, a; theta_j)).
        Evaluates candidate actions using 1-step simulator lookahead.
        """
        pref = particle.actor_types.get(actor)
        if not pref:
            return 0.0

        # Candidate actions for comparison
        candidates = [
            observed_action,
            Action(actor, "Hold", observed_action.target),
            Action(actor, "ImposeTariff", "Semiconductors", delta_weight=10.0),
        ]

        q_values = []
        for cand in candidates:
            # 1-step lookahead
            sim_p = TransitionSimulator(params=particle.params)
            s_next = sim_p.step(particle.state, [cand])
            u = pref.evaluate_utility(s_next, actor)
            q_values.append(u)

        q_arr = np.array(q_values) * self.rationality_lambda
        # Log-sum-exp
        max_q = np.max(q_arr)
        log_denom = max_q + np.log(np.sum(np.exp(q_arr - max_q)))
        # Target action is candidates[0]
        log_prob = q_arr[0] - log_denom
        return float(log_prob)

    def update(
        self,
        observed_claims: List[Claim],
        observed_actions: List[Action],
        coverage_triples: Optional[Set[Tuple[str, str, str]]] = None
    ) -> float:
        """
        Propagates particles, applies text evidence and IRL action likelihood,
        resamples if ESS < K/2, and rejuvenates static parameters.
        Returns average unnormalized weight for scenario marginal likelihood.
        """
        unnormalized_weights = np.zeros(self.num_particles)

        for k in range(self.num_particles):
            p = self.particles[k]
            # 1. State propagation
            p.state = self.simulator.step(p.state, observed_actions)

            # 2. Text evidence likelihood p(o_t | s_t)
            log_p_obs = self.obs_model.compute_log_likelihood(observed_claims, p.state, coverage_triples)

            # 3. Action likelihood (Bayesian IRL)
            log_p_act = 0.0
            for act in observed_actions:
                log_p_act += self._myopic_action_log_prob(p, act, act.actor)

            # Weight update
            log_w_update = log_p_obs + log_p_act
            # Bound log update to prevent underflow/overflow
            bounded_log = np.clip(log_w_update, -50.0, 50.0)
            update_factor = np.exp(bounded_log)
            p.weight *= update_factor
            unnormalized_weights[k] = p.weight

        # Normalize weights
        total_w = np.sum(unnormalized_weights)
        if total_w > 1e-20:
            for p in self.particles:
                p.weight /= total_w
        else:
            uniform_w = 1.0 / self.num_particles
            for p in self.particles:
                p.weight = uniform_w

        avg_unnorm_weight = float(np.mean(unnormalized_weights))
        self.marginal_likelihood_history.append(avg_unnorm_weight)

        # 4. Effective Sample Size (ESS) check & Resampling
        w_vec = np.array([p.weight for p in self.particles])
        ess = 1.0 / np.sum(w_vec ** 2)

        if ess < (self.num_particles / 2.0):
            self._resample_and_rejuvenate()

        return avg_unnorm_weight

    def _resample_and_rejuvenate(self) -> None:
        """Systematic resampling + Gilks & Berzuini MCMC rejuvenation kernel."""
        weights = [p.weight for p in self.particles]
        indices = self.rng.choice(self.num_particles, size=self.num_particles, p=weights)
        new_particles = [self.particles[idx].clone() for idx in indices]

        # MCMC rejuvenation kernel on static parameters theta and psi
        for p in new_particles:
            p.weight = 1.0 / self.num_particles
            # Jiggle theta
            for actor, pref in p.actor_types.items():
                for kw in pref.weights:
                    noise = float(self.rng.normal(0, 0.02))
                    pref.weights[kw] = float(np.clip(pref.weights[kw] + noise, 0.01, 1.0))
            # Jiggle psi
            p.params.tariff_growth_drag = float(np.clip(
                p.params.tariff_growth_drag + self.rng.normal(0, 0.002), 0.01, 0.15
            ))

        self.particles = new_particles

    def get_latent_summary(self) -> Dict:
        """Summarizes current belief over graph facts, actor types, and indicators."""
        w_vec = np.array([p.weight for p in self.particles])
        gdp_us = [p.state.get_indicator("gdp_growth_USA", 2.0) for p in self.particles]
        brent = [p.state.get_indicator("brent_crude", 80.0) for p in self.particles]

        # Average actor preference
        actor_weights = {}
        for a in self.particles[0].actor_types:
            sec_vals = [p.actor_types[a].weights.get("security", 0.3) for p in self.particles]
            actor_weights[a] = {
                "security_mean": float(np.average(sec_vals, weights=w_vec)),
                "security_std": float(np.sqrt(np.cov(sec_vals, aweights=w_vec))),
            }

        return {
            "gdp_growth_USA_mean": float(np.average(gdp_us, weights=w_vec)),
            "brent_crude_mean": float(np.average(brent, weights=w_vec)),
            "actor_types": actor_weights,
            "num_particles": len(self.particles),
        }
