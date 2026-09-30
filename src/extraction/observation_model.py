"""
Calibrated observation model for the LLM sensor (Section 3.2).
Likelihood p(o_t | s_t) accounts for sensitivity r_k, false-positive rate f_k,
intensity measurement error sigma_k, and document coverage C_t.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import math
import numpy as np
from src.core.state import WorldState


@dataclass(frozen=True)
class Claim:
    """An extracted claim c = (i, k, j, w_hat, t, span, source)."""
    source_entity: str     # i
    relation: str          # k
    target_entity: str     # j
    reported_weight: float # w_hat
    time_step: int         # t
    evidence_span: str     # verbatim evidence span
    source_outlet: str     # reporting source class

    def triple(self) -> Tuple[str, str, str]:
        return (self.source_entity, self.relation, self.target_entity)


@dataclass
class RelationErrorRates:
    sensitivity: float = 0.85          # r_k: P(claim | fact in F_t)
    false_positive_rate: float = 0.05  # f_k: P(claim | fact not in F_t)
    weight_sigma: float = 2.5          # sigma_k: std dev of reported intensity
    bg_mean: float = 10.0              # g_k background intensity mean
    bg_std: float = 15.0               # g_k background intensity std


class ObservationModel:
    """Calibrated likelihood model p(o_t | s_t) for present-state text reporting."""

    def __init__(self, error_rates: Optional[Dict[str, RelationErrorRates]] = None):
        self.error_rates = error_rates or {
            "Tariff": RelationErrorRates(sensitivity=0.88, false_positive_rate=0.04, weight_sigma=2.0, bg_mean=15.0, bg_std=10.0),
            "Sanctions": RelationErrorRates(sensitivity=0.82, false_positive_rate=0.06, weight_sigma=0.08, bg_mean=0.3, bg_std=0.2),
            "TradeAgreement": RelationErrorRates(sensitivity=0.90, false_positive_rate=0.03, weight_sigma=0.05, bg_mean=0.5, bg_std=0.2),
            "AllianceCohesion": RelationErrorRates(sensitivity=0.75, false_positive_rate=0.08, weight_sigma=0.10, bg_mean=0.5, bg_std=0.25),
            "TechExportControl": RelationErrorRates(sensitivity=0.85, false_positive_rate=0.05, weight_sigma=0.08, bg_mean=0.4, bg_std=0.2),
            "DiplomaticTension": RelationErrorRates(sensitivity=0.80, false_positive_rate=0.07, weight_sigma=0.09, bg_mean=0.3, bg_std=0.2),
        }
        self.default_rates = RelationErrorRates()

    def get_rates(self, relation: str) -> RelationErrorRates:
        return self.error_rates.get(relation, self.default_rates)

    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        if std <= 1e-6:
            return 1.0 if abs(x - mean) < 1e-4 else 1e-8
        var = std ** 2
        denom = math.sqrt(2.0 * math.pi * var)
        num = math.exp(-((x - mean) ** 2) / (2.0 * var))
        return max(num / denom, 1e-12)

    def claim_likelihood(self, claim: Claim, state: WorldState) -> float:
        """
        p(c | s_t) = r_k * N(w_hat; w_ikj, sigma_k^2) if (i, k, j) in F_t
                     f_k * g_k(w_hat)                 otherwise
        """
        rates = self.get_rates(claim.relation)
        g = state.graph
        if g.has_triple(claim.source_entity, claim.relation, claim.target_entity):
            true_w = g.get_weight(claim.source_entity, claim.relation, claim.target_entity, default=0.0)
            density = self._gaussian_pdf(claim.reported_weight, true_w, rates.weight_sigma)
            return rates.sensitivity * density
        else:
            bg_density = self._gaussian_pdf(claim.reported_weight, rates.bg_mean, rates.bg_std)
            return rates.false_positive_rate * bg_density

    def compute_log_likelihood(
        self,
        claims: List[Claim],
        state: WorldState,
        coverage_triples: Optional[Set[Tuple[str, str, str]]] = None
    ) -> float:
        """
        Computes log p(o_t | s_t) = sum_{c in o_t} log p(c | s_t) + sum_{(i,k,j) in C_t cap F_t unclaimed} log(1 - r_k)
        """
        log_prob = 0.0
        claimed_triples = set()

        for c in claims:
            claimed_triples.add(c.triple())
            prob_c = self.claim_likelihood(c, state)
            log_prob += math.log(max(prob_c, 1e-15))

        # Coverage penalty for unclaimed facts covered by documents
        coverage = coverage_triples or set()
        for fact in state.graph.all_facts():
            triple = fact.triple()
            if triple in coverage and triple not in claimed_triples:
                rates = self.get_rates(fact.relation)
                unclaimed_prob = max(1.0 - rates.sensitivity, 1e-6)
                log_prob += math.log(unclaimed_prob)

        return log_prob
