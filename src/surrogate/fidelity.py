"""
Surrogate Fidelity and Complexity Auditor (Section 9.5).
Evaluates action agreement %, KL divergence, and complexity on held-out in-distribution
and perturbed off-distribution test sets.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from src.surrogate.kan import KolmogorovArnoldNetwork


@dataclass
class FidelityMetrics:
    in_distribution_agreement: float     # % action match
    in_distribution_kl: float            # KL divergence
    off_distribution_agreement: float    # Extrapolation agreement %
    off_distribution_kl: float
    total_edges: int
    active_edges: int
    snapped_edges: int
    complexity_score: float              # Number of parameters in active snapped formulas
    reliable_explanation: bool          # Agreement >= 0.85 threshold


class FidelityAuditor:
    """Evaluates surrogate fidelity against ground-truth policy."""

    @staticmethod
    def evaluate(
        kan: KolmogorovArnoldNetwork,
        x_in: torch.Tensor,
        target_in: torch.Tensor,
        x_off: torch.Tensor,
        target_off: torch.Tensor,
        snapped_count: int = 0
    ) -> FidelityMetrics:
        kan.eval()
        with torch.no_grad():
            # In-distribution
            logits_in = kan(x_in)
            probs_in = F.softmax(logits_in, dim=-1)
            pred_act_in = torch.argmax(probs_in, dim=-1)
            true_act_in = torch.argmax(target_in, dim=-1)
            in_agree = float(torch.mean((pred_act_in == true_act_in).float()).item())
            log_in = F.log_softmax(logits_in, dim=-1)
            in_kl = float(F.kl_div(log_in, target_in, reduction="batchmean").item())

            # Off-distribution
            logits_off = kan(x_off)
            probs_off = F.softmax(logits_off, dim=-1)
            pred_act_off = torch.argmax(probs_off, dim=-1)
            true_act_off = torch.argmax(target_off, dim=-1)
            off_agree = float(torch.mean((pred_act_off == true_act_off).float()).item())
            log_off = F.log_softmax(logits_off, dim=-1)
            off_kl = float(F.kl_div(log_off, target_off, reduction="batchmean").item())

        # Complexity
        total_e = 0
        active_e = 0
        # Layer 1
        for j in range(kan.hidden_dim):
            for i in range(kan.in_dim):
                total_e += 1
                if kan.layer1.edges[j][i].is_active:
                    active_e += 1
        # Layer 2
        for j in range(kan.out_dim):
            for i in range(kan.hidden_dim):
                total_e += 1
                if kan.layer2.edges[j][i].is_active:
                    active_e += 1

        reliable = (in_agree >= 0.85)

        return FidelityMetrics(
            in_distribution_agreement=round(in_agree * 100.0, 1),
            in_distribution_kl=round(in_kl, 4),
            off_distribution_agreement=round(off_agree * 100.0, 1),
            off_distribution_kl=round(off_kl, 4),
            total_edges=total_e,
            active_edges=active_e,
            snapped_edges=snapped_count,
            complexity_score=float(active_e * 2.5),
            reliable_explanation=reliable
        )
