"""
Sparse Kolmogorov-Arnold Network (KAN) Policy Surrogate (Section 9).
Maps (g(s), tau) to action logits using learnable univariate spline + SiLU edge activations.
Implements L1 edge sparsification and pruning.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional


class KANSplineEdge(nn.Module):
    """
    Univariate learnable edge function:
    phi(x) = w_b * silu(x) + w_s * sum_{m} c_m B_m(x)
    """

    def __init__(self, num_grid: int = 5, spline_order: int = 3, grid_range: Tuple[float, float] = (-2.0, 2.0)):
        super().__init__()
        self.num_grid = num_grid
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Spline grid knots
        grid = torch.linspace(grid_range[0], grid_range[1], num_grid + 2 * spline_order + 1)
        self.register_buffer("grid", grid)

        # Spline coefficients and base weight
        num_bases = num_grid + spline_order
        self.spline_coeffs = nn.Parameter(torch.randn(num_bases) * 0.1)
        self.w_base = nn.Parameter(torch.randn(1) * 0.1)
        self.w_spline = nn.Parameter(torch.ones(1) * 0.5)

        # Active flag for pruning
        self.is_active = True
        self.snapped_formula: Optional[str] = None
        self.r2_score: float = 0.0

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates Cox-de Boor B-spline basis functions."""
        # x: (batch_size, 1)
        grid = self.grid
        x = x.unsqueeze(-1)  # (batch, 1, 1)

        # 0th-order basis
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Higher-order recursive basis
        for k in range(1, self.spline_order + 1):
            denom1 = grid[k:-1] - grid[:-k-1]
            denom2 = grid[k+1:] - grid[1:-k]
            denom1 = torch.where(denom1 == 0, torch.ones_like(denom1), denom1)
            denom2 = torch.where(denom2 == 0, torch.ones_like(denom2), denom2)

            term1 = (x - grid[:-k-1]) / denom1 * bases[:, :, :-1]
            term2 = (grid[k+1:] - x) / denom2 * bases[:, :, 1:]
            bases = term1 + term2

        return bases.squeeze(1)  # (batch, num_bases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            x_flat = x.view(-1)
        else:
            x_flat = x

        batch_size = x_flat.shape[0]
        if not self.is_active:
            return torch.zeros(batch_size, device=x.device)

        base = (self.w_base * F.silu(x_flat)).view(-1)
        bases = self._b_splines(x_flat.unsqueeze(-1))
        spline = torch.matmul(bases, self.spline_coeffs).view(-1)
        return base + self.w_spline * spline

    def get_activation_l1(self, test_x: torch.Tensor) -> torch.Tensor:
        """Measures empirical L1 edge activation magnitude."""
        with torch.no_grad():
            out = self.forward(test_x)
            return torch.mean(torch.abs(out))


class KANLayer(nn.Module):
    """Bipartite KAN layer from n_in concepts to n_out nodes."""

    def __init__(self, in_features: int, out_features: int, num_grid: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edges = nn.ModuleList([
            nn.ModuleList([KANSplineEdge(num_grid=num_grid) for _ in range(in_features)])
            for _ in range(out_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_features)
        batch_size = x.shape[0]
        outputs = []
        for j in range(self.out_features):
            node_out = torch.zeros(batch_size, device=x.device)
            for i in range(self.in_features):
                node_out = node_out + self.edges[j][i](x[:, i])
            outputs.append(node_out)
        return torch.stack(outputs, dim=-1)  # (batch, out_features)


class KolmogorovArnoldNetwork(nn.Module):
    """
    Two-layer sparse KAN mapping (g(s), tau) -> action logits.
    Follows Section 9 formulation.
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 4, out_dim: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.layer1 = KANLayer(in_dim, hidden_dim)
        self.layer2 = KANLayer(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        logits = self.layer2(h)
        return logits

    def compute_l1_penalty(self, sample_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes sum of empirical edge activation / weight L1 norms."""
        penalty = torch.tensor(0.0, device=self.layer1.edges[0][0].w_base.device)
        for layer in [self.layer1, self.layer2]:
            for row in layer.edges:
                for edge in row:
                    if edge.is_active:
                        penalty = penalty + torch.abs(edge.w_base).sum() + torch.abs(edge.w_spline).sum() * torch.abs(edge.spline_coeffs).mean()
        return penalty

    def prune_edges(self, threshold: float = 0.05, sample_x: Optional[torch.Tensor] = None) -> int:
        """Prunes edges whose activation magnitude is below threshold."""
        pruned = 0
        if sample_x is None:
            sample_x = torch.randn(100, self.in_dim)

        # Layer 1
        for j in range(self.hidden_dim):
            for i in range(self.in_dim):
                mag = self.layer1.edges[j][i].get_activation_l1(sample_x[:, i]).item()
                if mag < threshold:
                    self.layer1.edges[j][i].is_active = False
                    pruned += 1

        # Layer 2
        with torch.no_grad():
            h = self.layer1(sample_x)
        for j in range(self.out_dim):
            for i in range(self.hidden_dim):
                mag = self.layer2.edges[j][i].get_activation_l1(h[:, i]).item()
                if mag < threshold:
                    self.layer2.edges[j][i].is_active = False
                    pruned += 1

        return pruned
