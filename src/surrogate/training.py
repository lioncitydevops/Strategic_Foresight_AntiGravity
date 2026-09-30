"""
KAN Policy Surrogate Training & Distillation (Section 9.3).
Trains KAN to match target policy distribution with L1 edge activation penalty:
L = E_s [KL(pi*(· | s) || pi_hat(· | s))] + lambda_1 * sum ||phi_{l,j,i}||_1
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from src.surrogate.kan import KolmogorovArnoldNetwork


class KANTrainer:
    """Trains sparse KAN surrogate on policy visitation data."""

    def __init__(
        self,
        kan: KolmogorovArnoldNetwork,
        l1_penalty_weight: float = 0.005,
        learning_rate: float = 0.01
    ):
        self.kan = kan
        self.l1_lambda = l1_penalty_weight
        self.lr = learning_rate

    def train_on_dataset(
        self,
        x_data: torch.Tensor,       # shape: (N, in_dim)
        target_probs: torch.Tensor, # shape: (N, out_dim)
        epochs: int = 120,
        batch_size: int = 32
    ) -> List[float]:
        """Runs policy distillation training loop."""
        optimizer = optim.Adam(self.kan.parameters(), lr=self.lr, weight_decay=1e-5)
        num_samples = x_data.shape[0]
        loss_history = []

        for ep in range(epochs):
            perm = torch.randperm(num_samples)
            epoch_loss = 0.0
            num_batches = 0

            for b_start in range(0, num_samples, batch_size):
                b_idx = perm[b_start:b_start + batch_size]
                bx = x_data[b_idx]
                by_target = target_probs[b_idx]

                optimizer.zero_grad()
                logits = self.kan(bx)
                log_probs = F.log_softmax(logits, dim=-1)

                # KL divergence: sum p * (log p - log q) = sum p * log p - sum p * log q
                # In PyTorch F.kl_div(log_probs, target_probs) expects target in same form
                kl_loss = F.kl_div(log_probs, by_target, reduction="batchmean")

                # L1 penalty on edge activations
                l1_loss = self.kan.compute_l1_penalty(bx)

                total_loss = kl_loss + self.l1_lambda * l1_loss
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            loss_history.append(epoch_loss / max(num_batches, 1))

        # Prune dead edges after training
        self.kan.prune_edges(threshold=0.03, sample_x=x_data[:50])
        return loss_history
