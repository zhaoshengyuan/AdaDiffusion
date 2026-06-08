"""Learnable tile gating network (v0.3).

A lightweight CNN that predicts which tiles can be frozen during
diffusion sampling. ~50K parameters — trainable on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TileGatingNetwork(nn.Module):
    """Lightweight tile-level gate predictor.

    Input:  4-channel tile feature map [B, 4, n_tiles_h, n_tiles_w]
            (amplitude, change, step_normalized, snr_normalized)

    Output: per-tile logit [B, 1, n_tiles_h, n_tiles_w]
            > 0 → active, < 0 → frozen
    """

    def __init__(self, in_channels: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(
        self,
        amp_score: torch.Tensor,
        chg_score: torch.Tensor,
        step_idx: int,
        total_steps: int,
        noise_std: float,
    ) -> torch.Tensor:
        """Predict active-tile logits.

        Args:
            amp_score: [B, H_t, W_t] amplitude score
            chg_score: [B, H_t, W_t] change score
            step_idx: current sampling step
            total_steps: total steps
            noise_std: noise level

        Returns:
            logits: [B, 1, H_t, W_t] (unconstrained, use >0 as active)
        """
        B, H_t, W_t = amp_score.shape
        step_norm = torch.full((B, 1, H_t, W_t), step_idx / max(total_steps, 1),
                               device=amp_score.device, dtype=amp_score.dtype)
        snr_norm = torch.full((B, 1, H_t, W_t), min(noise_std / 3.0, 1.0),
                              device=amp_score.device, dtype=amp_score.dtype)

        features = torch.stack([
            amp_score, chg_score,
            step_norm.squeeze(1), snr_norm.squeeze(1)
        ], dim=1)  # [B, 4, H_t, W_t]

        return self.net(features)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def gate_loss(
    logits: torch.Tensor,
    target_active: torch.Tensor,
    freeze_penalty: float = 0.1,
) -> torch.Tensor:
    """Training loss for the gating network.

    BCE to match ground-truth active decisions + L1 penalty
    to encourage freezing (sparsity).

    Args:
        logits: [B, 1, H_t, W_t] predicted logits
        target_active: [B, H_t, W_t] bool ground truth
        freeze_penalty: weight for sparsity regularization

    Returns:
        scalar loss
    """
    bce = F.binary_cross_entropy_with_logits(
        logits.squeeze(1), target_active.float()
    )
    sparsity = torch.sigmoid(logits).mean()  # Lower = more frozen
    return bce + freeze_penalty * sparsity
