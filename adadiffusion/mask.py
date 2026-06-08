"""Tile mask construction and masked update for AdaDiffusion."""

from typing import Optional
import torch


def build_active_mask(
    amplitude_score: torch.Tensor,
    change_score: Optional[torch.Tensor],
    tau_amp: float,
    tau_chg: float,
) -> torch.Tensor:
    """Build active-tile mask from confidence scores.

    A tile is FROZEN (mask=0) only when BOTH:
      - amplitude < tau_amp  (no significant content)
      - change < tau_chg     (residual has converged)

    Args:
        amplitude_score: a_q [B, H_t, W_t]
        change_score: d_q [B, H_t, W_t] or None (amplitude-only)
        tau_amp, tau_chg: thresholds

    Returns:
        mask: bool [B, H_t, W_t], True = active
    """
    active = amplitude_score >= tau_amp
    if change_score is not None:
        active = active | (change_score >= tau_chg)
    return active


def apply_masked_update(
    proposal: torch.Tensor,
    current: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked residual update.

    R_new = active_mask * proposal + (1 - active_mask) * current

    Args:
        proposal: denoised candidate (any shape)
        current: previous value (same shape)
        active_mask: bool mask broadcastable to proposal shape

    Returns:
        updated tensor (same shape as proposal)
    """
    return torch.where(active_mask, proposal, current)
