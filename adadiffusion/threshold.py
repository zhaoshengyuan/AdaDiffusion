"""Adaptive threshold scheduling for AdaDiffusion."""

from typing import Tuple
import numpy as np


def compute_adaptive_thresholds(
    noise_std: float,
    step_idx: int,
    tau_scale: float,
    total_steps: int,
    snr_aggr_min: float = 0.75,
    snr_aggr_max: float = 1.35,
    step_aggr_base: float = 0.70,
    step_aggr_slope: float = 0.18,
    change_thresh_base: float = 0.10,
) -> Tuple[float, float]:
    """SNR- and step-adaptive double thresholds.

    tau_amp = tau_scale * noise_std * snr_aggression * step_aggression
    tau_chg = change_thresh_base * noise_std * (1 + 0.20 * step_idx)

    Args:
        noise_std: noise standard deviation (domain-specific)
        step_idx: current sampling step (0-based)
        tau_scale: the ONE free hyperparameter
        total_steps: total sampling steps

    Returns:
        (tau_amp, tau_chg): amplitude and change thresholds
    """
    snr_aggr = 1.0 / max(noise_std, 1e-6)
    snr_aggr = float(np.clip(snr_aggr, snr_aggr_min, snr_aggr_max))

    step_aggr = step_aggr_base + step_aggr_slope * step_idx

    tau_amp = tau_scale * noise_std * snr_aggr * step_aggr
    tau_chg = change_thresh_base * noise_std * (1.0 + 0.20 * step_idx)

    return tau_amp, tau_chg
