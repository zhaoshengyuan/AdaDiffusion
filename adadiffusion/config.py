"""Configuration for AdaDiffusion adaptive inference."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AdaptiveConfig:
    """Hyperparameters for adaptive tile-freezing.

    Only one free parameter (tau_scale). All others are fixed heuristics.
    """

    # ── Tile geometry ──
    tile_spatial: Tuple[int, ...] = (4, 4)  # (H, W) tile size

    # ── Core hyperparameter ──
    tau_scale: float = 0.24

    # ── Schedule ──
    warmup_steps: int = 2
    snr_aggr_min: float = 0.75
    snr_aggr_max: float = 1.35
    step_aggr_base: float = 0.70
    step_aggr_slope: float = 0.18
    change_thresh_base: float = 0.10

    # ── Refresh ──
    force_refresh_every: int = 0
    use_change_score: bool = True

    # ── Learnable gate (v0.3) ──
    use_learnable_gate: bool = False
    gate_checkpoint: str = ""

    # ── Evaluation ──
    tau_sweep_values: Tuple[float, ...] = (
        0.16, 0.20, 0.22, 0.24, 0.26, 0.28, 0.32
    )
    target_loss_db: float = 0.30
