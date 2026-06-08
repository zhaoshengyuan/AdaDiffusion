"""AdaDiffusion: Adaptive Tile-Gated Inference for Efficient Diffusion Models.

A lightweight, plug-and-play framework that accelerates diffusion model
inference by adaptively freezing converged spatial tiles during sampling.
Works with any pre-trained diffusion model without retraining.
"""

from .gate import (
    tile_mean,
    expand_tile_mask,
    compute_amplitude_score,
    compute_change_score,
)
from .threshold import compute_adaptive_thresholds
from .mask import build_active_mask, apply_masked_update
from .estimator import AdaptiveDiffusionEstimator
from .config import AdaptiveConfig

__version__ = "0.2.0"
