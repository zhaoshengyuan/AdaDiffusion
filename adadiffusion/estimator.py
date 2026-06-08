"""Domain-agnostic adaptive diffusion estimator.

Wraps ANY pre-trained diffusion pipeline with tile-level adaptive inference.
No model retraining required.
"""

from __future__ import annotations

import time
import math
from typing import Optional, Dict, List, Callable, Tuple

import torch
import numpy as np

from .config import AdaptiveConfig
from .gate import tile_mean, expand_tile_mask, compute_amplitude_score, compute_change_score
from .threshold import compute_adaptive_thresholds
from .mask import build_active_mask, apply_masked_update


class AdaptiveDiffusionEstimator:
    """Adaptive tile-freezing wrapper for diffusion model inference.

    Design: plug-and-play. The user provides:
      - A model_fn(sample, timestep, **condition_kwargs) → denoised_output
      - A scheduler that supports .step() and .timesteps
      - An optional noise_std_fn(conditions) → float

    Usage:
        estimator = AdaptiveDiffusionEstimator(
            model_fn=unet.forward,
            scheduler=ddim_scheduler,
            spatial_shape=(H, W),
            noise_std_fn=lambda cond: math.sqrt(1 / 10**(snr/10)),
            cfg=AdaptiveConfig(tau_scale=0.24),
        )
        result, info = estimator.sample(
            initial_noise, steps=5, conditions={'snr': 10.0}
        )
    """

    def __init__(
        self,
        model_fn: Callable[..., torch.Tensor],
        scheduler,
        spatial_shape: Tuple[int, int],
        noise_std_fn: Optional[Callable[..., float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cfg: Optional[AdaptiveConfig] = None,
    ):
        self.model_fn = model_fn
        self.scheduler = scheduler
        self.spatial_shape = spatial_shape  # (H, W) of the latent/grid
        self.noise_std_fn = noise_std_fn or (lambda **kw: 0.1)
        self.device = torch.device(device)
        self.cfg = cfg or AdaptiveConfig

    def sample(
        self,
        initial_sample: torch.Tensor,
        num_steps: int = 5,
        conditions: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Run adaptive sampling.

        Args:
            initial_sample: starting noise [B, C, H, W]
            num_steps: total DDIM steps
            conditions: dict passed to model_fn and noise_std_fn

        Returns:
            final_sample: [B, C, H, W]
            info: dict with active_ratio, active_history, etc.
        """
        conditions = conditions or {}
        cfg = self.cfg
        self.scheduler.set_timesteps(num_steps)

        noise_std = self.noise_std_fn(**conditions)
        B = initial_sample.shape[0]
        H, W = self.spatial_shape
        tH, tW = cfg.tile_spatial
        n_tiles_h, n_tiles_w = H // tH, W // tW

        sample = initial_sample.to(self.device)
        prev_sample = sample.clone()
        active_history: List[float] = []
        last_mask = None
        t0 = time.perf_counter()

        for step_idx, t in enumerate(self.scheduler.timesteps):
            t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

            # Full-grid model forward
            model_output = self.model_fn(sample, t_tensor, **conditions)

            # Scheduler step → candidate
            candidate = self.scheduler.step(model_output, t, sample).prev_sample
            candidate = torch.clamp(candidate, -100, 100)

            # Adaptive mask decision
            is_warmup = step_idx < cfg.warmup_steps
            is_refresh = (cfg.force_refresh_every > 0 and
                          step_idx % cfg.force_refresh_every == 0)

            if is_warmup or is_refresh:
                active_tile = torch.ones(
                    B, n_tiles_h, n_tiles_w, dtype=torch.bool, device=self.device
                )
            else:
                amp = compute_amplitude_score(sample, tH, tW)
                chg = None
                if cfg.use_change_score:
                    chg = compute_change_score(sample, prev_sample, tH, tW)

                tau_amp, tau_chg = compute_adaptive_thresholds(
                    noise_std=noise_std,
                    step_idx=step_idx,
                    tau_scale=cfg.tau_scale,
                    total_steps=num_steps,
                    snr_aggr_min=cfg.snr_aggr_min,
                    snr_aggr_max=cfg.snr_aggr_max,
                    step_aggr_base=cfg.step_aggr_base,
                    step_aggr_slope=cfg.step_aggr_slope,
                    change_thresh_base=cfg.change_thresh_base,
                )
                active_tile = build_active_mask(amp, chg, tau_amp, tau_chg)

            # Expand to pixel level + channel broadcast
            active_grid = expand_tile_mask(active_tile, tH, tW)
            active_bc = active_grid.unsqueeze(1).expand(-1, sample.shape[1], -1, -1)

            # Masked update
            sample = apply_masked_update(candidate, sample, active_bc)

            prev_sample = sample.clone()
            active_history.append(float(active_grid.float().mean().item()))
            last_mask = active_tile.detach().cpu()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(B, 1)

        return sample, {
            "active_ratio": float(np.mean(active_history)),
            "active_history": active_history,
            "last_mask": last_mask,
            "per_frame_ms": elapsed_ms,
        }
