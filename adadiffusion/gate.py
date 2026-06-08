"""Spatial tile operations and confidence scores for AdaDiffusion.

All functions are domain-agnostic — they work on any [B, C, H, W] tensor.
No OTFS or communication-specific dependencies.
"""

from typing import Optional
import torch


def tile_mean(x: torch.Tensor, tile_h: int, tile_w: int) -> torch.Tensor:
    """Average a [B, H, W] spatial grid into [B, H//tH, W//tW] tiles.

    Args:
        x: [B, H, W] real-valued spatial map
        tile_h, tile_w: tile size in pixels

    Returns:
        tiled: [B, H//tile_h, W//tile_w]
    """
    B, H, W = x.shape
    assert H % tile_h == 0, f"H={H} not divisible by tile_h={tile_h}"
    assert W % tile_w == 0, f"W={W} not divisible by tile_w={tile_w}"
    r = x.reshape(B, H // tile_h, tile_h, W // tile_w, tile_w)
    return r.mean(dim=(2, 4))


def expand_tile_mask(
    mask: torch.Tensor, tile_h: int, tile_w: int
) -> torch.Tensor:
    """Expand tile-level mask to pixel-level.

    Args:
        mask: [B, n_tiles_h, n_tiles_w] bool
        tile_h, tile_w: tile size

    Returns:
        expanded: [B, H, W] — each tile value repeated
    """
    m = torch.repeat_interleave(mask, tile_h, dim=1)
    return torch.repeat_interleave(m, tile_w, dim=2)


def compute_amplitude_score(
    sample: torch.Tensor, tile_h: int, tile_w: int
) -> torch.Tensor:
    """Per-tile mean magnitude (amplitude score a_q).

    For image diffusion: mean(|x|) across channels.
    For other modalities: override this function.

    Args:
        sample: [B, C, H, W] current sample (image, channel estimate, etc.)
        tile_h, tile_w: tile size

    Returns:
        score: [B, H//tile_h, W//tile_w]
    """
    magnitude = sample.abs().mean(dim=1)  # [B, H, W]
    return tile_mean(magnitude, tile_h, tile_w)


def compute_change_score(
    curr: torch.Tensor, prev: torch.Tensor, tile_h: int, tile_w: int
) -> torch.Tensor:
    """Per-tile mean residual change (change score d_q).

    Args:
        curr: [B, C, H, W] current residual / latent
        prev: [B, C, H, W] previous step residual / latent
        tile_h, tile_w: tile size

    Returns:
        score: [B, H//tile_h, W//tile_w]
    """
    diff = (curr - prev).abs().mean(dim=1)  # [B, H, W]
    return tile_mean(diff, tile_h, tile_w)
