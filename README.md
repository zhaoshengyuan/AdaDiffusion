# AdaDiffusion

**Adaptive Tile-Gated Inference for Efficient Diffusion Models**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, plug-and-play framework that accelerates diffusion model inference
by adaptively freezing converged spatial tiles during sampling. Works with **any**
pre-trained diffusion model without retraining.

## Key Idea

Diffusion sampling is expensive because every denoising step processes the entire
spatial grid. But not all regions need the same number of updates:

- **Background / easy regions** converge early → can be frozen
- **Foreground / complex regions** need more refinement → stay active

AdaDiffusion detects which tiles have converged (via amplitude + residual-change
scores) and freezes them, allocating computation only where needed.

## Installation

```bash
pip install torch numpy
git clone https://github.com/zhaoshengyuan/AdaDiffusion.git
cd AdaDiffusion
```

## Quick Start

```python
import torch, math
from diffusers import DDIMScheduler, DDPMPipeline
from adadiffusion import AdaptiveDiffusionEstimator, AdaptiveConfig

# Load any pre-trained diffusion model
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
unet = pipe.unet

# Wrap with AdaDiffusion
cfg = AdaptiveConfig(tau_scale=0.24, tile_spatial=(4, 4))
estimator = AdaptiveDiffusionEstimator(
    model_fn=lambda x, t, **kw: unet(x, t).sample,
    scheduler=DDIMScheduler(num_train_timesteps=1000),
    spatial_shape=(32, 32),
    cfg=cfg,
)

# Sample with adaptive freezing
noise = torch.randn(4, 3, 32, 32)
image, info = estimator.sample(noise, num_steps=5)
print(f"Computation saved: {(1-info['active_ratio']):.1%}")
```

## Use Cases

| Domain | Example |
|--------|---------|
| Image Generation | Freeze background tiles, focus on objects |
| Wireless Communications | Freeze empty delay-Doppler bins |
| Video Diffusion | Freeze static regions across frames |
| Scientific Imaging | Freeze noise-only regions |

## Project Structure

```
adadiffusion/
├── __init__.py          # Public API
├── config.py            # AdaptiveConfig dataclass
├── gate.py              # Tile ops & confidence scores
├── threshold.py         # SNR/step-adaptive thresholds
├── mask.py              # Mask construction & update
├── estimator.py         # AdaptiveDiffusionEstimator
└── learnable_gate.py    # Learnable gating network (v0.3)

tests/
└── test_core.py         # Unit tests

examples/
└── otfs/                # OTFS channel estimation example
```

## Citation

```bibtex
@misc{adadiffusion2026,
  title={AdaDiffusion: Adaptive Tile-Gated Inference for Efficient Diffusion Models},
  author={Zhao Shengyuan},
  year={2026},
  url={https://github.com/zhaoshengyuan/AdaDiffusion}
}
```

## License

MIT License.
