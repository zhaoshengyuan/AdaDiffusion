# AdaDiffusion for OTFS Channel Estimation

This example applies AdaDiffusion to OTFS delay-Doppler channel estimation.
See the full communication-focused repo at:
https://github.com/zhaoshengyuan/Adaptive-DD-Inference-otfs

## Quick Start

```python
from adadiffusion import AdaptiveDiffusionEstimator, AdaptiveConfig
from diffusers import DDIMScheduler

# 1. Load your trained UNet
unet = load_otfs_unet("best_model.pth")

# 2. Wrap with AdaDiffusion
cfg = AdaptiveConfig(tau_scale=0.24, tile_spatial=(4, 4))
estimator = AdaptiveDiffusionEstimator(
    model_fn=lambda x, t, **kw: unet(x, t, speed_kmh=kw['speed'], snr_db=kw['snr']),
    scheduler=DDIMScheduler(num_train_timesteps=1000, prediction_type='v_prediction'),
    spatial_shape=(64, 128),  # (doppler, delay)
    noise_std_fn=lambda **kw: math.sqrt(1 / 10**(kw['snr']/10)),
    cfg=cfg,
)

# 3. Sample
H_est, info = estimator.sample(
    initial_noise=torch.randn(1, 8, 64, 128),
    num_steps=5,
    conditions={'speed': 60.0, 'snr': 5.0},
)
print(f"Active ratio: {info['active_ratio']:.1%}")
```
