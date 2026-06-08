"""Unit tests for AdaDiffusion core — no external deps beyond torch + numpy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from adadiffusion import (
    tile_mean, expand_tile_mask,
    compute_amplitude_score, compute_change_score,
    compute_adaptive_thresholds,
    build_active_mask, apply_masked_update,
    AdaptiveConfig, AdaptiveDiffusionEstimator,
)


class TestTileOps:
    def test_tile_mean_basic(self):
        g = torch.tensor([[1.,1.,2.,2.],[1.,1.,2.,2.],[3.,3.,4.,4.],[3.,3.,4.,4.]]).unsqueeze(0)
        r = tile_mean(g, 2, 2)
        assert torch.allclose(r, torch.tensor([[[1.,2.],[3.,4.]]]))

    def test_expand_mask_roundtrip(self):
        m = torch.rand(2, 8, 16) > 0.5
        e = expand_tile_mask(m, 4, 4)
        assert e.shape == (2, 32, 64)
        recovered = tile_mean(e.float(), 4, 4)
        assert torch.allclose(recovered, m.float())

class TestScores:
    def test_amplitude(self):
        x = torch.ones(1, 3, 8, 8)
        x[0, :, 0:4, 0:4] = 10.0
        s = compute_amplitude_score(x, 4, 4)
        assert s[0, 0, 0] > s[0, 1, 1]

    def test_change(self):
        c = torch.ones(1, 3, 4, 4)
        p = torch.zeros(1, 3, 4, 4)
        c[:, :, 0:2, 0:2] = 3.0
        s = compute_change_score(c, p, 2, 2)
        assert s[0, 0, 0] > s[0, 1, 1]

class TestThreshold:
    def test_increase_with_step(self):
        t1, _ = compute_adaptive_thresholds(0.3, 2, 0.24, 5)
        t2, _ = compute_adaptive_thresholds(0.3, 4, 0.24, 5)
        assert t2 > t1

class TestMask:
    def test_both_scores(self):
        a = torch.tensor([[[0.05, 0.30]]])
        c = torch.tensor([[[0.01, 0.02]]])
        m = build_active_mask(a, c, 0.10, 0.05)
        assert m[0,0,0] == 0
        assert m[0,0,1] == 1

    def test_masked_update(self):
        prop = torch.ones(1, 2, 2, 2)
        curr = torch.zeros(1, 2, 2, 2)
        mask = torch.tensor([[[True, False],[False, True]]]).unsqueeze(1).expand(1,2,2,2)
        r = apply_masked_update(prop, curr, mask)
        assert r[0,0,0,0] == 1.0
        assert r[0,0,0,1] == 0.0

class TestConfig:
    def test_defaults(self):
        c = AdaptiveConfig()
        assert c.tau_scale == 0.24
        assert c.warmup_steps == 2

class TestLearnableGate:
    def test_forward_shape(self):
        try:
            from adadiffusion.learnable_gate import TileGatingNetwork
        except ImportError:
            import pytest; pytest.skip("torch.nn not available")
        net = TileGatingNetwork()
        amp = torch.randn(2, 16, 32)
        chg = torch.randn(2, 16, 32)
        out = net(amp, chg, step_idx=2, total_steps=5, noise_std=0.5)
        assert out.shape == (2, 1, 16, 32)

    def test_param_count(self):
        from adadiffusion.learnable_gate import TileGatingNetwork
        net = TileGatingNetwork()
        assert net.num_params < 50000
