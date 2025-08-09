import os
import pytest

# Skip if MLX not available
_HAS_MLX = True
try:
    import mlx.core as mx  # type: ignore
    import mlx.nn as nn  # type: ignore
except Exception:
    _HAS_MLX = False

pytestmark = pytest.mark.skipif(not _HAS_MLX, reason="MLX not installed")

import numpy as np

from models.hrm_mlx.hrm_act_v1_mlx import HierarchicalReasoningModel_ACTV1 as MxHRM
from models.hrm_mlx.diffusion.ddim_mlx import MxDDIMScheduler
from models.hrm_mlx.diffusion.hrm_denoiser_mlx import HRMDenoiserMx


def test_mlx_ddim_shapes():
    config = dict(
        batch_size=1,
        seq_len=16,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=256,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,
        L_layers=0,
        hidden_size=64,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
    )
    model = MxHRM(config)
    denoiser = HRMDenoiserMx(model.inner, hidden_size=64)
    sched = MxDDIMScheduler(timesteps=4)
    mx.random.seed(123)  # type: ignore[attr-defined]
    z0 = sched.sample(steps=4, shape=(1,16,64), denoiser=lambda x,t,c: denoiser(x,t,c), seed=123)
    assert z0.shape == (1,16,64)
