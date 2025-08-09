import os
import torch

from models.diffusion import DDIMScheduler


def test_ddim_cpu_determinism():
    sched = DDIMScheduler(timesteps=10)
    shape = (2, 8, 16)
    torch.manual_seed(42)
    z1 = sched.sample(steps=10, shape=shape, denoiser=lambda x, t, c: torch.zeros_like(x), seed=123)
    torch.manual_seed(42)
    z2 = sched.sample(steps=10, shape=shape, denoiser=lambda x, t, c: torch.zeros_like(x), seed=123)
    assert torch.allclose(z1, z2)
