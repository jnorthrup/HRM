"""
MLX implementation of a deterministic DDIM scheduler (eta=0) for HRM latent diffusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import math
import mlx.core as mx


def _cosine_beta_schedule(T: int, s: float = 0.008):
    steps = T + 1
    x = mx.linspace(0, T, steps)
    alphas_cumprod = mx.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return mx.clip(betas, 1e-8, 0.999)


def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return mx.linspace(beta_start, beta_end, T)


@dataclass
class MxDDIMScheduler:
    timesteps: int
    beta_schedule: str = "cosine"

    def __post_init__(self):
        if self.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.timesteps)
        elif self.beta_schedule == "linear":
            betas = _linear_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")
        alphas = 1.0 - betas
        self.alphas_cumprod = mx.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)

    def step(self, x_t, eps, t: int):
        a_t = self.sqrt_alphas_cumprod[t].astype(x_t.dtype)
        om_a_t = self.sqrt_one_minus_alphas_cumprod[t].astype(x_t.dtype)
        x0 = (x_t - om_a_t * eps) / (a_t + 1e-12)
        if t == 0:
            return x0
        a_tm1 = self.sqrt_alphas_cumprod[t - 1].astype(x_t.dtype)
        x_prev = a_tm1 * x0 + self.sqrt_one_minus_alphas_cumprod[t - 1].astype(x_t.dtype) * eps
        return x_prev

    def sample(
        self,
        steps: int,
        shape: Tuple[int, int, int],
        denoiser: Callable[[mx.array, mx.array, Optional[dict]], mx.array],
        cond: Optional[dict] = None,
        seed: Optional[int] = None,
        dtype=mx.float32,
    ):
        if steps != self.timesteps:
            raise ValueError("steps must equal scheduler.timesteps for fixed grid determinism")
        if seed is not None:
            mx.random.seed(seed)
        x_t = mx.random.normal(shape=shape).astype(dtype)
        for t in range(steps - 1, -1, -1):
            t_arr = mx.full((shape[0],), t, dtype=mx.int32)
            eps = denoiser(x_t, t_arr, cond)
            x_t = self.step(x_t, eps, t)
        return x_t
