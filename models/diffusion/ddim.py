"""
Deterministic DDIM scheduler (eta=0) for HRM latent diffusion.

Shapes:
- x_t: [B, L, H]
- eps: [B, L, H]

API:
- DDIMScheduler(timesteps: int, beta_schedule: str = "cosine")
- step(x_t, eps, t) -> x_{t-1}
- sample(steps, shape, denoiser, cond=None, device=None, dtype=torch.float32)

Numerics are chosen for repeatability on Torch CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import math
import torch


def _cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    # https://openreview.net/forum?id=-NEXDKk8gZ
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = (torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


def _linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


@dataclass
class DDIMScheduler:
    timesteps: int
    beta_schedule: str = "cosine"  # or "linear"

    def __post_init__(self):
        if self.beta_schedule == "cosine":
            betas = _cosine_beta_schedule(self.timesteps)
        elif self.beta_schedule == "linear":
            betas = _linear_beta_schedule(self.timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register(alphas_cumprod)

    def register(self, alphas_cumprod: torch.Tensor):
        # Keep as buffers not nn.Buffers to avoid module dependency; tensors live on CPU by default
        self.alphas_cumprod = alphas_cumprod.to(torch.float64)  # high precision host ref
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, eps: torch.Tensor, t: int) -> torch.Tensor:
        # Deterministic DDIM (eta=0)
        a_t = self.sqrt_alphas_cumprod[t].to(x_t.dtype)
        om_a_t = self.sqrt_one_minus_alphas_cumprod[t].to(x_t.dtype)
        if t == 0:
            # Return x0 directly
            x0 = (x_t - om_a_t * eps) / (a_t + 1e-12)
            return x0
        a_tm1 = self.sqrt_alphas_cumprod[t - 1].to(x_t.dtype)
        # Estimate x0
        x0 = (x_t - om_a_t * eps) / (a_t + 1e-12)
        x_prev = a_tm1 * x0 + self.sqrt_one_minus_alphas_cumprod[t - 1].to(x_t.dtype) * eps
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        steps: int,
        shape: Tuple[int, int, int],
        denoiser: Callable[[torch.Tensor, torch.Tensor, Optional[dict]], torch.Tensor],
        cond: Optional[dict] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if steps != self.timesteps:
            raise ValueError("steps must equal scheduler.timesteps for fixed grid determinism")
        if device is None:
            device = torch.device("cpu")
        if seed is not None:
            torch.manual_seed(seed)
        x_t = torch.randn(*shape, device=device, dtype=dtype)
        for t in reversed(range(steps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.int64)
            eps = denoiser(x_t, t_tensor, cond)
            x_t = self.step(x_t, eps, t)
        return x_t
