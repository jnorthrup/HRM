"""
HRM denoiser wrapper that conditions on diffusion timestep via simple sinusoidal time embedding
and FiLM-like scaling on the inputs to each reasoning level entry.

Contract:
- forward(x_t: [B,L,H], t: [B], cond: Optional[dict]) -> eps_pred: [B,L,H]

Assumptions:
- We reuse HRM inner blocks' hidden_size and add a time adapter that produces a residual injected to z_H and z_L inputs.
- We bypass token embeddings: the denoiser operates directly in hidden space matching HRM.hidden_size.
"""
from __future__ import annotations

from typing import Optional, Dict

import math
import torch
from torch import nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    # t: [B] integer timesteps
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half)
    args = t.to(torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimeAdapter(nn.Module):
    def __init__(self, hidden_size: int, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.proj = nn.Sequential(
            nn.Linear(time_dim, hidden_size * 2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2, bias=True),
        )
        # Zero init the final layer to start as a no-op
        last = self.proj[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        temb = sinusoidal_time_embedding(t, self.time_dim)
        return self.proj(temb)


class HRMDenoiser(nn.Module):
    def __init__(self, hrm_inner: nn.Module, hidden_size: int):
        super().__init__()
        self.hrm_inner = hrm_inner
        self.hidden_size = hidden_size
        self.time_adapter = TimeAdapter(hidden_size)
        # Simple input/output projections to match HRM hidden
        self.in_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.in_proj.weight.copy_(torch.eye(hidden_size))
            self.out_proj.weight.copy_(torch.eye(hidden_size))

    @torch.no_grad()
    def _zero_grad_mode(self):
        for p in self.hrm_inner.parameters():
            p.requires_grad = False

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[Dict] = None) -> torch.Tensor:
        # x_t: [B,L,H]
        B, L, H = x_t.shape
        assert H == self.hidden_size

        # Prepare time FiLM
        film = self.time_adapter(t.to(x_t.device))  # [B, 2H]
        scale, shift = film.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        # Create input with time FiLM
        x = self.in_proj(x_t) * (1 + torch.tanh(scale)) + shift

        # Run a minimal reasoning pass using HRM's levels; skip RoPE for strict simplicity
        with torch.no_grad():
            z_H, z_L = x, x
            z_L = self.hrm_inner.L_level(z_L, z_H + x, cos_sin=None)  # type: ignore
            z_H = self.hrm_inner.H_level(z_H, z_L, cos_sin=None)      # type: ignore

        eps = self.out_proj(z_H)
        return eps
