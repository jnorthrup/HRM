"""
MLX HRM-based denoiser with 4x2 FiLM (time embedding -> [B, 4H])
- Splits into (scale_H, shift_H, scale_L, shift_L)
- Applies FiLM to both z_H and z_L, then one pass through H/L levels.
"""
from __future__ import annotations

from typing import Optional, Dict, Tuple

import math
import mlx.core as mx
import mlx.nn as nn


def sinusoidal_time_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half) / half)
    args = t.astype(mx.float32).reshape((-1, 1)) * freqs.reshape((1, -1))
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
    return emb


class TimeAdapter4x2(nn.Module):
    def __init__(self, hidden_size: int, time_dim: int = 128, zero_init: bool = True):
        super().__init__()
        self.time_dim = time_dim
        self.fc1 = nn.Linear(time_dim, hidden_size * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size * 4)
        if zero_init:
            self.fc2.weight = mx.zeros_like(self.fc2.weight)
            self.fc2.bias = mx.zeros_like(self.fc2.bias)
        else:
            self.fc2.weight = 1e-2 * mx.random.normal(self.fc2.weight.shape)
            self.fc2.bias = mx.zeros_like(self.fc2.bias)

    def __call__(self, t: mx.array) -> mx.array:
        temb = sinusoidal_time_embedding(t, self.time_dim)
        return self.fc2(self.act(self.fc1(temb)))


class HRMDenoiserMx(nn.Module):
    def __init__(self, hrm_inner, hidden_size: int, small_time_init: bool = False):
        super().__init__()
        self.hrm_inner = hrm_inner
        self.hidden_size = hidden_size
        self.time_adapter = TimeAdapter4x2(hidden_size, zero_init=not small_time_init)
        self.inW = mx.eye(hidden_size)
        self.outW = mx.eye(hidden_size)

    def __call__(self, x_t: mx.array, t: mx.array, cond: Optional[Dict] = None) -> mx.array:
        B, L, H = x_t.shape
        film = self.time_adapter(t)
        sH, bH, sL, bL = mx.split(film, 4, axis=-1)
        sH = sH.reshape((B, 1, H))
        bH = bH.reshape((B, 1, H))
        sL = sL.reshape((B, 1, H))
        bL = bL.reshape((B, 1, H))

        z_H = x_t @ self.inW.T
        z_L = x_t @ self.inW.T

        z_H = z_H * (1 + mx.tanh(sH)) + bH
        z_L = z_L * (1 + mx.tanh(sL)) + bL

        # One reasoning pass
        z_L = self.hrm_inner.L_level(z_L, z_H + x_t, cos_sin=None)
        z_H = self.hrm_inner.H_level(z_H, z_L, cos_sin=None)

        eps = z_H @ self.outW.T
        return eps
