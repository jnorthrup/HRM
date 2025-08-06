from typing import Tuple
import math

import mlx.core as mx
import mlx.nn as nn

from .common import trunc_normal_init


CosSin = Tuple[mx.array, mx.array]


def _find_multiple(a, b):
    return (-(a // -b)) * b


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = trunc_normal_init((out_features, in_features), std=1.0 / (in_features ** 0.5))

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        weight = self.weight.astype(x.dtype)
        bias = self.bias.astype(x.dtype) if self.bias is not None else None
        return nn.linear(x, weight, bias)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: mx.Dtype):
        super().__init__()
        self.cast_to = cast_to
        self.weight = trunc_normal_init((num_embeddings, embedding_dim), std=init_std)

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x].astype(self.cast_to)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base):
        super().__init__()

        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        t = mx.arange(max_position_embeddings, dtype=mx.float32)
        freqs = mx.outer(t, inv_freq)

        emb = mx.concatenate([freqs, freqs], axis=-1)
        self.cos_cached = mx.cos(emb)
        self.sin_cached = mx.sin(emb)

    def __call__(self):
        return self.cos_cached, self.sin_cached


def rotate_half(x: mx.array):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.astype(cos.dtype)
    k = k.astype(cos.dtype)

    q_embed = (q * cos[None, :, None, :]) + (rotate_half(q) * sin[None, :, None, :])
    k_embed = (k * cos[None, :, None, :]) + (rotate_half(k) * sin[None, :, None, :])

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def __call__(self, cos_sin: CosSin, hidden_states: mx.array) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Transpose for attention
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_output = nn.scaled_dot_product_attention(query, key, value, mask=None, is_causal=self.causal)

        # Transpose back and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def __call__(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up)


def rms_norm(hidden_states: mx.array, variance_epsilon: float) -> mx.array:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(mx.float32)

    variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * mx.rsqrt(variance + variance_epsilon)
    return hidden_states.astype(input_dtype)
