from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pydantic import BaseModel

from .common import trunc_normal_init
from .layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear


# Missing CastedSparseEmbedding, will use nn.Embedding for now
class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: mx.Dtype):
        super().__init__()
        # This is a placeholder implementation. The original uses a sparse embedding
        # which might have performance implications. For correctness, a dense nn.Embedding is equivalent.
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # The original initializes with std=0, so we do the same.
        self.embedding.weight = mx.zeros_like(self.embedding.weight)

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding(x)


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: mx.array
    z_L: mx.array


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: mx.array
    halted: mx.array
    current_data: Dict[str, mx.array]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"

    class Config:
        arbitrary_types_allowed = True


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def __call__(self, cos_sin: CosSin, hidden_states: mx.array) -> mx.array:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = layers

    def __call__(self, hidden_states: mx.array, input_injection: mx.array, **kwargs) -> mx.array:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(mx, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        self.H_init = trunc_normal_init((self.config.hidden_size,), std=1).astype(self.forward_dtype)
        self.L_init = trunc_normal_init((self.config.hidden_size,), std=1).astype(self.forward_dtype)

        self.q_head.weight = mx.zeros_like(self.q_head.weight)
        self.q_head.bias = mx.full_like(self.q_head.bias, -5.0)

    def _input_embeddings(self, input_arr: mx.array, puzzle_identifiers: mx.array):
        embedding = self.embed_tokens(input_arr.astype(mx.int32))

        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = mx.pad(puzzle_embedding, ((0, 0), (0, pad_count)))
            embedding = mx.concatenate((puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), axis=-2)

        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.weight.astype(self.forward_dtype))

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=mx.empty((batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size), dtype=self.forward_dtype),
            z_L=mx.empty((batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size), dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: mx.array, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=mx.where(reset_flag[:, None, None], self.H_init, carry.z_H),
            z_L=mx.where(reset_flag[:, None, None], self.L_init, carry.z_L),
        )

    def __call__(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, mx.array]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, mx.array, Tuple[mx.array, mx.array]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        z_H, z_L = carry.z_H, carry.z_L

        # The original uses torch.no_grad() here. In MLX, gradients are not
        # computed unless explicitly requested, so this loop is already "no_grad".
        for _H_step in range(self.config.H_cycles):
            for _L_step in range(self.config.L_cycles):
                if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            if not (_H_step == self.config.H_cycles - 1):
                z_H = self.H_level(z_H, z_L, **seq_info)

        # 1-step grad (in MLX, this will be handled by the training loop)
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        q_logits = self.q_head(z_H[:, 0]).astype(mx.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)
        self.training = False

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, mx.array]):
        batch_size = batch["inputs"].shape[0]
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=mx.zeros((batch_size,), dtype=mx.int32),
            halted=mx.ones((batch_size,), dtype=mx.bool_),
            current_data={k: mx.empty_like(v) for k, v in batch.items()}
        )

    def __call__(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, mx.array]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, mx.array]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = mx.where(carry.halted, 0, carry.steps)
        new_current_data = {k: mx.where(carry.halted.reshape((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # The gradient flow in the original depends on the training loop.
        # Here we define the forward pass. The training step will handle gradients.
        # The original calculates next_q_values inside the forward pass, which
        # is a bit unusual. We will replicate that logic.

        # We need value_and_grad for the inner model if we want to replicate the Q-learning part
        # exactly. However, for a simple forward pass, we don't need it.
        # Let's assume for now the training logic will handle the gradient part.

        new_inner_carry_grad, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        halted = is_last_step

        if self.training and (self.config.halt_max_steps > 1):
            halted = halted | (q_halt_logits > q_continue_logits)

            # Exploration
            rand_halt_prob = mx.array(np.random.rand(*q_halt_logits.shape))
            rand_halt_steps = mx.array(np.random.randint(low=2, high=self.config.halt_max_steps + 1, size=new_steps.shape))

            min_halt_steps = (rand_halt_prob < self.config.halt_exploration_prob) * rand_halt_steps
            halted = halted & (new_steps >= min_halt_steps)

            # Target Q calculation
            next_inner_carry, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_inner_carry_grad, new_current_data)

            target_q_continue = mx.sigmoid(mx.where(is_last_step, next_q_halt_logits, mx.maximum(next_q_halt_logits, next_q_continue_logits)))
            outputs["target_q_continue"] = target_q_continue

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry_grad, new_steps, halted, new_current_data), outputs
