"""Mixture‑of‑Depths HRM.

Three depth branches (shallow/medium/deep) plus a gating head.
"""
import torch
from torch import nn
from typing import Dict, Tuple

from .hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Carry,
)


class MixtureOfDepthsHRM(nn.Module):
    """Wrap three HRM instances with a learned gating head."""

    def __init__(
        self,
        depth_configs: Tuple[HierarchicalReasoningModel_ACTV1Config, ...],
    ):
        super().__init__()
        assert len(depth_configs) == 3, "Need exactly three depth configs"

        # One HRM per depth
        self.branches = nn.ModuleList(
            [HierarchicalReasoningModel_ACTV1(cfg) for cfg in depth_configs]
        )

        # Gating head: 2 halt logits -> 3 depth probabilities
        self.gating_head = nn.Sequential(
            nn.Linear(2, 3), nn.Softmax(dim=-1)
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Run all depth branches
        outputs = []
        for branch in self.branches:
            out_carry, out_logits, (q_halt, q_cont) = branch(carry, batch)
            outputs.append((out_carry, out_logits, q_halt, q_cont))

        # Shallow branch provides gating logits
        _, _, q_halt_shallow, q_cont_shallow = Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:  # [B,3]

        # Weighted sum across depths
        logits = sum(w * o[1] for w, o in zip(gating_weights.split(1, -1), outputs))
        q_halt = sum(w * o[2] for w, o in zip(gating_weights.split(1, -1), outputs))
        q_cont = sum(w * o[3] for w, o in zip(gating_weights.split(1, -1), outputs))

        # Use deepest carry (most computation)
        final_carry = outputs[-1][0]

        return final_carry, {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
            "gating_weights": gating_weights,
        }
