#!/usr/bin/env python3
"""
Export HRM (PyTorch) to a fixed-shape Core ML model for ANE tests.

Notes
- Static B=1, fixed seq_len, hidden, heads, etc.
- Halting disabled (halt_max_steps=1) and cycles fixed.
- Inputs: input_ids [1, seq_len] (int32), puzzle_ids [1] (int32, unused if puzzle_emb_ndim=0).
- Output: logits [1, seq_len, vocab_size] (float32).

Requires: coremltools
"""
from __future__ import annotations

import argparse
import sys
from typing import Tuple

import torch


def build_model(cfg: dict):
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    model = HierarchicalReasoningModel_ACTV1(cfg).eval()
    return model


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, puzzle_ids: torch.Tensor) -> torch.Tensor:
        # Shapes: input_ids [1, S] int32, puzzle_ids [1] int32
        batch = {
            "inputs": input_ids.to(torch.int32),
            "puzzle_identifiers": puzzle_ids.to(torch.int32),
        }
        carry = self.model.initial_carry(batch)
        _carry_out, out = self.model(carry, batch)
        return out["logits"].to(torch.float32)


def main():
    p = argparse.ArgumentParser(description="Export HRM to Core ML (fixed shapes)")
    p.add_argument("--out", required=True, help="Output path (.mlpackage or .mlmodel)")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--H-layers", type=int, default=0)
    p.add_argument("--L-layers", type=int, default=0)
    p.add_argument("--pos-enc", choices=["rope", "learned"], default="rope")
    args = p.parse_args()

    cfg = dict(
        batch_size=1,
        seq_len=args.seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=args.vocab,
        H_cycles=1,
        L_cycles=1,
        H_layers=args.H_layers,
        L_layers=args.L_layers,
        hidden_size=args.hidden,
        expansion=4,
        num_heads=args.heads,
        pos_encodings=args.pos_enc,
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
    )

    model = build_model(cfg)
    wrapper = ExportWrapper(model).eval()

    # Example inputs (static shapes)
    input_ids = torch.zeros((1, args.seq_len), dtype=torch.int32)
    puzzle_ids = torch.zeros((1,), dtype=torch.int32)

    try:
        import coremltools as ct
        import numpy as np
    except Exception as e:
        print("coremltools not installed. Install with: pip install coremltools", file=sys.stderr)
        sys.exit(2)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (input_ids, puzzle_ids))

    # Describe inputs for Core ML
    inputs = [
        ct.TensorType(name="input_ids", shape=input_ids.shape, dtype=np.int32),
        ct.TensorType(name="puzzle_ids", shape=puzzle_ids.shape, dtype=np.int32),
    ]

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.save(args.out)
    print(f"Saved Core ML model to {args.out}")


if __name__ == "__main__":
    main()
