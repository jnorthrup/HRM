"""
Sample HRM latent diffusion using deterministic DDIM.
This is a minimal wiring assuming hrm_act_v1 config and inner model availability.
"""
from __future__ import annotations

import argparse
import torch

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.diffusion import DDIMScheduler, HRMDenoiser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")

    # Minimal HRM config for inner-only usage
    config = dict(
        batch_size=1,
        seq_len=args.seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=256,
        H_cycles=1,
        L_cycles=1,
    H_layers=0,
    L_layers=0,
        hidden_size=args.hidden,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
    )

    hrm = HierarchicalReasoningModel_ACTV1(config)
    hrm.eval()

    denoiser = HRMDenoiser(hrm.inner, hidden_size=args.hidden)
    sched = DDIMScheduler(timesteps=args.steps, beta_schedule="cosine")

    # Sample latent
    z0 = sched.sample(
        steps=args.steps,
        shape=(1, args.seq_len, args.hidden),
        denoiser=lambda x, t, c: denoiser(x, t, c),
        device=device,
        dtype=torch.float32,
        seed=args.seed,
    )

    print("z0 mean/std:", z0.mean().item(), z0.std().item())


if __name__ == "__main__":
    main()
