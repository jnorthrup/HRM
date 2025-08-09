#!/usr/bin/env python3
"""
Deterministic benchmark for BEFORE (no diffusion) vs AFTER (DDIM + HRM denoiser).

- Runs each pipeline twice with identical seeds and fixed iteration count.
- Measures elapsed time and tokens/sec.
- Streams outputs into SHA256 to verify identical results across runs.

Before = sample Gaussian z ~ N(0,1) directly, then decode via lm_head.
After  = DDIM sampling in hidden space using HRMDenoiser, then decode via lm_head.

This is eval-only; uses CPU and float32 for determinism.
"""
from __future__ import annotations

import argparse
import sys
import pathlib
import hashlib
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

# Ensure repo root is on sys.path for direct script execution
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from models.diffusion import DDIMScheduler, HRMDenoiser


def set_all_seeds(seed: int):
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_model(hidden_size: int, batch_size: int, seq_len: int) -> HierarchicalReasoningModel_ACTV1:
    config = dict(
        batch_size=batch_size,
        seq_len=seq_len,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=256,
        H_cycles=1,
        L_cycles=1,
        H_layers=0,  # keep light for CPU determinism bench
        L_layers=0,
        hidden_size=hidden_size,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
    )
    model = HierarchicalReasoningModel_ACTV1(config).eval()
    return model


def sha256_update(h: "hashlib._Hash", t: torch.Tensor):
    h.update(t.detach().cpu().contiguous().numpy().tobytes())


@torch.no_grad()
def run_before(model: HierarchicalReasoningModel_ACTV1, iters: int, seed: int, shape: tuple[int, int, int]):
    B, L, H = shape
    h1, h2 = hashlib.sha256(), hashlib.sha256()

    # warmup
    _ = model.inner.lm_head(torch.randn(B, L, H, dtype=torch.float32))

    def one_pass(base_seed: int):
        tokens = 0
        start = time.time()
        for i in range(iters):
            g = torch.Generator(device="cpu").manual_seed(base_seed + i)
            z = torch.randn(B, L, H, dtype=torch.float32, generator=g)
            toks = model.inner.lm_head(z).argmax(dim=-1)
            tokens += int(toks.numel())
            yield toks
        return time.time() - start, tokens

    # run 1
    t0 = time.time()
    tokens1 = 0
    for toks in one_pass(seed):
        sha256_update(h1, toks)
        tokens1 += int(toks.numel())
    elapsed1 = time.time() - t0

    # run 2
    t1 = time.time()
    tokens2 = 0
    for toks in one_pass(seed):
        sha256_update(h2, toks)
        tokens2 += int(toks.numel())
    elapsed2 = time.time() - t1

    return {
        "iters": iters,
        "tokens_run1": tokens1,
        "tokens_run2": tokens2,
        "elapsed_run1_s": round(elapsed1, 4),
        "elapsed_run2_s": round(elapsed2, 4),
        "hash_run1": h1.hexdigest(),
        "hash_run2": h2.hexdigest(),
        "identical": h1.digest() == h2.digest(),
        "toks_per_s_run1": round(tokens1 / elapsed1, 2) if elapsed1 > 0 else None,
        "toks_per_s_run2": round(tokens2 / elapsed2, 2) if elapsed2 > 0 else None,
    }


@torch.no_grad()
def run_after(model: HierarchicalReasoningModel_ACTV1, iters: int, seed: int, shape: tuple[int, int, int], steps: int):
    B, L, H = shape
    device = torch.device("cpu")
    denoiser = HRMDenoiser(model.inner, hidden_size=H)
    sched = DDIMScheduler(timesteps=steps, beta_schedule="cosine")
    h1, h2 = hashlib.sha256(), hashlib.sha256()

    # warmup
    _ = sched.sample(steps, (B, L, H), lambda x, t, c: denoiser(x, t, c), device=device, dtype=torch.float32, seed=seed)

    def one_pass(base_seed: int):
        tokens = 0
        start = time.time()
        for i in range(iters):
            z0 = sched.sample(steps, (B, L, H), lambda x, t, c: denoiser(x, t, c), device=device, dtype=torch.float32, seed=base_seed + i)
            toks = model.inner.lm_head(z0).argmax(dim=-1)
            tokens += int(toks.numel())
            yield toks
        return time.time() - start, tokens

    # run 1
    t0 = time.time()
    tokens1 = 0
    for toks in one_pass(seed):
        sha256_update(h1, toks)
        tokens1 += int(toks.numel())
    elapsed1 = time.time() - t0

    # run 2
    t1 = time.time()
    tokens2 = 0
    for toks in one_pass(seed):
        sha256_update(h2, toks)
        tokens2 += int(toks.numel())
    elapsed2 = time.time() - t1

    return {
        "iters": iters,
        "tokens_run1": tokens1,
        "tokens_run2": tokens2,
        "elapsed_run1_s": round(elapsed1, 4),
        "elapsed_run2_s": round(elapsed2, 4),
        "hash_run1": h1.hexdigest(),
        "hash_run2": h2.hexdigest(),
        "identical": h1.digest() == h2.digest(),
        "toks_per_s_run1": round(tokens1 / elapsed1, 2) if elapsed1 > 0 else None,
        "toks_per_s_run2": round(tokens2 / elapsed2, 2) if elapsed2 > 0 else None,
    }


def _avg_time(fn, n: int = 5) -> float:
    """Average wall time for running fn() n times."""
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n


def main():
    p = argparse.ArgumentParser(description="Before/After diffusion deterministic benchmark")
    p.add_argument("--iters", type=int, default=int(os.environ.get("HRM_BENCH_ITERS", 64)), help="Fixed iterations per run (ignored if --seconds or --seconds-total provided)")
    p.add_argument("--seed", type=int, default=int(os.environ.get("HRM_BENCH_SEED", 123)))
    p.add_argument("--steps", type=int, default=int(os.environ.get("HRM_BENCH_STEPS", 20)))
    p.add_argument("--B", type=int, default=int(os.environ.get("HRM_BENCH_B", 1)))
    p.add_argument("--L", type=int, default=int(os.environ.get("HRM_BENCH_L", 128)))
    p.add_argument("--H", type=int, default=int(os.environ.get("HRM_BENCH_H", 512)))
    p.add_argument("--seconds", type=float, default=os.environ.get("HRM_BENCH_SECONDS"), help="Target seconds per run (each of the 4 runs). If provided, overrides --iters.")
    p.add_argument("--seconds-total", dest="seconds_total", type=float, default=os.environ.get("HRM_BENCH_SECONDS_TOTAL"), help="Target total seconds across all 4 runs (before x2 + after x2). Overrides --seconds and --iters.")
    args = p.parse_args()

    shape = (args.B, args.L, args.H)
    set_all_seeds(args.seed)
    model = build_model(args.H, args.B, args.L)

    # Determine iteration counts
    seconds_per_run: float | None = None
    if args.seconds_total:
        try:
            total = float(args.seconds_total)
            seconds_per_run = max(0.01, total / 4.0)
        except Exception:
            seconds_per_run = None
    elif args.seconds:
        try:
            seconds_per_run = max(0.01, float(args.seconds))
        except Exception:
            seconds_per_run = None

    if seconds_per_run is not None:
        # Calibrate per pipeline to hit approximately seconds_per_run deterministically
        B, L, H = shape
        device = torch.device("cpu")
        denoiser = HRMDenoiser(model.inner, hidden_size=H)
        sched = DDIMScheduler(timesteps=args.steps, beta_schedule="cosine")

        # Prepare single-iteration callables using disjoint seeds to avoid overlapping with actual runs
        calib_seed = args.seed + 10_000_000

        def one_before():
            g = torch.Generator(device="cpu").manual_seed(calib_seed)
            z = torch.randn(B, L, H, dtype=torch.float32, generator=g)
            _ = model.inner.lm_head(z).argmax(dim=-1)

        def one_after():
            z0 = sched.sample(args.steps, (B, L, H), lambda x, t, c: denoiser(x, t, c), device=device, dtype=torch.float32, seed=calib_seed)
            _ = model.inner.lm_head(z0).argmax(dim=-1)

        t_iter_before = _avg_time(one_before, n=3)
        t_iter_after = _avg_time(one_after, n=3)
        iters_before = max(1, int(seconds_per_run / max(t_iter_before, 1e-9)))
        iters_after = max(1, int(seconds_per_run / max(t_iter_after, 1e-9)))
    else:
        iters_before = args.iters
        iters_after = args.iters

    before = run_before(model, iters_before, args.seed, shape)
    after = run_after(model, iters_after, args.seed, shape, args.steps)

    summary = {
        "shape": list(shape),
        "steps": args.steps,
        "iters": {
            "requested": args.iters,
            "before": iters_before,
            "after": iters_after,
        },
        "seconds": {
            "per_run": seconds_per_run,
            "total_requested": float(args.seconds_total) if args.seconds_total else None,
        },
        "seed": args.seed,
        "before": before,
        "after": after,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
