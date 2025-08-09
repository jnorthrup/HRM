# Discussion: 4‑bit Nibble Modulation & Dead‑Energy Accumulator

## Overview

We explored how a **4‑bit index** can act as a learned modulator for an 8‑bit byte stream in the HRM model. The key ideas are:

1. **Dual embeddings** – a standard 8‑bit byte embedding (`E8`) and a compact 4‑bit embedding (`E4`).
2. **FiLM‑style gating** – project the 4‑bit embedding to scale (`γ`) and shift (`β`) parameters that modulate the byte embedding: `modulated = γ * E8 + β`.
3. **Tiling on M3 Pro** – 8‑bit GEMM uses 16 × 16 tiles, while the 4‑bit FiLM projection fits the ANE’s 32 × 32 tiles, giving comparable throughput with half the weight storage.
4. **Unicode handling** – the model still consumes UTF‑8 byte streams; the 4‑bit channel can encode context such as “inside a multibyte sequence” without needing a Unicode‑aware tokenizer.

## Capturing “Dead‑Energy” from Rare Nibble Pairs

Rare nibble‑pair patterns (e.g., byte pairs that never appear in natural language) can be turned into a **tiny accumulator** that provides an extra signal:

- Pre‑compute a static table of the least‑frequent 4‑bit values.
- During inference, update an 8‑bit counter `acc` whenever a rare nibble occurs: `acc = (acc + 1) & 0xFF`.
- Embed `acc` with a small linear layer and use it to FiLM‑modulate the main embedding, giving the model a learned “energy” context that only grows on rare sequences.
- This operation is cheap (single integer add & mask) and maps cleanly onto the ANE’s tiled kernels.

## Memory & Throughput Impact

| Component | 8‑bit only | 8‑bit + 4‑bit modulator | Dead‑energy accumulator |
|----------|------------|--------------------------|------------------------|
| Embedding matrix | `256 × H` | `256 × H` + `16 × H` (≈ 6 % extra) | `256 × H` (for `acc` embedding) |
| Output head | `256 × H` | unchanged | unchanged |
| Hidden states | `B × L × H` | unchanged | unchanged |
| Bandwidth | reads `256×H` weights per batch | adds `16×H` + one `H×2H` projection per token | adds one extra `256×H` projection per step (tiny) |
| Tile utilisation | 16 × 16 (int8) | 32 × 32 (int4) for FiLM, 16 × 16 for main path | same as modulator |

## Integration Checklist

1. Add a second input tensor (`idx_ids`) to the data pipeline.
2. Replace the original byte embedding with `DualModulatedEmbedding` (see code snippet).
3. Generate the 4‑bit index (rule‑based or learned) alongside the byte stream.
4. Implement the rare‑nibble accumulator and its embedding.
5. Re‑compile with `mlcore.compile(target="ane")` – the compiler will emit tiled kernels for both int8 and int4 paths.

---
*This file records the design discussion for future reference and version control.*
