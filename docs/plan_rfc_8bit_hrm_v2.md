# Revised Plan: Adding the IETF RFC Collection to the 8‑bit HRM

## Why a revision?
The original plan assumed a full download of all ~900 RFCs and a pure byte‑level tokeniser.  
Based on the feedback we propose a **lighter, more flexible** approach that still gives the model solid English exposure while keeping the pipeline simple.

## 1. Scope of the RFC Corpus
| Option | Description | Approx. Size |
|--------|-------------|--------------|
| **A – Full set** | All RFCs (~900 files, 120 MiB gzipped) | 150 MiB (raw) |
| **B – Representative subset** | 100 most‑cited RFCs (covers ~80 % of terminology) | 15 MiB (raw) |
| **C – Incremental sampling** | 10 % random sample, refreshed each training run | 12 MiB (raw) |

*We recommend **Option B** for a quick proof‑of‑concept and to keep download time short.*

## 2. Tokenisation Strategy
| Strategy | Pros | Cons |
|----------|------|------|
| **Byte‑level (256‑token)** | No extra vocab, matches existing binary pipeline | No sub‑word semantics, longer sequences |
| **Byte‑pair‑encoding (BPE, 30 k vocab)** | Captures common words, reduces sequence length | Requires a separate tokenizer, adds a non‑byte vocab to the 8‑bit model |
| **Hybrid** – keep byte‑level for binary, BPE for text (dual‑head) | Best of both worlds | More complex model architecture |

*For the first iteration we stick to **byte‑level** (no extra vocab) to keep the model unchanged.*

## 3. Dataset Construction
1. **Download** the chosen RFC subset (script will accept a flag `--subset` to pick A/B/C).  
2. **Pre‑process** each file:  
   * Strip headers/footers (`RFC xxxx`, `Copyright`).  
   * Replace non‑ASCII bytes with `0xFF`.  
   * Convert to `torch.uint8` tensor.  
3. **Mix with binary data**:  
   * Use a **balanced sampler** that yields 50 % binary and 50 % RFC batches.  
   * Store as two files: `data/mixed_train.pt` and `data/mixed_val.pt`.  

## 4. Config Changes
```yaml
# config/cfg_mixed_binary_rfc.yaml
defaults:
  - arch: hrm_binary
  - _self_
train_data_path: data/mixed_train.pt
val_data_path:   data/mixed_val.pt
batch_size: 64
halt_max_steps: 20          # allow a few more steps for longer text
halt_exploration_prob: 0.05
```

## 5. Training Procedure
| Phase | Description | Epochs |
|-------|-------------|--------|
| **Binary‑only warm‑up** | Train on existing binary data to stabilise the low‑level encoder. | 2 |
| **Mixed training** | Continue training on the mixed dataset. | 4–6 |
| **Fine‑tune on text** (optional) | Slightly lower learning rate, focus on RFC data only to improve language fluency. | 1–2 |

## 6. Evaluation
* **Binary reconstruction loss** – ensure no regression on original tasks.  
* **RFC perplexity** – compute byte‑level perplexity on a held‑out set of RFCs.  
* **Qualitative test** – prompt the model with “RFC 2119 – ” and sample 200 bytes; check for coherent English fragments.

## 7. Optional CoreML Export
The model remains **byte‑level**, so conversion with `convert_weights.py` works unchanged.  
If we later adopt a BPE vocab, we would need a small embedding conversion step.

## 8. Risks & Mitigations (updated)
| Risk | Mitigation |
|------|------------|
| **Catastrophic forgetting of binary skills** | Keep a binary‑only warm‑up phase; use a balanced sampler throughout training. |
| **Sequence length explosion** (text → longer than binary) | Limit `seq_len` to 512 bytes; truncate longer RFC paragraphs or split into windows. |
| **Download overhead** | Offer the three corpus‑size options; default to the 100‑RFC subset for quick iteration. |
| **Tokenizer mismatch** (if later switch to BPE) | Design the config to allow swapping the embedding layer without touching the rest of the model. |

---

### Next Steps
1. **Confirm** which RFC subset (A/B/C) you’d like to use.  
2. Approve the revised plan (or suggest further tweaks).  
3. Once approved, I’ll create the required scripts (`download_rfc.sh`, `build_mixed_dataset.py`) and update the config files.

Please let me know your preferences!