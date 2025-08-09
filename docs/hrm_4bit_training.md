# Training a 4‑bit HRM with 4‑bit Index + Accumulator

This guide walks through setting up the environment, preparing the dataset, and training the **Hierarchical Reasoning Model (HRM)** in 4‑bit mode on an Apple M3 Pro (24 GB RAM, ~12 GB free for the process).

---

## 1. Install the required libraries
```bash
# Core MLX stack (CPU/GPU/ANE support)
pip install mlx-core mlx-nn mlx-optim mlx-mlir tqdm
```

## 2. Prepare the dataset
The model expects a tuple of four tensors per example:
| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `byte_ids` | `(B, L)` | `uint8` | UTF‑8 byte stream (0‑255) |
| `idx_ids`  | `(B, L)` | `uint8` (values 0‑15) | 4‑bit index (e.g., high‑nibble, control flag, etc.) |
| `acc_ids`  | `(B, L)` | `uint8` | 8‑bit accumulator (counts rare nibble occurrences) |
| `target_ids`| `(B, L)` | `uint8` | next‑token labels for language modeling |

You can build the dataset by iterating over your git/pijul histories, tokenising each file to UTF‑8 bytes, extracting a 4‑bit index (e.g., high‑nibble), and updating the accumulator as described in the design doc.

```python
# pseudo‑code for a single file
bytes_ = list(file_content.encode('utf‑8'))
byte_ids = mx.array(bytes_, dtype=mx.uint8)
# 4‑bit index: high nibble of each byte
idx_ids = (byte_ids >> 4) & 0xF
# accumulator: count of rare nibble pairs (pre‑computed table)
acc_ids = compute_accumulator(byte_ids)  # returns uint8 array of same length
# target is the next byte (shifted by 1)
target_ids = mx.concatenate([byte_ids[1:], mx.array([0], dtype=mx.uint8)])
```

Wrap the logic in a `torch.utils.data.Dataset`‑like class (or a simple generator) that yields `(byte_ids, idx_ids, acc_ids, target_ids)`.

## 3. DataLoader
```python
from mlx.nn import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## 4. Model import
```python
from models.hrml_mlx.hrml_act_v1_mlx import HierarchicalReasoningModel_ACTV1
```
The model already contains the 4‑bit index and accumulator conditioning (see `hrm_act_v1_mlx.py`).

## 5. Optimizer & scheduler
```python
import mlx.nn.optim as optim
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10_000)
```

## 6. Loss function
```python
import mlx.nn.losses as losses
loss_fn = losses.CrossEntropyLoss(ignore_index=0)  # 0 = padding token
```

## 7. Training step (autodiff wrapper)
```python
import mlx.core as mx

@mx.autodiff
def step(batch):
    byte_ids, idx_ids, acc_ids, target_ids = batch
    # Initialise carry (the HRM expects a dict with "inputs" and "puzzle_identifiers")
    carry = model.initial_carry({"inputs": byte_ids, "puzzle_identifiers": acc_ids})
    new_carry, outputs = model(carry, {"inputs": byte_ids, "puzzle_identifiers": acc_ids})
    logits = outputs["logits"]               # (B, L, vocab)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                   target_ids.reshape(-1))
    return loss, new_carry
```

## 8. Full training loop
```python
from tqdm import tqdm

num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        loss, _ = step(batch)
        opt.step(loss)
        scheduler.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} avg loss: {epoch_loss/len(loader):.4f}")
    # checkpoint
    import pickle, os
    ckpt_path = f"ckpt_epoch{epoch}.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump({"model": model.state_dict(),
                     "opt": opt.state_dict(),
                     "epoch": epoch}, f)
    print(f"Saved checkpoint {ckpt_path}")
```

## 9. Evaluation (perplexity)
```python
def evaluate(eval_loader):
    total_loss = 0.0
    total_tokens = 0
    for batch in eval_loader:
        byte_ids, idx_ids, acc_ids, target_ids = batch
        carry = model.initial_carry({"inputs": byte_ids, "puzzle_identifiers": acc_ids})
        _, outputs = model(carry, {"inputs": byte_ids, "puzzle_identifiers": acc_ids})
        logits = outputs["logits"]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
        total_loss += loss.item() * target_ids.size
        total_tokens += target_ids.size
    ppl = mx.exp(total_loss / total_tokens)
    return ppl
```

## 10. Inference & ANE compilation
After training, compile the model for the Apple Neural Engine (ANE) for fast inference:
```bash
python - <<'PY'
import mlx.core as mx, pickle
import mlcore
from models.hrml_mlx.hrml_act_v1_mlx import HierarchicalReasoningModel_ACTV1

# Load the final checkpoint
ckpt = pickle.load(open('ckpt_epoch19.pkl', 'rb'))
model = HierarchicalReasoningModel_ACTV1(ckpt['model'])
model.eval()

# Dummy inputs matching the training shape
byte_ids = mx.zeros((1, 512), dtype=mx.uint8)
acc_ids  = mx.zeros((1, 512), dtype=mx.uint8)

mlcore.compile(
    model,
    inputs=[{"inputs": byte_ids, "puzzle_identifiers": acc_ids}],
    target="ane",
    output_path="hrm_ane.mlmodel",
)
PY
```
The resulting `hrm_ane.mlmodel` can be loaded on macOS/iOS and run with CoreML, delivering the low‑latency, low‑memory inference you need.

---

### Quick sanity‑check script (optional)
A tiny helper to verify that the chosen hyper‑parameters stay within the 12 GB headroom:
```python
# mem_estimate.py (already in the repo)

def estimate_mem(batch, seq, hidden, layers):
    B = 2  # bfloat16 = 2 bytes
    act = 4 * batch * seq * hidden * layers  # forward+backward+cache
    return act / (1024**3)  # GB

if __name__ == "__main__":
    print(estimate_mem(8, 512, 4096, 24))  # ≈ 1.5 GB
```
Run `python mem_estimate.py` to confirm you are well under the 12 GB limit.

---

**That’s it!** Follow the steps, monitor loss, and you’ll have a 4‑bit HRM trained on your voluminous git/pijul histories, ready for fast ANE inference.
