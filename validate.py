import torch
import mlx.core as mx
import numpy as np
import yaml
import argparse
from pathlib import Path

# PyTorch imports
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1 as HierarchicalReasoningModel_ACTV1_torch
from models.losses import ACTLossHead

# MLX imports
from models.hrm_mlx.hrm_act_v1_mlx import HierarchicalReasoningModel_ACTV1 as HierarchicalReasoningModel_ACTV1_mlx
from evaluate_mlx import MLXPuzzleDataset # Reuse the dataset loader from evaluate_mlx

def run_torch_model(model, batch):
    """Runs the PyTorch model until halting and returns logits."""
    batch_torch = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else torch.from_numpy(v) for k, v in batch.items()}

    carry = model.initial_carry(batch_torch)
    while True:
        carry, outputs, _, _, all_finish = model(carry=carry, batch=batch_torch, return_keys=[])
        if all_finish:
            break
    return outputs['logits'].detach().numpy()

def run_mlx_model(model, batch):
    """Runs the MLX model until halting and returns logits."""
    batch_mlx = {k: mx.array(v) for k, v in batch.items()}

    carry = model.initial_carry(batch_mlx)
    while True:
        carry, outputs = model(carry, batch_mlx)
        if carry.halted.all():
            break
    mx.eval(outputs['logits']) # Make sure computation is done
    return np.array(outputs['logits'])

def main():
    parser = argparse.ArgumentParser(description="Validate MLX port against the original PyTorch model.")
    parser.add_argument("--torch_checkpoint_dir", type=str, required=True, help="Path to the directory containing the PyTorch checkpoint and config.")
    parser.add_argument("--mlx_weights_path", type=str, required=True, help="Path to the converted MLX weights (.safetensors).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for validation.")

    args = parser.parse_args()

    # --- Load Config ---
    config_path = Path(args.torch_checkpoint_dir) / "all_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    arch_config = config['arch']
    model_cfg_dict = arch_config.get('__pydantic_extra__', {})
    model_cfg_dict.update({
        'batch_size': args.batch_size,
        'vocab_size': config.get('vocab_size', 28),
        'seq_len': config.get('seq_len', 400),
        'num_puzzle_identifiers': config.get('num_puzzle_identifiers', 1120),
        'causal': False,
    })

    # --- Load PyTorch Model ---
    print("Loading PyTorch model...")
    torch_checkpoint_path = next((Path(args.torch_checkpoint_dir).glob("step_*.pt")), None)
    if not torch_checkpoint_path:
        torch_checkpoint_path = next(Path(args.torch_checkpoint_dir).glob("*.pt"))

    torch_model = HierarchicalReasoningModel_ACTV1_torch(model_cfg_dict)
    torch_model_with_loss = ACTLossHead(torch_model, **arch_config['loss']['__pydantic_extra__'])
    state_dict = torch.load(torch_checkpoint_path, map_location="cpu")
    torch_model_with_loss.load_state_dict(state_dict)
    torch_model_with_loss.eval()

    # --- Load MLX Model ---
    print("Loading MLX model...")
    mlx_model = HierarchicalReasoningModel_ACTV1_mlx(model_cfg_dict)
    mlx_model.load_weights(args.mlx_weights_path)
    mlx_model.eval()
    mx.eval(mlx_model.parameters())

    # --- Load Data ---
    print("Loading data...")
    # Use a modified MLXPuzzleDataset that yields numpy arrays
    class NumpyPuzzleDataset(MLXPuzzleDataset):
        def _collate_batch(self, batch: dict) -> dict:
            # Override to return numpy arrays instead of mlx arrays
            if self.metadata.get('ignore_label_id') is not None:
                batch["labels"][batch["labels"] == self.metadata['ignore_label_id']] = IGNORE_LABEL_ID

            actual_batch_size = batch["inputs"].shape[0]
            if actual_batch_size < self.batch_size:
                pad_size = self.batch_size - actual_batch_size
                pad_values = {"inputs": self.metadata['pad_id'], "labels": IGNORE_LABEL_ID, "puzzle_identifiers": self.metadata['blank_identifier_id']}
                batch = {k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), 'constant', constant_values=pad_values[k]) for k, v in batch.items()}
            return batch

    dataset = NumpyPuzzleDataset(args.dataset_path, "test", args.batch_size)
    _, batch_np, _ = next(iter(dataset))

    print("Running inference on both models...")

    # --- Run Inference ---
    # The batch from NumpyPuzzleDataset is already in numpy format
    torch_logits = run_torch_model(torch_model_with_loss, batch_np)
    mlx_logits = run_mlx_model(mlx_model, batch_np)

    # --- Compare Outputs ---
    print("\n--- Validation Results ---")
    print(f"PyTorch logits shape: {torch_logits.shape}")
    print(f"MLX logits shape: {mlx_logits.shape}")

    if torch_logits.shape != mlx_logits.shape:
        print("Error: Output shapes do not match!")
        return

    abs_diff = np.abs(torch_logits - mlx_logits)
    mse = np.mean(np.square(torch_logits - mlx_logits))

    print(f"Mean Absolute Difference: {np.mean(abs_diff):.6f}")
    print(f"Max Absolute Difference: {np.max(abs_diff):.6f}")
    print(f"Mean Squared Error: {mse:.6f}")

    # Set a tolerance for the check
    tolerance = 1e-4
    if np.allclose(torch_logits, mlx_logits, atol=tolerance):
        print(f"\n✅ Outputs are numerically close (tolerance={tolerance}). The port is likely correct.")
    else:
        print(f"\n❌ Outputs are not close enough (tolerance={tolerance}). There might be an issue with the port.")

if __name__ == "__main__":
    main()
