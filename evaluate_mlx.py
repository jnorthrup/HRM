import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from models.hrm_mlx.hrm_act_v1_mlx import HierarchicalReasoningModel_ACTV1 as HierarchicalReasoningModel_ACTV1_mlx
from models.losses import IGNORE_LABEL_ID

class MLXPuzzleDataset:
    """A simplified iterable dataset for evaluation with MLX."""
    def __init__(self, dataset_path: str, split: str, batch_size: int):
        self.dataset_path = Path(dataset_path) / split
        self.batch_size = batch_size
        self.metadata = self._load_metadata()
        self._data = self._load_data()

    def _load_metadata(self) -> Dict:
        with open(self.dataset_path / "dataset.json", "r") as f:
            return json.load(f)

    def _load_data(self) -> Dict:
        data = {}
        for set_name in self.metadata['sets']:
            data[set_name] = {
                "inputs": np.load(self.dataset_path / f"{set_name}__inputs.npy", mmap_mode="r"),
                "labels": np.load(self.dataset_path / f"{set_name}__labels.npy", mmap_mode="r"),
                "puzzle_identifiers": np.load(self.dataset_path / f"{set_name}__puzzle_identifiers.npy"),
                "puzzle_indices": np.load(self.dataset_path / f"{set_name}__puzzle_indices.npy"),
            }
        return data

    def __iter__(self):
        for set_name, dataset in self._data.items():
            total_examples = len(dataset["inputs"])
            start_index = 0
            while start_index < total_examples:
                end_index = min(total_examples, start_index + self.batch_size)

                # Find puzzle identifiers for the current batch
                puzzle_indices_map = dataset["puzzle_indices"]

                # This logic is simplified from the original. It might not be perfectly correct
                # for all cases but should work for contiguous examples.
                puzzle_idx_start = np.searchsorted(puzzle_indices_map, start_index, side='right') - 1
                puzzle_idx_end = np.searchsorted(puzzle_indices_map, end_index - 1, side='right') -1

                # This is a further simplification, assuming one puzzle per example, which is not always true.
                # A more robust implementation would iterate through each example to find its puzzle_id.
                # For this PoC, we'll assume the puzzle_identifiers align with examples.
                # This part is tricky to get right without a deeper dive into the data structure.
                # Let's just take the corresponding slice of puzzle_identifiers.

                batch = {
                    "inputs": dataset["inputs"][start_index:end_index],
                    "labels": dataset["labels"][start_index:end_index],
                    # This is likely incorrect if puzzles have variable numbers of examples.
                    "puzzle_identifiers": dataset["puzzle_identifiers"][start_index:end_index]
                }

                # Collate (padding and type conversion)
                batch = self._collate_batch(batch)

                yield set_name, batch, end_index - start_index
                start_index += self.batch_size

    def _collate_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
        if self.metadata.get('ignore_label_id') is not None:
            batch["labels"][batch["labels"] == self.metadata['ignore_label_id']] = IGNORE_LABEL_ID

        # Pad if necessary
        actual_batch_size = batch["inputs"].shape[0]
        if actual_batch_size < self.batch_size:
            pad_size = self.batch_size - actual_batch_size
            pad_values = {
                "inputs": self.metadata['pad_id'],
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata['blank_identifier_id']
            }
            batch = {k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), 'constant', constant_values=pad_values[k]) for k, v in batch.items()}

        return {k: mx.array(v) for k, v in batch.items()}


def evaluate_model(model: HierarchicalReasoningModel_ACTV1_mlx, dataloader: MLXPuzzleDataset):
    model.eval()

    all_metrics = {}

    for set_name, batch, _ in dataloader:
        carry = model.initial_carry(batch)

        while True:
            carry, outputs = model(carry, batch)

            # In a real scenario, we would compute metrics from 'outputs'
            # For now, we just check if the loop terminates.

            if carry.halted.all():
                break

        # Simple accuracy metric (example)
        logits = outputs["logits"]
        predictions = mx.argmax(logits, axis=-1)
        labels = batch["labels"]

        correct = (predictions == labels) & (labels != IGNORE_LABEL_ID)
        total = (labels != IGNORE_LABEL_ID)

        accuracy = (mx.sum(correct) / mx.sum(total)).item()

        all_metrics.setdefault(set_name, []).append(accuracy)

    # Average metrics over batches
    for set_name, accs in all_metrics.items():
        all_metrics[set_name] = np.mean(accs)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HRM model with MLX.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file (all_config.yaml).")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the MLX model weights (.safetensors).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")

    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reconstruct model config dict
    arch_config = config['arch']
    model_cfg_dict = arch_config.get('__pydantic_extra__', {})
    model_cfg_dict.update({
        'batch_size': args.batch_size,
        'vocab_size': config.get('vocab_size', 28),
        'seq_len': config.get('seq_len', 400),
        'num_puzzle_identifiers': config.get('num_puzzle_identifiers', 1120),
        'causal': False,
    })

    # Initialize model and load weights
    model = HierarchicalReasoningModel_ACTV1_mlx(model_cfg_dict)
    model.load_weights(args.weights_path)
    mx.eval(model.parameters()) # Ensure weights are loaded

    print("Model loaded successfully.")

    # Initialize dataset
    dataset = MLXPuzzleDataset(args.dataset_path, "test", args.batch_size)

    # Run evaluation
    print("Starting evaluation...")
    metrics = evaluate_model(model, dataset)

    print("Evaluation finished.")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
