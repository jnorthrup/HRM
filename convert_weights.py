import torch
import mlx.core as mx
import numpy as np
from pathlib import Path
import argparse
import yaml

# Adjust the import paths for both PyTorch and MLX models
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1 as HierarchicalReasoningModel_ACTV1_torch
from models.hrm_mlx.hrm_act_v1_mlx import HierarchicalReasoningModel_ACTV1 as HierarchicalReasoningModel_ACTV1_mlx
from models.losses import ACTLossHead

def convert_and_save_weights(pytorch_model, mlx_model):
    """
    Converts PyTorch weights to MLX format and loads them into the MLX model.
    Note: This function modifies mlx_model in-place.
    """

    # The saved state_dict is from the model wrapped in ACTLossHead
    # So the keys have a 'model.' prefix.
    pytorch_weights = pytorch_model.state_dict()

    mlx_weights = {}

    for key, value in pytorch_weights.items():
        # Strip the 'model.' prefix
        if key.startswith("model."):
            key = key[6:]

        # The buffer 'H_init' and 'L_init' are not in the state_dict, but are parameters in MLX model
        # We don't need to convert them as they are part of the model definition.
        # Also, the rotary embedding caches are not in the state dict.

        # PyTorch `CastedSparseEmbedding` has a buffer `embedding_weight` which is not in the state_dict
        # but the MLX version has a nn.Embedding with a `weight` parameter.
        # This part of the conversion might need manual handling if there are issues.
        # The placeholder `CastedSparseEmbedding` in MLX has a standard nn.Embedding.
        # The original has a buffer, which is not in the state_dict. Let's assume it's not learned
        # and the zero-init in the MLX model is correct. The name is `puzzle_emb`.

        # Let's check the keys and convert
        if key.endswith(".embedding_weight"):
            # This is from CastedEmbedding in PyTorch, which is a Parameter.
            # In MLX, it's a direct array.
            key = key.replace(".embedding_weight", ".weight")

        mlx_weights[key] = mx.array(value.numpy())

    mlx_model.update(nn.utils.tree_unflatten(mlx_weights.items()))


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch HRM weights to MLX format.")
    parser.add_argument("--torch_checkpoint_dir", type=str, required=True, help="Path to the directory containing the PyTorch checkpoint and config.")
    parser.add_argument("--mlx_weights_path", type=str, required=True, help="Path to save the converted MLX weights (.safetensors).")

    args = parser.parse_args()

    checkpoint_dir = Path(args.torch_checkpoint_dir)
    config_path = checkpoint_dir / "all_config.yaml"

    # Find the checkpoint file
    checkpoint_files = list(checkpoint_dir.glob("step_*.pt"))
    if not checkpoint_files:
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}")

    # Use the first one found
    torch_checkpoint_path = checkpoint_files[0]

    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Prepare model configs
    arch_config = config['arch']
    model_config = dict(
        **arch_config,
        batch_size=1, # Batch size for inference can be 1
        vocab_size=config.get('vocab_size', 28), # Provide defaults if not in config
        seq_len=config.get('seq_len', 256),
        num_puzzle_identifiers=config.get('num_puzzle_identifiers', 1000),
    )
    # The pydantic model in MLX needs the dict, not the other way around.
    # The original config is nested. Let's reconstruct the model_cfg dict
    # as it is done in pretrain.py

    # A bit of a hack to get the config right.
    # We load the metadata from the dataset to get vocab_size etc.
    # For conversion, we can hardcode some of these if we know them, or
    # make them arguments.
    # For now, let's assume the defaults are ok for loading the arch.

    # Let's rebuild the config dict for the model
    model_cfg_dict = arch_config['__pydantic_extra__']
    model_cfg_dict.update({
        'batch_size': 1,
        'vocab_size': 28, # Default for ARC
        'seq_len': 400, # Default for ARC
        'num_puzzle_identifiers': 1120, # Default for ARC-2
        'causal': False,
    })


    # Load the PyTorch model
    print("Loading PyTorch model...")
    # The saved model includes the loss head, so we need to wrap it
    torch_model = HierarchicalReasoningModel_ACTV1_torch(model_cfg_dict)
    torch_model_with_loss = ACTLossHead(torch_model, **arch_config['loss']['__pydantic_extra__'])

    state_dict = torch.load(torch_checkpoint_path, map_location="cpu")
    torch_model_with_loss.load_state_dict(state_dict)

    # Instantiate the MLX model
    print("Instantiating MLX model...")
    mlx_model = HierarchicalReasoningModel_ACTV1_mlx(model_cfg_dict)

    # Convert weights
    print("Converting weights...")
    convert_and_save_weights(torch_model_with_loss, mlx_model)

    # Save the MLX model weights
    print(f"Saving MLX weights to {args.mlx_weights_path}...")
    mlx_model.save_weights(args.mlx_weights_path)

    print("Conversion complete.")

if __name__ == "__main__":
    main()
