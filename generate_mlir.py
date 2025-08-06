import torch
import yaml
import argparse
from pathlib import Path

from torch_mlir import compile
from torch_mlir.dialects.torch import register_dialect as register_torch_dialect

# We need to register the torch dialect to be able to save the MLIR
# This is a bit of a quirk in the torch-mlir library
from mlir.ir import Context
register_torch_dialect(Context())


# Import the PyTorch model definition
from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1_Inner, HierarchicalReasoningModel_ACTV1Config


def main():
    parser = argparse.ArgumentParser(description="Generate MLIR from a PyTorch HRM model.")
    parser.add_argument("--torch_checkpoint_dir", type=str, required=True, help="Path to the directory containing the PyTorch checkpoint and config.")
    parser.add_argument("--output_mlir_path", type=str, required=True, help="Path to save the generated MLIR file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the example input.")

    args = parser.parse_args()

    # --- Load Config ---
    print("Loading configuration...")
    config_path = Path(args.torch_checkpoint_dir) / "all_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Reconstruct the model config dictionary
    arch_config = config['arch']
    model_cfg_dict = arch_config.get('__pydantic_extra__', {})
    model_cfg_dict.update({
        'batch_size': args.batch_size,
        'vocab_size': config.get('vocab_size', 28),
        'seq_len': config.get('seq_len', 400),
        'num_puzzle_identifiers': config.get('num_puzzle_identifiers', 1120),
        'causal': False,
    })
    model_config = HierarchicalReasoningModel_ACTV1Config(**model_cfg_dict)

    # --- Instantiate PyTorch Model ---
    print("Instantiating PyTorch model...")
    # We compile the 'Inner' module because its forward pass is a simple
    # computational graph without complex Python control flow like the outer class.
    model = HierarchicalReasoningModel_ACTV1_Inner(model_config)
    model.eval()

    # --- Create Example Inputs ---
    print("Creating example inputs for tracing...")
    # The model expects a 'carry' and a 'batch' dictionary

    # 1. Create example carry
    inner_carry = model.empty_carry(args.batch_size)

    # 2. Create example batch dictionary
    batch = {
        "inputs": torch.randint(0, model_config.vocab_size, (args.batch_size, model_config.seq_len)),
        "puzzle_identifiers": torch.randint(0, model_config.num_puzzle_identifiers, (args.batch_size,))
    }

    # --- Compile to MLIR ---
    print("Compiling model to MLIR...")
    mlir_module = compile(
        model,
        (inner_carry, batch),
        output_type="torch",  # Export to the 'torch' dialect of MLIR
        use_tracing=True      # Use tracing to capture the graph
    )

    # --- Save the MLIR Module ---
    print(f"Saving MLIR to {args.output_mlir_path}...")
    with open(args.output_mlir_path, 'w') as f:
        f.write(str(mlir_module))

    print("MLIR generation complete.")
    print("\nTo inspect the output, you can view the generated .mlir file.")
    print("This MLIR can then be lowered to other dialects (like TOSA or Linalg) and compiled for different hardware targets.")


if __name__ == "__main__":
    main()
