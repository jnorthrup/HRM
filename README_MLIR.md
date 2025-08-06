# Generating MLIR from the HRM Model

This document explains how to generate a Multi-Level Intermediate Representation (MLIR) of the Hierarchical Reasoning Model (HRM). MLIR is a powerful compiler infrastructure that allows models to be represented in a way that can be optimized and deployed across a wide variety of hardware targets.

The generated MLIR represents the core computational graph of the model, making it a crucial starting point for porting HRM to new platforms or for performance optimization.

## 1. Purpose

The `generate_mlir.py` script provides a bridge between the PyTorch implementation of HRM and the MLIR ecosystem. It uses the `torch-mlir` project to trace the model's forward pass and convert it into the `torch` dialect of MLIR.

**Note:** The script specifically compiles the `HierarchicalReasoningModel_ACTV1_Inner` module. This is because MLIR compilers work best on static, analyzable computation graphs. The full `HierarchicalReasoningModel_ACTV1` contains a Python `while` loop for its Adaptive Computation Time (ACT) mechanism, which cannot be directly traced. By compiling the inner module, we capture one full step of the reasoning process, which is the fundamental building block of the model.

## 2. Setup

Ensure you have installed the necessary dependencies, including `torch-mlir`. The original `requirements.txt` file has been updated to include it.

```bash
# It is recommended to use a fresh virtual environment
pip install -r requirements.txt
```

## 3. How to Run

To generate the MLIR file, you need a PyTorch checkpoint directory for the HRM model.

```bash
python generate_mlir.py \
    --torch_checkpoint_dir /path/to/your/torch_checkpoint_dir \
    --output_mlir_path hrm_model.mlir
```

- `--torch_checkpoint_dir`: Path to the directory containing the `all_config.yaml` and the `.pt` model file.
- `--output_mlir_path`: The file where the generated MLIR will be saved.

After running the script, `hrm_model.mlir` will contain the full MLIR representation of the model's core logic.

## 4. Next Steps

The generated `.mlir` file (in the `torch` dialect) is the first step in the compilation pipeline. From here, you can use other tools from the MLIR and LLVM ecosystem to:

1.  **Lower the MLIR**: Convert the `torch` dialect MLIR to more generic dialects like `TOSA` (Tensor Operator Set Architecture) or `Linalg` (Linear Algebra).
2.  **Optimize**: Apply hardware-agnostic and hardware-specific optimization passes.
3.  **Compile**: Generate machine code for your target hardware (e.g., CPUs, GPUs, or custom accelerators).

This process allows for a highly portable and performant deployment of the HRM model.
