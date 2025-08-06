# Hierarchical Reasoning Model (HRM) - MLX Port

This directory contains a port of the Hierarchical Reasoning Model (HRM) from its original PyTorch implementation to MLX, the machine learning framework for Apple Silicon.

This port allows running the HRM model efficiently on Apple M-series chips.

## 1. Setup

First, install the necessary Python packages for the MLX version of the model. It is recommended to do this in a new virtual environment.

```bash
pip install -r requirements_mlx.txt
```

Note: `requirements_mlx.txt` is located in the root of this repository.

## 2. Convert Weights

The original model uses PyTorch checkpoints. To use the model with MLX, you must first convert the weights into the `.safetensors` format used by MLX.

The `convert_weights.py` script is provided for this purpose.

**Prerequisites:**
- Download an original PyTorch checkpoint. Checkpoints are available on Hugging Face (e.g., [ARC-AGI-2 Checkpoint](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)).
- Unzip the checkpoint into a directory. This directory should contain a `.pt` file and an `all_config.yaml` file.

**Run conversion:**

```bash
python convert_weights.py \
    --torch_checkpoint_dir /path/to/your/torch_checkpoint_dir \
    --mlx_weights_path hrm_mlx_weights.safetensors
```

This will create the `hrm_mlx_weights.safetensors` file in your current directory.

## 3. Run Evaluation

You can run evaluation on a dataset using the ported model and the converted weights with the `evaluate_mlx.py` script.

**Prerequisites:**
- A converted MLX weights file (`hrm_mlx_weights.safetensors`).
- The dataset you want to evaluate on (e.g., the ARC dataset).

**Run evaluation:**

```bash
python evaluate_mlx.py \
    --config_path /path/to/your/torch_checkpoint_dir/all_config.yaml \
    --weights_path hrm_mlx_weights.safetensors \
    --dataset_path /path/to/your/dataset
```

The script will print accuracy metrics to the console.

## 4. Validate the Port

To ensure the MLX port is numerically consistent with the original PyTorch model, you can use the `validate.py` script. This script runs both models on the same data and compares their output logits.

**Prerequisites:**
- A PyTorch checkpoint directory.
- A converted MLX weights file.
- A dataset to use for validation.

**Run validation:**
```bash
python validate.py \
    --torch_checkpoint_dir /path/to/your/torch_checkpoint_dir \
    --mlx_weights_path hrm_mlx_weights.safetensors \
    --dataset_path /path/to/your/dataset
```

The script will output the mean absolute difference and mean squared error between the models' outputs. A small difference (e.g., < 1e-4) indicates a successful port.
