# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Hierarchical Reasoning Model (HRM) is a novel recurrent architecture for complex reasoning tasks. The codebase includes PyTorch training/evaluation, MLX port for Apple Silicon, and MLIR generation for hardware portability.

## Common Development Commands

### Dataset Preparation
```bash
# Initialize submodules for ARC data
git submodule update --init --recursive

# Build ARC-1 dataset (960 examples)
python dataset/build_arc_dataset.py

# Build ARC-2 dataset (1120 examples) 
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

# Build Sudoku dataset (1000 examples)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Build Maze dataset (1000 examples)
python dataset/build_maze_dataset.py
```

### Training
```bash
# Single GPU training (Sudoku example)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU training (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py

# ARC-2 training
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

### Evaluation
```bash
# Standard evaluation
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>

# MLX evaluation (Apple Silicon)
python evaluate_mlx.py --config_path /path/to/checkpoint/all_config.yaml --weights_path hrm_mlx_weights.safetensors --dataset_path /path/to/dataset
```

### MLX Conversion (Apple Silicon)
```bash
# Install MLX dependencies
pip install -r requirements_mlx.txt

# Convert PyTorch weights to MLX format
python convert_weights.py --torch_checkpoint_dir /path/to/torch/checkpoint --mlx_weights_path hrm_mlx_weights.safetensors

# Validate MLX port accuracy
python validate.py --torch_checkpoint_dir /path/to/torch/checkpoint --mlx_weights_path hrm_mlx_weights.safetensors --dataset_path /path/to/dataset
```

### MLIR Generation
```bash
# Generate MLIR for hardware portability
python generate_mlir.py --torch_checkpoint_dir /path/to/checkpoint --output_mlir_path hrm_model.mlir
```

## Architecture Overview

### Core Model Structure
- **HierarchicalReasoningModel_ACTV1**: Main model with Adaptive Computation Time (ACT)
- **Two-level hierarchy**: High-level (slow, abstract) and Low-level (fast, detailed) modules
- **Recurrent processing**: H_cycles for high-level, L_cycles for low-level reasoning
- **Q-learning halting**: Dynamic computation depth with halt_max_steps and exploration

### Key Components
- **models/hrm/hrm_act_v1.py**: Core PyTorch HRM implementation
- **models/hrm_mlx/**: MLX port for Apple Silicon efficiency
- **models/layers.py**: Shared transformer components (Attention, SwiGLU, etc.)
- **models/sparse_embedding.py**: Puzzle-specific sparse embeddings
- **puzzle_dataset.py**: Dataset handling with augmentation and tokenization

### Configuration System
- **config/arch/hrm_v1.yaml**: Model architecture defaults
- **config/cfg_pretrain.yaml**: Training hyperparameters
- Uses Hydra/OmegaConf for configuration management
- Key configs: H_cycles=2, L_cycles=2, hidden_size=512, num_heads=8

### Dataset Processing
- **Tokenization**: Grid-based puzzles converted to sequences
- **Augmentation**: 8 dihedral transformations for data efficiency
- **Puzzle identifiers**: Sparse embeddings for task-specific adaptation
- **Metadata tracking**: vocab_size, seq_len, num_puzzle_identifiers

## Dependencies and Environment

### PyTorch Dependencies (requirements.txt)
- torch, adam-atan2, einops, flash-attn
- wandb (experiment tracking), hydra-core (config)
- torch-mlir (MLIR generation)

### MLX Dependencies (requirements_mlx.txt) 
- mlx, safetensors (Apple Silicon port)
- Compatible PyTorch for weight conversion

### GPU Requirements
- CUDA 12.6+ for PyTorch training
- FlashAttention 2/3 depending on GPU architecture
- Multi-GPU support via torchrun

## Key Implementation Details

### Adaptive Computation Time (ACT)
- Dynamic halting based on Q-learning
- halt_exploration_prob controls exploration vs exploitation
- Inner module compiled separately for MLIR (static graphs)

### Training Specifics
- AdamATan2 optimizer with separate puzzle embedding learning rates
- Gradient clipping and weight decay for stability
- Cosine learning rate schedule with warmup
- Distributed training support with proper synchronization

### Evaluation and Checkpointing  
- Regular checkpointing with all_config.yaml preservation
- Multiple output formats for analysis (logits, halt decisions)
- ARC-specific evaluation notebook (arc_eval.ipynb)

## Visualization and Analysis
- **puzzle_visualizer.html**: Interactive puzzle viewer
- **arc_eval.ipynb**: ARC evaluation analysis
- Supports uploading dataset folders for visual exploration