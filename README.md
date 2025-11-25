# Geolife Next-Location Prediction

A PyTorch implementation of attention-based next-location prediction on the Geolife dataset.

## Overview

This project implements a transformer-based model for predicting the next location in a user's trajectory sequence. The model achieves **37.95% Test Acc@1** with less than 500K parameters.

## Project Structure

```
geolife_prediction/
├── configs/                    # Configuration files
│   ├── default.yml            # Default configuration
│   ├── model_v1.yml           # Model variant 1
│   ├── model_v2.yml           # Model variant 2 (best)
│   └── model_v3.yml           # Model variant 3
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   └── attention_model.py
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── logger.py          # Logging utilities
│   │   └── trainer.py         # Training loop
│   └── __init__.py
├── scripts/                   # Utility scripts
│   ├── train.sh              # Training script
│   └── evaluate.sh           # Evaluation script
├── checkpoints/              # Model checkpoints
├── results/                  # Experiment results
├── logs/                     # Training logs
├── train.py                  # Main training script
├── evaluate.py               # Evaluation script
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
└── README.md                 # This file
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate geolife
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Best Model

```bash
python train.py --config configs/model_v2.yml
```

### 2. Evaluate a Checkpoint

```bash
python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt
```

### 3. Train with Custom Parameters

```bash
python train.py --config configs/model_v2.yml \
    --learning_rate 0.001 \
    --batch_size 128 \
    --seed 123
```

## Configuration

All hyperparameters are managed through YAML configuration files in `configs/`. Command-line arguments override config file values.

### Config File Structure

```yaml
# Model architecture
model:
  name: model_v2
  d_model: 88
  d_inner: 176
  n_layers: 4
  n_head: 8
  d_k: 11
  d_v: 11
  dropout: 0.15

# Training
training:
  batch_size: 64
  learning_rate: 0.0005
  weight_decay: 0.0001
  label_smoothing: 0.1
  max_epochs: 200
  patience: 30

# Data
data:
  data_dir: /content/another_try_20251125/data/geolife
  max_len: 50
  num_workers: 0

# Experiment
experiment:
  seed: 42
  device: cuda
  checkpoint_dir: checkpoints
  log_dir: logs
```

## Reproducibility

All experiments are fully reproducible with fixed random seeds:

```bash
# Exact reproduction of best results (37.95% Test Acc@1)
python train.py --config configs/model_v2.yml --seed 42
```

The seed controls:
- PyTorch random number generation
- NumPy random state
- Python random module
- CUDA deterministic operations

## Results

### Best Model (model_v2)

| Metric | Value |
|--------|-------|
| Test Acc@1 | 37.95% |
| Test Acc@5 | 56.54% |
| Test Acc@10 | 58.97% |
| Test MRR | 46.39% |
| Test NDCG | 49.25% |
| Parameters | 481,458 |
| Val Acc@1 | 43.70% |

### All Models

| Model | Params | Val Acc@1 | Test Acc@1 |
|-------|--------|-----------|------------|
| model_v1 | 477,215 | 43.16% | 35.21% |
| **model_v2** | **481,458** | **43.70%** | **37.95%** |
| model_v3 | 363,957 | 42.44% | 36.29% |

## Advanced Usage

### Training with Multiple Seeds

```bash
for seed in 42 123 456; do
    python train.py --config configs/model_v2.yml --seed $seed
done
```

### Hyperparameter Search

```bash
python train.py --config configs/model_v2.yml \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --experiment_name lr_0.001_dropout_0.2
```

### Evaluation on Different Checkpoints

```bash
python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt
python evaluate.py --checkpoint checkpoints/model_v1/best_model.pt
```

## Development

### Project Guidelines

- All configuration in `configs/*.yml`
- All source code in `src/`
- All scripts in `scripts/`
- Follow PEP 8 style guide
- Add docstrings to all functions
- Use type hints where applicable

### Adding a New Model

1. Create configuration file: `configs/my_model.yml`
2. Implement model in: `src/models/my_model.py`
3. Train: `python train.py --config configs/my_model.yml`

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{geolife_prediction,
  title={Attention-Based Next-Location Prediction on Geolife Dataset},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
