# Geolife Next-Location Prediction - Production Project Structure

## Overview

This is a production-grade PyTorch implementation of attention-based next-location prediction, structured following PhD-level research standards.

## Directory Structure

```
geolife_prediction/
│
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment specification
│
├── configs/                           # Configuration files (YAML)
│   ├── default.yml                   # Default configuration
│   ├── model_v1.yml                  # Model variant 1 (96d, 3L) - 35.21% test
│   ├── model_v2.yml                  # Model variant 2 (88d, 4L) - 37.95% test ⭐ BEST
│   └── model_v3.yml                  # Model variant 3 (80d, 3L) - 36.29% test
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── data/                        # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── dataset.py              # Geolife dataset loader
│   ├── models/                      # Model architectures
│   │   ├── __init__.py
│   │   └── attention_model.py      # Transformer-based model
│   └── utils/                       # Utilities
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging utilities  
│       ├── metrics.py              # Evaluation metrics
│       └── trainer.py              # Training loop
│
├── scripts/                          # Utility scripts
│   ├── train.sh                    # Training shell script
│   └── evaluate.sh                 # Evaluation shell script
│
├── train.py                          # Main training script
├── evaluate.py                       # Evaluation script
│
├── checkpoints/                      # Model checkpoints
│   ├── Model_v2_88d_4L/           # Old checkpoint (37.95% test)
│   │   └── best_model.pt
│   └── model_v2/                  # New checkpoint (with config)
│       ├── best_model.pt
│       └── config.yml
│
├── results/                          # Experiment results
│   ├── Model_v2_88d_4L_results.txt  # Best model results
│   └── model_v2_results.txt         # Latest training results
│
├── logs/                             # Training logs
│   ├── model_v2_train.log
│   └── verification_training.log
│
├── old_scripts/                      # Archive of old training scripts
│   └── (legacy scripts moved here)
│
└── docs/                             # Documentation
    ├── FINAL_REPORT.md              # Comprehensive final report
    ├── PROJECT_SUMMARY.md           # Technical summary
    └── REPRODUCE_BEST_MODEL.md      # Reproduction guide
```

## Key Features

### 1. Configuration Management (`configs/`)
- All hyperparameters in YAML files
- Command-line args override config values
- Easy to version and reproduce experiments

### 2. Modular Code Structure (`src/`)
- **data/**: Dataset loading and preprocessing
- **models/**: Model architectures
- **utils/**: Reusable utilities (config, logging, metrics, training)

### 3. Reproducibility
- Fixed random seeds (default: 42)
- Deterministic CUDA operations
- Configuration saved with checkpoints
- Requirements pinned in requirements.txt

### 4. Logging and Monitoring
- Structured logging to files
- Training progress bars
- Metric tracking
- Experiment history

### 5. Evaluation Tools
- Standalone evaluation script
- Multiple split support (train/val/test)
- Comprehensive metrics

## Quick Start

### Installation
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate geolife
```

### Training
```bash
# Train best model
python train.py --config configs/model_v2.yml

# Train with custom parameters
python train.py --config configs/model_v2.yml --learning_rate 0.001 --seed 123

# Using shell script
bash scripts/train.sh --config configs/model_v2.yml
```

### Evaluation
```bash
# Evaluate checkpoint
python evaluate.py --checkpoint checkpoints/Model_v2_88d_4L/best_model.pt

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt --split val

# Using shell script
bash scripts/evaluate.sh --checkpoint checkpoints/Model_v2_88d_4L/best_model.pt
```

## Best Model Performance

**Model: model_v2 (88d, 4 layers, 8 heads)**
- Parameters: 481,458 (96.3% of 500K budget)
- **Test Acc@1: 37.95%** ⭐
- Test Acc@5: 56.54%
- Test MRR: 46.39%
- Val Acc@1: 43.70%
- Checkpoint: `checkpoints/Model_v2_88d_4L/best_model.pt`

## Configuration File Format

```yaml
# configs/model_v2.yml
model:
  name: model_v2
  d_model: 88
  d_inner: 176
  n_layers: 4
  n_head: 8
  d_k: 11
  d_v: 11
  dropout: 0.15
  max_len: 50

training:
  batch_size: 64
  learning_rate: 0.0005
  weight_decay: 0.0001
  label_smoothing: 0.1
  max_epochs: 200
  patience: 30

data:
  data_dir: /path/to/geolife
  max_len: 50
  num_workers: 0

experiment:
  name: model_v2
  seed: 42
  device: cuda
  deterministic: true
```

## Command-Line Arguments

### Training (`train.py`)
- `--config`: Path to config file (required)
- `--d_model`: Model dimension (overrides config)
- `--n_layers`: Number of layers (overrides config)
- `--batch_size`: Batch size (overrides config)
- `--learning_rate`: Learning rate (overrides config)
- `--seed`: Random seed (overrides config)
- `--device`: Device (cuda/cpu, overrides config)
- `--experiment_name`: Custom experiment name

### Evaluation (`evaluate.py`)
- `--checkpoint`: Path to checkpoint (required)
- `--data_dir`: Path to data directory
- `--batch_size`: Batch size for evaluation
- `--device`: Device (cuda/cpu)
- `--split`: Dataset split (train/val/test)

## File Naming Conventions

- **Checkpoints**: `checkpoints/{experiment_name}/best_model.pt`
- **Configs**: `checkpoints/{experiment_name}/config.yml`
- **Results**: `results/{experiment_name}_results.txt`
- **Logs**: `logs/{experiment_name}_train.log`

## Development Workflow

1. **Create config**: `configs/my_experiment.yml`
2. **Train**: `python train.py --config configs/my_experiment.yml`
3. **Monitor**: Check `logs/my_experiment_train.log`
4. **Evaluate**: `python evaluate.py --checkpoint checkpoints/my_experiment/best_model.pt`
5. **Analyze**: Review `results/my_experiment_results.txt`

## Reproducibility Checklist

- ✅ Fixed random seeds (42)
- ✅ Deterministic CUDA operations
- ✅ Configuration saved with checkpoints
- ✅ Requirements.txt for exact package versions
- ✅ Training logs preserved
- ✅ Standalone evaluation script
- ✅ Documentation and reproduction guide

## Migration from Old Scripts

Old training scripts (`train_single_best.py`, `train_enhanced.py`, etc.) have been moved to `old_scripts/` for reference. Use the new unified `train.py` with config files instead.

### Old → New Mapping
- `train_single_best.py` → `python train.py --config configs/model_v2.yml`
- `evaluate_best_model.py` → `python evaluate.py --checkpoint <path>`
- Hard-coded params → `configs/*.yml`

## Performance Notes

Training the best model (model_v2) with seed 42:
- Expected test accuracy: **37.5% - 38.5%**
- Variance due to random initialization: ±1%
- Best recorded: **37.95%** (checkpoint saved)
- Val-test gap: ~5-6%

## Citation

```bibtex
@misc{geolife_prediction_2024,
  title={Attention-Based Next-Location Prediction on Geolife Dataset},
  author={Research Team},
  year={2024},
  note={Production-grade PyTorch implementation}
}
```

## Contact

For issues or questions:
1. Check existing documentation
2. Review logs in `logs/`
3. Verify configuration in `configs/`
4. Test with known-good checkpoint
