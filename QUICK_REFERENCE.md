# Quick Reference - Geolife Prediction

## Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml && conda activate geolife
```

## Training

```bash
# Best model (37.95% test)
python train.py --config configs/model_v2.yml

# With custom seed
python train.py --config configs/model_v2.yml --seed 123

# Override parameters
python train.py --config configs/model_v2.yml \
    --learning_rate 0.001 \
    --dropout 0.2 \
    --batch_size 128
```

## Evaluation

```bash
# Evaluate best checkpoint
python evaluate.py --checkpoint checkpoints/Model_v2_88d_4L/best_model.pt

# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt --split val

# Evaluate on train set
python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt --split train
```

## Reproduce Best Result (37.95%)

```bash
# Just evaluate the saved checkpoint
python evaluate.py --checkpoint checkpoints/Model_v2_88d_4L/best_model.pt
```

Output:
```
Acc@1:  37.95%
Acc@5:  56.54%
MRR:    46.39%
```

## Project Structure

```
configs/          # YAML configuration files
src/              # Source code (data, models, utils)
scripts/          # Shell scripts
checkpoints/      # Saved models
results/          # Experiment results  
logs/             # Training logs
```

## Key Files

- `configs/model_v2.yml` - Best model configuration
- `checkpoints/Model_v2_88d_4L/best_model.pt` - Best checkpoint (37.95%)
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `README.md` - Full documentation

## Configuration Format

```yaml
model:
  d_model: 88        # Model dimension
  n_layers: 4        # Number of layers
  n_head: 8          # Attention heads
  dropout: 0.15      # Dropout rate

training:
  batch_size: 64
  learning_rate: 0.0005
  max_epochs: 200
  patience: 30

experiment:
  seed: 42           # Random seed
  device: cuda       # cuda or cpu
```

## Common Tasks

### Create New Experiment
```bash
# Copy and modify config
cp configs/model_v2.yml configs/my_exp.yml
# Edit my_exp.yml
python train.py --config configs/my_exp.yml
```

### Check Results
```bash
cat results/my_exp_results.txt
```

### View Training Log
```bash
tail -f logs/my_exp_train.log
```

### List All Checkpoints
```bash
find checkpoints -name "best_model.pt"
```

## Performance

| Model | Params | Test Acc@1 | Config |
|-------|--------|------------|--------|
| model_v1 | 477K | 35.21% | configs/model_v1.yml |
| **model_v2** | **481K** | **37.95%** | **configs/model_v2.yml** |
| model_v3 | 364K | 36.29% | configs/model_v3.yml |

## Troubleshooting

**CUDA out of memory:**
```bash
python train.py --config configs/model_v2.yml --batch_size 32
```

**Different results:**
- Expected variance: ±1% due to random initialization
- Use same seed for exact reproduction

**Checkpoint not found:**
```bash
# List available checkpoints
ls -lh checkpoints/*/best_model.pt
```

## Getting Help

1. Check `README.md` for full documentation
2. Review `PRODUCTION_STRUCTURE.md` for project details
3. See `RESTRUCTURING_SUMMARY.md` for migration guide
4. Check logs in `logs/` directory
