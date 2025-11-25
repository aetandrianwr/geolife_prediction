# How to Reproduce Best Model Results (37.95% Test Acc@1)

This guide provides step-by-step instructions to reproduce the best model results:
- **Test Acc@1**: 37.95%
- **Test Acc@5**: 56.54%
- **Test MRR**: 46.39%
- **Validation Acc@1**: 43.70%

## Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- Geolife dataset in `/content/another_try_20251125/data/geolife`

## Method 1: Quick Reproduction (Using Saved Checkpoint)

If the checkpoint exists, you can directly evaluate it:

```bash
cd /content/geolife_prediction
python3 evaluate_best_model.py
```

This will load the best checkpoint and evaluate on the test set.

## Method 2: Full Training Reproduction

### Step 1: Setup Environment

```bash
# Navigate to project directory
cd /content/geolife_prediction

# Install dependencies (if needed)
pip install torch numpy pandas scikit-learn tqdm
```

### Step 2: Train Best Model Configuration

```bash
# Run the training script that will try all configurations
python3 train.py
```

This will:
1. Try Model 1 (96d, 3L) - will get ~35% test acc
2. Try Model 2 (88d, 4L) - **will get ~38% test acc** ✓ BEST
3. Try Model 3 (80d, 3L) - will get ~36% test acc

The script automatically trains all models and selects the best one.

**Expected output:**
```
Model_v2_88d_4L:
  Parameters: 481,458
  Budget OK: True

Training Model_v2_88d_4L...
[Training progress bars...]

Best validation epoch: 18
Test Acc@1: 37.95%
Test Acc@5: 56.54%
Test MRR: 46.39%
```

### Step 3: Verify Results

Check the results file:
```bash
cat results/Model_v2_88d_4L_results.txt
```

Expected content:
```
Configuration: Model_v2_88d_4L
================================================================================

Test Acc@1: 37.95%
Test Acc@5: 56.54%
Test Acc@10: 58.97%
Test MRR: 46.39%
Test NDCG: 49.25%

Model Parameters: 481,458
Best Val Acc@1: 43.70%
Best Epoch: 18
```

## Method 3: Train Only the Best Model

To skip other configurations and train only Model 2:

```bash
python3 train_single_best.py
```

This trains only the best configuration directly.

## Expected Training Details

### Model Configuration
- **Name**: Model_v2_88d_4L
- **Architecture**:
  - d_model: 88
  - d_inner: 176
  - n_layers: 4
  - n_head: 8
  - d_k: 11
  - d_v: 11
  - dropout: 0.15
- **Parameters**: 481,458 (96.3% of 500K budget)

### Training Hyperparameters
- Learning rate: 5e-4
- Weight decay: 1e-4
- Label smoothing: 0.1
- Batch size: 64
- Max epochs: 200
- Patience: 30
- Optimizer: AdamW
- LR Schedule: Cosine annealing

### Expected Training Behavior
- **Total epochs**: ~48 (early stopping)
- **Best epoch**: Around epoch 18
- **Best validation accuracy**: 43.70%
- **Training time**: ~10-15 minutes on GPU

### Performance Checkpoints
You should see these approximate accuracies during training:

| Epoch | Val Acc@1 | Notes |
|-------|-----------|-------|
| 1-5   | 35-40%    | Initial learning |
| 10-15 | 40-42%    | Rapid improvement |
| 18    | 43.70%    | **Best checkpoint** |
| 20-30 | 42-43%    | Plateau |
| 31-48 | 40-42%    | Decline (overfitting) |

## Verification Checklist

After training completes, verify:

- ✅ Checkpoint saved: `checkpoints/Model_v2_88d_4L/best_model.pt`
- ✅ Results file: `results/Model_v2_88d_4L_results.txt`
- ✅ Test Acc@1: **37.95% ± 0.5%** (slight variance due to randomness)
- ✅ Val Acc@1: **43.70% ± 0.5%**
- ✅ Parameters: **481,458**

## Troubleshooting

### If results differ slightly (±1%):
This is normal due to:
- Random initialization
- Data shuffling
- GPU non-determinism

To get exact reproduction, ensure:
```python
# In train.py, seed is set to 42
set_seed(42)
torch.backends.cudnn.deterministic = True
```

### If Test Acc@1 is much lower (<35%):
Check:
1. Dataset path is correct
2. All features are being used (location, user, temporal)
3. Model loaded from correct checkpoint (epoch 18, not last epoch)

### If training doesn't converge:
- Verify GPU is being used: Check `device='cuda'` in logs
- Check batch size (should be 64)
- Verify data loading (should see 116 training batches)

## Files Created During Training

```
checkpoints/Model_v2_88d_4L/
  └── best_model.pt              # Best model weights (epoch 18)

results/
  └── Model_v2_88d_4L_results.txt  # Final test results

experiments/logs/
  └── training_run2.log          # Full training log
```

## Quick Evaluation Only

If you just want to evaluate the existing checkpoint without retraining:

```python
import torch
from src.models.attention_model import LocationPredictionModel
from src.data.dataset import get_dataloaders

# Load data
_, _, test_loader, dataset_info = get_dataloaders(
    data_dir='/content/another_try_20251125/data/geolife',
    batch_size=64,
    max_len=50
)

# Create model
model = LocationPredictionModel(
    num_locations=dataset_info['num_locations'],
    num_users=dataset_info['num_users'],
    d_model=88,
    d_inner=176,
    n_layers=4,
    n_head=8,
    d_k=11,
    d_v=11,
    dropout=0.15,
    max_len=50
)

# Load checkpoint
checkpoint = torch.load('checkpoints/Model_v2_88d_4L/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate (code in trainer.py evaluate method)
```

## Expected Timeline

- **Setup**: 1-2 minutes
- **Training Model 1**: 10-15 minutes
- **Training Model 2** (best): 10-15 minutes
- **Training Model 3**: 8-12 minutes
- **Total time**: ~30-40 minutes for all models

To save time, use `train_single_best.py` for just Model 2 (~15 minutes).

## Contact

If you encounter issues reproducing these results, check:
1. Dataset integrity (1,187 locations, 46 users)
2. Train/val/test split (7424/3334/3502)
3. Correct configuration loaded
4. Checkpoint from best epoch (not final epoch)

The model is deterministic given the same seed, so exact reproduction should be possible.
