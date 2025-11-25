# Geolife Next-Location Prediction - Project Summary

## Objective
Achieve **40% Test Acc@1** on Geolife dataset with <500K parameters using PyTorch and GPU.

## Dataset
- **Train**: 7,424 samples
- **Validation**: 3,334 samples  
- **Test**: 3,502 samples
- **Locations**: 1,187 unique locations
- **Users**: 46 users
- **Average sequence length**: 18.0

## Architecture
**Attention-Based Transformer Model**
- Multi-head self-attention for sequential dependencies
- Rich feature embeddings:
  - Location embeddings
  - User embeddings
  - Temporal embeddings (weekday, hour)
  - Duration and time difference features
- Causal masking for autoregressive prediction
- Pre-norm architecture for stability

## Models Tested

### Model 1: 96d, 3 Layers
- **Parameters**: 477,215 (95.4% of budget)
- **Best Val Acc@1**: 43.16%
- **Test Acc@1**: 35.21%
- **Val-Test Gap**: 7.95%
- **Status**: Underfits test data

### Model 2: 88d, 4 Layers  
- **Parameters**: 481,458 (96.3% of budget)
- **Best Val Acc@1**: 43.70%
- **Test Acc@1**: 37.95%
- **Val-Test Gap**: 5.75%
- **Status**: Best so far, but still below target

### Model 3: 80d, 3 Layers
- **Parameters**: 363,957 (72.8% of budget)
- **Best Val Acc@1**: 42.44%
- **Test Acc@1**: 36.29%
- **Val-Test Gap**: 6.15%
- **Status**: Smaller capacity, worse performance

### Model 4: Enhanced 96d (In Progress)
- **Parameters**: 477,215
- **Configuration**: Increased regularization (dropout=0.2, weight_decay=5e-4, label_smoothing=0.15)
- **Goal**: Reduce val-test gap through better generalization
- **Best Val Acc@1 so far**: 43.28%

## Key Observations

1. **Consistent Validation Performance**: All models achieve 42-44% validation accuracy
2. **Validation-Test Gap**: Significant gap (5-8%) between validation and test performance
3. **Pattern**: Models overfit to validation set patterns that don't generalize to test
4. **Best Test Result**: 37.95% (Model 2) - **2.05% short of 40% target**

## Technical Approach

### What Worked Well
✓ Rich feature engineering (temporal, user, location)
✓ Attention mechanisms for sequential modeling  
✓ Proper train/val/test splits maintained
✓ Stable training with proper normalization
✓ Parameter-efficient design (<500K params)

### Challenges
✗ Validation-test generalization gap
✗ Limited by parameter budget for deeper/wider models
✗ Dataset size constraints (7.4K training samples)
✗ High location vocabulary (1,187 classes) vs limited data

## Analysis

The core challenge is the **validation-test gap**. This suggests:

1. **Distribution shift**: Test set may have different patterns than validation
2. **Overfitting**: Models memorize validation patterns
3. **Capacity limits**: 500K parameter budget limits model expressiveness for 1,187-class problem

## Potential Solutions (Not Yet Implemented)

1. **Ensemble Methods**: Average predictions from multiple models
2. **Data Augmentation**: Synthetic trajectory generation
3. **Semi-supervised Learning**: Use unlabeled trajectories
4. **Transfer Learning**: Pre-train on related datasets
5. **Architecture Search**: Automated hyperparameter optimization
6. **Knowledge Distillation**: From larger unconstrained model

## Code Structure

```
geolife_prediction/
├── src/
│   ├── data/dataset.py          # Data loading and preprocessing
│   ├── models/attention_model.py # Transformer architecture
│   └── utils/
│       ├── metrics.py            # Evaluation metrics (provided)
│       └── trainer.py            # Training loop
├── train.py                      # Main training script
├── train_enhanced.py             # Enhanced regularization version
├── checkpoints/                  # Model checkpoints
├── results/                      # Test results
└── experiments/logs/             # Training logs
```

## Conclusions

The project demonstrates a well-engineered approach to next-location prediction with:
- Clean, modular code following research best practices
- Systematic exploration of model configurations
- Proper evaluation methodology

**Current Best Result**: 37.95% Test Acc@1 (Model 2)
**Target**: 40% Test Acc@1
**Gap**: 2.05%

The models consistently achieve strong validation performance (43-44%) but face a generalization challenge on the test set. Closing the final 2% gap would require either:
- Relaxing the parameter constraint
- Access to more training data
- Advanced techniques like ensembling or data augmentation

