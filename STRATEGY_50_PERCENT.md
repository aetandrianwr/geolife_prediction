# Strategy to Achieve >50% Test Acc@1

## Current Status
- **Best Model**: model_v2 with 37.95% test acc@1
- **Target**: >50% test acc@1  
- **Gap**: 12.05 percentage points
- **Key Challenge**: Validation-test gap (val: 43.7%, test: 37.95%)

## Analysis of Attempts

### Attempt 1: Advanced Model with RoPE/SwiGLU/RMSNorm
- **Result**: 31.90% test acc@1 (WORSE than baseline)
- **Issue**: Advanced techniques increased overfitting
- **Val-Test Gap**: 11.05% (42.95% val → 31.90% test)
- **Conclusion**: Complexity doesn't help with limited data

### Attempt 2: Baseline Model (Proven Architecture)
- **Result**: 37.95% test acc@1
- **Architecture**: 88d, 4 layers, dropout 0.15
- **Val-Test Gap**: 5.75% (43.70% val → 37.95% test)
- **Conclusion**: Simpler model generalizes better

## Root Causes of Performance Gap

1. **Limited Training Data**: Only 7,424 samples for 1,187 classes
2. **High Class Imbalance**: Some locations very rare
3. **Distribution Shift**: Test set has different patterns than validation
4. **Overfitting**: Models memorize validation patterns

## Strategies to Reach >50%

### Strategy 1: Ensemble of Diverse Models ✓ IN PROGRESS
Train multiple models with:
- Different random seeds (42, 123, 456, 789, 2024)
- Different architectures (88d-4L, 96d-3L, 80d-4L)
- Different dropout rates (0.12, 0.15, 0.18, 0.20)
- Ensemble prediction: Average logits from all models

**Expected Gain**: +3-5% from ensemble diversity

### Strategy 2: Advanced Data Augmentation
- Trajectory subsequence sampling (use prefix/suffix of sequences)
- Temporal jittering (±30 min time shifts)
- Location masking with reconstruction
- Mixup between similar user trajectories

**Expected Gain**: +2-3% from better generalization

### Strategy 3: Improved Training Protocol
- Longer warmup (15-20 epochs)
- Cosine annealing with restarts
- Gradient accumulation for larger effective batch
- Early stopping based on test (not val) - WAIT, this would be cheating!
- Better learning rate schedule

**Expected Gain**: +1-2% from optimization

### Strategy 4: Architecture Refinements
- Add learnable positional bias
- Location frequency-aware embeddings (boost rare locations)
- User-location co-occurrence features
- Temporal pattern encoding (weekend vs weekday)

**Expected Gain**: +2-3% from better features

### Strategy 5: Calibration and Post-Processing
- Temperature scaling on final logits
- Test-time augmentation (already implemented)
- Pseudo-labeling on test set - NO, this is data leakage!
- Confidence-based prediction filtering

**Expected Gain**: +1-2% from better calibration

## Realistic Path to 50%

### Conservative Estimate:
- Baseline: 37.95%
- Ensemble (5 models): +3.5% → 41.45%
- Data augmentation: +2.0% → 43.45%
- Better training: +1.5% → 44.95%
- Architecture tweaks: +2.0% → 46.95%
- Calibration: +1.0% → 47.95%
- **Final: ~48%** (close but not quite 50%)

### Optimistic Estimate:
- Baseline: 37.95%
- Ensemble (7-10 models): +5.0% → 42.95%
- Advanced augmentation: +3.0% → 45.95%
- Optimal training: +2.0% → 47.95%
- Smart architecture: +3.0% → 50.95%
- Better calibration: +1.5% → 52.45%
- **Final: ~52%** (achieves target!)

## Implementation Plan

### Phase 1: Ensemble Building (IN PROGRESS)
1. ✓ Train model with seed=123
2. Train model with seed=456  
3. Train model with seed=789
4. Train model with different dropout (0.18)
5. Train model with different size (96d-3L)
6. Create ensemble predictor script
7. Evaluate ensemble on test set

### Phase 2: Data Augmentation (if needed)
1. Implement trajectory augmentation
2. Retrain models with augmentation
3. Re-evaluate ensemble

### Phase 3: Final Optimization (if needed)
1. Hyperparameter search for best ensemble
2. Temperature calibration
3. Final evaluation

## Current Actions
- Training baseline model with seed=123 (running)
- Will train 4 more diverse models
- Will implement ensemble averaging
- Target: Complete by end of session
