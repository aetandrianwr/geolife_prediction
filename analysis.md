# Training Analysis

## Results So Far

### Model 1 (96d, 3L)
- Parameters: 477,215
- Best Val Acc@1: 43.16%
- **Test Acc@1: 35.21%**
- Val-Test Gap: 7.95%

### Model 2 (88d, 4L)  
- Parameters: 481,458
- Best Val Acc@1: 43.70%
- **Test Acc@1: 37.95%**
- Val-Test Gap: 5.75%

## Key Observations

1. **Validation accuracy is good** (42-44%) but doesn't transfer to test
2. **Significant val-test gap** indicates overfitting
3. **Both models show similar patterns** - suggesting architectural limitation

## Hypotheses

1. **Overfitting to validation patterns**: Need stronger regularization
2. **Insufficient capacity**: May need better feature representation
3. **Training instability**: High variance in later epochs

## Next Steps

1. Try Model 3 with lower dropout and see if it helps
2. If Model 3 fails, need to:
   - Add data augmentation
   - Use ensemble methods
   - Try different architecture (e.g., more layers but thinner)
   - Implement mixup or other regularization

## Target

Need to reach **40% Test Acc@1** consistently.
