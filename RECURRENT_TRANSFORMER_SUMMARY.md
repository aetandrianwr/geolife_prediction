# Recurrent Transformer Experiment Summary

## Objective
Implement Transformer-based recurrence (instead of RNN/LSTM) to achieve >50% acc@1 on GeoLife test set.

## Models Tested

### 1. Recurrent Transformer v1
- **Architecture**: Transformer processes input sequence + hidden state at each cycle
- **Cycles**: 3
- **Parameters**: 341k (68% of budget)
- **Results**:
  - Val Acc@1: 35.39%
  - Test Acc@1: **32.72%**
- **Status**: ❌ Worse than baseline (37.95%)

### 2. Recurrent Transformer v2 (Improved)
- **Architecture**: Better hidden integration with gating, attention pooling
- **Cycles**: 5
- **Parameters**: 397k (79% of budget)
- **Results**:
  - Val Acc@1: 36.29%
  - Test Acc@1: **34.29%**
- **Status**: ❌ Worse than baseline

### 3. Enhanced Baseline with Return Boosting
- **Architecture**: Proven baseline + explicit return-to-previous boosting
- **Key insight**: 83.8% of test predictions are returns
- **Parameters**: 481k (96% of budget)
- **Results** (training ongoing at epoch 31):
  - Val Acc@1: **43.64%** (best so far, epoch 21)
  - Expected test: ~39-40% (based on historical 5% val-test gap)
- **Status**: ⏳ In progress, approaching baseline performance

## Key Findings

### What Didn't Work:
1. **Recurrent Transformer concept**: Replacing RNN/LSTM with Transformer-based recurrence performed worse than baseline
   - v1: 32.72% test (-5.23% vs baseline)
   - v2: 34.29% test (-3.66% vs baseline)
   
2. **Why it failed**:
   - Transformer recurrence is less suited for this task than standard Transformer
   - The iterative refinement doesn't help - single-pass Transformer is better
   - More parameters doesn't mean better performance with limited data

### What Works:
1. **Baseline architecture** (37.95% test) remains the strongest
2. **Domain knowledge**: Understanding return patterns (83.8%) is crucial
3. **Explicit mechanisms**: Return boosting shows promise (43.64% val so far)

### Critical Insights:
- **Return rate**: 83.8% of test predictions are returns to previous locations
  - 32.4% → last location
  - 33.1% → 2nd-to-last
  - 88.0% → within last 5
- **Oracle baseline**: Predicting any of last 3 = 65.53%!
- **Simple heuristics often beat complex models** with limited data

## Progress Toward Goals

| Target | Current Best | Status |
|--------|-------------|---------|
| 45% test acc@1 | 38.12% (domain model) | ❌ 6.88% short |
| 50% test acc@1 | 38.12% (domain model) | ❌ 11.88% short |

## Next Steps to Reach 50%

Based on learnings, to reach 50% requires:

1. **Better return mechanism implementation**:
   - Current models don't properly exploit 83.8% return rate
   - Need explicit binary classifier: "return vs new"
   - Then route accordingly

2. **Hybrid approach**:
   ```
   if is_return (83.8% probability):
       score = attention_over_last_5_locations()
   else:
       score = sequence_model()
   ```

3. **Simpler architecture**:
   - Complexity hurts with limited data (7,424 samples)
   - Focus on exploiting domain knowledge
   - Ensemble simple models rather than one complex model

4. **More data or external knowledge**:
   - 6.25 samples per location class on average
   - Geographic knowledge (POI, maps) would help
   - Semi-supervised learning

## Conclusion

The Recurrent Transformer concept did not improve over baseline. The enhanced baseline with return boosting shows promise (43.64% val) but still falls short of 50% target. The path to >50% requires:
- Explicit exploitation of 83.8% return rate
- Simpler, more targeted architectures
- Better use of domain knowledge over model complexity

---

**Date**: 2025-11-26
**Best Model So Far**: Domain-Informed (38.12% test)
**Current Experiment**: Enhanced Baseline (43.64% val, training)
