# Next-Location Prediction on Geolife Dataset

**Objective**: Achieve stable 40% Test Acc@1 with <500K parameters

## Project Structure
```
geolife_prediction/
├── data/                    # Dataset and preprocessing
├── src/
│   ├── models/             # Model architectures
│   ├── utils/              # Training utilities and metrics
│   └── data/               # Data loaders and preprocessing
├── configs/                # Configuration files
├── experiments/            # Training scripts and logs
├── checkpoints/            # Saved models
└── results/                # Evaluation results
```

## Approach
- Systematic exploration of architectures
- Early stopping on unpromising approaches
- Focus on proven techniques for sequential prediction
- Maintain strict train/val/test separation
