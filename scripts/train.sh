#!/bin/bash
# Training script for Geolife next-location prediction

# Default values
CONFIG="configs/model_v2.yml"
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--config CONFIG] [--seed SEED]"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "Training Geolife Next-Location Prediction Model"
echo "================================================================================"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "================================================================================"

python train.py --config "$CONFIG" --seed "$SEED"
