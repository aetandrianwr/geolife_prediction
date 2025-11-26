#!/bin/bash
# Evaluation script for trained models

# Default values
CHECKPOINT="checkpoints/model_v2/best_model.pt"
SPLIT="test"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--checkpoint CHECKPOINT] [--split SPLIT]"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "Evaluating Geolife Next-Location Prediction Model"
echo "================================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Split: $SPLIT"
echo "================================================================================"

python evaluate.py --checkpoint "$CHECKPOINT" --split "$SPLIT"
