#!/bin/bash

echo "================================================================================"
echo "GEOLIFE PREDICTION SYSTEM - SETUP VERIFICATION"
echo "================================================================================"

# Check Python
echo -e "\n📦 Python & Packages:"
python3 --version
echo -n "PyTorch: "
python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED"
echo -n "CUDA Available: "
python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A"

# Check dataset
echo -e "\n📁 Dataset:"
DATA_DIR="/content/another_try_20251125/data/geolife"
if [ -d "$DATA_DIR" ]; then
    echo "✓ Dataset found: $DATA_DIR"
    echo "  Files:"
    ls -lh "$DATA_DIR"/*.pkl 2>/dev/null | awk '{print "    " $9 " (" $5 ")"}'
else
    echo "✗ Dataset NOT found at: $DATA_DIR"
fi

# Check project structure
echo -e "\n📂 Project Structure:"
[ -d "src" ] && echo "✓ src/" || echo "✗ src/"
[ -d "src/data" ] && echo "✓ src/data/" || echo "✗ src/data/"
[ -d "src/models" ] && echo "✓ src/models/" || echo "✗ src/models/"
[ -d "src/utils" ] && echo "✓ src/utils/" || echo "✗ src/utils/"
[ -f "train.py" ] && echo "✓ train.py" || echo "✗ train.py"
[ -f "train_single_best.py" ] && echo "✓ train_single_best.py" || echo "✗ train_single_best.py"
[ -f "evaluate_best_model.py" ] && echo "✓ evaluate_best_model.py" || echo "✗ evaluate_best_model.py"

# Check for trained models
echo -e "\n🎯 Trained Models:"
if [ -d "checkpoints" ]; then
    for model_dir in checkpoints/*/; do
        if [ -f "${model_dir}best_model.pt" ]; then
            model_name=$(basename "$model_dir")
            size=$(du -h "${model_dir}best_model.pt" | cut -f1)
            echo "✓ $model_name (checkpoint: $size)"
        fi
    done
    
    if [ ! -f "checkpoints/Model_v2_88d_4L/best_model.pt" ]; then
        echo "⚠ Best model (Model_v2_88d_4L) not found - need to train"
    fi
else
    echo "⚠ No checkpoints directory - models not trained yet"
fi

# Check results
echo -e "\n📊 Results:"
if [ -d "results" ]; then
    for result_file in results/*.txt; do
        if [ -f "$result_file" ]; then
            echo "✓ $(basename "$result_file")"
            # Extract test acc
            acc=$(grep "Test Acc@1:" "$result_file" | head -1)
            [ -n "$acc" ] && echo "    $acc"
        fi
    done
else
    echo "⚠ No results directory"
fi

# Summary
echo -e "\n================================================================================"
echo "QUICK START COMMANDS:"
echo "================================================================================"
if [ -f "checkpoints/Model_v2_88d_4L/best_model.pt" ]; then
    echo "✓ Best model exists - evaluate it:"
    echo "  python3 evaluate_best_model.py"
else
    echo "⚠ Best model not trained yet - train it:"
    echo "  python3 train_single_best.py"
fi
echo ""
echo "Or train all models:"
echo "  python3 train.py"
echo ""
echo "================================================================================"
