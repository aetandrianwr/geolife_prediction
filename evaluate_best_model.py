"""
Evaluate the best saved model on test set
Loads the checkpoint and reports metrics
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


def evaluate_checkpoint(checkpoint_path, data_dir, device='cuda'):
    """Evaluate a saved checkpoint on test set."""
    
    print("="*80)
    print("EVALUATING BEST MODEL CHECKPOINT")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    _, _, test_loader, dataset_info = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    print(f"✓ Test set loaded: {dataset_info['num_test']:,} samples")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model with saved configuration
    print("\nCreating model...")
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (from epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Evaluate
    print("\nEvaluating on test set...")
    
    all_results = []
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device
            batch_device = {
                'locations': batch['locations'].to(device),
                'users': batch['users'].to(device),
                'weekdays': batch['weekdays'].to(device),
                'start_mins': batch['start_mins'].to(device),
                'mask': batch['mask'].to(device)
            }
            target = batch['target'].to(device)
            
            # Forward pass
            logits = model(batch_device)
            
            # Calculate metrics
            result, true_labels, pred_labels = calculate_correct_total_prediction(logits, target)
            all_results.append(result)
            all_true_labels.append(true_labels)
            all_pred_labels.append(pred_labels)
    
    # Aggregate results
    all_results = np.sum(all_results, axis=0)
    return_dict = {
        "correct@1": all_results[0],
        "correct@3": all_results[1],
        "correct@5": all_results[2],
        "correct@10": all_results[3],
        "rr": all_results[4],
        "ndcg": all_results[5],
        "f1": 0,  # Not used
        "total": all_results[6],
    }
    
    metrics = get_performance_dict(return_dict)
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Acc@1:  {metrics['acc@1']:.2f}%")
    print(f"  Acc@5:  {metrics['acc@5']:.2f}%")
    print(f"  Acc@10: {metrics['acc@10']:.2f}%")
    print(f"\n📈 Ranking Metrics:")
    print(f"  MRR:    {metrics['mrr']:.2f}%")
    print(f"  NDCG:   {metrics['ndcg']:.2f}%")
    print(f"\n📝 Evaluation:")
    print(f"  Total samples: {int(metrics['total']):,}")
    print(f"  Correct@1: {int(return_dict['correct@1']):,}")
    
    print("\n" + "="*80)
    if metrics['acc@1'] >= 40.0:
        print("🎉 TARGET ACHIEVED! Test Acc@1 >= 40%")
    elif metrics['acc@1'] >= 37.5:
        print(f"✓ Expected performance! Test Acc@1 = {metrics['acc@1']:.2f}%")
        print(f"  (Best model achieved: 37.95%)")
    else:
        print(f"⚠ Lower than expected: {metrics['acc@1']:.2f}%")
    print("="*80 + "\n")
    
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/Model_v2_88d_4L/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='/content/another_try_20251125/data/geolife',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"\nAvailable checkpoints:")
        checkpoint_dir = Path('checkpoints')
        if checkpoint_dir.exists():
            for model_dir in checkpoint_dir.iterdir():
                if model_dir.is_dir():
                    ckpt = model_dir / 'best_model.pt'
                    if ckpt.exists():
                        print(f"  ✓ {ckpt}")
        else:
            print(f"  (No checkpoints directory found)")
        print(f"\nPlease train a model first using:")
        print(f"  python3 train_single_best.py")
        return
    
    # Evaluate
    metrics = evaluate_checkpoint(args.checkpoint, args.data_dir, args.device)
    
    # Save results
    results_file = Path('results') / 'evaluated_checkpoint_results.txt'
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("CHECKPOINT EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        f.write(f"Test Acc@1:  {metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5:  {metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR:    {metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG:   {metrics['ndcg']:.2f}%\n")
    
    print(f"💾 Results saved to: {results_file}")


if __name__ == '__main__':
    main()
