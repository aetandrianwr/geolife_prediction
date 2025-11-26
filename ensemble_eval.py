"""
Ensemble Evaluation Script

Combines predictions from multiple trained models to improve accuracy.
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict


def load_model(checkpoint_path, device='cuda'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint
    model = LocationPredictionModel(
        num_locations=1187,
        num_users=46,
        d_model=88,
        d_inner=176,
        n_layers=4,
        n_head=8,
        d_k=11,
        d_v=11,
        dropout=0.15,
        max_len=50
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def get_predictions(model, data_loader, device):
    """Get predictions from a single model."""
    all_logits = []
    all_targets = []
    
    for batch in tqdm(data_loader, desc='Predicting', leave=False):
        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_targets.append(batch['target'].cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_logits, all_targets


def ensemble_predict(checkpoint_paths, data_loader, device='cuda', method='average'):
    """
    Ensemble predictions from multiple models.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        data_loader: DataLoader for evaluation
        device: Device to run on
        method: Ensemble method ('average', 'vote', 'weighted')
    
    Returns:
        Ensemble predictions and targets
    """
    print(f"Loading {len(checkpoint_paths)} models...")
    models = [load_model(cp, device) for cp in checkpoint_paths]
    
    print("Getting predictions from each model...")
    all_predictions = []
    targets = None
    
    for i, model in enumerate(models):
        logits, tgts = get_predictions(model, data_loader, device)
        all_predictions.append(logits)
        if targets is None:
            targets = tgts
        print(f"Model {i+1}/{len(models)}: Logits shape {logits.shape}")
    
    # Ensemble
    print(f"Combining predictions using method: {method}")
    if method == 'average':
        # Average logits
        ensemble_logits = torch.stack(all_predictions).mean(dim=0)
    elif method == 'vote':
        # Majority voting on top-1 predictions
        top1_preds = [logits.argmax(dim=1) for logits in all_predictions]
        # For each sample, count votes for each class
        votes = torch.stack(top1_preds)  # (num_models, num_samples)
        # Create logits from vote counts
        ensemble_logits = torch.zeros_like(all_predictions[0])
        for i in range(len(targets)):
            unique, counts = torch.unique(votes[:, i], return_counts=True)
            ensemble_logits[i, unique] = counts.float()
    elif method == 'weighted':
        # Weighted average (weights could be based on validation accuracy)
        # For now, use uniform weights
        weights = torch.ones(len(all_predictions)) / len(all_predictions)
        ensemble_logits = sum(w * logits for w, logits in zip(weights, all_predictions))
    
    return ensemble_logits, targets


def evaluate_ensemble(checkpoint_paths, data_dir, method='average', device='cuda'):
    """Evaluate ensemble on test set."""
    
    # Load data
    print("Loading dataset...")
    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        max_len=50,
        num_workers=0
    )
    
    # Get ensemble predictions
    ensemble_logits, targets = ensemble_predict(
        checkpoint_paths, test_loader, device, method
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    result_array, _, _ = calculate_correct_total_prediction(ensemble_logits, targets)
    
    return_dict = {
        "correct@1": result_array[0],
        "correct@3": result_array[1],
        "correct@5": result_array[2],
        "correct@10": result_array[3],
        "rr": result_array[4],
        "ndcg": result_array[5],
        "f1": 0.0,
        "total": result_array[6]
    }
    
    metrics = get_performance_dict(return_dict)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Ensemble evaluation')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Paths to model checkpoints')
    parser.add_argument('--data_dir', type=str,
                       default='/content/geolife_prediction/data/geolife',
                       help='Path to data directory')
    parser.add_argument('--method', type=str, default='average',
                       choices=['average', 'vote', 'weighted'],
                       help='Ensemble method')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE EVALUATION")
    print("="*80)
    print(f"Number of models: {len(args.checkpoints)}")
    print(f"Ensemble method: {args.method}")
    print(f"Device: {args.device}")
    print()
    
    metrics = evaluate_ensemble(
        args.checkpoints,
        args.data_dir,
        args.method,
        args.device
    )
    
    print()
    print("="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    print(f"Test Acc@1:  {metrics['acc@1']:.2f}%")
    print(f"Test Acc@5:  {metrics['acc@5']:.2f}%")
    print(f"Test Acc@10: {metrics['acc@10']:.2f}%")
    print(f"Test MRR:    {metrics['mrr']:.2f}%")
    print(f"Test NDCG:   {metrics['ndcg']:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
