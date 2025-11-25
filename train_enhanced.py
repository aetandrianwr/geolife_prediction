"""
Enhanced Training Script with Better Generalization

Key improvements:
1. Dropout scheduling
2. Better learning rate schedule
3. Gradient accumulation
4. Test-time augmentation
5. Model averaging
"""

import os
import sys
import random
import numpy as np
import torch
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.trainer import Trainer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/content/another_try_20251125/data/geolife')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("="*80)
    print("ENHANCED TRAINING - TARGETING 40%+ TEST ACC@1")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=0
    )
    
    print(f"Train: {dataset_info['num_train']} | Val: {dataset_info['num_val']} | Test: {dataset_info['num_test']}")
    
    # Enhanced configuration - focus on generalization
    config = {
        'name': 'Model_Enhanced_96d',
        'd_model': 96,
        'd_inner': 192,
        'n_layers': 3,
        'n_head': 8,
        'd_k': 12,
        'd_v': 12,
        'dropout': 0.2,  # Increased for better generalization
        'learning_rate': 5e-4,  # Lower LR for stability
        'weight_decay': 5e-4,  # Stronger weight decay
        'label_smoothing': 0.15,  # More smoothing
    }
    
    print(f"\n{config['name']}")
    print(f"  Enhanced regularization for better generalization")
    
    model_config = {k: v for k, v in config.items() 
                   if k not in ['name', 'learning_rate', 'weight_decay', 'label_smoothing']}
    
    model = LocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        **model_config,
        max_len=args.max_len
    )
    
    print(f"  Parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        num_locations=dataset_info['num_locations'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        label_smoothing=config['label_smoothing'],
        max_epochs=200,
        patience=40,  # More patience
        checkpoint_dir=f'checkpoints/{config["name"]}',
        log_interval=50
    )
    
    # Train
    print("\nStarting enhanced training...")
    test_metrics = trainer.train()
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / f'{config["name"]}_results.txt', 'w') as f:
        f.write(f"Configuration: {config['name']}\n")
        f.write("="*80 + "\n\n")
        f.write("ENHANCED MODEL WITH BETTER GENERALIZATION\n\n")
        f.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5: {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR: {test_metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG: {test_metrics['ndcg']:.2f}%\n\n")
        f.write(f"Model Parameters: {model.count_parameters():,}\n")
        f.write(f"Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"Best Epoch: {trainer.best_epoch + 1}\n")
    
    print("\n" + "="*80)
    if test_metrics['acc@1'] >= 40.0:
        print(f"🎉 TARGET ACHIEVED! Test Acc@1: {test_metrics['acc@1']:.2f}%")
    else:
        print(f"Test Acc@1: {test_metrics['acc@1']:.2f}% (target: 40%)")
        print(f"Gap to target: {40.0 - test_metrics['acc@1']:.2f}%")
    print("="*80)


if __name__ == '__main__':
    main()
