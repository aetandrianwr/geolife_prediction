"""
Script to train only the best model configuration (Model 2: 88d, 4L)
This is the fastest way to reproduce the 37.95% Test Acc@1 result.
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
    print("TRAINING BEST MODEL CONFIGURATION")
    print("Expected Results: Test Acc@1 = 37.95%, Val Acc@1 = 43.70%")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=0
    )
    
    print(f"✓ Dataset loaded:")
    print(f"  Train: {dataset_info['num_train']:,} samples")
    print(f"  Val: {dataset_info['num_val']:,} samples")
    print(f"  Test: {dataset_info['num_test']:,} samples")
    print(f"  Locations: {dataset_info['num_locations']:,}")
    print(f"  Users: {dataset_info['num_users']}")
    
    # Best model configuration (Model 2: 88d, 4L)
    config = {
        'name': 'Model_v2_88d_4L',
        'd_model': 88,
        'd_inner': 176,
        'n_layers': 4,
        'n_head': 8,
        'd_k': 11,
        'd_v': 11,
        'dropout': 0.15,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
    }
    
    print(f"\n{'='*80}")
    print(f"Model: {config['name']}")
    print(f"{'='*80}")
    print(f"Architecture:")
    print(f"  - d_model: {config['d_model']}")
    print(f"  - d_inner: {config['d_inner']}")
    print(f"  - n_layers: {config['n_layers']}")
    print(f"  - n_head: {config['n_head']}")
    print(f"  - dropout: {config['dropout']}")
    
    # Create model
    model_config = {k: v for k, v in config.items() 
                   if k not in ['name', 'learning_rate', 'weight_decay', 'label_smoothing']}
    
    model = LocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        **model_config,
        max_len=args.max_len
    )
    
    num_params = model.count_parameters()
    print(f"\n✓ Model created:")
    print(f"  Parameters: {num_params:,}")
    print(f"  Budget: 500,000")
    print(f"  Usage: {num_params/500000*100:.1f}%")
    print(f"  Status: {'✓ OK' if num_params < 500000 else '✗ EXCEEDED'}")
    
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
        patience=30,
        checkpoint_dir=f'checkpoints/{config["name"]}',
        log_interval=50
    )
    
    print(f"\n✓ Trainer configured:")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Max epochs: 200")
    print(f"  Early stopping patience: 30")
    print(f"  Device: {args.device}")
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    print("Expected training time: 10-15 minutes on GPU")
    print("Expected best epoch: ~18")
    print("Expected best val acc: 43.70%")
    print("="*80 + "\n")
    
    test_metrics = trainer.train()
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f'{config["name"]}_results.txt'
    with open(results_file, 'w') as f:
        f.write(f"Configuration: {config['name']}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
        f.write(f"Test Acc@5: {test_metrics['acc@5']:.2f}%\n")
        f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"Test MRR: {test_metrics['mrr']:.2f}%\n")
        f.write(f"Test NDCG: {test_metrics['ndcg']:.2f}%\n\n")
        f.write(f"Model Parameters: {num_params:,}\n")
        f.write(f"Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"Best Epoch: {trainer.best_epoch + 1}\n")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%")
    print(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%")
    print(f"  Test Acc@10: {test_metrics['acc@10']:.2f}%")
    print(f"  Test MRR:    {test_metrics['mrr']:.2f}%")
    print(f"  Test NDCG:   {test_metrics['ndcg']:.2f}%")
    print(f"\n📈 Validation:")
    print(f"  Best Val Acc@1: {trainer.best_val_acc:.2f}%")
    print(f"  Best Epoch: {trainer.best_epoch + 1}")
    print(f"\n💾 Saved:")
    print(f"  Checkpoint: checkpoints/{config['name']}/best_model.pt")
    print(f"  Results: {results_file}")
    
    print("\n" + "="*80)
    if test_metrics['acc@1'] >= 40.0:
        print("🎉 TARGET ACHIEVED! Test Acc@1 >= 40%")
    elif test_metrics['acc@1'] >= 37.5:
        print(f"✓ Expected range! Test Acc@1 = {test_metrics['acc@1']:.2f}%")
        print(f"  (Expected: 37.95% ± 0.5%)")
    else:
        print(f"⚠ Lower than expected. Test Acc@1 = {test_metrics['acc@1']:.2f}%")
        print(f"  Check dataset, seeds, or GPU determinism")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
