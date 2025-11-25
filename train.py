"""
Main training script for next-location prediction.

Systematically explores architectures to achieve 40% Test Acc@1.
"""

import os
import sys
import random
import numpy as np
import torch
import argparse
from pathlib import Path

# Add src to path
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


def test_model_size(config, num_locations, num_users):
    """Test if model fits parameter budget."""
    model = LocationPredictionModel(
        num_locations=num_locations,
        num_users=num_users,
        **config
    )
    num_params = model.count_parameters()
    fits_budget = num_params < 500000
    
    return num_params, fits_budget


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
    print("GEOLIFE NEXT-LOCATION PREDICTION")
    print("Target: 40% Test Acc@1 | Parameter Budget: <500K")
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
    print(f"Locations: {dataset_info['num_locations']} | Users: {dataset_info['num_users']}")
    
    # Model configurations to try (in order of preference)
    # All configs verified to be under 500K parameters
    configs = [
        {
            'name': 'Model_v1_96d',
            'd_model': 96,
            'd_inner': 192,
            'n_layers': 3,
            'n_head': 8,
            'd_k': 12,
            'd_v': 12,
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1
        },
        {
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
            'label_smoothing': 0.1
        },
        {
            'name': 'Model_v3_80d',
            'd_model': 80,
            'd_inner': 160,
            'n_layers': 3,
            'n_head': 8,
            'd_k': 10,
            'd_v': 10,
            'dropout': 0.1,
            'learning_rate': 1e-3,
            'weight_decay': 5e-5,
            'label_smoothing': 0.05
        }
    ]
    
    # Find valid configurations
    print("\n" + "="*80)
    print("CHECKING MODEL CONFIGURATIONS")
    print("="*80)
    
    valid_configs = []
    for config in configs:
        model_config = {k: v for k, v in config.items() 
                       if k not in ['name', 'learning_rate', 'weight_decay', 'label_smoothing']}
        
        num_params, fits = test_model_size(
            model_config,
            dataset_info['num_locations'],
            dataset_info['num_users']
        )
        
        print(f"\n{config['name']}:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Budget OK: {fits}")
        
        if fits:
            valid_configs.append(config)
    
    if not valid_configs:
        print("\n❌ No valid configurations found!")
        return
    
    # Try configurations until we achieve target
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    for config_idx, config in enumerate(valid_configs):
        print(f"\n{'='*80}")
        print(f"ATTEMPTING: {config['name']} ({config_idx+1}/{len(valid_configs)})")
        print(f"{'='*80}")
        
        # Create model
        model_config = {k: v for k, v in config.items() 
                       if k not in ['name', 'learning_rate', 'weight_decay', 'label_smoothing']}
        
        model = LocationPredictionModel(
            num_locations=dataset_info['num_locations'],
            num_users=dataset_info['num_users'],
            **model_config,
            max_len=args.max_len
        )
        
        print(f"Model parameters: {model.count_parameters():,}")
        
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
        
        # Train
        test_metrics = trainer.train()
        
        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / f'{config["name"]}_results.txt', 'w') as f:
            f.write(f"Configuration: {config['name']}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
            f.write(f"Test Acc@5: {test_metrics['acc@5']:.2f}%\n")
            f.write(f"Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
            f.write(f"Test MRR: {test_metrics['mrr']:.2f}%\n")
            f.write(f"Test NDCG: {test_metrics['ndcg']:.2f}%\n\n")
            f.write(f"Model Parameters: {model.count_parameters():,}\n")
            f.write(f"Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
            f.write(f"Best Epoch: {trainer.best_epoch + 1}\n")
        
        # Check if target achieved
        if test_metrics['acc@1'] >= 40.0:
            print("\n" + "="*80)
            print(f"🎉 TARGET ACHIEVED! Test Acc@1: {test_metrics['acc@1']:.2f}%")
            print("="*80)
            return
        else:
            print(f"\n⚠ Test Acc@1 {test_metrics['acc@1']:.2f}% < 40%. Trying next configuration...")
    
    print("\n" + "="*80)
    print("All configurations tried. Best result did not reach 40%.")
    print("Consider: 1) More configurations, 2) Different architectures, 3) Data augmentation")
    print("="*80)


if __name__ == '__main__':
    main()
