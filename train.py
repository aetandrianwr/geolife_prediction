"""
Main training script for Geolife next-location prediction.

Usage:
    python train.py --config configs/model_v2.yml
    python train.py --config configs/model_v2.yml --seed 123 --batch_size 128
    python train.py --config configs/model_v2.yml --learning_rate 0.001
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.trainer import Trainer
from src.utils.config import get_config, save_config, print_config
from src.utils.logger import setup_logger, log_experiment_info


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train next-location prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument('--config', type=str, default='configs/default.yml',
                       help='Path to config file')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=None,
                       help='Model dimension')
    parser.add_argument('--d_inner', type=int, default=None,
                       help='Inner dimension')
    parser.add_argument('--n_layers', type=int, default=None,
                       help='Number of layers')
    parser.add_argument('--n_head', type=int, default=None,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=None,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to data directory')
    
    # Experiment arguments
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config, args)
    
    # Set random seed
    set_seed(config.experiment['seed'], config.experiment.get('deterministic', True))
    
    # Setup directories
    checkpoint_dir = Path(config.experiment['checkpoint_dir']) / config.experiment['name']
    log_dir = Path(config.experiment['log_dir'])
    result_dir = Path(config.experiment['result_dir'])
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = log_dir / f"{config.experiment['name']}_train.log"
    logger = setup_logger('geolife', str(log_file))
    
    # Print configuration
    print_config(config)
    
    # Save configuration
    config_save_path = checkpoint_dir / 'config.yml'
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Set device
    device = config.experiment['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=config.data['data_dir'],
        batch_size=config.training['batch_size'],
        max_len=config.data['max_len'],
        num_workers=config.data.get('num_workers', 0)
    )
    logger.info("Dataset loaded successfully")
    
    # Create model
    logger.info("Creating model...")
    model = LocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config.model['d_model'],
        d_inner=config.model['d_inner'],
        n_layers=config.model['n_layers'],
        n_head=config.model['n_head'],
        d_k=config.model['d_k'],
        d_v=config.model['d_v'],
        dropout=config.model['dropout'],
        max_len=config.model['max_len']
    )
    
    # Check parameter budget
    num_params = model.count_parameters()
    if num_params >= 500000:
        logger.warning(f"Model has {num_params:,} parameters (exceeds 500K budget!)")
    else:
        logger.info(f"Model has {num_params:,} parameters ({num_params/500000*100:.1f}% of budget)")
    
    # Log experiment info
    log_experiment_info(logger, config, model, dataset_info)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_locations=dataset_info['num_locations'],
        learning_rate=config.training['learning_rate'],
        weight_decay=config.training['weight_decay'],
        label_smoothing=config.training['label_smoothing'],
        max_epochs=config.training['max_epochs'],
        patience=config.training['patience'],
        checkpoint_dir=str(checkpoint_dir),
        log_interval=config.training.get('log_interval', 50),
        logger=logger
    )
    
    # Train
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    test_metrics = trainer.train()
    
    # Save results
    result_file = result_dir / f'{config.experiment["name"]}_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Experiment: {config.experiment['name']}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Seed: {config.experiment['seed']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("MODEL:\n")
        f.write(f"  d_model: {config.model['d_model']}\n")
        f.write(f"  n_layers: {config.model['n_layers']}\n")
        f.write(f"  n_head: {config.model['n_head']}\n")
        f.write(f"  dropout: {config.model['dropout']}\n")
        f.write(f"  Parameters: {num_params:,}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%\n")
        f.write(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%\n")
        f.write(f"  Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"  Test MRR:    {test_metrics['mrr']:.2f}%\n")
        f.write(f"  Test NDCG:   {test_metrics['ndcg']:.2f}%\n\n")
        
        f.write(f"  Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"  Best Epoch: {trainer.best_epoch + 1}\n")
    
    logger.info("")
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {result_file}")
    logger.info(f"Checkpoint saved to: {checkpoint_dir / 'best_model.pt'}")
    logger.info("")
    logger.info("FINAL TEST RESULTS:")
    logger.info(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%")
    logger.info(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%")
    logger.info(f"  Test MRR:    {test_metrics['mrr']:.2f}%")
    logger.info(f"  Val Acc@1:   {trainer.best_val_acc:.2f}%")
    logger.info("="*80)
    
    return test_metrics


if __name__ == '__main__':
    main()
