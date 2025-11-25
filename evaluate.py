"""
Evaluation script for trained models.

Usage:
    python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt
    python evaluate.py --checkpoint checkpoints/model_v2/best_model.pt --data_dir /path/to/data
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.attention_model import LocationPredictionModel
from src.utils.metrics import calculate_correct_total_prediction, get_performance_dict
from src.utils.logger import setup_logger
from src.utils.config import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                       default='/content/another_try_20251125/data/geolife',
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    
    return parser.parse_args()


def evaluate_model(model, dataloader, device, logger):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to use
        logger: Logger
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_results = []
    all_true_labels = []
    all_pred_labels = []
    
    logger.info("Running evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
        "f1": 0,
        "total": all_results[6],
    }
    
    metrics = get_performance_dict(return_dict)
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger('geolife_eval', console=True)
    
    logger.info("="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Try to load config from checkpoint directory
    config_path = checkpoint_path.parent / 'config.yml'
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        config = load_config(str(config_path))
        model_config = config['model']
    else:
        logger.warning("Config not found, using defaults from checkpoint")
        # Try to infer from checkpoint or use defaults
        model_config = {
            'd_model': 88,
            'd_inner': 176,
            'n_layers': 4,
            'n_head': 8,
            'd_k': 11,
            'd_v': 11,
            'dropout': 0.15,
            'max_len': 50
        }
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    # Load data
    logger.info(f"Loading data from: {args.data_dir}")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_len=model_config.get('max_len', 50),
        num_workers=0
    )
    
    # Select split
    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    else:
        dataloader = test_loader
    
    logger.info(f"Evaluating on {args.split} set ({dataset_info[f'num_{args.split}']} samples)")
    
    # Create model
    logger.info("Creating model...")
    model = LocationPredictionModel(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=model_config['d_model'],
        d_inner=model_config['d_inner'],
        n_layers=model_config['n_layers'],
        n_head=model_config['n_head'],
        d_k=model_config['d_k'],
        d_v=model_config['d_v'],
        dropout=model_config['dropout'],
        max_len=model_config['max_len']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    logger.info(f"Parameters: {model.count_parameters():,}")
    
    # Evaluate
    metrics = evaluate_model(model, dataloader, device, logger)
    
    # Print results
    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    logger.info(f"Split: {args.split}")
    logger.info("")
    logger.info("Accuracy Metrics:")
    logger.info(f"  Acc@1:  {metrics['acc@1']:.2f}%")
    logger.info(f"  Acc@5:  {metrics['acc@5']:.2f}%")
    logger.info(f"  Acc@10: {metrics['acc@10']:.2f}%")
    logger.info("")
    logger.info("Ranking Metrics:")
    logger.info(f"  MRR:    {metrics['mrr']:.2f}%")
    logger.info(f"  NDCG:   {metrics['ndcg']:.2f}%")
    logger.info("")
    logger.info(f"Total samples: {int(metrics['total']):,}")
    logger.info("="*80)
    
    # Save results
    result_file = checkpoint_path.parent / f'eval_{args.split}_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Acc@1:  {metrics['acc@1']:.2f}%\n")
        f.write(f"Acc@5:  {metrics['acc@5']:.2f}%\n")
        f.write(f"Acc@10: {metrics['acc@10']:.2f}%\n")
        f.write(f"MRR:    {metrics['mrr']:.2f}%\n")
        f.write(f"NDCG:   {metrics['ndcg']:.2f}%\n\n")
        f.write(f"Total: {int(metrics['total'])}\n")
    
    logger.info(f"Results saved to: {result_file}")
    
    return metrics


if __name__ == '__main__':
    main()
