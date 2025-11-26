"""
Train properly researched model with deep understanding of:
- Data characteristics
- Architectural best practices  
- Task-specific requirements
"""

import os
import sys
import random
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import get_dataloaders
from src.models.properly_researched_model import ProperlyResearchedModel
from src.utils.trainer import Trainer
from src.utils.config import get_config, save_config, print_config
from src.utils.logger import setup_logger, log_experiment_info


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/properly_researched.yml')
    args = parser.parse_args()
    
    config = get_config(args.config, args)
    set_seed(config.experiment['seed'])
    
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
    
    print_config(config)
    save_config(config, str(checkpoint_dir / 'config.yml'))
    
    device = config.experiment['device']
    if device == 'cuda' and not torch.cuda.is_available():
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
    
    # Create properly researched model
    logger.info("="*80)
    logger.info("CREATING PROPERLY RESEARCHED MODEL")
    logger.info("="*80)
    logger.info("Improvements:")
    logger.info("  1. ✓ Pre-Layer Normalization (critical for stability)")
    logger.info("  2. ✓ GeGLU activation in FFN (better than standard GELU)")
    logger.info("  3. ✓ Relative position bias (better than absolute positions)")
    logger.info("  4. ✓ Multi-scale temporal encoding (discrete + continuous)")
    logger.info("  5. ✓ Proper weight initialization (Xavier)")
    logger.info("="*80)
    
    model = ProperlyResearchedModel(
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
    
    num_params = model.count_parameters()
    logger.info(f"✓ Model has {num_params:,} parameters ({num_params/500000*100:.1f}% of budget)")
    
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
    
    logger.info("="*80)
    logger.info("STARTING TRAINING - PROPERLY RESEARCHED MODEL")
    logger.info("="*80)
    
    test_metrics = trainer.train()
    
    # Save results
    result_file = result_dir / f'{config.experiment["name"]}_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Experiment: {config.experiment['name']}\n")
        f.write(f"Seed: {config.experiment['seed']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("PROPERLY RESEARCHED IMPROVEMENTS:\n")
        f.write("  1. Pre-Layer Normalization\n")
        f.write("  2. GeGLU activation in FFN\n")
        f.write("  3. Relative position bias\n")
        f.write("  4. Multi-scale temporal encoding\n")
        f.write("  5. Proper Xavier initialization\n\n")
        
        f.write("MODEL:\n")
        f.write(f"  Parameters: {num_params:,}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%\n")
        f.write(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%\n")
        f.write(f"  Test Acc@10: {test_metrics['acc@10']:.2f}%\n")
        f.write(f"  Test MRR:    {test_metrics['mrr']:.2f}%\n")
        f.write(f"  Test NDCG:   {test_metrics['ndcg']:.2f}%\n\n")
        
        f.write(f"  Best Val Acc@1: {trainer.best_val_acc:.2f}%\n")
        f.write(f"  Best Epoch: {trainer.best_epoch}\n")
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {result_file}")
    logger.info(f"Test Acc@1: {test_metrics['acc@1']:.2f}%")
    logger.info(f"Val Acc@1: {trainer.best_val_acc:.2f}%")
    logger.info("="*80)


if __name__ == '__main__':
    main()
