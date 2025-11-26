"""
Train Recurrent Transformer for Next Location Prediction

Key Innovation:
- Transformer-based recurrence (not RNN/LSTM)
- Processes same input sequence multiple times
- Refines hidden state at each cycle
- Maintains global context throughout

Target: >50% acc@1 on GeoLife test set
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
from src.models.recurrent_transformer import RecurrentTransformer
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
    parser.add_argument('--config', type=str, default='configs/recurrent_transformer.yml')
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
    
    # Create model
    logger.info("="*80)
    logger.info("CREATING RECURRENT TRANSFORMER")
    logger.info("="*80)
    logger.info("Architecture:")
    logger.info(f"  - Transformer-based recurrence (not RNN/LSTM)")
    logger.info(f"  - Recurrent cycles: {config.model['n_cycles']}")
    logger.info(f"  - d_model: {config.model['d_model']}")
    logger.info(f"  - n_head: {config.model['n_head']}")
    logger.info(f"  - dim_feedforward: {config.model['dim_feedforward']}")
    logger.info("="*80)
    
    model = RecurrentTransformer(
        num_locations=dataset_info['num_locations'],
        num_users=dataset_info['num_users'],
        d_model=config.model['d_model'],
        n_head=config.model['n_head'],
        dim_feedforward=config.model['dim_feedforward'],
        n_cycles=config.model['n_cycles'],
        dropout=config.model['dropout']
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
    logger.info("STARTING TRAINING - RECURRENT TRANSFORMER")
    logger.info("="*80)
    logger.info(f"Target: >50% test acc@1")
    logger.info(f"Minimum: >45% test acc@1")
    logger.info("="*80)
    
    test_metrics = trainer.train()
    
    # Save results
    result_file = result_dir / f'{config.experiment["name"]}_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Experiment: {config.experiment['name']}\n")
        f.write(f"Seed: {config.experiment['seed']}\n")
        f.write("="*80 + "\n\n")
        
        f.write("ARCHITECTURE:\n")
        f.write("  Recurrent Transformer (Transformer-based recurrence)\n")
        f.write(f"  Cycles: {config.model['n_cycles']}\n")
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
    
    if test_metrics['acc@1'] >= 45.0:
        logger.info("🎯 MINIMUM TARGET REACHED: >45% test acc@1!")
    if test_metrics['acc@1'] >= 50.0:
        logger.info("🎉 STRETCH GOAL REACHED: >50% test acc@1!")
    
    logger.info("="*80)


if __name__ == '__main__':
    main()
