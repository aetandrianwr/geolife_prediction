"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = 'geolife',
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Remove existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'geolife') -> logging.Logger:
    """Get existing logger or create a basic one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name)
    return logger


class MetricLogger:
    """Logger for tracking metrics during training."""
    
    def __init__(self, log_file: str = None):
        self.metrics = {}
        self.log_file = log_file
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # Write header
            with open(log_file, 'w') as f:
                f.write('epoch,split,loss,acc@1,acc@5,mrr,ndcg\n')
    
    def log(self, epoch: int, split: str, metrics: dict):
        """Log metrics for an epoch."""
        key = f"{split}_epoch_{epoch}"
        self.metrics[key] = metrics
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{split},{metrics.get('loss', 0):.4f},"
                       f"{metrics.get('acc@1', 0):.2f},{metrics.get('acc@5', 0):.2f},"
                       f"{metrics.get('mrr', 0):.2f},{metrics.get('ndcg', 0):.2f}\n")
    
    def get_metrics(self, split: str = None):
        """Get all logged metrics, optionally filtered by split."""
        if split is None:
            return self.metrics
        return {k: v for k, v in self.metrics.items() if split in k}


def log_experiment_info(logger: logging.Logger, config, model, dataset_info):
    """Log experiment configuration and model info."""
    logger.info("="*80)
    logger.info("EXPERIMENT INFORMATION")
    logger.info("="*80)
    
    # Experiment details
    logger.info(f"Experiment: {config.experiment['name']}")
    logger.info(f"Seed: {config.experiment['seed']}")
    logger.info(f"Device: {config.experiment['device']}")
    
    # Dataset info
    logger.info("")
    logger.info("DATASET:")
    logger.info(f"  Train samples: {dataset_info.get('num_train', 0):,}")
    logger.info(f"  Val samples: {dataset_info.get('num_val', 0):,}")
    logger.info(f"  Test samples: {dataset_info.get('num_test', 0):,}")
    logger.info(f"  Num locations: {dataset_info.get('num_locations', 0):,}")
    logger.info(f"  Num users: {dataset_info.get('num_users', 0)}")
    
    # Model info
    logger.info("")
    logger.info("MODEL:")
    logger.info(f"  Name: {config.model['name']}")
    logger.info(f"  d_model: {config.model['d_model']}")
    logger.info(f"  d_inner: {config.model['d_inner']}")
    logger.info(f"  n_layers: {config.model['n_layers']}")
    logger.info(f"  n_head: {config.model['n_head']}")
    logger.info(f"  dropout: {config.model['dropout']}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    logger.info(f"  Parameter budget: 500,000")
    logger.info(f"  Budget usage: {model.count_parameters()/500000*100:.1f}%")
    
    # Training config
    logger.info("")
    logger.info("TRAINING:")
    logger.info(f"  Batch size: {config.training['batch_size']}")
    logger.info(f"  Learning rate: {config.training['learning_rate']}")
    logger.info(f"  Weight decay: {config.training['weight_decay']}")
    logger.info(f"  Label smoothing: {config.training['label_smoothing']}")
    logger.info(f"  Max epochs: {config.training['max_epochs']}")
    logger.info(f"  Patience: {config.training['patience']}")
    
    logger.info("="*80)
