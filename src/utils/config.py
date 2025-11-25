"""
Configuration management utilities.
Handles loading and merging of YAML configs with command-line arguments.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration container with dict-like access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    def __getattr__(self, key):
        if key.startswith('_'):
            return super().__getattribute__(key)
        return self._config.get(key)
    
    def __getitem__(self, key):
        return self._config[key]
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config.copy()
    
    def __repr__(self):
        return f"Config({self._config})"


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override_config into base_config."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def get_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    """
    Load configuration from file and override with command-line arguments.
    
    Args:
        config_path: Path to YAML config file
        args: Command-line arguments from argparse
        
    Returns:
        Config object with all settings
    """
    # Load default config
    default_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yml'
    if default_path.exists():
        config = load_config(str(default_path))
    else:
        config = {}
    
    # Load specified config
    if config_path:
        custom_config = load_config(config_path)
        config = merge_configs(config, custom_config)
    
    # Override with command-line arguments
    if args is not None:
        overrides = {}
        
        # Model overrides
        if hasattr(args, 'd_model') and args.d_model is not None:
            overrides.setdefault('model', {})['d_model'] = args.d_model
        if hasattr(args, 'd_inner') and args.d_inner is not None:
            overrides.setdefault('model', {})['d_inner'] = args.d_inner
        if hasattr(args, 'n_layers') and args.n_layers is not None:
            overrides.setdefault('model', {})['n_layers'] = args.n_layers
        if hasattr(args, 'n_head') and args.n_head is not None:
            overrides.setdefault('model', {})['n_head'] = args.n_head
        if hasattr(args, 'dropout') and args.dropout is not None:
            overrides.setdefault('model', {})['dropout'] = args.dropout
        
        # Training overrides
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            overrides.setdefault('training', {})['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            overrides.setdefault('training', {})['learning_rate'] = args.learning_rate
        if hasattr(args, 'weight_decay') and args.weight_decay is not None:
            overrides.setdefault('training', {})['weight_decay'] = args.weight_decay
        if hasattr(args, 'max_epochs') and args.max_epochs is not None:
            overrides.setdefault('training', {})['max_epochs'] = args.max_epochs
        if hasattr(args, 'patience') and args.patience is not None:
            overrides.setdefault('training', {})['patience'] = args.patience
        
        # Data overrides
        if hasattr(args, 'data_dir') and args.data_dir is not None:
            overrides.setdefault('data', {})['data_dir'] = args.data_dir
        
        # Experiment overrides
        if hasattr(args, 'seed') and args.seed is not None:
            overrides.setdefault('experiment', {})['seed'] = args.seed
        if hasattr(args, 'device') and args.device is not None:
            overrides.setdefault('experiment', {})['device'] = args.device
        if hasattr(args, 'experiment_name') and args.experiment_name is not None:
            overrides.setdefault('experiment', {})['name'] = args.experiment_name
        
        # Merge overrides
        config = merge_configs(config, overrides)
    
    # Set experiment name if not specified
    if config.get('experiment', {}).get('name') is None:
        config['experiment']['name'] = config.get('model', {}).get('name', 'experiment')
    
    return Config(config)


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def print_config(config: Config):
    """Pretty print configuration."""
    print("="*80)
    print("CONFIGURATION")
    print("="*80)
    
    config_dict = config.to_dict()
    
    for section, values in config_dict.items():
        print(f"\n{section.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
    
    print("="*80)
