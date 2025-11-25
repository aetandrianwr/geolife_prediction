"""
Geolife dataset loader and preprocessing.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GeolifeDataset(Dataset):
    """Geolife trajectory dataset."""
    
    def __init__(self, data_path, max_len=50):
        """
        Args:
            data_path: Path to pickle file
            max_len: Maximum sequence length
        """
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = self.data[idx]
        
        # Extract features
        locations = sample['X']  # Location IDs
        users = sample['user_X']  # User IDs
        weekdays = sample['weekday_X']  # Day of week
        start_mins = sample['start_min_X']  # Time of day (minutes)
        durations = sample['dur_X']  # Duration at location
        time_diffs = sample['diff']  # Time difference to next location
        target = sample['Y']  # Target location
        
        seq_len = len(locations)
        
        # Pad or truncate to max_len
        if seq_len > self.max_len:
            locations = locations[-self.max_len:]
            users = users[-self.max_len:]
            weekdays = weekdays[-self.max_len:]
            start_mins = start_mins[-self.max_len:]
            durations = durations[-self.max_len:]
            time_diffs = time_diffs[-self.max_len:]
            seq_len = self.max_len
        
        # Create tensors
        batch = {
            'locations': torch.zeros(self.max_len, dtype=torch.long),
            'users': torch.zeros(self.max_len, dtype=torch.long),
            'weekdays': torch.zeros(self.max_len, dtype=torch.long),
            'start_mins': torch.zeros(self.max_len, dtype=torch.long),
            'durations': torch.zeros(self.max_len, dtype=torch.float32),
            'time_diffs': torch.zeros(self.max_len, dtype=torch.long),
            'mask': torch.zeros(self.max_len, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.long),
            'seq_len': torch.tensor(seq_len, dtype=torch.long)
        }
        
        # Fill in actual values
        batch['locations'][:seq_len] = torch.from_numpy(locations)
        batch['users'][:seq_len] = torch.from_numpy(users)
        batch['weekdays'][:seq_len] = torch.from_numpy(weekdays)
        batch['start_mins'][:seq_len] = torch.from_numpy(start_mins)
        batch['durations'][:seq_len] = torch.from_numpy(durations)
        batch['time_diffs'][:seq_len] = torch.from_numpy(time_diffs)
        batch['mask'][:seq_len] = 1.0
        
        return batch


def get_dataloaders(data_dir, batch_size=64, max_len=50, num_workers=2):
    """Create train, validation, and test dataloaders."""
    
    train_dataset = GeolifeDataset(
        f'{data_dir}/geolife_transformer_7_train.pk',
        max_len=max_len
    )
    
    val_dataset = GeolifeDataset(
        f'{data_dir}/geolife_transformer_7_validation.pk',
        max_len=max_len
    )
    
    test_dataset = GeolifeDataset(
        f'{data_dir}/geolife_transformer_7_test.pk',
        max_len=max_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataset_info = {
        'num_train': len(train_dataset),
        'num_val': len(val_dataset),
        'num_test': len(test_dataset),
        'num_locations': 1187,  # From data analysis
        'num_users': 46  # From data analysis
    }
    
    return train_loader, val_loader, test_loader, dataset_info
