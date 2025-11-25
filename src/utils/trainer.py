"""
Training utilities and trainer class.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .metrics import calculate_correct_total_prediction, get_performance_dict


class Trainer:
    """Trainer for next-location prediction."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        num_locations,
        learning_rate=1e-3,
        weight_decay=1e-4,
        label_smoothing=0.1,
        max_epochs=200,
        patience=30,
        checkpoint_dir='checkpoints',
        log_interval=50,
        logger=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_locations = num_locations
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = logger
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_results = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                results, _, _ = calculate_correct_total_prediction(logits, batch['target'])
                all_results.append(results)
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Aggregate results
        all_results = np.sum(all_results, axis=0)
        return_dict = {
            'correct@1': all_results[0],
            'correct@3': all_results[1],
            'correct@5': all_results[2],
            'correct@10': all_results[3],
            'rr': all_results[4],
            'ndcg': all_results[5],
            'f1': 0.0,
            'total': all_results[6]
        }
        
        metrics = get_performance_dict(return_dict)
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate(self, data_loader, split_name='Val'):
        """Evaluate on validation or test set."""
        self.model.eval()
        total_loss = 0
        all_results = []
        
        pbar = tqdm(data_loader, desc=f'{split_name}')
        for batch in pbar:
            # Move to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # Forward pass
            logits = self.model(batch)
            loss = self.criterion(logits, batch['target'])
            
            total_loss += loss.item()
            
            # Calculate metrics
            results, _, _ = calculate_correct_total_prediction(logits, batch['target'])
            all_results.append(results)
        
        # Aggregate results
        all_results = np.sum(all_results, axis=0)
        return_dict = {
            'correct@1': all_results[0],
            'correct@3': all_results[1],
            'correct@5': all_results[2],
            'correct@10': all_results[3],
            'rr': all_results[4],
            'ndcg': all_results[5],
            'f1': 0.0,
            'total': all_results[6]
        }
        
        metrics = get_performance_dict(return_dict)
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, val_acc, filename='best_model.pt'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, filename='best_model.pt'):
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.evaluate(self.val_loader, 'Validation')
            
            # Update scheduler
            self.scheduler.step()
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc@1: {train_metrics['acc@1']:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc@1: {val_metrics['acc@1']:.2f}%")
            print(f"Val Acc@5: {val_metrics['acc@5']:.2f}% | Val MRR: {val_metrics['mrr']:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check for improvement
            if val_metrics['acc@1'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc@1']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics['acc@1'])
                print(f"✓ New best model! Val Acc@1: {self.best_val_acc:.2f}%")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epochs (best: {self.best_val_acc:.2f}%)")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Early diagnostic: if val acc is very low after 10 epochs, stop
            if epoch >= 9 and val_metrics['acc@1'] < 15.0:
                print(f"\n⚠ EARLY DIAGNOSTIC: Val Acc@1 ({val_metrics['acc@1']:.2f}%) too low after 10 epochs")
                print("This approach is unlikely to reach 40%. Stopping to try different architecture.")
                break
        
        # Load best model and evaluate on test set
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)
        
        self.load_checkpoint()
        test_loss, test_metrics = self.evaluate(self.test_loader, 'Test')
        
        print(f"\nTest Results:")
        print(f"  Acc@1: {test_metrics['acc@1']:.2f}%")
        print(f"  Acc@5: {test_metrics['acc@5']:.2f}%")
        print(f"  Acc@10: {test_metrics['acc@10']:.2f}%")
        print(f"  MRR: {test_metrics['mrr']:.2f}%")
        print(f"  NDCG: {test_metrics['ndcg']:.2f}%")
        print(f"\nBest validation epoch: {self.best_epoch + 1}")
        print(f"Best validation Acc@1: {self.best_val_acc:.2f}%")
        
        return test_metrics
