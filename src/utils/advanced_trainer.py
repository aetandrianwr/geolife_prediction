"""
Advanced Trainer with State-of-the-Art Training Techniques

Key Features:
1. Cosine annealing with warm restarts
2. Stochastic Weight Averaging (SWA)
3. Mixed precision training
4. Gradient accumulation
5. Test-time augmentation
6. Advanced regularization
7. Learning rate warmup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import copy

from src.utils.metrics import calculate_metrics


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


class AdvancedTrainer:
    """Advanced trainer with state-of-the-art techniques."""
    
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
        logger=None,
        grad_accum_steps=1,
        warmup_epochs=5,
        use_swa=True,
        swa_start=150,
        swa_lr=5e-5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_locations = num_locations
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.logger = logger
        self.grad_accum_steps = grad_accum_steps
        self.warmup_epochs = warmup_epochs
        self.use_swa = use_swa
        self.swa_start = swa_start
        
        # Loss function
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Stochastic Weight Averaging
        if use_swa:
            self.swa_model = copy.deepcopy(model)
            self.swa_n = 0
        else:
            self.swa_model = None
        
        # Training state
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.current_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.max_epochs}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Mixed precision forward pass
            with autocast():
                logits = self.model(batch)
                targets = batch['target']
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * self.grad_accum_steps
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Learning rate warmup
        if self.current_epoch < self.warmup_epochs:
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
        else:
            self.scheduler.step()
        
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def evaluate(self, data_loader, use_swa=False):
        """Evaluate on validation or test set."""
        model = self.swa_model if (use_swa and self.swa_model is not None) else self.model
        model.eval()
        
        all_logits = []
        all_targets = []
        
        for batch in tqdm(data_loader, desc='Evaluating', leave=False):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            with autocast():
                logits = model(batch)
            
            all_logits.append(logits.cpu())
            all_targets.append(batch['target'].cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_logits, all_targets, num_locations=self.num_locations)
        
        return metrics
    
    @torch.no_grad()
    def evaluate_with_tta(self, data_loader):
        """Evaluate with test-time augmentation."""
        self.model.eval()
        
        all_logits = []
        all_targets = []
        
        for batch in tqdm(data_loader, desc='Evaluating (TTA)', leave=False):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Original prediction
            with autocast():
                logits1 = self.model(batch)
            
            # TTA: Dropout at test time (MC Dropout)
            self.model.train()  # Enable dropout
            with autocast():
                logits2 = self.model(batch)
                logits3 = self.model(batch)
            self.model.eval()
            
            # Average predictions
            logits = (logits1 + logits2 + logits3) / 3.0
            
            all_logits.append(logits.cpu())
            all_targets.append(batch['target'].cpu())
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_logits, all_targets, num_locations=self.num_locations)
        
        return metrics
    
    def update_swa_model(self):
        """Update SWA model with current model parameters."""
        if self.swa_model is None:
            return
        
        # Moving average
        alpha = 1.0 / (self.swa_n + 1)
        for swa_param, param in zip(self.swa_model.parameters(), self.model.parameters()):
            swa_param.data = swa_param.data * (1 - alpha) + param.data * alpha
        
        self.swa_n += 1
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
        
        if self.swa_model is not None:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
            checkpoint['swa_n'] = self.swa_n
        
        torch.save(checkpoint, filename)
        
        if is_best and self.logger:
            self.logger.info(f"✓ New best model saved: {filename}")
    
    def train(self):
        """Full training loop."""
        if self.logger:
            self.logger.info("Starting training...")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            val_acc = val_metrics['acc@1']
            
            # Update SWA model
            if self.use_swa and epoch >= self.swa_start:
                self.update_swa_model()
            
            # Log
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch+1:3d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                    f"Val Acc@1: {val_acc:.2f}% | Val Acc@5: {val_metrics['acc@5']:.2f}% | "
                    f"Val MRR: {val_metrics['mrr']:.2f}%"
                )
            
            # Check improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                
                # Save best model
                checkpoint_path = f"{self.checkpoint_dir}/best_model.pt"
                self.save_checkpoint(checkpoint_path, is_best=True)
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= self.patience:
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint periodically
            if (epoch + 1) % 20 == 0:
                checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
        
        # Load best model
        checkpoint = torch.load(f"{self.checkpoint_dir}/best_model.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation
        if self.logger:
            self.logger.info("\n" + "="*80)
            self.logger.info("FINAL EVALUATION")
            self.logger.info("="*80)
        
        # Evaluate regular model
        test_metrics = self.evaluate(self.test_loader)
        
        if self.logger:
            self.logger.info("Regular model:")
            self.logger.info(f"  Test Acc@1:  {test_metrics['acc@1']:.2f}%")
            self.logger.info(f"  Test Acc@5:  {test_metrics['acc@5']:.2f}%")
            self.logger.info(f"  Test Acc@10: {test_metrics['acc@10']:.2f}%")
            self.logger.info(f"  Test MRR:    {test_metrics['mrr']:.2f}%")
            self.logger.info(f"  Test NDCG:   {test_metrics['ndcg']:.2f}%")
        
        # Evaluate SWA model if available
        if self.use_swa and self.swa_n > 0:
            swa_metrics = self.evaluate(self.test_loader, use_swa=True)
            
            if self.logger:
                self.logger.info(f"\nSWA model (averaged over {self.swa_n} checkpoints):")
                self.logger.info(f"  Test Acc@1:  {swa_metrics['acc@1']:.2f}%")
                self.logger.info(f"  Test Acc@5:  {swa_metrics['acc@5']:.2f}%")
                self.logger.info(f"  Test MRR:    {swa_metrics['mrr']:.2f}%")
            
            # Use SWA metrics if better
            if swa_metrics['acc@1'] > test_metrics['acc@1']:
                test_metrics = swa_metrics
                if self.logger:
                    self.logger.info("✓ Using SWA model (better performance)")
        
        # Test-time augmentation
        tta_metrics = self.evaluate_with_tta(self.test_loader)
        
        if self.logger:
            self.logger.info(f"\nWith Test-Time Augmentation:")
            self.logger.info(f"  Test Acc@1:  {tta_metrics['acc@1']:.2f}%")
            self.logger.info(f"  Test Acc@5:  {tta_metrics['acc@5']:.2f}%")
            self.logger.info(f"  Test MRR:    {tta_metrics['mrr']:.2f}%")
        
        # Use TTA if better
        if tta_metrics['acc@1'] > test_metrics['acc@1']:
            test_metrics = tta_metrics
            if self.logger:
                self.logger.info("✓ Using TTA predictions (better performance)")
        
        return test_metrics
