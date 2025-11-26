"""
Recurrent Transformer for Next Location Prediction

Key Idea: Replace RNN/LSTM recurrence with Transformer recurrence.
- At each cycle, process entire input sequence with Transformer
- Combine with hidden state from previous cycle
- Output updated hidden state for next cycle
- Configurable number of cycles for iterative refinement

This allows the model to:
1. Refine understanding of sequence through multiple passes
2. Maintain global context (unlike RNN sequential processing)
3. Adaptively focus on different parts at each cycle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentTransformerBlock(nn.Module):
    """
    A single Transformer block that processes input + hidden state.
    """
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual
        ff_out = self.feedforward(x)
        x = self.norm2(x + ff_out)
        
        return x


class RecurrentTransformer(nn.Module):
    """
    Recurrent Transformer: Processes input sequence multiple times (cycles),
    refining hidden state at each iteration.
    
    Architecture:
    1. Embed input sequence (locations + temporal features)
    2. For each cycle:
       - Combine current input embeddings with hidden state
       - Process through Transformer block
       - Extract updated hidden state
    3. Final hidden state used for prediction
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        d_model=128,
        n_head=8,
        dim_feedforward=256,
        n_cycles=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_cycles = n_cycles
        
        # Input embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.hour_emb = nn.Embedding(25, d_model // 4, padding_idx=0)
        self.weekday_emb = nn.Embedding(8, d_model // 4, padding_idx=0)
        
        # Input projection (loc + temporal features → d_model)
        temporal_size = (d_model // 4) * 3
        self.input_proj = nn.Linear(d_model + temporal_size, d_model)
        
        # Positional encoding (standard sinusoidal)
        self.register_buffer('pos_encoding', self._create_positional_encoding(100, d_model))
        
        # Cycle-specific embeddings (different for each recurrent cycle)
        self.cycle_emb = nn.Embedding(n_cycles, d_model)
        
        # Initial hidden state (learnable)
        self.init_hidden = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer blocks (shared across cycles)
        self.transformer_block = RecurrentTransformerBlock(
            d_model, n_head, dim_feedforward, dropout
        )
        
        # Hidden state update (combines old hidden + transformer output)
        self.hidden_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_locations)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        mask = batch['mask']
        
        batch_size, seq_len = locations.shape
        device = locations.device
        
        # Extract hours
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 24)
        
        # Create input embeddings (same for all cycles)
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        hour_emb = self.hour_emb(hours)
        weekday_emb = self.weekday_emb(weekdays)
        
        # Combine temporal features
        temporal = torch.cat([user_emb, hour_emb, weekday_emb], dim=-1)
        combined = torch.cat([loc_emb, temporal], dim=-1)
        
        # Project to d_model
        input_emb = self.input_proj(combined)  # (batch, seq, d_model)
        
        # Add positional encoding
        input_emb = input_emb + self.pos_encoding[:, :seq_len, :]
        input_emb = self.dropout(input_emb)
        
        # Initialize hidden state (broadcast to batch size)
        hidden = self.init_hidden.expand(batch_size, 1, -1)  # (batch, 1, d_model)
        
        # Padding mask for attention (True = padding)
        attn_mask = ~mask.bool() if mask is not None else None
        
        # RECURRENT CYCLES
        for cycle in range(self.n_cycles):
            # Add cycle-specific embedding to input
            cycle_idx = torch.tensor([cycle], device=device)
            cycle_embed = self.cycle_emb(cycle_idx)  # (1, d_model)
            cycle_embed = cycle_embed.unsqueeze(0).expand(batch_size, seq_len, -1)
            
            # Combine input with cycle embedding
            cycle_input = input_emb + cycle_embed
            
            # Concatenate hidden state with input sequence
            # Hidden state acts as "memory" token
            combined_seq = torch.cat([hidden, cycle_input], dim=1)  # (batch, 1+seq, d_model)
            
            # Extend mask for hidden state (hidden is never masked)
            if attn_mask is not None:
                hidden_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
                extended_mask = torch.cat([hidden_mask, attn_mask], dim=1)
            else:
                extended_mask = None
            
            # Process through Transformer
            transformed = self.transformer_block(combined_seq, extended_mask)
            
            # Extract new hidden state (first token) and sequence output
            new_hidden_candidate = transformed[:, 0:1, :]  # (batch, 1, d_model)
            
            # Update hidden state (combine old and new)
            hidden_concat = torch.cat([hidden, new_hidden_candidate], dim=-1)
            hidden = self.hidden_update(hidden_concat)  # (batch, 1, d_model)
        
        # Final prediction from refined hidden state
        hidden_final = hidden.squeeze(1)  # (batch, d_model)
        logits = self.output_proj(hidden_final)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
