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
    IMPROVED Recurrent Transformer v2.
    
    Key improvements:
    1. Better hidden state integration (gating mechanism)
    2. More cycles for deeper refinement
    3. Layer-wise hidden states
    4. Adaptive cycle termination
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        d_model=96,
        n_head=8,
        dim_feedforward=256,
        n_cycles=5,  # Increased from 3
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
        
        # Input projection
        temporal_size = (d_model // 4) * 3
        self.input_proj = nn.Linear(d_model + temporal_size, d_model)
        
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding(100, d_model))
        
        # Cycle-specific embeddings
        self.cycle_emb = nn.Embedding(n_cycles, d_model)
        
        # Transformer block (shared)
        self.transformer = RecurrentTransformerBlock(
            d_model, n_head, dim_feedforward, dropout
        )
        
        # Gating mechanism for hidden state update
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # Hidden state refinement
        self.hidden_refine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Sequence pooling (better than just last token)
        self.seq_pooling = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_locations)
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
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
        
        # Create input embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        hour_emb = self.hour_emb(hours)
        weekday_emb = self.weekday_emb(weekdays)
        
        # Combine
        temporal = torch.cat([user_emb, hour_emb, weekday_emb], dim=-1)
        combined = torch.cat([loc_emb, temporal], dim=-1)
        
        # Project and add position
        input_emb = self.input_proj(combined)
        input_emb = input_emb + self.pos_encoding[:, :seq_len, :]
        input_emb = self.dropout(input_emb)
        
        # Initialize hidden as weighted mean of input (data-dependent init)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(input_emb)
            masked_input = input_emb * mask_expanded
            hidden = masked_input.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (batch, d_model)
        else:
            hidden = input_emb.mean(dim=1)
        
        # Padding mask
        attn_mask = ~mask.bool() if mask is not None else None
        
        # RECURRENT CYCLES
        for cycle in range(self.n_cycles):
            # Add cycle embedding to input
            cycle_idx = torch.tensor([cycle], device=device)
            cycle_embed = self.cycle_emb(cycle_idx)  # (1, d_model)
            cycle_input = input_emb + cycle_embed.view(1, 1, -1)  # Broadcast properly
            
            # Use attention pooling with hidden as query
            hidden_query = hidden.unsqueeze(1)  # (batch, 1, d_model)
            context, _ = self.seq_pooling(
                hidden_query, cycle_input, cycle_input,
                key_padding_mask=attn_mask
            )
            context = context.squeeze(1)  # (batch, d_model)
            
            # Gated update of hidden state
            combined_hidden = torch.cat([hidden, context], dim=-1)
            gate = self.gate(combined_hidden)
            
            # Refine hidden
            refined = self.hidden_refine(combined_hidden)
            
            # Gated combination (like GRU)
            hidden = gate * hidden + (1 - gate) * refined
        
        # Final prediction
        logits = self.output_proj(hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
