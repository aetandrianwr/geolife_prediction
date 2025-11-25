"""
Attention-Based Model for Next-Location Prediction

Design Philosophy:
- Multi-head self-attention for capturing sequential dependencies
- Rich feature embeddings (location, user, temporal)
- Parameter-efficient architecture (<500K params)
- Stable training with proper normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional causal masking."""
    
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting
        
        q, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        
        q = self.layer_norm(q)
        
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        
        x = self.layer_norm(x)
        
        return x


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


import numpy as np


class LocationPredictionModel(nn.Module):
    """
    Transformer-based next-location prediction model.
    
    Features:
    - Location embeddings
    - User embeddings
    - Temporal embeddings (weekday, hour)
    - Multi-head self-attention
    - Feed-forward layers
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=128,
        d_inner=256,
        n_layers=3,
        n_head=8,
        d_k=16,
        d_v=16,
        dropout=0.1,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Temporal embeddings
        self.weekday_emb = nn.Embedding(8, d_model // 8, padding_idx=0)
        self.hour_emb = nn.Embedding(25, d_model // 8, padding_idx=0)
        
        # Positional encoding
        self.position_enc = PositionalEncoding(d_model, n_position=max_len)
        
        # Feature projection
        feat_dim = d_model + d_model // 4 + 2 * (d_model // 8)
        self.feat_proj = nn.Linear(feat_dim, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Encoder layers
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        
        # Output layers
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
    
    def forward(self, batch):
        """
        Args:
            batch: Dictionary containing:
                - locations: (batch_size, seq_len)
                - users: (batch_size, seq_len)
                - weekdays: (batch_size, seq_len)
                - start_mins: (batch_size, seq_len)
                - mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, num_locations)
        """
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        mask = batch['mask']
        
        # Extract hours from start_mins
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 24)
        
        # Get embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        weekday_emb = self.weekday_emb(weekdays)
        hour_emb = self.hour_emb(hours)
        
        # Concatenate features
        combined = torch.cat([loc_emb, user_emb, weekday_emb, hour_emb], dim=-1)
        
        # Project to model dimension
        enc_output = self.feat_proj(combined)
        enc_output = self.layer_norm(enc_output)
        enc_output = self.dropout(enc_output)
        
        # Add positional encoding
        enc_output = self.position_enc(enc_output)
        
        # Create causal attention mask
        slf_attn_mask = self.get_subsequent_mask(locations)
        
        # Pass through encoder layers
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=slf_attn_mask)
        
        # Get last valid position for each sequence
        seq_lens = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(locations.size(0), device=locations.device)
        last_hidden = enc_output[batch_idx, seq_lens]
        
        # Output projection
        logits = self.output_fc(last_hidden)
        
        return logits
    
    def get_subsequent_mask(self, seq):
        """Create causal mask for autoregressive prediction."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(
            torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
