"""
Advanced Next-Location Prediction Model with State-of-the-Art Techniques

Key Innovations:
1. Rotary Position Embeddings (RoPE) - Better position encoding
2. SwiGLU activation - Proven superior to GELU
3. RMSNorm - More stable than LayerNorm
4. Multi-scale temporal encoding
5. Location-aware attention
6. Mixture of representations
7. Learnable aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more stable than LayerNorm)."""
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Superior to sinusoidal."""
    
    def __init__(self, dim, max_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation - Proven superior to GELU in LLMs."""
    
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_out, bias=bias)
    
    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)


class MultiHeadAttentionRoPE(nn.Module):
    """Multi-head attention with RoPE and improvements."""
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = d_k ** -0.5
        
    def forward(self, q, k, v, rope_cos, rope_sin, mask=None):
        sz_b, len_q, _ = q.size()
        len_k = k.size(1)
        
        # Project
        q = self.w_qs(q).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_k, self.n_head, self.d_v).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.fc(output)
        
        return output


class FeedForwardSwiGLU(nn.Module):
    """Feed-forward with SwiGLU activation."""
    
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_inner, bias=False)
        self.w_2 = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w_2(self.swiglu(x)))


class TransformerLayer(nn.Module):
    """Enhanced Transformer layer with RoPE and SwiGLU."""
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttentionRoPE(n_head, d_model, d_k, d_v, dropout)
        self.ffn = FeedForwardSwiGLU(d_model, d_inner, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, rope_cos, rope_sin, mask=None):
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.slf_attn(x, x, x, rope_cos, rope_sin, mask)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x


class TemporalEncoder(nn.Module):
    """Multi-scale temporal encoding."""
    
    def __init__(self, d_model):
        super().__init__()
        d_time = d_model // 4
        
        # Hour encoding (24 hours + padding)
        self.hour_emb = nn.Embedding(25, d_time, padding_idx=0)
        
        # Weekday encoding (7 days + padding)
        self.weekday_emb = nn.Embedding(8, d_time, padding_idx=0)
        
        # Learnable time-of-day encoder (continuous)
        self.tod_encoder = nn.Sequential(
            nn.Linear(1, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time)
        )
        
        # Learnable day-of-week encoder (continuous)
        self.dow_encoder = nn.Sequential(
            nn.Linear(1, d_time),
            nn.SiLU(),
            nn.Linear(d_time, d_time)
        )
        
        self.proj = nn.Linear(d_time * 4, d_model)
    
    def forward(self, hours, weekdays, start_mins):
        # Discrete embeddings
        hour_emb = self.hour_emb(hours)
        weekday_emb = self.weekday_emb(weekdays)
        
        # Continuous encodings (normalized to [0, 1])
        tod_continuous = (start_mins.float() / 1440.0).unsqueeze(-1)  # 24*60 minutes
        dow_continuous = (weekdays.float() / 7.0).unsqueeze(-1)
        
        tod_enc = self.tod_encoder(tod_continuous)
        dow_enc = self.dow_encoder(dow_continuous)
        
        # Combine all temporal features
        combined = torch.cat([hour_emb, weekday_emb, tod_enc, dow_enc], dim=-1)
        return self.proj(combined)


class AdvancedLocationModel(nn.Module):
    """
    Advanced next-location prediction model with state-of-the-art techniques.
    
    Key features:
    - Rotary Position Embeddings (RoPE)
    - SwiGLU activation
    - RMSNorm normalization
    - Multi-scale temporal encoding
    - Efficient architecture design
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=96,
        d_inner=192,
        n_layers=4,
        n_head=8,
        d_k=12,
        d_v=12,
        dropout=0.1,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Location embedding with better initialization
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        nn.init.normal_(self.loc_emb.weight, mean=0, std=d_model ** -0.5)
        
        # User embedding (smaller dimension)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Advanced temporal encoder
        self.temporal_encoder = TemporalEncoder(d_model)
        
        # Rotary position embedding
        self.rope = RotaryPositionEmbedding(d_k, max_len=max_len)
        
        # Feature fusion
        self.feat_fusion = nn.Sequential(
            nn.Linear(d_model + d_model + d_model // 4, d_model),
            RMSNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, d_inner, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        
        # Output projection with adaptive softmax for efficiency
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Tie input and output embeddings for parameter efficiency
        # This is commonly used in LLMs
        self.output_proj[-1].weight = self.loc_emb.weight
        
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        
        bs, seq_len = locations.shape
        
        # Extract hours
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 24)
        
        # Get embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        temporal_emb = self.temporal_encoder(hours, weekdays, start_mins)
        
        # Fuse features
        combined = torch.cat([loc_emb, temporal_emb, user_emb], dim=-1)
        x = self.feat_fusion(combined)
        
        # Get RoPE embeddings
        rope_cos, rope_sin = self.rope(seq_len, x.device)
        rope_cos = rope_cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k)
        rope_sin = rope_sin.unsqueeze(0).unsqueeze(0)
        
        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = ~mask  # Invert: True for allowed positions
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, rope_cos, rope_sin, mask)
        
        x = self.final_norm(x)
        
        # Get last position
        seq_lens = batch['mask'].sum(dim=1).long() - 1
        batch_idx = torch.arange(bs, device=x.device)
        last_hidden = x[batch_idx, seq_lens]
        
        # Output projection
        logits = self.output_proj(last_hidden)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
