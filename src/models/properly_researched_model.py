"""
Properly Researched Advanced Architecture for Next-Location Prediction

Based on deep research and understanding of:
1. Data characteristics (short sequences, class imbalance)
2. Proven architectural improvements (Pre-LN, GeGLU, relative positions)
3. Task-specific design (temporal patterns, user behavior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionBias(nn.Module):
    """
    Relative position bias for attention (T5-style).
    Learns "how far apart" matters more than absolute position.
    """
    def __init__(self, num_heads, max_distance=32, bidirectional=False):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        
        # Learnable relative position embeddings
        num_buckets = 2 * max_distance + 1 if bidirectional else max_distance + 1
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    
    def _relative_position_bucket(self, relative_position):
        """Convert relative position to bucket index."""
        num_buckets = 2 * self.max_distance + 1 if self.bidirectional else self.max_distance + 1
        ret = 0
        n = -relative_position
        
        if self.bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        
        # Clamp to max_distance
        ret += torch.clamp(n, 0, num_buckets - 1)
        return ret
    
    def forward(self, seq_len):
        """
        Compute relative position bias for attention.
        Returns: (1, num_heads, seq_len, seq_len)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        context_position = positions[:, None]
        memory_position = positions[None, :]
        
        # Relative positions
        relative_position = memory_position - context_position
        
        # Get bucket indices
        rp_bucket = self._relative_position_bucket(relative_position)
        
        # Get biases
        values = self.relative_attention_bias(rp_bucket)  # (seq_len, seq_len, num_heads)
        values = values.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, seq_len, seq_len)
        
        return values


class GeGLU(nn.Module):
    """
    Gated GELU activation for better FFN.
    From "GLU Variants Improve Transformer" (Shazeer, 2020)
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class ImprovedFFN(nn.Module):
    """
    Feed-Forward Network with GeGLU activation.
    More effective than standard FFN.
    """
    def __init__(self, d_model, d_inner, dropout=0.1):
        super().__init__()
        # GeGLU needs 2x hidden dim for gating, but we reduce d_inner to compensate
        # Use d_inner directly (not *2) and split internally
        self.w_1 = nn.Linear(d_model, d_inner)
        self.geglu = GeGLU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_inner // 2, d_model)  # GeGLU halves the dimension
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.w_1(x)
        x = self.geglu(x)  # This halves the dimension
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class ImprovedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative position bias.
    Uses Pre-LN architecture for stability.
    """
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1, use_relative_pos=True):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Relative position bias
        self.use_relative_pos = use_relative_pos
        if use_relative_pos:
            self.relative_position_bias = RelativePositionBias(n_head, max_distance=32)
        
        self.scale = d_k ** -0.5
    
    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()
        
        # Linear projections and split heads
        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_k, self.n_head, self.d_v).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        if self.use_relative_pos:
            rel_pos_bias = self.relative_position_bias(len_q)
            attn = attn + rel_pos_bias
        
        # Apply mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Compute output
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, self.n_head * self.d_v)
        output = self.fc(output)
        
        return output, attn


class ImprovedEncoderLayer(nn.Module):
    """
    Encoder layer with Pre-Layer Normalization.
    CRITICAL: Pre-LN is more stable than Post-LN!
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        
        # Pre-LN: LayerNorm before sublayers
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.slf_attn = ImprovedMultiHeadAttention(
            d_model, n_head, d_k, d_v, dropout=dropout, use_relative_pos=True
        )
        
        self.ffn = ImprovedFFN(d_model, d_inner, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-LN architecture
        # Attention block
        residual = x
        x = self.norm1(x)
        x, attn = self.slf_attn(x, x, x, mask=mask)
        x = self.dropout(x)
        x = residual + x
        
        # FFN block
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, attn


class MultiScaleTemporalEncoding(nn.Module):
    """
    Proper multi-scale temporal encoding.
    Combines discrete embeddings + continuous cyclical features.
    """
    def __init__(self, d_model):
        super().__init__()
        d_time = d_model // 3
        
        # Discrete embeddings
        self.hour_emb = nn.Embedding(25, d_time, padding_idx=0)  # 0-23 + padding
        self.weekday_emb = nn.Embedding(8, d_time, padding_idx=0)  # 0-6 + padding
        
        # Continuous encoders
        self.continuous_proj = nn.Linear(4, d_time)  # 2 for hour + 2 for day
        
        # Final projection
        self.output_proj = nn.Linear(d_time * 3, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, hours, weekdays, start_mins):
        batch_size, seq_len = hours.shape
        
        # Discrete embeddings
        hour_emb = self.hour_emb(hours)
        weekday_emb = self.weekday_emb(weekdays)
        
        # Continuous cyclical features
        # Hour of day (0-23) -> angle
        hour_angle = (start_mins.float() / 1440.0) * 2 * math.pi  # Normalize to 24 hours
        hour_sin = torch.sin(hour_angle).unsqueeze(-1)
        hour_cos = torch.cos(hour_angle).unsqueeze(-1)
        
        # Day of week (0-6) -> angle
        day_angle = (weekdays.float() / 7.0) * 2 * math.pi
        day_sin = torch.sin(day_angle).unsqueeze(-1)
        day_cos = torch.cos(day_angle).unsqueeze(-1)
        
        # Combine continuous features
        continuous_features = torch.cat([hour_sin, hour_cos, day_sin, day_cos], dim=-1)
        continuous_emb = self.continuous_proj(continuous_features)
        
        # Combine all temporal features
        combined = torch.cat([hour_emb, weekday_emb, continuous_emb], dim=-1)
        output = self.output_proj(combined)
        output = self.layer_norm(output)
        
        return output


class ProperlyResearchedModel(nn.Module):
    """
    Next-location prediction model with properly researched improvements.
    
    Key improvements over baseline:
    1. Pre-Layer Normalization (critical for stability)
    2. GeGLU activation in FFN (better than GELU)
    3. Relative position bias (better than absolute positions)
    4. Multi-scale temporal encoding (discrete + continuous)
    5. Proper initialization
    """
    def __init__(
        self,
        num_locations,
        num_users,
        d_model=88,
        d_inner=176,
        n_layers=4,
        n_head=8,
        d_k=11,
        d_v=11,
        dropout=0.15,
        max_len=50
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        
        # Multi-scale temporal encoding
        self.temporal_encoder = MultiScaleTemporalEncoding(d_model)
        
        # Sinusoidal positional encoding (as fallback/additional signal)
        self.register_buffer('pos_encoding', self._get_sinusoidal_encoding(max_len, d_model))
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_model + d_model + d_model // 4, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Improved encoder layers with Pre-LN
        self.layers = nn.ModuleList([
            ImprovedEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm (Pre-LN architecture)
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Proper initialization
        self._init_weights()
    
    def _get_sinusoidal_encoding(self, max_len, d_model):
        """Sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def _init_weights(self):
        """Proper weight initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Special initialization for embeddings
        nn.init.normal_(self.loc_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        mask = batch['mask']
        
        batch_size, seq_len = locations.shape
        
        # Extract hours from start_mins
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 24)
        
        # Get embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        temporal_emb = self.temporal_encoder(hours, weekdays, start_mins)
        
        # Combine features
        combined = torch.cat([loc_emb, temporal_emb, user_emb], dim=-1)
        x = self.input_proj(combined)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Causal mask for autoregressive prediction
        causal_mask = self._get_causal_mask(seq_len, x.device)
        
        # Apply encoder layers
        for layer in self.layers:
            x, _ = layer(x, mask=causal_mask)
        
        # Final layer norm (Pre-LN architecture)
        x = self.final_norm(x)
        
        # Get last position representation
        seq_lens = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(batch_size, device=x.device)
        last_hidden = x[batch_idx, seq_lens]
        
        # Output projection
        logits = self.output_fc(last_hidden)
        
        return logits
    
    def _get_causal_mask(self, seq_len, device):
        """Generate causal mask."""
        mask = torch.triu(torch.ones(1, seq_len, seq_len, device=device), diagonal=1)
        return (mask == 0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
