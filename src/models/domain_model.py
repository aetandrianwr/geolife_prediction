"""
Domain-Informed Mobility Prediction Model

Key Insights from Data:
1. 83.8% of test predictions return to previous location
2. Strong transition patterns (loc_i → loc_j)
3. Users have identifiable home/work locations
4. Time-of-day routing matters
5. Last location is highly predictive

This model explicitly incorporates domain knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from pathlib import Path


class ReturnProbabilityModule(nn.Module):
    """
    Predicts probability that user will return to a previously visited location.
    This is CRITICAL - 83.8% of predictions are returns!
    """
    def __init__(self, d_model):
        super().__init__()
        # Simpler predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_state, sequence_summary):
        """
        Args:
            current_state: (batch, d_model) - last position representation
            sequence_summary: (batch, d_model) - summary of whole sequence
        Returns:
            return_prob: (batch, 1) - probability of returning to previous location
        """
        combined = torch.cat([current_state, sequence_summary], dim=-1)
        return_prob = self.predictor(combined)
        return return_prob


class TransitionModule(nn.Module):
    """
    Models P(next_loc | current_loc, time, user).
    Learned transition patterns are very strong in the data.
    """
    def __init__(self, num_locations, num_users, d_model):
        super().__init__()
        
        # Share embeddings with sequence encoder (save parameters)
        self.loc_emb = nn.Embedding(num_locations, d_model // 2, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.time_emb = nn.Embedding(8, d_model // 4, padding_idx=0)  # 8 time slots (not hours)
        
        # Simpler transition predictor
        self.transition_net = nn.Linear(d_model, num_locations)
    
    def forward(self, last_location, last_time, user):
        """
        Predict next location based on current location, time, and user.
        
        Args:
            last_location: (batch,) - current location ID
            last_time: (batch,) - time slot (0-7)
            user: (batch,) - user ID
        Returns:
            transition_logits: (batch, num_locations)
        """
        # Convert hour to time slot (0-7)
        time_slot = torch.div(last_time, 3, rounding_mode='floor').clamp(0, 7)
        
        loc_emb = self.loc_emb(last_location)
        time_emb = self.time_emb(time_slot)
        user_emb = self.user_emb(user)
        
        combined = torch.cat([loc_emb, time_emb, user_emb], dim=-1)
        logits = self.transition_net(combined)
        
        return logits


class UserProfileModule(nn.Module):
    """
    Models user's personal preferences (home, work, frequent locations).
    """
    def __init__(self, num_users, num_locations, d_model):
        super().__init__()
        
        # Simpler user preference
        self.user_profile_emb = nn.Embedding(num_users, d_model // 2, padding_idx=0)
        self.profile_net = nn.Linear(d_model // 2, num_locations)
    
    def forward(self, user):
        """
        Predict locations based on user's long-term preferences.
        
        Args:
            user: (batch,) - user ID
        Returns:
            profile_logits: (batch, num_locations)
        """
        profile_emb = self.user_profile_emb(user)
        logits = self.profile_net(profile_emb)
        return logits


class SequenceEncoderModule(nn.Module):
    """
    Encodes recent trajectory for context.
    Simpler than full transformer - just captures recent patterns.
    """
    def __init__(self, num_locations, num_users, d_model, n_layers=1, n_head=4):
        super().__init__()
        
        self.d_model = d_model
        
        # Shared embeddings to save parameters
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.hour_emb = nn.Embedding(25, d_model // 4, padding_idx=0)
        self.weekday_emb = nn.Embedding(8, d_model // 4, padding_idx=0)
        
        # Position encoding (not learned)
        pos = self._get_sinusoidal_encoding(50, d_model)
        self.register_buffer('pos_encoding', pos)
        
        # Input projection
        temporal_size = (d_model // 4) * 3  # user + hour + weekday
        self.input_proj = nn.Linear(d_model + temporal_size, d_model)
        
        # Simple LSTM encoder (lighter than transformer) - only 1 layer
        self.encoder = nn.LSTM(
            d_model,
            d_model,
            n_layers,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def _get_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, locations, users, hours, weekdays, mask):
        """
        Encode sequence.
        
        Returns:
            last_state: (batch, d_model) - final state
            sequence_summary: (batch, d_model) - summary of sequence
        """
        batch_size, seq_len = locations.shape
        
        # Embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        hour_emb = self.hour_emb(hours)
        weekday_emb = self.weekday_emb(weekdays)
        
        # Combine
        temporal = torch.cat([user_emb, hour_emb, weekday_emb], dim=-1)
        combined = torch.cat([loc_emb, temporal], dim=-1)
        x = self.input_proj(combined)
        
        # Add position encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Encode
        output, (hidden, cell) = self.encoder(x)
        
        # Get last valid position
        seq_lens = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(batch_size, device=x.device)
        last_state = output[batch_idx, seq_lens]
        
        # Sequence summary (mean pooling)
        mask_expanded = mask.unsqueeze(-1).expand_as(output)
        masked_output = output * mask_expanded
        sequence_summary = masked_output.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
        last_state = self.layer_norm(last_state)
        sequence_summary = self.layer_norm(sequence_summary)
        
        return last_state, sequence_summary


class DomainInformedModel(nn.Module):
    """
    Mobility prediction model informed by domain knowledge.
    
    Components:
    1. Sequence encoder - understands recent trajectory
    2. Return probability - predicts if returning to previous location
    3. Transition module - models loc_i → loc_j transitions
    4. User profile - long-term user preferences
    5. Ensemble - combines all signals
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        d_model=96,
        n_layers=2,
        n_head=4,
        dropout=0.15
    ):
        super().__init__()
        
        self.num_locations = num_locations
        
        # Core modules
        self.sequence_encoder = SequenceEncoderModule(
            num_locations, num_users, d_model, n_layers, n_head
        )
        
        self.return_module = ReturnProbabilityModule(d_model)
        self.transition_module = TransitionModule(num_locations, num_users, d_model)
        self.user_profile_module = UserProfileModule(num_users, num_locations, d_model)
        
        # Attention over previous locations (for return prediction)
        self.return_attention = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Location embedding for return candidates
        self.return_loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # Final ensemble weights (learnable)
        self.ensemble_weights = nn.Parameter(torch.ones(4) / 4)  # 4 components
        
        # Output projection from sequence
        self.seq_output = nn.Linear(d_model, num_locations)
    
    def forward(self, batch):
        locations = batch['locations']
        users = batch['users']
        weekdays = batch['weekdays']
        start_mins = batch['start_mins']
        mask = batch['mask']
        
        batch_size = locations.shape[0]
        device = locations.device
        
        # Extract hours
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 24)
        
        # Get last location and time for each sequence
        seq_lens = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(batch_size, device=device)
        last_location = locations[batch_idx, seq_lens]
        last_hour = hours[batch_idx, seq_lens]
        user_id = users[:, 0]  # User is same across sequence
        
        # 1. Encode sequence
        last_state, sequence_summary = self.sequence_encoder(
            locations, users, hours, weekdays, mask
        )
        
        # 2. Predict return probability
        return_prob = self.return_module(last_state, sequence_summary)
        
        # 3. Get transition-based predictions
        transition_logits = self.transition_module(last_location, last_hour, user_id)
        
        # 4. Get user profile predictions
        profile_logits = self.user_profile_module(user_id)
        
        # 5. Get sequence-based predictions
        sequence_logits = self.seq_output(last_state)
        
        # 6. Return-to-previous prediction
        # Create attention over previous locations
        prev_loc_emb = self.return_loc_emb(locations)  # (batch, seq, d_model)
        query = last_state.unsqueeze(1)  # (batch, 1, d_model)
        
        attn_output, attn_weights = self.return_attention(
            query, prev_loc_emb, prev_loc_emb, key_padding_mask=~mask.bool()
        )
        
        # Convert attention weights to location logits
        return_logits = torch.zeros(batch_size, self.num_locations, device=device)
        for b in range(batch_size):
            valid_locs = locations[b, mask[b] > 0]
            valid_weights = attn_weights[b, 0, mask[b] > 0]
            return_logits[b].scatter_add_(0, valid_locs, valid_weights)
        
        # Scale return logits by return probability
        return_logits = return_logits * return_prob * 5.0  # Boost return signal
        
        # 7. Ensemble all predictions
        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        final_logits = (
            weights[0] * sequence_logits +
            weights[1] * transition_logits +
            weights[2] * profile_logits +
            weights[3] * return_logits
        )
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
