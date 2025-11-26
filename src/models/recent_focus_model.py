"""
Recent-Location Focused Model

KEY INSIGHT: 88% of returns are to last 5 locations!
- 32.4% → last location
- 33.1% → 2nd-to-last  
- Oracle baseline: Predict any of last 3 = 65.53%!

Strategy: Learn to intelligently select among recent locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecentLocationSelector(nn.Module):
    """
    Learns to select which recent location (1-5 back) user will return to.
    This is the KEY to high accuracy!
    """
    def __init__(self, d_model, max_recent=5):
        super().__init__()
        self.max_recent = max_recent
        
        # Features for selection: location emb + time diff + position
        self.scorer = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2),  # location + time_diff + position
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, loc_embeddings, time_diffs, positions):
        """
        Args:
            loc_embeddings: (batch, max_recent, d_model) - embeddings of recent locations
            time_diffs: (batch, max_recent, 1) - time since each location
            positions: (batch, max_recent, 1) - relative position (0=last, 1=2nd-last, etc)
        Returns:
            scores: (batch, max_recent) - score for each recent location
        """
        features = torch.cat([loc_embeddings, time_diffs, positions], dim=-1)
        scores = self.scorer(features).squeeze(-1)  # (batch, max_recent)
        return scores


class RecentFocusedModel(nn.Module):
    """
    Model that focuses on predicting returns to recent locations.
    
    Two-stage approach:
    1. Score recent locations (last 5)
    2. Predict new location if not in recent set
    
    Then combine with learned weights.
    """
    def __init__(
        self,
        num_locations=1187,
        num_users=46,
        d_model=96,
        dropout=0.15,
        max_recent=5
    ):
        super().__init__()
        
        self.num_locations = num_locations
        self.max_recent = max_recent
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 4, padding_idx=0)
        self.hour_emb = nn.Embedding(25, d_model // 4, padding_idx=0)
        self.weekday_emb = nn.Embedding(8, d_model // 4, padding_idx=0)
        
        # Context encoder (lightweight)
        temporal_size = (d_model // 4) * 3  # user + hour + weekday
        self.context_lstm = nn.LSTM(
            d_model + temporal_size,
            d_model,
            num_layers=1,
            batch_first=True
        )
        
        # Recent location selector
        self.recent_selector = RecentLocationSelector(d_model, max_recent)
        
        # New location predictor (for non-returns)
        self.new_loc_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_locations)
        )
        
        # Mixing weight: recent vs new
        self.mix_predictor = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    
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
        
        # Get embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        hour_emb = self.hour_emb(hours)
        weekday_emb = self.weekday_emb(weekdays)
        
        # Combine
        temporal = torch.cat([user_emb, hour_emb, weekday_emb], dim=-1)
        combined = torch.cat([loc_emb, temporal], dim=-1)
        
        # Encode sequence
        output, (hidden, cell) = self.context_lstm(combined)
        
        # Get last state
        seq_lens = mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(batch_size, device=device)
        context = output[batch_idx, seq_lens]
        
        # === PART 1: Score recent locations ===
        recent_logits = torch.full((batch_size, self.num_locations), -1e9, device=device)
        
        for b in range(batch_size):
            seq_len_b = int(seq_lens[b]) + 1
            recent_count = min(self.max_recent, seq_len_b)
            
            if recent_count > 0:
                # Get recent locations (last N)
                recent_locs = locations[b, seq_len_b - recent_count:seq_len_b]
                recent_times = start_mins[b, seq_len_b - recent_count:seq_len_b]
                
                # Get their embeddings
                recent_embs = self.loc_emb(recent_locs)  # (recent_count, d_model)
                
                # Time differences (current time - each past time)
                current_time = start_mins[b, seq_len_b - 1]
                time_diffs = (current_time - recent_times).float().unsqueeze(-1)  # (recent_count, 1)
                time_diffs = time_diffs / 1440.0  # Normalize to days
                
                # Positions (0 = most recent)
                positions = torch.arange(recent_count - 1, -1, -1, device=device).float().unsqueeze(-1)  # (recent_count, 1)
                
                # Expand to batch dimension
                recent_embs_batch = recent_embs.unsqueeze(0)  # (1, recent_count, d_model)
                time_diffs_batch = time_diffs.unsqueeze(0)  # (1, recent_count, 1)
                positions_batch = positions.unsqueeze(0)  # (1, recent_count, 1)
                
                # Score recent locations
                scores = self.recent_selector(recent_embs_batch, time_diffs_batch, positions_batch)  # (1, recent_count)
                
                # Assign scores to location logits
                recent_logits[b, recent_locs] = scores[0]
        
        # === PART 2: Predict new locations ===
        new_logits = self.new_loc_predictor(context)
        
        # === PART 3: Mix recent and new ===
        # Predict mixing weight (higher = more weight on recent)
        mix_weight = self.mix_predictor(context)  # (batch, 1)
        
        # Boost recent predictions significantly (they're 88% of test!)
        mix_weight = mix_weight * 0.9 + 0.05  # Shift to 0.05-0.95 range
        
        # Combine
        final_logits = mix_weight * recent_logits + (1 - mix_weight) * new_logits
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
