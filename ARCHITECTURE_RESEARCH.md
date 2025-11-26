# Deep Research: Architecture Improvements for Next-Location Prediction

## Data Characteristics (Critical for Architecture Design)

### Key Findings:
1. **Sequence lengths**: Mean 18, Median 14-16 (short sequences!)
2. **Extreme class imbalance**: Location 14 appears 17,949 times vs many locations appear only once
3. **Target imbalance**: Top target (location 7) appears 783 times, some never appear as targets
4. **Sequential patterns**: Strong transition patterns exist (same location pairs repeat)
5. **Temporal patterns**: Clear hour-of-day and day-of-week patterns
6. **User-specific behavior**: Different users have different mobility patterns

## Architecture Research: What Actually Works

### 1. PROPER Relative Position Encoding (NOT RoPE - that was wrong!)

**Research**: For sequential data with strong positional dependencies:
- **Relative position bias** (T5-style) > Absolute positional encoding
- **Learnable relative positions** capture "how far apart" matters more than "absolute position"
- **Key insight**: Location at position i relative to position i-1 matters more than absolute position

**Proper Implementation**:
```python
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=32):
        # Learnable relative position embeddings
        # For each head, learn biases for relative distances
        self.relative_attention_bias = nn.Embedding(2 * max_distance + 1, num_heads)
```

### 2. GLU Variants in FFN (Gated Linear Units)

**Research**: 
- **GeGLU** (Gated GELU) from "GLU Variants Improve Transformer" (Shazeer, 2020)
- Increases effective parameters without increasing model size
- Better gradient flow than standard FFN

**Proper Implementation**:
```python
class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

# In FFN:
hidden = d_model * 4  # Expand
output = d_model
self.wi = nn.Linear(d_model, hidden * 2)  # *2 for gating
self.geglu = GeGLU()
self.wo = nn.Linear(hidden, d_model)
```

### 3. Pre-Layer Normalization (CRITICAL!)

**Research**:
- **Pre-LN** > Post-LN for training stability
- "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
- Smoother gradients, better convergence

**Current baseline uses POST-LN** (wrong!):
```python
# Current (POST-LN - unstable):
out = LayerNorm(x + Attention(x))

# Should be (PRE-LN - stable):
out = x + Attention(LayerNorm(x))
```

### 4. Multi-Scale Temporal Encoding

**Research**:
- Locations depend on MULTIPLE time scales: hour, day, week
- Cyclical encoding (sin/cos) captures periodicity
- Learned embeddings capture discrete patterns

**Proper Implementation**:
```python
# Combine discrete + continuous
hour_discrete = Embedding(24, d)
hour_continuous = sin/cos encoding
day_discrete = Embedding(7, d)
Combined temporal = Concat[hour_discrete, hour_continuous, day_discrete]
```

### 5. Location Embedding Strategy

**Research**: For highly imbalanced classes:
- **Adaptive input embeddings** (Baevski & Auli, 2019)
- Rare locations get more capacity
- Frequent locations share parameters

**Current issue**: All locations get same embedding size (wasteful!)

**Better approach**:
```python
# Group locations by frequency
frequent_locs (>1000 occurrences): smaller embeddings
medium_locs (100-1000): medium embeddings  
rare_locs (<100): larger embeddings + special handling
```

### 6. Cross-Attention Between Features

**Research**: User + Location + Time are different modalities
- **Cross-modal attention** lets them interact
- User patterns inform location predictions
- Time context modulates both

**Implementation**:
```python
# Instead of simple concatenation:
loc_features = SelfAttention(locations)
user_context = CrossAttention(query=loc_features, key=users, value=users)
time_context = CrossAttention(query=loc_features, key=time, value=time)
combined = loc_features + user_context + time_context
```

### 7. Mixture of Experts (Lightweight)

**Research**: Different patterns for different scenarios
- Some locations are habitual (home, work)
- Some are exploratory (tourism)
- **Solution**: Learn which "expert" to use

**Lightweight implementation**:
```python
class LightweightMoE(nn.Module):
    def __init__(self, d_model, num_experts=4):
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        # Route to experts
        router_logits = self.router(x)  # (batch, seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Combine expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(expert_outputs * router_probs.unsqueeze(-2), dim=-1)
        return output
```

### 8. Attention Pattern Improvements

**Research**: Not all positions are equally important
- **Local attention** for recent context
- **Global attention** for key landmarks
- **Strided attention** for long sequences

**For our task** (sequences ~18 long):
- Recent locations (last 3-5) matter most
- First location (starting point) matters
- **Solution**: Attention bias favoring recent + first

### 9. Loss Function Improvements

**Research**: Standard cross-entropy ignores class imbalance
- **Focal Loss** (Lin et al., 2017): Focus on hard examples
- **Class-balanced loss**: Weight by inverse frequency
- **Label smoothing**: Prevent overconfidence

**Proper implementation**:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 10. Output Layer Design

**Research**: Prediction head matters!
- **Mixture of Softmaxes** for multi-modal distributions
- **Adaptive softmax** for large vocab
- **Temperature scaling** for calibration

**Current baseline**: Simple linear layer (too simple!)

**Better**:
```python
# Multi-head prediction
class MultiHeadPredictor(nn.Module):
    def __init__(self, d_model, num_locations, num_heads=4):
        self.heads = nn.ModuleList([
            nn.Linear(d_model, num_locations) for _ in range(num_heads)
        ])
        self.head_weights = nn.Linear(d_model, num_heads)
    
    def forward(self, x):
        # Multiple prediction heads
        head_outputs = [head(x) for head in self.heads]
        head_weights = F.softmax(self.head_weights(x), dim=-1)
        
        # Weighted combination
        output = sum(w.unsqueeze(-1) * out for w, out in zip(head_weights.unbind(-1), head_outputs))
        return output
```

## Key Architectural Decisions

### What to DEFINITELY implement:
1. ✅ Pre-Layer Normalization (easy, proven)
2. ✅ GeGLU in FFN (proven improvement)
3. ✅ Relative position bias (crucial for sequences)
4. ✅ Proper multi-scale temporal encoding
5. ✅ Focal loss or class-balanced loss

### What to implement CAREFULLY:
6. ⚠️ Lightweight MoE (test parameter budget)
7. ⚠️ Cross-attention between modalities (if fits in budget)
8. ⚠️ Adaptive embeddings (complex but potentially high impact)

### What to AVOID:
- ❌ RoPE (designed for long sequences, ours are short)
- ❌ Heavy data augmentation (can cause distribution shift)
- ❌ Very deep models (4 layers is probably optimal for this data size)

## Parameter Budget Allocation

With <500k parameters:
- Embeddings: ~110k (location) + 1k (user) = 111k
- Encoder layers (4 layers): ~250k
- Output layer: ~110k
- Remaining: ~30k for improvements

**Strategy**: Keep same model size, but use parameters MORE EFFECTIVELY.
