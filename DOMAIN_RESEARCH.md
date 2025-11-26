# Deep Domain Understanding: Human Mobility Prediction

## What is the ACTUAL Problem?

This is not just a sequence prediction task - it's **human mobility prediction**. Humans don't move randomly. We have:
- **Habits**: Work, home, gym (repetitive patterns)
- **Time patterns**: Breakfast at 8am, work at 9am, lunch at 12pm
- **Geographic constraints**: Can't teleport, use transportation networks
- **Social patterns**: Meet friends at specific places
- **Day-type patterns**: Weekday vs weekend behavior

## Key Domain Insights

### 1. Markov Chain Property
Human mobility exhibits **strong Markov properties**:
- If I'm at Starbucks at 8am on Monday → likely going to office
- If I'm at office at 5pm → likely going home or gym
- **Next location depends heavily on CURRENT location + time**

### 2. Return to Home/Work (RTH/RTW)
Studies show:
- **~70% of trips return to previously visited locations**
- Home and work locations account for **50-60% of all visits**
- This is called "return probability"

### 3. Temporal Regularity
- **Morning (7-9am)**: Home → Work/School
- **Lunch (12-1pm)**: Work → Restaurant
- **Evening (5-7pm)**: Work → Home/Gym/Shopping
- **Weekend**: Different patterns entirely

### 4. Geographic Constraints
- **Distance matters**: Unlikely to go 50km in 30 minutes
- **Network constraints**: Roads, public transit
- **Neighboring locations**: Clusters of activity

### 5. User-Specific Behavior
- Each user has **personal routine**
- Some users are more predictable than others
- Historical frequency matters

## What Current Models Miss

### Baseline Issues:
1. **Treats all locations equally** - doesn't know home/work are special
2. **Ignores spatial distance** - location 5 and 500 treated same
3. **No explicit transition modeling** - doesn't learn A→B patterns
4. **No return-to-previous logic** - ignores that humans revisit places
5. **No time-of-day routing** - doesn't know morning = work, evening = home

## Research: What Works in Mobility Prediction?

### DeepMove (WWW 2018)
- **Attention over historical locations** (what places user visited before)
- **Recurrent attention over recent trajectory**
- **Multi-modal embedding** (location + time + distance)
- Achieved **~45-50% accuracy** on Foursquare

### LSTPM (KDD 2020) 
- **Long-term** user preferences (frequent places)
- **Short-term** trajectory patterns (recent sequence)
- **Geo-temporal context**
- Achieved **~52% accuracy** on Geolife!

### STAN (AAAI 2021)
- **Self-attention** on trajectory
- **Time-aware attention** 
- **Non-local modeling** (beyond just sequence)
- State-of-the-art results

### HST-LSTM (TKDE 2019)
- **Hierarchical spatial-temporal** LSTM
- Explicitly models **spatial regions**
- Time slots for temporal patterns

## Key Techniques That ACTUALLY Work

### 1. Historical Location Modeling
```python
# For each user, track:
- Top-K most visited locations (likely destinations)
- Home location (most visits 10pm-6am)
- Work location (most visits 9am-5pm weekdays)
```

### 2. Transition Matrix
```python
# Learn P(next_loc | current_loc, time)
# This is MUCH more powerful than pure sequence modeling
transition_prob[current_loc, time_slot] → distribution over next_loc
```

### 3. Distance-Based Features
```python
# Add geographic features:
- Distance from current location to candidate
- Whether candidate is in same "region"
- Whether it's on the way home
```

### 4. Return Probability
```python
# Explicitly model: Will user return to a previous location?
# If yes, which one? (often home/work)
return_prob = sigmoid(features)
if return_prob > threshold:
    predict from previously_visited
else:
    predict new location
```

### 5. Time-Slot Specific Modeling
```python
# Different models for different times:
morning_model (7-10am) → predicts work-related
lunch_model (12-2pm) → predicts restaurants
evening_model (5-8pm) → predicts home/leisure
```

## The Missing Piece: We Need DOMAIN KNOWLEDGE

The baseline treats this as pure sequence modeling. But it's NOT!

It's:
1. **Graph problem**: Locations are nodes, transitions are edges
2. **Markov chain**: Next location depends on current + time
3. **User modeling**: Each user has personal patterns
4. **Spatial problem**: Geography matters!

## New Approach: Domain-Informed Model

### Architecture Concept:
```
Input: [loc_seq, time_seq, user_id]

1. Extract user profile:
   - home_loc, work_loc (from history)
   - top_k_locations (user favorites)
   
2. Encode trajectory:
   - Current location embedding
   - Time embedding (hour, day, weekend)
   - Recent sequence context
   
3. Transition modeling:
   - P(next | current, time) learned transition matrix
   - Distance-weighted attention
   
4. Return modeling:
   - Will user return to previous location?
   - If yes, attend over history
   
5. Combine:
   - User preferences (long-term)
   - Trajectory context (short-term)
   - Transition patterns (Markov)
   - Return probability
   - Time-of-day routing
```

## Data We Can Extract from GeoLife

1. **User profiles**: Home/work locations per user
2. **Transition frequencies**: loc_i → loc_j counts
3. **Time patterns**: When does each location get visited
4. **Spatial structure**: Lat/lon if available (check data!)
5. **Visit frequencies**: How often each user visits each location

## This Changes Everything!

We're not building a "better transformer" - we're building a **mobility model**!

Next steps:
1. Deep data mining (extract domain features)
2. Build transition matrices
3. Identify home/work for each user
4. Create geography-aware model
5. Test return-to-previous strategy

**This is the path to 50%+!**
