"""
Compute location frequencies from training data for frequency-aware embeddings.
"""

import pickle
import numpy as np
from collections import Counter

def compute_location_frequencies(data_path, num_locations=1187):
    """Compute frequency of each location in training data."""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Count occurrences
    location_counts = Counter()
    for sample in data:
        locs = sample['X']
        for loc in locs:
            location_counts[loc] += 1
    
    # Convert to array
    frequencies = np.zeros(num_locations)
    for loc_id, count in location_counts.items():
        if loc_id < num_locations:
            frequencies[loc_id] = count
    
    return frequencies.tolist()


if __name__ == '__main__':
    train_path = '/content/geolife_prediction/data/geolife/geolife_transformer_7_train.pk'
    freqs = compute_location_frequencies(train_path)
    
    print(f"Total locations: {len(freqs)}")
    print(f"Locations with data: {sum(1 for f in freqs if f > 0)}")
    print(f"Max frequency: {max(freqs)}")
    print(f"Min frequency (non-zero): {min(f for f in freqs if f > 0)}")
    
    # Save
    import json
    with open('/content/geolife_prediction/data/geolife/location_frequencies.json', 'w') as f:
        json.dump(freqs, f)
    
    print("Saved to location_frequencies.json")
