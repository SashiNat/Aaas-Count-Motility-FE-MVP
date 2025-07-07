import json
import pandas as pd

# Load JSON
with open('annotations/motility/motility_annotations.json', 'r') as f:
    motility_data = json.load(f)

print("✅ Loaded JSON motility trajectories:")
print(motility_data[:1])

# Load CSV
motility_df = pd.read_csv('annotations/motility/motility_clip_labels.csv')
print("\n✅ Loaded CSV clip-level motility labels:")
print(motility_df.head())
