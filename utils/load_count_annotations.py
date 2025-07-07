import json
import pandas as pd

# Load JSON
with open('annotations/count/count_annotations.json', 'r') as f:
    count_data = json.load(f)

print("✅ Loaded JSON count annotations:")
print(count_data[:1])

# Load CSV
count_df = pd.read_csv('annotations/count/count_annotations.csv')
print("\n✅ Loaded CSV count annotations:")
print(count_df.head())
