import json
import glob
import os
from pathlib import Path

# Set the directory containing the batch files
augmented_data_dir = "augmented_data/batched"
output_dir = "augmented_data"

# Find all batch*.json files and sort them numerically
batch_files = sorted(
    glob.glob(os.path.join(augmented_data_dir, "batch*.json")),
    key=lambda x: int(Path(x).stem.replace('batch', ''))
)

print(f"Found {len(batch_files)} batch files to merge")

# Merge all batches into one list
all_pairs = []

for batch_file in batch_files:
    batch_num = Path(batch_file).stem.replace('batch', '')
    print(f"Loading batch{batch_num}.json...")
    
    with open(batch_file, 'r') as f:
        batch_data = json.load(f)
        all_pairs.extend(batch_data)
        print(f"  Added {len(batch_data)} entries (total: {len(all_pairs)})")

# Save merged data
output_file = os.path.join(output_dir, "augmented_train.json")
with open(output_file, 'w') as f:
    json.dump(all_pairs, f, indent=2)

print(f"\nMerge complete!")
print(f"Total pairs: {len(all_pairs)}")
print(f"Saved to: {output_file}")