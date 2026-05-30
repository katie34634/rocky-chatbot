"""
Extract Grace-to-Rocky training pairs.

Usage:
    python preprocessing/extract_pairs_dialogue_only.py DATA_I_CLEANED_BY_HAND/book_pairs.txt

Outputs:
    pairs.json          - training pairs {"input": ..., "output": ...}
"""

import json
import re
import sys

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    lines_path = sys.argv[1]

    speaker = False
    rocky_lines = []
    grace_lines = []

    print(f"Loading lines: {lines_path}")
    with open(lines_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not speaker:
                grace_lines.append(line)
            else:
                rocky_lines.append(line)

            speaker = not speaker
            
    print(f"  {len(rocky_lines)} Rocky lines")
    print(f"  {len(grace_lines)} Grace lines")

    print("\n--- Sample lines ---")
    for i in range(3):
        print(f"Rocky: {rocky_lines[i]}")
        print(f"Grace: {grace_lines[i]}")
        print()

    pairs = []

    for i, line in enumerate(grace_lines):
        if line.strip():
            pairs.append({
                "input": line,
                "output": rocky_lines[i],
            })

    with open("pairs.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved pairs.json ({len(pairs)} training pairs)")


if __name__ == "__main__":
    main()