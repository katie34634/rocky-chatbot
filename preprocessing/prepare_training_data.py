"""
Merge book and movie pairs, format for T5, and split into train/val.

Usage:
    python preprocessing/prepare_training_data.py datasets/dialogue/pairs.json datasets/dialogue/movie_pairs.json

Outputs:
    train.json  - training set
    val.json    - validation set
"""

import json
import re
import random
import sys


# ---------------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------------

def load_pairs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_and_deduplicate(book_pairs: list[dict], movie_pairs: list[dict]) -> list[dict]:
    seen_outputs = set()
    merged = []

    # Book pairs first — they're higher quality
    for pair in book_pairs + movie_pairs:
        input_text  = pair["input"].strip()
        output_text = pair["output"].strip()

        if not input_text or not output_text:
            continue

        # Deduplicate on output — same Rocky line appearing in both sources
        output_norm = re.sub(r"\s+", " ", output_text.lower().strip(".,!? "))
        if output_norm in seen_outputs:
            continue
        seen_outputs.add(output_norm)

        merged.append({"input": input_text, "output": output_text})

    return merged


# ---------------------------------------------------------------------------
# Format for T5
# ---------------------------------------------------------------------------

def format_for_t5(pairs: list[dict]) -> list[dict]:
    formatted = []
    for pair in pairs:
        formatted.append({
            "input_text":  f"question: {pair['input']}",
            "target_text": pair["output"],
        })
    return formatted


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split(pairs: list[dict], val_ratio: float = 0.1, seed: int = 42) -> tuple:
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * val_ratio))
    return shuffled[split_idx:], shuffled[:split_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python prepare_training_data.py pairs.json movie_pairs.json")
        sys.exit(1)

    book_path  = sys.argv[1]
    movie_path = sys.argv[2]

    print(f"Loading {book_path}...")
    book_pairs = load_pairs(book_path)
    print(f"  {len(book_pairs)} book pairs")

    print(f"Loading {movie_path}...")
    movie_pairs = load_pairs(movie_path)
    print(f"  {len(movie_pairs)} movie pairs")

    print("\nMerging and deduplicating...")
    merged = merge_and_deduplicate(book_pairs, movie_pairs)
    print(f"  {len(merged)} pairs after deduplication")

    print("Formatting for T5...")
    formatted = format_for_t5(merged)

    print("Splitting into train/val...")
    train, val = split(formatted, val_ratio=0.1)
    print(f"  {len(train)} train / {len(val)} val")

    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    print("\nSaved train.json")

    with open("val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print("Saved val.json")

    # Print a few examples
    print("\n--- Sample training pairs ---")
    for pair in train[:5]:
        print(f"  INPUT : {pair['input_text'][:80]}")
        print(f"  TARGET: {pair['target_text']}")
        print()


if __name__ == "__main__":
    main()