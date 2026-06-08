import json
import random

with open("rocky_lines/rocky_lines_clean.txt", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

pairs = [{"input": "rocky:", "output": line} for line in lines]

random.seed(42)
random.shuffle(pairs)
split_idx = max(1, int(len(pairs) * 0.1))
val = pairs[:split_idx]
train = pairs[split_idx:]

with open("train.json", "w", encoding="utf-8") as f:
    json.dump([{"input_text": f"question: {p['input']}", "target_text": p["output"]} for p in train], f, indent=2, ensure_ascii=False)

with open("val.json", "w", encoding="utf-8") as f:
    json.dump([{"input_text": f"question: {p['input']}", "target_text": p["output"]} for p in val], f, indent=2, ensure_ascii=False)