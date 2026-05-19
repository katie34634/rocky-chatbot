"""
Extract Grace->Rocky training pairs from the movie script.

Uses rocky_movie_lines.txt as anchors and matches them back to the
script to find the preceding Grace dialogue as context.

Usage:
    python extract_movie_pairs.py movie_transcript.txt rocky_movie_lines.txt

Outputs:
    movie_pairs.json        - training pairs {"input": ..., "output": ...}
    movie_pairs_review.txt  - human-readable for inspection
    movie_unmatched.txt     - Rocky lines that couldn't be found
"""

import re
import json
import sys


TURN_RE = re.compile(
    r'^(GRACE|ROCKY)(?:\s*\[[^\]]*\])?\s*:\s*(.*)',
    re.IGNORECASE
)
NONVERBAL_RE = re.compile(
    r'^\s*\[(musical|panicked chitter|chitter|screeching|screeches|chirp|aggressive)',
    re.IGNORECASE
)
STAGE_ONLY      = re.compile(r'^\[.*\]\.?$')
BRACKET_CONTENT = re.compile(r'\[.*?\]')
LEADING_CONT    = re.compile(r'^[—–\-…]+\s*')
TRAILING_CONT   = re.compile(r'\s*[—–\-]+$')


def clean_rocky_line(line):
    line = BRACKET_CONTENT.sub("", line).strip()
    line = LEADING_CONT.sub("", line)
    line = TRAILING_CONT.sub("", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line.strip(",. ")


def is_droppable(line):
    return clean_rocky_line(line) == ""


def normalize(text):
    text = BRACKET_CONTENT.sub("", text)
    text = LEADING_CONT.sub("", text)
    text = TRAILING_CONT.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.rstrip(".,!?…—-")


def parse_script(text):
    turns = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        m = TURN_RE.match(line)
        if m:
            speaker  = m.group(1).upper()
            dialogue = m.group(2).strip()
            if speaker == "ROCKY" and (NONVERBAL_RE.match(dialogue) or STAGE_ONLY.match(dialogue)):
                continue
            turns.append({"speaker": speaker, "text": dialogue})
        elif line.startswith("—") or line.startswith("–"):
            if turns:
                turns[-1]["text"] += " " + line.lstrip("—–").strip()
    return turns


def find_rocky_turn(rocky_norm, turns, search_from=0):
    if not rocky_norm or len(rocky_norm) < 4:
        return -1
    for i in range(search_from, len(turns)):
        if turns[i]["speaker"] != "ROCKY":
            continue
        turn_norm = normalize(turns[i]["text"])
        if len(turn_norm) < 4:
            continue
        if rocky_norm in turn_norm or turn_norm in rocky_norm:
            return i
    return -1


def get_grace_context(turns, rocky_idx, max_turns=3):
    context = []
    count   = 0
    i       = rocky_idx - 1
    while i >= 0 and count < max_turns:
        turn = turns[i]
        if turn["speaker"] == "ROCKY":
            break
        if turn["speaker"] != "GRACE":
            i -= 1
            continue
        grace_text = BRACKET_CONTENT.sub("", turn["text"]).strip()
        grace_text = re.sub(r"\s+", " ", grace_text).strip()
        if not grace_text:
            i -= 1
            continue
        context.insert(0, grace_text)
        count += 1
        i -= 1
    return " ".join(context).strip()


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_movie_pairs.py movie_transcript.txt rocky_movie_lines.txt")
        sys.exit(1)

    script_path = sys.argv[1]
    rocky_path  = sys.argv[2]

    print(f"Loading script: {script_path}")
    with open(script_path, encoding="utf-8") as f:
        script_text = f.read()

    turns = parse_script(script_text)
    print(f"  {len(turns)} total turns parsed")
    print(f"  {sum(1 for t in turns if t['speaker'] == 'ROCKY')} Rocky turns")
    print(f"  {sum(1 for t in turns if t['speaker'] == 'GRACE')} Grace turns")

    print(f"\nLoading Rocky lines: {rocky_path}")
    with open(rocky_path, encoding="utf-8") as f:
        rocky_lines = [l.rstrip("\n") for l in f if l.strip()]
    print(f"  {len(rocky_lines)} Rocky lines")

    pairs       = []
    unmatched   = []
    search_from = 0

    for raw_line in rocky_lines:
        if is_droppable(raw_line):
            continue
        output = clean_rocky_line(raw_line)
        if not output:
            continue
        rocky_norm = normalize(raw_line)
        idx = find_rocky_turn(rocky_norm, turns, search_from)
        if idx == -1:
            idx = find_rocky_turn(rocky_norm, turns, 0)
        if idx == -1:
            unmatched.append(raw_line)
            pairs.append({"input": "", "output": output, "raw": raw_line, "matched": False})
            continue
        search_from = max(search_from, idx)
        context = get_grace_context(turns, idx, max_turns=3)
        pairs.append({"input": context, "output": output, "raw": raw_line, "matched": True})

    matched    = sum(1 for p in pairs if p["matched"])
    no_context = sum(1 for p in pairs if p["matched"] and not p["input"])
    print(f"\n--- Results ---")
    print(f"Matched    : {matched} / {len(pairs)}")
    print(f"No context : {no_context}")
    print(f"Unmatched  : {len(unmatched)}")
    if unmatched:
        print("Unmatched lines:")
        for u in unmatched:
            print(f"  {repr(u)}")

    training = [
        {"input": p["input"], "output": p["output"]}
        for p in pairs
        if p["matched"] and p["input"]
    ]
    with open("movie_pairs.json", "w", encoding="utf-8") as f:
        json.dump(training, f, indent=2, ensure_ascii=False)
    print(f"\nSaved movie_pairs.json ({len(training)} training pairs)")

    with open("movie_pairs_review.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            status = "OK" if (p["matched"] and p["input"]) else ("no context" if p["matched"] else "unmatched")
            f.write(f"=== {i+1}/{len(pairs)} [{status}] ===\n")
            f.write(f"RAW    : {p['raw']}\n")
            f.write(f"OUTPUT : {p['output']}\n")
            f.write(f"INPUT  : {p['input']}\n\n")
    print("Saved movie_pairs_review.txt")

    if unmatched:
        with open("movie_unmatched.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(unmatched))
        print(f"Saved movie_unmatched.txt ({len(unmatched)} lines)")


if __name__ == "__main__":
    main()