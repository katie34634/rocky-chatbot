"""
Context extractor for Rocky's lines.

Uses rocky_book_lines.txt as anchors, searches the cleaned book text
for each line, then grabs the preceding Grace narration/dialogue as
the input context.

Usage:
    python extract_pairs.py book_raw.txt rocky_book_lines.txt

Outputs:
    pairs.json          - training pairs {"input": ..., "output": ...}
    pairs_review.txt    - human-readable for manual inspection
    unmatched.txt       - Rocky lines that couldn't be found in the book
"""

import re
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Curly quotes and straight quotes
OPEN_QUOTE  = "\u201c"
CLOSE_QUOTE = "\u201d"

ATTRIBUTION = re.compile(
    r',?\s*(he says|Rocky says|he asks|Rocky asks|Rocky calls|he adds'
    r'|he says again|Rocky says out of nowhere|he asks again)[^"]*',
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Normalize for fuzzy matching — strip quotes, attribution, whitespace."""
    text = text.strip()
    # Remove attribution fragments embedded in the line
    text = ATTRIBUTION.sub("", text)
    # Strip curly/straight outer quotes
    text = text.strip(OPEN_QUOTE + CLOSE_QUOTE + '"\'')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing punctuation for matching only
    text = text.rstrip(".!?,…")
    return text.lower()


def extract_speech(line: str) -> str:
    """
    Pull just the spoken words out of a Rocky line, removing attribution.
    Handles mid-line attribution like: "Many seconds," he says. "Why?"

    Strategy: extract all quoted fragments first, then join them.
    Falls back to attribution-stripping for unquoted lines (laptop screen text).
    """
    # Find all quoted fragments (curly or straight quotes)
    fragments = re.findall(
        rf'{OPEN_QUOTE}([^{CLOSE_QUOTE}]+){CLOSE_QUOTE}|"([^"]+)"',
        line
    )

    if fragments:
        # Each match is a tuple of (curly_match, straight_match) — take whichever is non-empty
        parts = [a or b for a, b in fragments]
        # Strip trailing comma/space artifacts from mid-line breaks like "Many seconds,"
        parts = [p.rstrip(", ") for p in parts]
        return " ".join(parts).strip()

    # Fallback for unquoted lines (laptop screen text) — strip attribution only
    cleaned = ATTRIBUTION.sub(" ", line)
    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# Load and split book into paragraphs, keeping char offsets
# ---------------------------------------------------------------------------

def load_paragraphs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    paras = []
    for m in re.finditer(r'[^\n]+', text):
        content = m.group().strip()
        if content:
            paras.append({
                "text": content,
                "start": m.start(),
                "end": m.end(),
            })
    return paras, text


# ---------------------------------------------------------------------------
# Find a Rocky line in the paragraph list
# ---------------------------------------------------------------------------

def find_rocky_para(rocky_norm: str, paragraphs: list[dict], search_from: int = 0) -> int:
    """
    Returns the index of the paragraph most likely containing this Rocky line.
    Uses substring matching on normalized text.
    Falls back to partial match if the line is very short.
    """
    for i in range(search_from, len(paragraphs)):
        para_norm = normalize(paragraphs[i]["text"])
        if rocky_norm and rocky_norm in para_norm:
            return i
    return -1


# ---------------------------------------------------------------------------
# Grab Grace context preceding a Rocky paragraph
# ---------------------------------------------------------------------------

ROCKY_SIGNALS = re.compile(
    r"(Rocky|comes his|his (translated|musical)|Eridian voice)",
    re.IGNORECASE,
)
GRACE_SPEECH = re.compile(
    r'\b(I say|I said|I ask|I tell|I explain|I shout|I reply|I answer|I call)\b',
    re.IGNORECASE,
)


def is_grace_para(text: str) -> bool:
    """Heuristic: is this paragraph Grace narration or dialogue?"""
    # Very short paragraphs that are just Rocky utterances
    if len(text) < 80 and not GRACE_SPEECH.search(text):
        if text.startswith(OPEN_QUOTE) or text.startswith('"'):
            return False
    return True


def get_grace_context(paragraphs: list[dict], rocky_idx: int, max_paras: int = 3) -> str:
    """
    Walk backwards from rocky_idx collecting Grace context.
    Stop when we hit another Rocky paragraph or exhaust max_paras.
    """
    context_parts = []
    count = 0
    i = rocky_idx - 1

    while i >= 0 and count < max_paras:
        text = paragraphs[i]["text"]

        # Stop if we've hit a previous Rocky line
        if ROCKY_SIGNALS.search(text) and not is_grace_para(text):
            break

        # Skip scene-break markers
        if re.match(r'^[\*\-–—=]+$', text.strip()):
            i -= 1
            continue

        context_parts.insert(0, text)
        count += 1
        i -= 1

    return " ".join(context_parts).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_pairs.py book_raw.txt rocky_book_lines.txt")
        sys.exit(1)

    book_path   = sys.argv[1]
    rocky_path  = sys.argv[2]

    print(f"Loading book: {book_path}")
    paragraphs, _ = load_paragraphs(book_path)
    print(f"  {len(paragraphs)} paragraphs")

    print(f"Loading Rocky lines: {rocky_path}")
    with open(rocky_path, encoding="utf-8") as f:
        rocky_lines = [l.rstrip("\n") for l in f if l.strip()]
    print(f"  {len(rocky_lines)} Rocky lines")

    pairs    = []
    unmatched = []
    search_from = 0  # keep search position advancing through the book

    for i, raw_line in enumerate(rocky_lines):
        rocky_norm = normalize(raw_line)

        if not rocky_norm or len(rocky_norm) < 2:
            # Too short to match reliably (e.g. bare "I", "Yes.")
            # Still include with empty context — useful for style training
            pairs.append({
                "input": "",
                "output": extract_speech(raw_line),
                "raw": raw_line,
                "matched": False,
                "note": "too short to anchor",
            })
            continue

        idx = find_rocky_para(rocky_norm, paragraphs, search_from)

        if idx == -1:
            # Try again from the beginning in case ordering is off
            idx = find_rocky_para(rocky_norm, paragraphs, 0)

        if idx == -1:
            unmatched.append(raw_line)
            pairs.append({
                "input": "",
                "output": extract_speech(raw_line),
                "raw": raw_line,
                "matched": False,
                "note": "not found in book",
            })
            continue

        # Advance search position (book is roughly sequential)
        search_from = max(search_from, idx)

        context = get_grace_context(paragraphs, idx, max_paras=3)

        pairs.append({
            "input": context,
            "output": extract_speech(raw_line),
            "raw": raw_line,
            "matched": True,
            "para_idx": idx,
        })

    # --- Stats ---
    matched   = sum(1 for p in pairs if p["matched"])
    no_context = sum(1 for p in pairs if p["matched"] and not p["input"])
    print(f"\n--- Results ---")
    print(f"Matched in book   : {matched} / {len(rocky_lines)}")
    print(f"Matched, no context found: {no_context}")
    print(f"Unmatched         : {len(unmatched)}")

    # --- Save pairs.json ---
    # For training, only include matched pairs with actual context
    training_pairs = [
        {"input": p["input"], "output": p["output"]}
        for p in pairs
        if p["matched"] and p["input"]
    ]
    with open("pairs.json", "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved pairs.json ({len(training_pairs)} training pairs)")

    # --- Save full pairs with metadata for review ---
    with open("pairs_review.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            status = "OK" if (p["matched"] and p["input"]) else p.get("note", "no context")
            f.write(f"=== {i+1}/{len(pairs)} [{status}] ===\n")
            f.write(f"RAW    : {p['raw']}\n")
            f.write(f"OUTPUT : {p['output']}\n")
            f.write(f"INPUT  : {p['input']}\n")
            f.write("\n")
    print("Saved pairs_review.txt")

    # --- Save unmatched ---
    if unmatched:
        with open("unmatched.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(unmatched))
        print(f"Saved unmatched.txt ({len(unmatched)} lines) — check these manually")


if __name__ == "__main__":
    main()