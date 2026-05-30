"""
Extract Grace-to-Rocky training pairs.

Uses rocky_book_lines.txt as anchors, searches the book text for each
Rocky line, then grabs the preceding Grace dialogue as the input.

Usage:
    python preprocessing/extract_pairs.py corpus/book_raw.txt rocky_lines/rocky_book_lines.txt

Outputs:
    pairs.json          - training pairs {"input": ..., "output": ...}
    pairs_review.txt    - human-readable for manual inspection
    unmatched.txt       - Rocky lines that couldn't be found in the book
"""

import json
import re
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Curly quotes and straight quotes
OPEN_QUOTE = "\u201c"
CLOSE_QUOTE = "\u201d"

ATTRIBUTION = re.compile(
    r',?\s*(he says|Rocky says|he asks|Rocky asks|Rocky calls|he adds'
    r'|he says again|Rocky says out of nowhere|he asks again)[^"]*',
    re.IGNORECASE,
)

GRACE_ATTRIBUTION = re.compile(
    r',?\s*(Grace says|Grace said|Grace asks|Grace asked|Grace replies|Grace replied|'
    r'Grace tells|Grace told|I say|I said|I ask|I asked|I tell|I told|'
    r'I explain|I explained|I shout|I shouted|I reply|I replied|'
    r'I answer|I answered|I call|I called)[^"]*',
    re.IGNORECASE,
)

ROCKY_SIGNALS = re.compile(
    r"(Rocky|comes his|his (translated|musical)|Eridian voice)",
    re.IGNORECASE,
)

GRACE_SPEECH = re.compile(
    r'\b(I say|I said|I ask|I asked|I tell|I told|I explain|I explained|'
    r'I shout|I shouted|I reply|I replied|I answer|I answered|I call|I called|'
    r'Grace says|Grace said|Grace asks|Grace asked|Grace replies|Grace replied|'
    r'Grace tells|Grace told)\b',
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Normalize for fuzzy matching."""
    text = text.strip()
    text = ATTRIBUTION.sub("", text)
    text = text.strip(OPEN_QUOTE + CLOSE_QUOTE + '"\'')
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower().rstrip(".!?,…")


def extract_rocky_speech(line: str) -> str:
    """
    Pull just the spoken words out of a Rocky line, removing attribution.
    Handles mid-line attribution like: "Many seconds," he says. "Why?"

    Strategy: extract all quoted fragments first, then join them.
    Falls back to attribution-stripping for unquoted lines.
    """
    fragments = re.findall(
        rf'{OPEN_QUOTE}([^{CLOSE_QUOTE}]+){CLOSE_QUOTE}|"([^"]+)"',
        line,
    )

    if fragments:
        parts = [a or b for a, b in fragments]
        parts = [p.rstrip(", ") for p in parts]
        return " ".join(parts).strip()

    cleaned = ATTRIBUTION.sub(" ", line)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_grace_speech(text: str, rocky_norm: str = "") -> str:
    """
    Pull Grace dialogue out of a paragraph.
    If the paragraph contains Rocky speech too, truncate at the first Rocky marker.
    """
    working = text

    if rocky_norm:
        rocky_plain = rocky_norm.strip()
        if rocky_plain:
            idx = normalize(working).find(rocky_plain)
            if idx != -1:
                working = working[:idx]

    rocky_markers = [
        r"\bRocky\b",
        r"\bquestion\b",
        r"\bunderstand\b",
        r"\bignore\b",
        r"\bno\b",
        r"\byes\b",
    ]

    for marker in rocky_markers:
        match = re.search(marker, working, re.IGNORECASE)
        if match:
            working = working[:match.start()]
            break

    fragments = re.findall(
        rf'{OPEN_QUOTE}([^{CLOSE_QUOTE}]+){CLOSE_QUOTE}|"([^"]+)"',
        working,
    )

    if fragments:
        parts = [a or b for a, b in fragments]
        parts = [p.rstrip(", ") for p in parts]
        cleaned = " ".join(parts).strip()
    else:
        cleaned = GRACE_ATTRIBUTION.sub(" ", working)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    cleaned = cleaned.strip(OPEN_QUOTE + CLOSE_QUOTE + '"\'')
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    if fragments:
        return cleaned

    if GRACE_SPEECH.search(cleaned):
        cleaned = GRACE_ATTRIBUTION.sub(" ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    return ""


def is_grace_dialogue_para(text: str, rocky_norm: str = "") -> bool:
    """Heuristic: is this paragraph likely Grace dialogue?"""
    return bool(extract_grace_speech(text, rocky_norm=rocky_norm))


def get_grace_context(paragraphs: list[dict], rocky_idx: int, rocky_norm: str, max_paras: int = 3) -> str:
    """
    Walk backwards from rocky_idx collecting Grace dialogue.
    Stop when we hit another Rocky paragraph or exhaust max_paras.
    """
    context_parts = []
    count = 0
    i = rocky_idx - 1

    while i >= 0 and count < max_paras:
        text = paragraphs[i]["text"]

        if ROCKY_SIGNALS.search(text) and not is_grace_dialogue_para(text, rocky_norm=rocky_norm):
            break

        if re.match(r"^[\*\-–—=]+$", text.strip()):
            i -= 1
            continue

        grace_text = extract_grace_speech(text, rocky_norm=rocky_norm)
        if grace_text:
            context_parts.insert(0, grace_text)
            count += 1

        i -= 1

    return " ".join(context_parts).strip()

# ---------------------------------------------------------------------------
# Load and split book into paragraphs
# ---------------------------------------------------------------------------

def load_paragraphs(path: str) -> tuple[list[dict], str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    paras = []
    for m in re.finditer(r"[^\n]+", text):
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
    """
    for i in range(search_from, len(paragraphs)):
        para_norm = normalize(paragraphs[i]["text"])
        if rocky_norm and rocky_norm in para_norm:
            return i
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_pairs.py book_raw.txt rocky_book_lines.txt")
        sys.exit(1)

    book_path = sys.argv[1]
    rocky_path = sys.argv[2]

    print(f"Loading book: {book_path}")
    paragraphs, _ = load_paragraphs(book_path)
    print(f"  {len(paragraphs)} paragraphs")

    print(f"Loading Rocky lines: {rocky_path}")
    with open(rocky_path, encoding="utf-8") as f:
        rocky_lines = [l.rstrip("\n") for l in f if l.strip()]
    print(f"  {len(rocky_lines)} Rocky lines")

    pairs = []
    unmatched = []
    search_from = 0

    for raw_line in rocky_lines:
        rocky_norm = normalize(raw_line)

        if not rocky_norm or len(rocky_norm) < 2:
            pairs.append({
                "input": "",
                "output": extract_rocky_speech(raw_line),
                "raw": raw_line,
                "matched": False,
                "note": "too short to anchor",
            })
            continue

        idx = find_rocky_para(rocky_norm, paragraphs, search_from)

        if idx == -1:
            idx = find_rocky_para(rocky_norm, paragraphs, 0)

        if idx == -1:
            unmatched.append(raw_line)
            pairs.append({
                "input": "",
                "output": extract_rocky_speech(raw_line),
                "raw": raw_line,
                "matched": False,
                "note": "not found in book",
            })
            continue

        search_from = max(search_from, idx)

        context = get_grace_context(paragraphs, idx, rocky_norm, max_paras=3)

        pairs.append({
            "input": context,
            "output": extract_rocky_speech(raw_line),
            "raw": raw_line,
            "matched": True,
            "para_idx": idx,
        })

    matched = sum(1 for p in pairs if p["matched"])
    no_context = sum(1 for p in pairs if p["matched"] and not p["input"])
    print(f"\n--- Results ---")
    print(f"Matched in book        : {matched} / {len(rocky_lines)}")
    print(f"Matched, no Grace input : {no_context}")
    print(f"Unmatched              : {len(unmatched)}")

    training_pairs = [
        {"input": p["input"], "output": p["output"]}
        for p in pairs
        if p["matched"] and p["input"]
    ]

    with open("pairs.json", "w", encoding="utf-8") as f:
        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
    print(f"\nSaved pairs.json ({len(training_pairs)} training pairs)")

    with open("pairs_review.txt", "w", encoding="utf-8") as f:
        for i, p in enumerate(pairs):
            status = "OK" if (p["matched"] and p["input"]) else p.get("note", "no Grace input")
            f.write(f"=== {i+1}/{len(pairs)} [{status}] ===\n")
            f.write(f"RAW    : {p['raw']}\n")
            f.write(f"OUTPUT : {p['output']}\n")
            f.write(f"INPUT  : {p['input']}\n\n")
    print("Saved pairs_review.txt")

    if unmatched:
        with open("unmatched.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(unmatched))
        print(f"Saved unmatched.txt ({len(unmatched)} lines) — check these manually")


if __name__ == "__main__":
    main()