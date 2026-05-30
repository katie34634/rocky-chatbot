#!/usr/bin/env python3
"""Generate question paraphrases for each pair in a JSON training file.

Usage: python scripts/augment_train.py train.json augmented_train.json

By default this will include the original entry and 10 paraphrased copies
for each input pair. The paraphraser is rule-based and deterministic by seed.
"""
import json
import random
import re
import sys
from pathlib import Path


CONTRACTIONS = {
    "it's": "it is",
    "we're": "we are",
    "i'm": "i am",
    "don't": "do not",
    "isn't": "is not",
    "you're": "you are",
    "that's": "that is",
}


TEMPLATES = [
    "{q}",
    "Can you tell me {q}",
    "Do you know {q}",
    "Could you explain {q}",
    "What is the answer to: {q}",
    "Is it true that {q}",
    "Would you say {q}",
    "How long will it take for {q_short}",
    "What's the expected time for {q_short}",
    "Please tell me: {q}",
]


SYNONYMS = {
    "bouncing": ["reflecting", "sending back", "rebounding"],
    "back": ["back", "toward us", "our way"],
    "hot": ["scorching", "very hot", "burning"],
    "emits": ["radiates", "gives off", "produces"],
    "we're": ["we are", "we're"],
    "we\'re": ["we are", "we're"],
    "we": ["we", "our team", "us"],
    "two": ["two", "a pair of", "2"],
    "still": ["still", "still yet", "remaining"],
    "have": ["contain", "hold", "have"],
    "how long": ["how long", "what is the duration", "how much time"],
}


POLITE_OPENERS = ["Hey,", "Quick question:", "Hi—", "Please,"]
SHORT_QUERIES = ["When?", "Why?", "Now?", "ETA?", "When", "Why", "Now", "How?"]



def clean_question(raw: str) -> str:
    # strip leading question: prefix if present
    s = raw.strip()
    s = re.sub(r"^question:\s*", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s


def make_q_short(q: str, max_words: int = 6) -> str:
    words = q.rstrip("?!. ").split()
    return " ".join(words[:max_words])


def apply_contraction_expansions(q: str) -> str:
    q_lower = q
    for k, v in CONTRACTIONS.items():
        q_lower = re.sub(r"\b" + re.escape(k) + r"\b", v, q_lower, flags=re.IGNORECASE)
    return q_lower


def generate_variants(q_raw: str, n: int = 10, seed: int = 42):
    q = clean_question(q_raw)
    q_short = make_q_short(q)
    rng = random.Random(seed + hash(q) & 0xFFFFFFFF)

    variants = []

    def replace_synonyms(text: str) -> str:
        words = re.split(r"(\W+)", text)
        out_words = []
        for w in words:
            key = w.lower()
            if key in SYNONYMS and rng.random() < 0.5:
                choice = rng.choice(SYNONYMS[key])
                out_words.append(choice)
            else:
                out_words.append(w)
        return "".join(out_words)

    def reorder_clauses(text: str) -> str:
        # split on comma/semicolon/dash/and/or
        parts = re.split(r"[,;—-]", text)
        if len(parts) <= 1:
            return text
        parts = [p.strip() for p in parts if p.strip()]
        rng.shuffle(parts)
        return ", ".join(parts)

    def shorten(text: str) -> str:
        words = text.split()
        if len(words) <= 6:
            return text
        cut = max(3, int(len(words) * 0.5))
        return " ".join(words[:cut])

    def expand(text: str) -> str:
        # add a clarifying clause
        extras = ["Can you confirm?", "Is that correct?", "Why is that?"]
        return f"{text} — {rng.choice(extras)}"

    def make_structural_variants(base: str):
        out = []
        # original
        out.append(base)
        # templates
        for t in TEMPLATES:
            out.append(t.format(q=base, q_short=q_short))
        # synonym-substituted
        out.append(replace_synonyms(base))
        # reordered clauses
        out.append(reorder_clauses(base))
        # shortened
        out.append(shorten(base))
        # expanded
        out.append(expand(base))
        # polite opener + base
        out.append(f"{rng.choice(POLITE_OPENERS)} {base}")
        # contraction-expanded
        out.append(apply_contraction_expansions(base))
        return out

    candidates = make_structural_variants(q)
    # add variations with small surface transformations
    more = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        # ensure question mark if it sounds interrogative
        if not c.endswith("?") and any(w in c.lower() for w in ["what", "how", "is", "do", "can", "why"]):
            c = c + "?"
        # sometimes prepend polite opener
        if rng.random() < 0.2:
            c = f"{rng.choice(POLITE_OPENERS)} {c}"
        # sometimes synonym replace again
        if rng.random() < 0.3:
            c = replace_synonyms(c)
        more.append(c)

    # shuffle deterministically and pick top n unique
    rng.shuffle(more)
    unique = []
    for v in more:
        v = re.sub(r"\s+", " ", v).strip()
        if v not in unique:
            unique.append(v)
        if len(unique) >= n:
            break
    # if still not enough, fill with polite + base
    while len(unique) < n:
        # include some very short 1-2 word queries to increase diversity
        if rng.random() < 0.4:
            cand = rng.choice(SHORT_QUERIES)
        else:
            cand = f"{rng.choice(POLITE_OPENERS)} {q}"
        if cand not in unique:
            unique.append(cand)
        else:
            unique.append(q)
    return unique[:n]


def augment_file(in_path: Path, out_path: Path, copies: int = 10, include_original: bool = True):
    data = json.loads(in_path.read_text(encoding="utf-8"))
    out = []
    for entry in data:
        input_text = entry.get("input_text", "")
        target_text = entry.get("target_text", "")

        # keep original if requested
        if include_original:
            out.append(entry)

        variants = generate_variants(input_text, n=copies)
        for v in variants:
            out.append({"input_text": f"question: {v}", "target_text": target_text})

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv):
    if len(argv) < 3:
        print("Usage: scripts/augment_train.py <input.json> <output.json> [--copies N] [--no-original]")
        return 2
    in_file = Path(argv[1])
    out_file = Path(argv[2])
    copies = 10
    include_original = True
    if "--copies" in argv:
        i = argv.index("--copies")
        try:
            copies = int(argv[i + 1])
        except Exception:
            pass
    if "--no-original" in argv:
        include_original = False

    augment_file(in_file, out_file, copies=copies, include_original=include_original)
    print(f"Wrote augmented file: {out_file} (copies={copies}, include_original={include_original})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
