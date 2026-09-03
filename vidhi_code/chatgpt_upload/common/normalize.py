"""Unicode normalisation for Devanagari text.

Devanagari can encode the same visible character in more than one way, and web text
contains all of the variants. If they are not unified, the tokenizer learns several
spellings of the same word, splitting probability mass across duplicates and wasting
vocabulary slots that a low-resource model cannot spare.

What this module unifies:

* **Nukta forms.** क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ each exist as a single precomposed codepoint
  (U+0958–U+095F) *and* as base letter + nukta (U+093C). NFC normalisation maps the
  precomposed forms to the decomposed pair, because those codepoints carry singleton
  decompositions that Unicode excludes from recomposition. Applying NFC therefore gives
  one consistent representation — the decomposed one — which is also what Indic NLP
  tooling standardises on.
* **Zero-width joiners.** ZWJ (U+200D) and ZWNJ (U+200C) control conjunct rendering. They
  are invisible, applied inconsistently across sites, and would otherwise produce
  distinct tokens for identical-looking words.
* **Invisible and format characters.** Soft hyphens, byte-order marks, directional marks
  and other zero-width characters that survive HTML extraction.
* **Punctuation variants.** Curly quotes, dashes and ellipses collapse to plain ASCII
  equivalents. The danda ``।`` and double danda ``॥`` are deliberately preserved — they
  are genuine Devanagari sentence punctuation, not substitutable with a full stop.
* **Whitespace.** Non-breaking and exotic spaces become ordinary spaces; runs collapse.

Devanagari digits (०–९) are **not** converted to ASCII. Nepali in particular uses them
throughout, including in dates, and rewriting them would distort the language being
modelled.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and format characters to delete outright.
INVISIBLE_CHARS = (
    "​"  # zero-width space
    "‌"  # zero-width non-joiner (ZWNJ)
    "‍"  # zero-width joiner (ZWJ)
    "‎"  # left-to-right mark
    "‏"  # right-to-left mark
    "­"  # soft hyphen
    "﻿"  # byte-order mark
    "⁠"  # word joiner
)
INVISIBLE_RE = re.compile(f"[{INVISIBLE_CHARS}]")

# Punctuation variants that add nothing but vocabulary pressure.
PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "　": " ",
}
PUNCT_RE = re.compile("|".join(re.escape(k) for k in PUNCT_MAP))

# Whitespace runs, and blank-line runs.
SPACE_RUN_RE = re.compile(r"[ \t]+")
NEWLINE_RUN_RE = re.compile(r"\n{3,}")

# Repeated danda or punctuation, common in scraped text ("।।।।").
REPEAT_PUNCT_RE = re.compile(r"([।॥!?.,-])\1{2,}")


def normalize_text(text: str) -> str:
    """Apply full Unicode normalisation to one document.

    The order matters: NFC first so that nukta forms are unified before anything else
    inspects the characters, then invisible-character removal, then punctuation and
    whitespace tidying.

    Args:
        text: Raw or extracted text.

    Returns:
        Normalised text, ready for language identification and tokenizer training.
    """
    # NFC unifies nukta forms and any other canonically equivalent sequences.
    text = unicodedata.normalize("NFC", text)

    text = INVISIBLE_RE.sub("", text)
    text = PUNCT_RE.sub(lambda m: PUNCT_MAP[m.group()], text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)

    # Tidy whitespace without destroying paragraph structure.
    lines = [SPACE_RUN_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = NEWLINE_RUN_RE.sub("\n\n", text)

    return text.strip()


def normalization_report(before: str, after: str) -> dict:
    """Summarise what normalisation changed, for the dataset statistics report.

    Args:
        before: Text prior to normalisation.
        after: Text after normalisation.

    Returns:
        Counts of characters removed and whether the document changed at all.
    """
    return {
        "chars_before": len(before),
        "chars_after": len(after),
        "chars_removed": len(before) - len(after),
        "changed": before != after,
    }
