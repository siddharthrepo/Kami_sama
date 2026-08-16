"""Post-extraction text cleaning and document quality gating.

trafilatura removes the obvious page furniture, but news sites still leak three things
into the extracted text:

* **Whitespace scaffolding** — hundreds of blank or space-only lines left behind where
  layout ``<div>``s used to be.
* **Leaked JSON** — CMS metadata blobs rendered into the page body.
* **Site boilerplate** — "विज्ञापन", app-download prompts, paywall teasers. These repeat
  on every page of a site, so leaving them in would teach the model that the most likely
  Hindi sentence is an advertisement notice.

This module strips all three, then decides whether what remains is a real article. The
quality gate is applied to the *cleaned* text, which matters: a page that looks like
12,000 characters of content before cleaning can be 400 characters of prose after it.

Site-wide boilerplate that these patterns miss is caught later, at the corpus level,
where any line repeating across many documents of the same source is removable by
frequency alone.
"""

from __future__ import annotations

import re

# A line that is entirely a JSON object — CMS metadata leaked into the body.
JSON_LINE_RE = re.compile(r'^\s*[\{\[].*[\}\]]\s*$')

# Runs of spaces, tabs and non-breaking spaces.
INLINE_SPACE_RE = re.compile(r"[ \t ​]+")

# Boilerplate lines, matched case-insensitively anywhere in the line.
# Hindi and Nepali patterns share one list; a Hindi phrase simply never fires on a
# Nepali page, and keeping one list makes the filter easier to audit.
BOILERPLATE_PATTERNS = [
    # --- Hindi news furniture ---
    r"^विज्ञापन$",
    r"एप डाउनलोड कर",
    r"ऐप डाउनलोड कर",
    r"वीडियो विज्ञापन",
    r"प्रीमियम मेंबरशिप",
    r"पढ़ना जारी रखने के लिए",
    r"खबरें लगातार पढ़ने",
    r"यह भी पढ़ें",
    r"संबंधित (खबरें|समाचार)",
    r"हमें फॉलो कर",
    r"शेयर कर",
    # --- Nepali news furniture ---
    r"^विज्ञापन",
    r"प्रकाशित मिति",
    r"सम्बन्धित समाचार",
    r"तपाईंको प्रतिक्रिया",
    r"^प्रतिक्रिया",
    r"फेसबुक(मा)? ",
    # --- Generic / English chrome that survives on Indic pages ---
    r"^\s*(Advertisement|ADVERTISEMENT|Sponsored|Read more|Share|Comments?)\s*$",
    r"^\s*(Copyright|All rights reserved|©)",
    r"^\s*(Home|Menu|Search|Login|Subscribe)\s*$",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

# A line needs at least this many words before it counts as prose rather than as a
# caption, byline, tag or navigation fragment.
PROSE_LINE_MIN_WORDS = 8

# Devanagari Unicode block, used by the script-ratio language guard.
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Latin letters, used to find residual English lines inside Indic documents.
LATIN_RE = re.compile(r"[A-Za-z]")


def devanagari_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that fall in the Devanagari block.

    A cheap first-pass language guard. It cannot separate Hindi from Nepali — both use
    this script — but it reliably rejects the English pages that leak into Indian and
    Nepali news feeds. Real language identification happens in the cleaning stage.
    """
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 0.0
    return len(DEVANAGARI_RE.findall(text)) / len(non_space)


def latin_ratio(text: str) -> float:
    """Share of alphabetic characters that are Latin rather than Devanagari.

    Computed over letters only, so digits and punctuation do not distort the measure.
    """
    latin = len(LATIN_RE.findall(text))
    devanagari = len(DEVANAGARI_RE.findall(text))
    total = latin + devanagari
    return latin / total if total else 0.0


def drop_foreign_lines(
    text: str, max_latin_ratio: float = 0.8, min_words: int = 4
) -> str:
    """Remove lines that are predominantly Latin script.

    This targets residue that survives HTML extraction — product-recommendation widgets,
    "related gadgets" strips and similar furniture that appear as whole English lines
    inside an otherwise Hindi or Nepali article. Measured on the Hindi corpus, such lines
    are 0.5% of the total and are consistently boilerplate rather than prose.

    It deliberately does **not** remove foreign *words*. Code-mixing is a genuine feature
    of Indian journalism — English party names, brands and headline prefixes appear in
    62% of Hindi documents — and deleting individual words would leave ungrammatical
    fragments while erasing a real property of the language the model is meant to learn.
    The ``min_words`` floor protects short inline mentions for the same reason.

    Args:
        text: Document text, one paragraph per line.
        max_latin_ratio: Drop a line whose letters are more than this fraction Latin.
        min_words: Only consider lines with at least this many words, so brief inline
            English (a product name, an acronym) is never removed.

    Returns:
        Text with predominantly-Latin lines removed.
    """
    kept = []
    for line in text.split("\n"):
        words = line.split()
        if len(words) >= min_words and latin_ratio(line) > max_latin_ratio:
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_text(text: str) -> str:
    """Strip whitespace scaffolding, leaked JSON and known boilerplate from a document.

    Args:
        text: Raw text as returned by the extractor.

    Returns:
        Cleaned text with one logical line per surviving paragraph, no blank runs.
    """
    kept: list[str] = []

    for raw_line in text.split("\n"):
        line = INLINE_SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            continue
        if JSON_LINE_RE.match(line):
            continue
        if BOILERPLATE_RE.search(line):
            continue
        kept.append(line)

    return "\n".join(kept)


def prose_word_count(text: str) -> int:
    """Count words that sit on genuine prose lines, ignoring short fragment lines.

    Counting only prose lines stops a page made of fifty two-word navigation links from
    passing a naive total-word-count threshold.
    """
    total = 0
    for line in text.split("\n"):
        words = line.split()
        if len(words) >= PROSE_LINE_MIN_WORDS:
            total += len(words)
    return total


def is_good_document(
    text: str,
    min_prose_words: int = 120,
    min_prose_lines: int = 3,
) -> bool:
    """Decide whether cleaned text is a usable article.

    Args:
        text: Text that has already been through :func:`clean_text`.
        min_prose_words: Minimum words sitting on prose lines.
        min_prose_lines: Minimum number of distinct prose lines, which rejects single
            run-on blobs such as a page of concatenated headlines.

    Returns:
        True if the document should be kept.
    """
    prose_lines = [l for l in text.split("\n") if len(l.split()) >= PROSE_LINE_MIN_WORDS]
    if len(prose_lines) < min_prose_lines:
        return False
    return prose_word_count(text) >= min_prose_words
