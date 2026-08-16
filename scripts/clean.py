"""Merge, normalise, language-filter and deduplicate one language's corpus.

This is the preprocessing stage. It takes the two halves collected earlier — the manual
scrape and the downloaded public corpora — and turns them into a single clean corpus,
recording exactly what was removed at each step so the dataset statistics report can be
generated from measurements rather than assertions.

Order of operations, and why:

1. **Manual documents are processed first.** Deduplication keeps whichever copy it sees
   first, so processing the manual half before the downloaded half means that when the
   same article exists in both, the *manual* copy survives. Without this the downloaded
   copy would win and the manual token share — which the assignment requires to stay
   above 20% — would silently erode.
2. **Normalise before hashing.** Two copies of an article that differ only in quote style
   or zero-width joiners are the same document; normalising first lets exact hashing
   catch them.
3. **Language ID after normalisation**, batched, because fastText is far faster on
   batches than per document.
4. **Deduplicate last**, so the hashes are computed on final text.

Examples:
    python -m scripts.clean --lang hi --config hindi/configs/dataset.json
    python -m scripts.clean --lang ne --config nepali/configs/dataset.json --limit 50000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.clean import drop_foreign_lines, is_good_document, prose_word_count
from common.langid import LanguageIdentifier
from common.normalize import normalize_text
from common.schema import MANUAL, Document, ShardWriter, read_shards

# Documents processed per batch. Batching is what makes the fastText pass fast.
BATCH_SIZE = 2000

# Characters used for the near-duplicate prefix signature. Two news articles republished
# with different boilerplate still share their opening paragraph, so hashing a normalised
# prefix catches most real-world near-duplicates at a fraction of MinHash's cost.
PREFIX_CHARS = 300
# Subword tokens per whitespace word. Measured, not guessed: a trial SentencePiece BPE
# model gives ~1.24 on held-out Hindi at a 16k vocabulary, falling towards ~1.15 as the
# vocabulary grows. An earlier placeholder of 1.8 overstated every token count by ~45%.
# These remain *estimates* — the authoritative count comes from encoding the train split
# with the trained tokenizer in scripts/corpus_stats.py.
TOKENS_PER_WORD = 1.15


def text_hash(text: str) -> str:
    """Return a stable hash of full document text, for exact deduplication."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def prefix_hash(text: str) -> str:
    """Return a hash of the document's opening, for cheap near-duplicate detection.

    Whitespace is stripped entirely before hashing so that reformatting alone does not
    defeat the match.
    """
    prefix = "".join(text[:PREFIX_CHARS].split())
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:16]


class CorpusCleaner:
    """Runs the preprocessing pipeline for one language and records what it removed."""

    def __init__(
        self,
        lang: str,
        min_prose_words: int = 120,
        langid_min_confidence: float = 0.5,
        near_dup: bool = True,
        max_latin_ratio: float = 0.8,
    ) -> None:
        """Configure the cleaner.

        Args:
            lang: Target language code, ``"hi"`` or ``"ne"``.
            min_prose_words: Quality gate re-applied after normalisation.
            langid_min_confidence: Minimum classifier confidence to accept a document.
            near_dup: Whether to apply prefix-based near-duplicate removal.
            max_latin_ratio: Drop lines whose letters exceed this Latin fraction. This
                removes English boilerplate residue without touching code-mixed prose.
        """
        self.lang = lang
        self.min_prose_words = min_prose_words
        self.langid_min_confidence = langid_min_confidence
        self.near_dup = near_dup
        self.max_latin_ratio = max_latin_ratio

        self.identifier = LanguageIdentifier()
        self.seen_exact: set[str] = set()
        self.seen_prefix: set[str] = set()

        self.stats = {
            "input": Counter(),
            "kept": Counter(),
            "removed_quality": Counter(),
            "removed_langid": Counter(),
            "removed_exact_dup": Counter(),
            "removed_near_dup": Counter(),
            "foreign_lines_dropped": Counter(),
            "predicted_langs": Counter(),
            "words_kept": Counter(),
            # Per-source breakdowns. The assignment asks for corpus statistics by
            # source, not only by manual/downloaded, so every site and dataset is
            # counted separately.
            "input_by_source": Counter(),
            "kept_by_source": Counter(),
            "words_by_source": Counter(),
            # Document-length histogram, for the length-distribution plot.
            "doc_length_hist": Counter(),
        }

    @staticmethod
    def length_bucket(words: int) -> str:
        """Bucket a document's word count for the length-distribution histogram."""
        for upper in (200, 400, 800, 1600, 3200, 6400):
            if words < upper:
                return f"<{upper}"
        return ">=6400"

    def process_batch(self, docs: list[Document]) -> list[Document]:
        """Normalise, language-check and deduplicate one batch of documents.

        Args:
            docs: Documents to process.

        Returns:
            The subset that survived, with normalised text.
        """
        texts = []
        for doc in docs:
            normalized = normalize_text(doc.text)
            filtered = drop_foreign_lines(normalized, self.max_latin_ratio)
            dropped = normalized.count("\n") - filtered.count("\n")
            if dropped > 0:
                self.stats["foreign_lines_dropped"][doc.source_type] += dropped
            texts.append(filtered)
        predictions = self.identifier.predict_batch(texts)
        kept: list[Document] = []

        for doc, text, (predicted, confidence) in zip(docs, texts, predictions):
            key = doc.source_type
            self.stats["input"][key] += 1
            self.stats["input_by_source"][doc.source] += 1

            # Re-apply the quality gate: normalisation can shrink a document below it.
            if not is_good_document(text, min_prose_words=self.min_prose_words):
                self.stats["removed_quality"][key] += 1
                continue

            self.stats["predicted_langs"][predicted] += 1
            if predicted != self.lang or confidence < self.langid_min_confidence:
                self.stats["removed_langid"][key] += 1
                continue

            exact = text_hash(text)
            if exact in self.seen_exact:
                self.stats["removed_exact_dup"][key] += 1
                continue

            if self.near_dup:
                prefix = prefix_hash(text)
                if prefix in self.seen_prefix:
                    self.stats["removed_near_dup"][key] += 1
                    continue
                self.seen_prefix.add(prefix)

            self.seen_exact.add(exact)
            doc.text = text
            doc.doc_id = exact
            kept.append(doc)

            words = prose_word_count(text)
            self.stats["kept"][key] += 1
            self.stats["words_kept"][key] += words
            self.stats["kept_by_source"][doc.source] += 1
            self.stats["words_by_source"][doc.source] += words
            self.stats["doc_length_hist"][self.length_bucket(words)] += 1

        return kept

    def run(self, input_dirs: list[Path], writer: ShardWriter, limit: int = 0) -> None:
        """Stream every input directory through the pipeline, in order.

        Args:
            input_dirs: Directories of shards. **Order matters** — put the manual
                directory first so manual documents win deduplication ties.
            writer: Destination for surviving documents.
            limit: Stop after reading this many documents. 0 means no limit.
        """
        started = time.monotonic()
        read = 0
        batch: list[Document] = []

        for input_dir in input_dirs:
            print(f"  reading {input_dir} ...", flush=True)
            for doc in read_shards(input_dir):
                batch.append(doc)
                read += 1

                if len(batch) >= BATCH_SIZE:
                    for kept_doc in self.process_batch(batch):
                        writer.write(kept_doc)
                    batch = []

                    if read % (BATCH_SIZE * 25) == 0:
                        self._progress(read, started)

                if limit and read >= limit:
                    break
            if limit and read >= limit:
                break

        for kept_doc in self.process_batch(batch):
            writer.write(kept_doc)
        self._progress(read, started)

    def _progress(self, read: int, started: float) -> None:
        """Print a one-line progress summary."""
        elapsed = max(time.monotonic() - started, 1e-6)
        kept = sum(self.stats["kept"].values())
        words = sum(self.stats["words_kept"].values())
        print(
            f"    read {read:,} | kept {kept:,} ({kept/max(read,1):.0%}) | "
            f"~{words*TOKENS_PER_WORD/1e6:.0f}M tokens | {read/elapsed:.0f} docs/s",
            flush=True,
        )

    def summary(self) -> dict:
        """Assemble the cleaning statistics, including the manual/downloaded split."""
        kept = self.stats["kept"]
        words = self.stats["words_kept"]
        total_words = sum(words.values())

        return {
            "language": self.lang,
            "documents": {
                "input": dict(self.stats["input"]),
                "kept": dict(kept),
                "removed_quality": dict(self.stats["removed_quality"]),
                "removed_langid": dict(self.stats["removed_langid"]),
                "removed_exact_duplicate": dict(self.stats["removed_exact_dup"]),
                "removed_near_duplicate": dict(self.stats["removed_near_dup"]),
            },
            "foreign_lines_dropped": dict(self.stats["foreign_lines_dropped"]),
            "_note_foreign_lines": (
                "Predominantly-Latin lines removed as boilerplate residue "
                "(product widgets, 'related gadgets' strips). Foreign WORDS are "
                "deliberately retained: code-mixing is genuine usage in Indian "
                "journalism, and removing words would leave ungrammatical text."
            ),
            "by_source": {
                source: {
                    "input": self.stats["input_by_source"][source],
                    "kept": self.stats["kept_by_source"][source],
                    "words_kept": self.stats["words_by_source"][source],
                    "estimated_tokens": round(self.stats["words_by_source"][source] * TOKENS_PER_WORD),
                    "keep_rate": (
                        self.stats["kept_by_source"][source]
                        / self.stats["input_by_source"][source]
                        if self.stats["input_by_source"][source] else 0.0
                    ),
                }
                for source in sorted(self.stats["input_by_source"])
            },
            "document_length_histogram_words": dict(
                sorted(self.stats["doc_length_hist"].items())
            ),
            "words_kept": dict(words),
            "estimated_tokens": {k: round(v * TOKENS_PER_WORD) for k, v in words.items()},
            "manual_fraction_of_words": (
                words.get(MANUAL, 0) / total_words if total_words else 0.0
            ),
            "predicted_language_distribution": dict(
                self.stats["predicted_langs"].most_common(12)
            ),
        }


def run(args: argparse.Namespace) -> None:
    """Execute cleaning according to parsed CLI arguments."""
    config = json.loads(Path(args.config).read_text())
    lang = config["lang"]
    cleaning = config.get("cleaning", {})

    out_dir = Path(args.out or config["outputs"]["clean_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = Path(config["outputs"]["stats_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Manual first — this is what makes manual documents win deduplication ties.
    input_dirs = [
        Path(config["inputs"]["manual_dir"]),
        Path(config["inputs"]["downloaded_dir"]),
    ]

    print(f"Cleaning {config['language_name']} corpus", flush=True)
    print(f"  inputs : {[str(p) for p in input_dirs]}", flush=True)
    print(f"  output : {out_dir}", flush=True)

    cleaner = CorpusCleaner(
        lang=lang,
        min_prose_words=cleaning.get("min_prose_words", 120),
        langid_min_confidence=cleaning.get("langid_min_confidence", 0.5),
        near_dup=not args.no_near_dup,
        max_latin_ratio=cleaning.get("max_latin_ratio", 0.8),
    )

    with ShardWriter(out_dir, f"{lang}-clean", max_docs_per_shard=25_000) as writer:
        cleaner.run(input_dirs, writer, limit=args.limit)

    summary = cleaner.summary()
    stats_path = stats_dir / "cleaning_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== cleaning summary ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"\nStats written to {stats_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", choices=["hi", "ne"],
                        help="Informational; the config file is authoritative.")
    parser.add_argument("--config", required=True,
                        help="Path to the language's dataset config JSON.")
    parser.add_argument("--out", default=None,
                        help="Override the output directory from the config.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many input documents. 0 = all.")
    parser.add_argument("--no-near-dup", action="store_true",
                        help="Skip prefix-based near-duplicate removal.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
