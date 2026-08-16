"""Partition a cleaned corpus into train / validation / test splits.

Two properties matter here, and both are easy to get wrong.

**Splitting is at the document level, never the line level.** If sentences from one
article were scattered across train and test, the model would be evaluated on text
whose surrounding context it had already memorised, and every downstream number —
perplexity, BPB, generation quality — would be optimistically biased.

**Assignment is deterministic, from a hash of the document id.** Rather than shuffling
the corpus (which would need the whole thing in memory) each document's split is decided
by hashing its id together with a fixed seed. The same document always lands in the same
split, the result is reproducible from the config alone, and the corpus can be streamed.

The tokenizer is trained afterwards on the **train split only**, so that validation and
test text never influences the vocabulary.

Example:
    python -m scripts.split --config hindi/configs/dataset.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.clean import prose_word_count
from common.schema import MANUAL, ShardWriter, read_shards

# Largest value of the hash prefix we take, used to map a hash into [0, 1).
HASH_SCALE = 0xFFFFFFFF
# Subword tokens per whitespace word. Measured, not guessed: a trial SentencePiece BPE
# model gives ~1.24 on held-out Hindi at a 16k vocabulary, falling towards ~1.15 as the
# vocabulary grows. An earlier placeholder of 1.8 overstated every token count by ~45%.
# These remain *estimates* — the authoritative count comes from encoding the train split
# with the trained tokenizer in scripts/corpus_stats.py.
TOKENS_PER_WORD = 1.15


def assign_split(doc_id: str, seed: int, train: float, validation: float) -> str:
    """Deterministically assign one document to a split.

    Hashing ``seed:doc_id`` gives a value that is stable across runs and uniformly
    distributed, so the split proportions come out right without any shuffling.

    Args:
        doc_id: The document's content hash.
        seed: Split seed from the config; changing it reshuffles every assignment.
        train: Fraction of documents for training.
        validation: Fraction for validation. The remainder becomes test.

    Returns:
        ``"train"``, ``"validation"`` or ``"test"``.
    """
    digest = hashlib.sha1(f"{seed}:{doc_id}".encode("utf-8")).hexdigest()[:8]
    position = int(digest, 16) / HASH_SCALE

    if position < train:
        return "train"
    if position < train + validation:
        return "validation"
    return "test"


def run(args: argparse.Namespace) -> None:
    """Split a cleaned corpus and write per-split statistics."""
    config = json.loads(Path(args.config).read_text())
    lang = config["lang"]
    split_config = config["splits"]

    clean_dir = Path(config["outputs"]["clean_dir"])
    splits_dir = Path(args.out or config["outputs"]["splits_dir"])
    stats_dir = Path(config["outputs"]["stats_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    train_frac = split_config["train"]
    val_frac = split_config["validation"]
    seed = split_config["seed"]

    print(f"Splitting {config['language_name']} corpus", flush=True)
    print(f"  input  : {clean_dir}", flush=True)
    print(f"  output : {splits_dir}", flush=True)
    print(f"  ratios : train={train_frac} validation={val_frac} "
          f"test={round(1 - train_frac - val_frac, 6)} seed={seed}", flush=True)

    writers = {
        name: ShardWriter(splits_dir / name, f"{lang}-{name}", max_docs_per_shard=25_000)
        for name in ("train", "validation", "test")
    }

    docs = Counter()
    words = Counter()
    words_by_type: dict[str, Counter] = {
        name: Counter() for name in ("train", "validation", "test")
    }
    sources_by_split: dict[str, Counter] = {
        name: Counter() for name in ("train", "validation", "test")
    }

    try:
        for index, doc in enumerate(read_shards(clean_dir), start=1):
            split = assign_split(doc.doc_id, seed, train_frac, val_frac)
            writers[split].write(doc)

            count = prose_word_count(doc.text)
            docs[split] += 1
            words[split] += count
            words_by_type[split][doc.source_type] += count
            sources_by_split[split][doc.source] += 1

            if index % 100_000 == 0:
                print(f"    {index:,} documents assigned", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    total_words = sum(words.values())
    summary = {
        "language": lang,
        "seed": seed,
        "granularity": "document",
        "ratios_requested": {
            "train": train_frac,
            "validation": val_frac,
            "test": round(1 - train_frac - val_frac, 6),
        },
        "splits": {
            name: {
                "documents": docs[name],
                "words": words[name],
                "estimated_tokens": round(words[name] * TOKENS_PER_WORD),
                "document_share": docs[name] / max(sum(docs.values()), 1),
                "manual_words": words_by_type[name].get(MANUAL, 0),
                "manual_fraction_of_words": (
                    words_by_type[name].get(MANUAL, 0) / words[name] if words[name] else 0.0
                ),
                "by_source": dict(sources_by_split[name]),
            }
            for name in ("train", "validation", "test")
        },
        "total_documents": sum(docs.values()),
        "total_words": total_words,
        "total_estimated_tokens": round(total_words * TOKENS_PER_WORD),
        "_note": (
            "Splits are assigned per document by hashing seed:doc_id, so the partition "
            "is reproducible and no article is divided across splits. The tokenizer is "
            "trained on the train split only, to keep validation and test text out of "
            "the vocabulary."
        ),
    }

    stats_path = stats_dir / "split_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== split summary ===", flush=True)
    for name in ("train", "validation", "test"):
        entry = summary["splits"][name]
        print(f"  {name:11} {entry['documents']:>9,} docs | "
              f"{entry['estimated_tokens']/1e6:>7.1f}M tokens | "
              f"{entry['document_share']:.2%} | "
              f"manual {entry['manual_fraction_of_words']:.2%}", flush=True)
    print(f"\nStats written to {stats_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True,
                        help="Path to the language's dataset config JSON.")
    parser.add_argument("--out", default=None,
                        help="Override the splits directory from the config.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
