"""Measure the final corpus with the trained tokenizer, and produce the report figures.

Everything before this stage worked in *estimated* tokens — word counts multiplied by an
assumed fertility. This script produces the real numbers by encoding the corpus with the
tokenizer that was actually trained on it, which is the only definition of "token" the
assignment's ~500M target can refer to.

It answers, per language:

* How many tokens the corpus really contains, split by **manual vs downloaded** and by
  **individual source** — the evidence for the >=20% manual-collection requirement.
* Fertility (tokens per word), characters per token, and byte-fallback rate on held-out
  text.
* Token frequency structure, including the Zipf curve and how much of the vocabulary is
  actually used.
* Worked tokenization examples, so a reader can see the segmentation rather than trust it.

Plots are written with a title, axis labels and a legend where applicable, as the
assignment requires.

Example:
    python -m scripts.corpus_stats --config hindi/configs/dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt

import sentencepiece as spm

from common.schema import MANUAL, read_shards

# Sentences shown as worked tokenization examples in the report.
EXAMPLE_SENTENCES = {
    "hi": [
        "भारत में शिक्षा का अधिकार एक मौलिक अधिकार है।",
        "प्रधानमंत्री ने नई दिल्ली में एक बैठक की अध्यक्षता की।",
        "iPhone 15 की कीमत भारत में 79,900 रुपये है।",
    ],
    "ne": [
        "नेपालमा शिक्षाको अधिकार मौलिक अधिकार हो।",
        "प्रधानमन्त्रीले काठमाडौंमा एउटा बैठकको अध्यक्षता गरे।",
        "नेपाल र भारतबीचको सम्बन्ध ऐतिहासिक छ।",
    ],
}


def measure_split(
    sp: spm.SentencePieceProcessor, split_dir: Path, count_pieces: bool = False
) -> dict:
    """Encode every document in a split and accumulate token statistics.

    Provenance is preserved because each document still carries the ``source_type`` and
    ``source`` fields stamped on it at collection time. Tokenizing does not touch that
    metadata, so tokens can be attributed back to where the text came from.

    Args:
        sp: The trained tokenizer.
        split_dir: Directory of split shards.
        count_pieces: Whether to accumulate a full piece-frequency table. Only needed
            for the train split, where the Zipf plot comes from.

    Returns:
        Counts of documents, words, characters and tokens, broken down by provenance.
    """
    stats = {
        "documents": 0,
        "words": 0,
        "characters": 0,
        "tokens": 0,
        "tokens_by_type": Counter(),
        "tokens_by_source": Counter(),
        "documents_by_source": Counter(),
        "words_by_type": Counter(),
    }
    piece_counts: Counter = Counter()

    for index, doc in enumerate(read_shards(split_dir), start=1):
        if count_pieces:
            pieces = sp.encode(doc.text, out_type=str)
            n_tokens = len(pieces)
            piece_counts.update(pieces)
        else:
            n_tokens = len(sp.encode(doc.text, out_type=int))

        n_words = len(doc.text.split())
        stats["documents"] += 1
        stats["words"] += n_words
        stats["characters"] += len(doc.text)
        stats["tokens"] += n_tokens
        stats["tokens_by_type"][doc.source_type] += n_tokens
        stats["tokens_by_source"][doc.source] += n_tokens
        stats["documents_by_source"][doc.source] += 1
        stats["words_by_type"][doc.source_type] += n_words

        if index % 100_000 == 0:
            print(f"    {index:,} documents encoded "
                  f"({stats['tokens']/1e6:.0f}M tokens)", flush=True)

    stats["piece_counts"] = piece_counts
    return stats


def tokenization_examples(sp: spm.SentencePieceProcessor, lang: str) -> list[dict]:
    """Produce worked tokenization examples for the report."""
    examples = []
    for sentence in EXAMPLE_SENTENCES.get(lang, []):
        pieces = sp.encode(sentence, out_type=str)
        examples.append({
            "text": sentence,
            "pieces": pieces,
            "ids": sp.encode(sentence, out_type=int),
            "n_words": len(sentence.split()),
            "n_tokens": len(pieces),
            "fertility": len(pieces) / max(len(sentence.split()), 1),
        })
    return examples


# ------------------------------------------------------------------------- plots

def plot_source_breakdown(stats: dict, lang_name: str, out_path: Path) -> None:
    """Horizontal bar chart of token contribution per source, coloured by provenance."""
    sources = sorted(stats["tokens_by_source"], key=stats["tokens_by_source"].get)
    values = [stats["tokens_by_source"][s] / 1e6 for s in sources]
    # Manual sources are the ones we scraped; everything else was downloaded.
    manual_names = {"jansatta", "thewirehindi", "onlinekhabar"}
    colours = ["#2a9d8f" if s in manual_names else "#264653" for s in sources]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(sources, values, color=colours)
    ax.set_title(f"{lang_name}: token contribution by source")
    ax.set_xlabel("Tokens (millions)")
    ax.set_ylabel("Source")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2a9d8f"),
        plt.Rectangle((0, 0), 1, 1, color="#264653"),
    ]
    ax.legend(handles, ["Manual (scraped)", "Downloaded (public corpora)"],
              loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_zipf(piece_counts: Counter, lang_name: str, out_path: Path) -> None:
    """Log-log rank-frequency plot, the standard check that a corpus looks like language."""
    frequencies = sorted(piece_counts.values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ranks, frequencies, color="#e76f51", label="Observed token frequency")
    # A true Zipf distribution is a straight line of slope -1 on log-log axes.
    if frequencies:
        reference = [frequencies[0] / r for r in ranks]
        ax.loglog(ranks, reference, "--", color="#264653",
                  label="Zipf reference (slope -1)")
    ax.set_title(f"{lang_name}: token frequency distribution (Zipf)")
    ax.set_xlabel("Token rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_histogram(histogram: dict, lang_name: str, out_path: Path) -> None:
    """Bar chart of document lengths in words."""
    order = ["<200", "<400", "<800", "<1600", "<3200", "<6400", ">=6400"]
    labels = [b for b in order if b in histogram]
    values = [histogram[b] for b in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="#457b9d", label="Documents")
    ax.set_title(f"{lang_name}: document length distribution")
    ax.set_xlabel("Document length (words)")
    ax.set_ylabel("Number of documents")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fertility_vs_vocab(candidates: list[dict], lang_name: str, out_path: Path) -> None:
    """Fertility and byte-fallback rate against vocabulary size.

    This is the evidence behind the vocabulary-size choice the assignment asks for.
    """
    sizes = [c["vocab_size"] for c in candidates]
    fertility = [c["fertility_tokens_per_word"] for c in candidates]
    fallback = [c["byte_fallback_rate"] * 100 for c in candidates]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, fertility, "o-", color="#2a9d8f", label="Fertility (tokens/word)")
    ax.set_title(f"{lang_name}: vocabulary size vs fertility and byte fallback")
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Fertility (tokens per word)")

    twin = ax.twinx()
    twin.plot(sizes, fallback, "s--", color="#e76f51", label="Byte-fallback rate (%)")
    twin.set_ylabel("Byte-fallback rate (%)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main

def run(args: argparse.Namespace) -> None:
    """Measure the corpus, write statistics JSON, and render all report figures."""
    config = json.loads(Path(args.config).read_text())
    lang = config["lang"]
    lang_name = config["language_name"]

    splits_dir = Path(config["outputs"]["splits_dir"])
    tokenizer_dir = Path(config["outputs"]["tokenizer_dir"])
    stats_dir = Path(config["outputs"]["stats_dir"])
    figures_dir = stats_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = tokenizer_dir / f"{lang}.model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No tokenizer at {model_path}. Run scripts.train_tokenizer first."
        )
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    print(f"Measuring {lang_name} corpus with {model_path.name} "
          f"(vocab {sp.get_piece_size():,})", flush=True)

    # --- encode every split ---
    measurements = {}
    for split in ("train", "validation", "test"):
        print(f"  encoding {split} split ...", flush=True)
        measurements[split] = measure_split(
            sp, splits_dir / split, count_pieces=(split == "train")
        )

    train = measurements["train"]
    total_tokens = sum(m["tokens"] for m in measurements.values())
    manual_tokens = sum(m["tokens_by_type"].get(MANUAL, 0) for m in measurements.values())

    summary = {
        "language": lang,
        "language_name": lang_name,
        "tokenizer": {
            "model": str(model_path),
            "vocab_size": sp.get_piece_size(),
        },
        "corpus_totals": {
            "documents": sum(m["documents"] for m in measurements.values()),
            "words": sum(m["words"] for m in measurements.values()),
            "characters": sum(m["characters"] for m in measurements.values()),
            "tokens": total_tokens,
            "manual_tokens": manual_tokens,
            "downloaded_tokens": total_tokens - manual_tokens,
            "manual_token_fraction": manual_tokens / total_tokens if total_tokens else 0.0,
            "meets_500M_target": total_tokens >= 500_000_000,
            "meets_20pct_manual": (
                (manual_tokens / total_tokens if total_tokens else 0.0) >= 0.20
            ),
        },
        "splits": {
            name: {
                "documents": m["documents"],
                "words": m["words"],
                "tokens": m["tokens"],
                "tokens_by_source_type": dict(m["tokens_by_type"]),
                "tokens_by_source": dict(m["tokens_by_source"]),
                "fertility_tokens_per_word": m["tokens"] / m["words"] if m["words"] else 0.0,
                "chars_per_token": m["characters"] / m["tokens"] if m["tokens"] else 0.0,
            }
            for name, m in measurements.items()
        },
        "token_frequency": {
            "distinct_tokens_used": len(train["piece_counts"]),
            "vocab_utilisation": len(train["piece_counts"]) / sp.get_piece_size(),
            "top_20_tokens": [
                {"piece": p, "count": c}
                for p, c in train["piece_counts"].most_common(20)
            ],
            "hapax_tokens": sum(1 for c in train["piece_counts"].values() if c == 1),
        },
        "tokenization_examples": tokenization_examples(sp, lang),
    }

    stats_path = stats_dir / "corpus_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # --- figures ---
    print("  rendering figures ...", flush=True)
    plot_source_breakdown(train, lang_name, figures_dir / f"{lang}_source_breakdown.png")
    plot_zipf(train["piece_counts"], lang_name, figures_dir / f"{lang}_zipf.png")

    cleaning_stats_path = stats_dir / "cleaning_stats.json"
    if cleaning_stats_path.exists():
        cleaning = json.loads(cleaning_stats_path.read_text())
        histogram = cleaning.get("document_length_histogram_words", {})
        if histogram:
            plot_length_histogram(
                histogram, lang_name, figures_dir / f"{lang}_doc_lengths.png"
            )

    tokenizer_stats_path = stats_dir / "tokenizer_stats.json"
    if tokenizer_stats_path.exists():
        tok = json.loads(tokenizer_stats_path.read_text())
        if tok.get("candidates"):
            plot_fertility_vs_vocab(
                tok["candidates"], lang_name, figures_dir / f"{lang}_fertility.png"
            )

    # --- report to console ---
    totals = summary["corpus_totals"]
    print(f"\n=== {lang_name} final corpus ===", flush=True)
    print(f"  documents        : {totals['documents']:,}", flush=True)
    print(f"  words            : {totals['words']:,}", flush=True)
    print(f"  TOKENS           : {totals['tokens']:,} "
          f"({totals['tokens']/1e6:.1f}M)  "
          f"{'OK' if totals['meets_500M_target'] else 'SHORT OF 500M'}", flush=True)
    print(f"  manual tokens    : {totals['manual_tokens']:,} "
          f"({totals['manual_token_fraction']:.2%})  "
          f"{'OK' if totals['meets_20pct_manual'] else 'BELOW 20%'}", flush=True)
    print(f"  fertility (train): "
          f"{summary['splits']['train']['fertility_tokens_per_word']:.3f} tokens/word",
          flush=True)
    print(f"  chars per token  : "
          f"{summary['splits']['train']['chars_per_token']:.2f}", flush=True)
    print(f"  vocab utilisation: "
          f"{summary['token_frequency']['vocab_utilisation']:.1%}", flush=True)
    print(f"\n  stats  : {stats_path}", flush=True)
    print(f"  figures: {figures_dir}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True,
                        help="Path to the language's dataset config JSON.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
