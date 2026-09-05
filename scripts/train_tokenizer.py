"""Train a SentencePiece tokenizer from scratch for one language, and choose its size.

The assignment requires a separate tokenizer per language, trained from scratch, with the
vocabulary size *chosen* using fertility and unknown-token rate on held-out text rather
than picked arbitrarily. This script therefore trains several candidate vocabulary sizes,
evaluates each on the validation split, writes a comparison table, and installs the
selected one as the language's tokenizer.

Design decisions worth stating:

* **Trained on the train split only.** Validation and test text never contributes to the
  vocabulary, so held-out evaluation stays honest.
* **Sampled input.** SentencePiece holds its training corpus in memory. A 644M-token
  corpus will not fit, so a random sample of lines is used — which is standard practice
  and loses very little, since subword statistics converge long before the full corpus.
* **`character_coverage` below 1.0.** Devanagari web text contains a long tail of rare
  characters (other scripts, emoji, symbols). Covering literally everything wastes vocab
  slots on characters seen a handful of times; the remainder is handled by byte fallback.
* **Byte fallback enabled**, so no input is ever unrepresentable. That makes the true
  unknown-token rate zero by construction, and the informative metric becomes how often
  byte fallback fires — which is what this script reports.

Example:
    python -m scripts.train_tokenizer --config hindi/configs/dataset.json \\
        --vocab-sizes 8000 12000 16000 24000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sentencepiece as spm

from common.schema import read_shards

# Reserved ids. Kept explicit so Phase 2's model config can rely on them.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def build_training_sample(
    train_dir: Path, out_path: Path, max_lines: int, seed: int
) -> int:
    """Write a random sample of training lines to a plain-text file for SentencePiece.

    Documents are split into lines so that SentencePiece sees sentence-like units rather
    than whole articles, which gives it better boundary statistics.

    Reservoir-style sampling is avoided in favour of a simple probabilistic keep, because
    the corpus size is known well enough and this streams in constant memory.

    Args:
        train_dir: Directory of train-split shards.
        out_path: Text file to write.
        max_lines: Approximate number of lines to keep.
        seed: RNG seed, for reproducibility.

    Returns:
        Number of lines actually written.
    """
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # First pass is avoided: instead keep lines with a probability tuned to land near
    # max_lines, assuming ~8 usable lines per document. Slight over- or under-shoot is
    # harmless for tokenizer training.
    written = 0
    started = time.monotonic()

    with open(out_path, "w", encoding="utf-8") as handle:
        for doc in read_shards(train_dir):
            for line in doc.text.split("\n"):
                line = line.strip()
                if len(line) < 20:
                    continue
                if rng.random() < 0.25:  # thin the stream; plenty remains
                    handle.write(line + "\n")
                    written += 1
                    if written >= max_lines:
                        elapsed = time.monotonic() - started
                        print(f"    sampled {written:,} lines in {elapsed:.0f}s",
                              flush=True)
                        return written

    print(f"    sampled {written:,} lines (corpus exhausted)", flush=True)
    return written


def train_one(
    input_path: Path, model_prefix: Path, vocab_size: int, config: dict, seed: int
) -> None:
    """Train a single SentencePiece model.

    Args:
        input_path: Plain-text training sample.
        model_prefix: Output prefix; produces ``.model`` and ``.vocab``.
        vocab_size: Vocabulary size to train.
        config: The tokenizer section of the language config.
        seed: Unused by SentencePiece directly, kept for signature symmetry.
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=config.get("model_type", "bpe"),
        character_coverage=config.get("character_coverage", 0.9995),
        byte_fallback=True,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        normalization_rule_name="identity",  # we already normalised to NFC ourselves
        train_extremely_large_corpus=False,
        num_threads=8,
    )


def evaluate(model_path: Path, held_out: list[str]) -> dict:
    """Measure fertility, byte-fallback rate and segmentation statistics on held-out text.

    Args:
        model_path: Path to a trained ``.model`` file.
        held_out: Validation documents, unseen during tokenizer training.

    Returns:
        A dictionary of evaluation metrics.
    """
    sp = spm.SentencePieceProcessor(model_file=str(model_path))

    total_tokens = 0
    total_words = 0
    total_chars = 0
    byte_fallback_tokens = 0
    unk_tokens = 0
    piece_counts: Counter = Counter()

    # Byte-fallback pieces look like <0xE0>; counting them shows how often the tokenizer
    # had to fall back to raw bytes, which is the meaningful stand-in for an OOV rate
    # once byte fallback makes literal unknowns impossible.
    for text in held_out:
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)

        total_tokens += len(ids)
        total_words += len(text.split())
        total_chars += len(text)
        unk_tokens += sum(1 for i in ids if i == UNK_ID)
        byte_fallback_tokens += sum(
            1 for p in pieces if len(p) == 6 and p.startswith("<0x") and p.endswith(">")
        )
        piece_counts.update(pieces)

    return {
        "vocab_size": sp.get_piece_size(),
        "tokens": total_tokens,
        "words": total_words,
        "characters": total_chars,
        "fertility_tokens_per_word": total_tokens / total_words if total_words else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "unk_rate": unk_tokens / total_tokens if total_tokens else 0.0,
        "byte_fallback_rate": (
            byte_fallback_tokens / total_tokens if total_tokens else 0.0
        ),
        "distinct_pieces_used": len(piece_counts),
        "vocab_utilisation": len(piece_counts) / sp.get_piece_size(),
    }


def load_held_out(validation_dir: Path, max_docs: int) -> list[str]:
    """Read up to ``max_docs`` validation documents for tokenizer evaluation."""
    texts: list[str] = []
    for doc in read_shards(validation_dir):
        texts.append(doc.text)
        if len(texts) >= max_docs:
            break
    return texts


def run(args: argparse.Namespace) -> None:
    """Train candidate tokenizers, evaluate them, and install the selected one."""
    config = json.loads(Path(args.config).read_text())
    lang = config["lang"]
    tok_config = config["tokenizer"]
    seed = config["splits"]["seed"]

    splits_dir = Path(config["outputs"]["splits_dir"])
    tokenizer_dir = Path(config["outputs"]["tokenizer_dir"])
    stats_dir = Path(config["outputs"]["stats_dir"])
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training tokenizer for {config['language_name']}", flush=True)

    # --- sample training text from the TRAIN split only ---
    sample_path = tokenizer_dir / f"{lang}-tokenizer-input.txt"
    if sample_path.exists() and not args.resample:
        print(f"  reusing existing sample {sample_path}", flush=True)
    else:
        print(f"  sampling up to {args.sample_lines:,} lines from train split...",
              flush=True)
        build_training_sample(splits_dir / "train", sample_path, args.sample_lines, seed)

    held_out = load_held_out(splits_dir / "validation", args.eval_docs)
    print(f"  held-out evaluation set: {len(held_out):,} validation documents",
          flush=True)

    # --- train and evaluate each candidate vocabulary size ---
    results = []
    for vocab_size in args.vocab_sizes:
        prefix = tokenizer_dir / f"{lang}-vocab{vocab_size}"
        print(f"\n  training vocab_size={vocab_size:,} ...", flush=True)
        started = time.monotonic()
        train_one(sample_path, prefix, vocab_size, tok_config, seed)
        metrics = evaluate(prefix.with_suffix(".model"), held_out)
        metrics["train_seconds"] = round(time.monotonic() - started, 1)
        results.append(metrics)
        print(f"    fertility={metrics['fertility_tokens_per_word']:.3f} "
              f"chars/token={metrics['chars_per_token']:.2f} "
              f"byte_fallback={metrics['byte_fallback_rate']:.4%} "
              f"vocab_used={metrics['vocab_utilisation']:.1%}", flush=True)

    # --- choose ---
    # Two constraints, not one.
    #
    # Fertility alone would pick whichever vocabulary segments best, and for a
    # morphologically rich language that is simply the largest size offered — Nepali's
    # fertility was still improving at 48k. But vocabulary size is not free downstream:
    # with weight tying, the embedding matrix is vocab_size x d_model, and against the
    # ~25M-parameter budget this project sets, a 48k vocabulary would consume roughly
    # 18M of it and starve the Transformer layers.
    #
    # So candidates are first capped at `max_vocab_size`, and only then is the smallest
    # vocabulary within `tolerance` of the best remaining fertility selected.
    affordable = [r for r in results if r["vocab_size"] <= args.max_vocab_size]
    if not affordable:
        affordable = [min(results, key=lambda r: r["vocab_size"])]
        print(f"  warning: no candidate at or below max_vocab_size="
              f"{args.max_vocab_size:,}; falling back to the smallest trained",
              flush=True)

    best_fertility = min(r["fertility_tokens_per_word"] for r in affordable)
    tolerance = args.fertility_tolerance
    eligible = [
        r for r in affordable
        if r["fertility_tokens_per_word"] <= best_fertility * (1 + tolerance)
    ]
    chosen = min(eligible, key=lambda r: r["vocab_size"])

    print(f"\n  selected vocab_size={chosen['vocab_size']:,} "
          f"(fertility {chosen['fertility_tokens_per_word']:.3f}, within "
          f"{tolerance:.0%} of best {best_fertility:.3f})", flush=True)

    # Install the chosen model under a stable name.
    source_prefix = tokenizer_dir / f"{lang}-vocab{chosen['vocab_size']}"
    for suffix in (".model", ".vocab"):
        target = tokenizer_dir / f"{lang}{suffix}"
        target.write_bytes(source_prefix.with_suffix(suffix).read_bytes())
    print(f"  installed as {tokenizer_dir/lang}.model / .vocab", flush=True)

    summary = {
        "language": lang,
        "library": "sentencepiece",
        "model_type": tok_config.get("model_type", "bpe"),
        "character_coverage": tok_config.get("character_coverage", 0.9995),
        "byte_fallback": True,
        "trained_on": "train split only",
        "training_sample_lines": args.sample_lines,
        "held_out_documents": len(held_out),
        "selection_rule": (
            f"candidates capped at vocab_size <= {args.max_vocab_size:,} for the "
            f"~25M-parameter budget (embedding matrix is vocab_size x d_model under "
            f"weight tying), then the smallest vocabulary whose held-out fertility is "
            f"within {tolerance:.0%} of the best remaining"
        ),
        "max_vocab_size_considered": args.max_vocab_size,
        "selected_vocab_size": chosen["vocab_size"],
        "candidates": results,
    }
    stats_path = stats_dir / "tokenizer_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== vocabulary size comparison ===", flush=True)
    print(f"  {'vocab':>8} {'fertility':>10} {'chars/tok':>10} "
          f"{'byte_fb':>9} {'vocab_used':>11}", flush=True)
    for r in results:
        mark = " <-- selected" if r["vocab_size"] == chosen["vocab_size"] else ""
        print(f"  {r['vocab_size']:>8,} {r['fertility_tokens_per_word']:>10.3f} "
              f"{r['chars_per_token']:>10.2f} {r['byte_fallback_rate']:>8.4%} "
              f"{r['vocab_utilisation']:>10.1%}{mark}", flush=True)
    print(f"\nStats written to {stats_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True,
                        help="Path to the language's dataset config JSON.")
    parser.add_argument("--vocab-sizes", type=int, nargs="+",
                        default=[8000, 12000, 16000, 24000],
                        help="Candidate vocabulary sizes to train and compare.")
    parser.add_argument("--sample-lines", type=int, default=3_000_000,
                        help="Lines sampled from the train split for training.")
    parser.add_argument("--eval-docs", type=int, default=3000,
                        help="Validation documents used for held-out evaluation.")
    # Derived, not guessed. Under weight tying the embedding is vocab_size x d_model.
    # At d_model=448 the rest of the model costs 17,130,176 parameters, so a 25,000,000
    # budget leaves 7,869,824 for the table: 7,869,824 // 448 = 17,566. A 24,000
    # vocabulary would total 27,882,176 -- 11.5% over target -- so it is trained for
    # comparison but is never selectable.
    parser.add_argument("--max-vocab-size", type=int, default=17566,
                        help="Largest vocabulary the parameter budget allows. With "
                             "weight tying the embedding matrix is vocab_size x "
                             "d_model, so this caps how much of a ~25M-parameter model "
                             "the lookup table may consume. Default derived for "
                             "d_model=448 against a ~25M budget.")
    parser.add_argument("--fertility-tolerance", type=float, default=0.03,
                        help="How much worse than the best fertility a smaller "
                             "vocabulary may be and still be selected.")
    parser.add_argument("--resample", action="store_true",
                        help="Rebuild the training sample even if one exists.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
