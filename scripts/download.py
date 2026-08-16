"""Download the public-corpus share of each language's dataset from HuggingFace.

This produces the ``source_type="downloaded"`` portion — the ~80% we do not collect
ourselves. Output is written in exactly the same shard format as the scraper, so the two
halves merge later by simply reading both directories.

Everything is **streamed**. The corpora involved are far larger than the target (CC-100
Hindi alone is ~21 GB) and larger than local disk, so records are pulled one at a time,
filtered immediately, and either written or dropped. Nothing is cached whole.

Two properties matter for running this on a Kaggle session that can be killed at any
time:

* **Progress is recorded per source**, as a count of records consumed from the stream. A
  resumed run skips that many with ``islice`` and continues, so no work is repeated.
* **Shards are written atomically**, so a shard file that exists is always complete.

Examples:
    # Check which sources are actually reachable, downloading almost nothing
    python -m scripts.download --lang ne --out data/downloaded/ne --probe

    # Collect up to 300M tokens of Nepali
    python -m scripts.download --lang ne --out data/downloaded/ne --max-tokens 300000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.clean import clean_text, devanagari_ratio, is_good_document
from common.hf_sources import HFSource, get_hf_sources
from common.schema import DOWNLOADED, AtomicJsonlWriter, Document

# Subword tokens per whitespace word, used to convert a token budget into a word budget
# while streaming. This was initially guessed at 1.8, which turned out to be badly wrong:
# measured on held-out Hindi with a SentencePiece BPE vocabulary, fertility is ~1.24 at
# 16k and falls towards ~1.15 as the vocabulary grows. Devanagari segments far more
# efficiently than the guess assumed.
#
# The conservative end of that range is used deliberately. Under-estimating fertility
# means collecting more words than strictly needed, which is recoverable; over-estimating
# it means discovering a corpus shortfall only after the tokenizer exists, which is not.
TOKENS_PER_WORD = 1.15
WORDS_PER_TOKEN = 1.0 / TOKENS_PER_WORD

# Documents per shard file. Small enough that losing an in-progress shard is cheap.
DOCS_PER_SHARD = 20_000


def load_progress(path: Path) -> dict:
    """Read the resume state, or return an empty state if there is none."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_progress(path: Path, progress: dict) -> None:
    """Persist resume state after each shard, so a kill loses at most one shard."""
    path.write_text(json.dumps(progress, indent=2))


def open_stream(source: HFSource, token: str | None = None):
    """Open a streaming iterator over one HuggingFace source.

    Args:
        source: The registry entry to stream.
        token: Optional HuggingFace auth token, needed for gated datasets.

    Returns:
        An iterable dataset, or None if the dataset could not be opened.
    """
    from datasets import load_dataset

    try:
        kwargs = {"split": source.split, "streaming": True}
        if source.config:
            kwargs["name"] = source.config
        if source.data_dir:
            kwargs["data_dir"] = source.data_dir
        if token:
            kwargs["token"] = token
        return load_dataset(source.dataset, **kwargs)
    except Exception as exc:
        print(f"  [{source.name}] unavailable: {type(exc).__name__}: "
              f"{str(exc)[:160]}", flush=True)
        return None


def probe_sources(sources: list[HFSource], token: str | None) -> None:
    """Report which sources can be opened and show one sample record from each.

    Run this before committing to a long download. Dataset ids and config names drift
    over time, and some corpora are gated — it is far cheaper to discover that now than
    forty minutes into a run.
    """
    print(f"Probing {len(sources)} sources...\n", flush=True)
    for source in sources:
        stream = open_stream(source, token)
        if stream is None:
            continue
        try:
            record = next(iter(stream))
        except Exception as exc:
            print(f"  [{source.name}] opened but could not read: "
                  f"{type(exc).__name__}", flush=True)
            continue

        text = record.get(source.text_field, "")
        if not isinstance(text, str):
            print(f"  [{source.name}] field {source.text_field!r} is not text; "
                  f"available fields: {list(record)[:8]}", flush=True)
            continue

        words = len(text.split())
        print(f"  [{source.name}] OK | fields={list(record)[:6]} | "
              f"first doc {words} words | devanagari={devanagari_ratio(text):.2f}",
              flush=True)
        print(f"      {text[:120]!r}", flush=True)


def collect_source(
    source: HFSource,
    lang: str,
    out_dir: Path,
    progress: dict,
    progress_path: Path,
    token: str | None,
    remaining_words: float,
    min_prose_words: int,
    min_script_ratio: float,
) -> int:
    """Stream one source until it is exhausted or the word budget is met.

    Args:
        source: Registry entry to stream.
        lang: Target language code stamped on every document.
        out_dir: Directory for shard files.
        progress: Mutable resume state, keyed by source name.
        progress_path: Where to persist ``progress``.
        token: Optional HuggingFace auth token.
        remaining_words: Word budget still to fill across all sources.
        min_prose_words: Quality gate applied to cleaned text.
        min_script_ratio: Minimum Devanagari fraction.

    Returns:
        Words kept from this source during this run.
    """
    stream = open_stream(source, token)
    if stream is None:
        return 0

    state = progress.setdefault(source.name, {"consumed": 0, "kept": 0, "shard": 0})
    if state["consumed"]:
        print(f"  [{source.name}] resuming, skipping {state['consumed']:,} records",
              flush=True)
        stream = islice(iter(stream), state["consumed"], None)
    else:
        stream = iter(stream)

    words_this_run = 0
    started = time.monotonic()
    buffer: list[Document] = []

    def flush(buffer: list[Document]) -> None:
        """Write one shard atomically and record progress."""
        if not buffer:
            return
        path = out_dir / f"shard-{source.name}-{state['shard']:05d}.jsonl.zst"
        with AtomicJsonlWriter(path) as writer:
            for doc in buffer:
                writer.write(doc)
        state["shard"] += 1
        save_progress(progress_path, progress)

    try:
        for record in stream:
            state["consumed"] += 1

            raw = record.get(source.text_field, "")
            if not isinstance(raw, str) or not raw:
                continue

            text = clean_text(raw)
            if not is_good_document(text, min_prose_words=min_prose_words):
                continue
            if devanagari_ratio(text) < min_script_ratio:
                continue

            buffer.append(
                Document(
                    text=text,
                    lang=lang,
                    source_type=DOWNLOADED,
                    source=source.name,
                    url=record.get("url", "") if isinstance(record.get("url"), str) else "",
                )
            )
            words = len(text.split())
            words_this_run += words
            state["kept"] += 1

            if len(buffer) >= DOCS_PER_SHARD:
                flush(buffer)
                buffer = []
                rate = words_this_run / max(time.monotonic() - started, 1e-6)
                print(f"    [{source.name}] {state['kept']:,} docs | "
                      f"{words_this_run/1e6:.1f}M words | "
                      f"~{words_this_run*TOKENS_PER_WORD/1e6:.1f}M tokens | "
                      f"{rate/1e3:.0f}k words/s", flush=True)

            if words_this_run >= remaining_words:
                print(f"  [{source.name}] budget met", flush=True)
                break
    except KeyboardInterrupt:
        print(f"  [{source.name}] interrupted, flushing partial shard", flush=True)
        raise
    finally:
        flush(buffer)
        save_progress(progress_path, progress)

    print(f"  [{source.name}] kept {state['kept']:,} docs, "
          f"~{words_this_run*TOKENS_PER_WORD/1e6:.1f}M tokens this run", flush=True)
    return words_this_run


def run(args: argparse.Namespace) -> None:
    """Probe or download according to parsed CLI arguments."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "download-progress.json"

    sources = get_hf_sources(args.lang, args.sources)
    token = args.hf_token or None

    if args.probe:
        probe_sources(sources, token)
        return

    progress = load_progress(progress_path)
    target_words = args.max_tokens * WORDS_PER_TOKEN
    done_words = sum(s.get("words", 0) for s in progress.values())

    print(f"Target ~{args.max_tokens/1e6:.0f}M tokens "
          f"(~{target_words/1e6:.0f}M words) for lang={args.lang}", flush=True)

    total_words = done_words
    for source in sources:
        if total_words >= target_words:
            print("Token target met; stopping.", flush=True)
            break
        print(f"\nStreaming {source.name} ({source.dataset})...", flush=True)
        got = collect_source(
            source, args.lang, out_dir, progress, progress_path, token,
            remaining_words=target_words - total_words,
            min_prose_words=args.min_prose_words,
            min_script_ratio=args.min_script_ratio,
        )
        total_words += got
        progress[source.name]["words"] = progress[source.name].get("words", 0) + got
        save_progress(progress_path, progress)

    print(f"\nDone. ~{total_words*TOKENS_PER_WORD/1e6:.1f}M tokens across "
          f"{len(list(out_dir.glob('*.jsonl.zst')))} shards.", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", required=True, choices=["hi", "ne"])
    parser.add_argument("--out", required=True,
                        help="Output directory for shards and resume state.")
    parser.add_argument("--sources", nargs="*", default=None,
                        help="Restrict to named HF sources.")
    parser.add_argument("--max-tokens", type=int, default=400_000_000,
                        help="Approximate token budget to stop at.")
    parser.add_argument("--min-prose-words", type=int, default=120,
                        help="Quality gate, applied after cleaning.")
    parser.add_argument("--min-script-ratio", type=float, default=0.5,
                        help="Minimum Devanagari character fraction.")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token, required for gated datasets.")
    parser.add_argument("--probe", action="store_true",
                        help="Report which sources are reachable, then exit.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
