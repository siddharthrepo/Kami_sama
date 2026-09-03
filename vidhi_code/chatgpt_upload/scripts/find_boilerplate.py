"""Find lines that repeat across the corpus, so cleaning can remove them.

The per-document filters in ``common/clean.py`` catch boilerplate we thought to write a
pattern for. This script catches the boilerplate we did not: any line that appears across
many separate documents is, almost by definition, site furniture rather than prose.

Measured on the Hindi corpus, lines repeating ten or more times account for 4.72% of all
lines; on Nepali, 1.17%. Inspection shows they are exactly what you would expect — table
headers from election result pages, "also read" promotional links, journalist bylines and
news-agency attributions.

**Why frequency alone is a safe rule here.** A worry when doing this is deleting genuine
content: a name like ``सलमान खान`` appears in hundreds of articles. But it only appears as
a *standalone line* when it is a photo caption or a tag — real prose mentioning him is a
full sentence, which will not repeat verbatim. Because this operates on whole lines and
requires an exact match after normalisation, article text is not at risk.

The output is a JSON file of line hashes, consumed by ``scripts/clean.py --boilerplate``.

Example:
    python -m scripts.find_boilerplate --config hindi/configs/dataset.json --min-count 10
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

from common.normalize import normalize_text
from common.schema import read_shards


def line_hash(line: str) -> str:
    """Return a short stable hash of one line.

    Hashing rather than storing the line keeps memory bounded: a corpus of this size has
    tens of millions of distinct lines, and holding them as strings would not fit.
    """
    return hashlib.sha1(line.encode("utf-8")).hexdigest()[:12]


def count_lines(input_dirs: list[Path], limit: int = 0) -> tuple[Counter, dict, int]:
    """Count how often each normalised line appears across the corpus.

    Lines are normalised before hashing so that two copies differing only in whitespace or
    quote style are recognised as the same line — the same reason normalisation precedes
    document hashing in the cleaning pipeline.

    Args:
        input_dirs: Directories of shards to scan.
        limit: Stop after this many documents. 0 means scan everything.

    Returns:
        A ``(counts, examples, total_lines)`` triple, where ``examples`` maps a hash to one
        instance of the line text so the output is human-auditable.
    """
    counts: Counter = Counter()
    examples: dict[str, str] = {}
    total_lines = 0
    documents = 0
    started = time.monotonic()

    for input_dir in input_dirs:
        print(f"  scanning {input_dir} ...", flush=True)
        for doc in read_shards(input_dir):
            documents += 1
            for raw_line in normalize_text(doc.text).split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                digest = line_hash(line)
                counts[digest] += 1
                total_lines += 1
                # Keep one example per hash, capped so a pathological corpus cannot
                # exhaust memory through the example table alone.
                if digest not in examples and len(examples) < 4_000_000:
                    examples[digest] = line

            if documents % 200_000 == 0:
                rate = documents / max(time.monotonic() - started, 1e-6)
                print(f"    {documents:,} documents, {total_lines:,} lines, "
                      f"{len(counts):,} distinct | {rate:.0f} docs/s", flush=True)

            if limit and documents >= limit:
                break
        if limit and documents >= limit:
            break

    return counts, examples, total_lines


def run(args: argparse.Namespace) -> None:
    """Scan a corpus and write the boilerplate line list."""
    config = json.loads(Path(args.config).read_text())
    lang = config["lang"]
    stats_dir = Path(config["outputs"]["stats_dir"])
    stats_dir.mkdir(parents=True, exist_ok=True)

    input_dirs = [
        Path(config["inputs"]["manual_dir"]),
        Path(config["inputs"]["downloaded_dir"]),
    ]

    print(f"Finding repeated lines in the {config['language_name']} corpus", flush=True)
    counts, examples, total_lines = count_lines(input_dirs, args.limit)

    boilerplate = {h: c for h, c in counts.items() if c >= args.min_count}
    removed_occurrences = sum(boilerplate.values())

    out_path = Path(args.out or stats_dir / "boilerplate_lines.json")
    payload = {
        "language": lang,
        "min_count": args.min_count,
        "documents_scanned": args.limit or "all",
        "total_lines": total_lines,
        "distinct_lines": len(counts),
        "boilerplate_lines": len(boilerplate),
        "line_occurrences_removed": removed_occurrences,
        "share_of_all_lines": removed_occurrences / total_lines if total_lines else 0.0,
        "hashes": sorted(boilerplate),
        # A readable sample, so a reviewer can confirm the rule is catching furniture
        # rather than prose without having to trust the hashes.
        "top_examples": [
            {"count": c, "line": examples.get(h, "")[:160]}
            for h, c in sorted(boilerplate.items(), key=lambda kv: -kv[1])[:60]
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== boilerplate summary ({lang}) ===", flush=True)
    print(f"  lines scanned          : {total_lines:,}", flush=True)
    print(f"  distinct lines         : {len(counts):,}", flush=True)
    print(f"  lines seen >= {args.min_count} times : {len(boilerplate):,}", flush=True)
    print(f"  occurrences removed    : {removed_occurrences:,} "
          f"({removed_occurrences/max(total_lines,1):.2%} of all lines)", flush=True)
    print(f"\n  written to {out_path}", flush=True)
    print("\n  most repeated:", flush=True)
    for entry in payload["top_examples"][:12]:
        print(f"    {entry['count']:>7}x  {entry['line'][:80]!r}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True,
                        help="Path to the language's dataset config JSON.")
    parser.add_argument("--min-count", type=int, default=10,
                        help="A line appearing at least this many times is boilerplate.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Scan at most this many documents. 0 = all.")
    parser.add_argument("--out", default=None,
                        help="Override the output path for the boilerplate list.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
