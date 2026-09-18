"""Generate the Phase 3 synthetic reasoning dataset for one language.

The assignment requires a finetuning corpus built programmatically rather than
downloaded, in the model's own language, covering comparative and transitive reasoning.
This script is the whole pipeline: it partitions the entity pools, generates the three
splits, writes them, and records the statistics the report has to quote.

Two guarantees it enforces, both verified and written into the statistics file rather
than merely asserted in prose:

**No entity name is shared between splits.** The pools are partitioned first, and each
split only ever sees its own names. A model cannot succeed on test by having memorised
that some particular name tends to be the tallest.

**No prompt is repeated, within or across splits.** Deduplication is on a hash of the
prompt text, so a question cannot be both trained on and tested on.

Output goes to ``<lang>/reasoning/`` — deliberately *not* under ``<lang>/data/``, which
Phase 1 gitignores. The dataset is a few megabytes and belongs in the repository, since
the assignment grades what is on the branch.

Example:
    python -m scripts.make_reasoning_data --lang hi
    python -m scripts.make_reasoning_data --lang ne --train 24000 --val 3000 --test 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.reasoning import (
    GENERATORS,
    LANGUAGE_PACKS,
    generate_split,
    partition_pool,
)

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}

# Share of each entity pool given to each split. Train gets the most names because it
# needs the most distinct questions; val and test need only enough to build their
# smaller sets without repeating.
POOL_FRACTIONS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def write_jsonl(examples: list, path: Path) -> None:
    """Write examples to a JSONL file, one compact JSON object per line.

    Written to a temporary name and renamed, so an interrupted run cannot leave a
    truncated file that looks complete — the same guarantee the Phase 1 shard writers
    give.

    Args:
        examples: Examples to write.
        path: Final destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    tmp.rename(path)


def describe_split(examples: list) -> dict:
    """Summarise one split: sizes, template mix, attribute mix, answer balance.

    Args:
        examples: The split's examples.

    Returns:
        A JSON-serialisable summary.
    """
    prompt_chars = [len(e.prompt) for e in examples]
    equality = [e for e in examples if e.template == "numeric_equality"]

    return {
        "examples": len(examples),
        "templates": dict(sorted(Counter(e.template for e in examples).items())),
        "attributes": dict(sorted(Counter(e.attribute for e in examples).items())),
        "entities_per_example": dict(
            sorted(Counter(e.n_entities for e in examples).items())
        ),
        "reasoning_hops": dict(sorted(Counter(e.hops for e in examples).items())),
        "distinct_answers": len({e.answer for e in examples}),
        "equality_answer_balance": dict(
            sorted(Counter(e.answer for e in equality).items())
        ),
        "prompt_chars_min": min(prompt_chars) if prompt_chars else 0,
        "prompt_chars_mean": round(sum(prompt_chars) / len(prompt_chars), 1)
        if prompt_chars
        else 0,
        "prompt_chars_max": max(prompt_chars) if prompt_chars else 0,
    }


def leakage_report(splits: dict[str, list], pools: dict[str, dict[str, list]]) -> dict:
    """Verify the two leakage guarantees and record the evidence.

    Checks that entity pools are pairwise disjoint and that no prompt hash appears in
    more than one split. Both should be zero; the numbers are written to the statistics
    file so the report can cite a measurement rather than an intention.

    Args:
        splits: Split name -> examples.
        pools: Pool kind ("person"/"object") -> split name -> names.

    Returns:
        A JSON-serialisable report. ``clean`` is True only if every overlap is zero.
    """
    names = list(splits)
    overlaps: dict[str, int] = {}

    for kind, by_split in pools.items():
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = set(by_split[a]) & set(by_split[b])
                overlaps[f"{kind}_names_{a}_vs_{b}"] = len(shared)

    hashes = {name: {e.example_id for e in items} for name, items in splits.items()}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"prompts_{a}_vs_{b}"] = len(hashes[a] & hashes[b])

    return {
        "strategy": (
            "Entity-name pools are partitioned before generation, so no person or "
            "object name is shared between splits. Prompts are additionally "
            "deduplicated on a SHA-1 hash of the prompt text, within and across splits."
        ),
        "pool_sizes": {
            kind: {split: len(v) for split, v in by_split.items()}
            for kind, by_split in pools.items()
        },
        "overlaps": overlaps,
        "clean": all(v == 0 for v in overlaps.values()),
    }


def run(args: argparse.Namespace) -> None:
    """Generate, verify and write all three splits for one language."""
    pack = LANGUAGE_PACKS[args.lang]
    lang_dir = Path(LANGUAGE_DIRS[args.lang])
    out_dir = Path(args.out_dir) if args.out_dir else lang_dir / "reasoning"

    sizes = {"train": args.train, "validation": args.val, "test": args.test}

    # Partition first, from a dedicated RNG, so the split assignment of names does not
    # shift when the requested example counts change.
    pool_rng = random.Random(args.seed)
    persons = partition_pool(pack.person_names, POOL_FRACTIONS, pool_rng)
    objects = partition_pool(pack.object_names, POOL_FRACTIONS, pool_rng)

    print(f"language={args.lang}  out={out_dir}  seed={args.seed}", flush=True)
    print(f"  person names: {len(pack.person_names)}  "
          f"object names: {len(pack.object_names)}", flush=True)

    splits: dict[str, list] = {}
    for i, (name, count) in enumerate(sizes.items()):
        # A distinct seed per split, so regenerating one split cannot perturb another.
        rng = random.Random(args.seed + 1000 * (i + 1))
        splits[name] = list(
            generate_split(pack, persons[name], objects[name], count, rng)
        )
        write_jsonl(splits[name], out_dir / f"{name}.jsonl")
        print(f"  {name:11s} {len(splits[name]):>6,} examples  "
              f"-> {out_dir / (name + '.jsonl')}", flush=True)

    pools = {"person": persons, "object": objects}
    leakage = leakage_report(splits, pools)

    stats = {
        "language": args.lang,
        "seed": args.seed,
        "generator_families": sorted(GENERATORS),
        "pool_fractions": POOL_FRACTIONS,
        "splits": {name: describe_split(items) for name, items in splits.items()},
        "leakage": leakage,
    }

    report_dir = Path("report") / lang_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    stats_path = report_dir / "reasoning_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    summary_path = out_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    print(f"  statistics -> {stats_path}", flush=True)
    if leakage["clean"]:
        print("  leakage check: PASS (no shared names, no shared prompts)", flush=True)
    else:
        nonzero = {k: v for k, v in leakage["overlaps"].items() if v}
        print(f"  leakage check: FAIL {nonzero}", flush=True)
        raise SystemExit(1)

    print("\n  sample from train:", flush=True)
    for example in splits["train"][:3]:
        print(f"    {example.prompt} {example.answer}".replace("\n", " "), flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate the Phase 3 synthetic reasoning dataset."
    )
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--train", type=int, default=24_000,
                        help="training examples (default: 24000)")
    parser.add_argument("--val", type=int, default=3_000,
                        help="validation examples (default: 3000)")
    parser.add_argument("--test", type=int, default=3_000,
                        help="test examples (default: 3000)")
    parser.add_argument("--seed", type=int, default=1337,
                        help="same seed as pretraining, for consistency (default: 1337)")
    parser.add_argument("--out-dir", default=None,
                        help="default: <language>/reasoning")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
