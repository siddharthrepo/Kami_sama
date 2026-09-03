"""Backfill the ``scheduler`` block into checkpoints written before it existed.

Why this exists
---------------
The learning-rate schedule in this project is a pure function of the step number, so
resuming has never needed stored scheduler state — the step alone is sufficient, and
that was the original design. The project specification, however, lists scheduler state
as a mandatory checkpoint field in its own right. ``lma/checkpoint.py`` now writes it,
but the Hindi run was already underway when that change landed. Rather than retrain, or
ship a Hindi checkpoint whose contents differ from the Nepali one, this script adds the
missing block to existing files.

The values are reconstructed from the ``train_config`` and ``step`` already inside the
checkpoint, so nothing is invented: the block written here is byte-for-byte what
``save_checkpoint`` would have produced at that step. Files that already carry a
``scheduler`` key are left untouched.

Usage
-----
    python -m scripts.upgrade_checkpoint --dry-run checkpoints/hindi
    python -m scripts.upgrade_checkpoint checkpoints/hindi
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from lma.config import TrainConfig
from lma.schedule import scheduler_state


def upgrade(path: Path, *, dry_run: bool) -> str:
    """Add a ``scheduler`` block to one checkpoint file.

    Args:
        path: A ``.pt`` checkpoint written by ``save_checkpoint``.
        dry_run: If true, report what would change without writing.

    Returns:
        A one-line human-readable description of the outcome.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)

    if "scheduler" in payload:
        return f"{path.name}: already has scheduler state, skipped"

    if "train_config" not in payload or "step" not in payload:
        return f"{path.name}: missing train_config/step, cannot upgrade"

    train = TrainConfig(**{
        k: v for k, v in payload["train_config"].items()
        if k in TrainConfig.__dataclass_fields__
    })
    step = int(payload["step"])
    block = scheduler_state(
        step,
        peak_lr=train.learning_rate,
        warmup_steps=train.warmup_steps,
        max_steps=train.max_steps,
        min_lr_ratio=train.min_lr_ratio,
    )

    if dry_run:
        return f"{path.name}: would add scheduler at step {step:,}, lr {block['last_lr']:.3e}"

    # Rebuild the dict so "scheduler" sits next to "optimizer", matching the key order
    # a freshly written checkpoint has. Ordering is cosmetic but makes a diff of two
    # checkpoints readable.
    rebuilt = {}
    for key, value in payload.items():
        rebuilt[key] = value
        if key == "optimizer":
            rebuilt["scheduler"] = block
    if "scheduler" not in rebuilt:
        rebuilt["scheduler"] = block

    # Same atomic-rename discipline as save_checkpoint: a half-written checkpoint is
    # worse than no checkpoint, and this script overwrites files that took hours to make.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(rebuilt, tmp)
    os.replace(tmp, path)
    return f"{path.name}: added scheduler at step {step:,}, lr {block['last_lr']:.3e}"


def refresh_sidecar(directory: Path, *, dry_run: bool) -> str:
    """Add the scheduler block to a checkpoint directory's ``config.json`` sidecar.

    Args:
        directory: Directory holding ``config.json`` and the ``.pt`` files.
        dry_run: If true, report without writing.

    Returns:
        A one-line description of the outcome.
    """
    sidecar = directory / "config.json"
    if not sidecar.exists():
        return "config.json: absent, nothing to refresh"

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if "scheduler" in data:
        return "config.json: already has scheduler state, skipped"
    if "train" not in data or "last_step" not in data:
        return "config.json: unexpected shape, skipped"

    train = TrainConfig(**{
        k: v for k, v in data["train"].items()
        if k in TrainConfig.__dataclass_fields__
    })
    block = scheduler_state(
        int(data["last_step"]),
        peak_lr=train.learning_rate,
        warmup_steps=train.warmup_steps,
        max_steps=train.max_steps,
        min_lr_ratio=train.min_lr_ratio,
    )
    if dry_run:
        return f"config.json: would add scheduler at step {data['last_step']:,}"

    rebuilt = {}
    for key, value in data.items():
        rebuilt[key] = value
        if key == "parameter_counts":
            rebuilt["scheduler"] = block
    if "scheduler" not in rebuilt:
        rebuilt["scheduler"] = block
    sidecar.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")
    return f"config.json: added scheduler at step {data['last_step']:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory", type=Path,
                        help="Checkpoint directory, e.g. checkpoints/hindi")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing anything")
    args = parser.parse_args()

    if not args.directory.is_dir():
        raise SystemExit(f"not a directory: {args.directory}")

    files = sorted(args.directory.glob("*.pt"))
    if not files:
        raise SystemExit(f"no .pt checkpoints in {args.directory}")

    print(f"{'DRY RUN — ' if args.dry_run else ''}{len(files)} checkpoint(s) "
          f"in {args.directory}\n")
    for path in files:
        print("  " + upgrade(path, dry_run=args.dry_run))
    print("  " + refresh_sidecar(args.directory, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
