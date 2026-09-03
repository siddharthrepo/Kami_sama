"""Crash-safe checkpointing for pretraining and finetuning.

The assignment makes this mandatory: *"Your pipeline must save and resume from
intermediate checkpoints that include model weights, optimizer state, scheduler state,
training step, and configuration. A run that cannot resume after interruption will lose
marks."* It is also a practical necessity — Kaggle terminates sessions at twelve hours
and can pre-empt them sooner.

Two design points are worth stating.

**Atomic writes.** A checkpoint is written to a temporary file and then renamed into
place. ``os.replace`` is atomic within a filesystem, so a checkpoint file either does
not exist or is complete; a session killed mid-write cannot leave a truncated file that
loads as garbage. This is the same idiom Phase 1 used to mark completed shards.

**Exact resume, not approximate resume.** Alongside the obvious state, the RNG states
of Python, NumPy and PyTorch are saved. Restoring them means dropout masks and batch
order continue exactly as they would have, so a run interrupted at step 9,000 and
resumed produces the same trajectory as one that was never interrupted. Without this,
"resume" silently means "restart the data stream", which quietly re-shows data and
makes the loss curve dishonest.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lma.config import ModelConfig, TrainConfig
from lma.schedule import scheduler_state

# Bumped if the on-disk layout changes incompatibly, so an old checkpoint fails loudly.
CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    scaler: Any | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a complete, resumable checkpoint atomically.

    Args:
        path: Destination ``.pt`` file. Parent directories are created.
        model: The model whose ``state_dict`` is saved.
        optimizer: AdamW instance; its moment estimates are saved.
        model_config: Architecture, embedded so a checkpoint is self-describing and
            can be loaded without hunting for the matching config file.
        train_config: Training hyperparameters, embedded for the same reason.
        step: Optimiser steps completed so far. The learning-rate schedule is a pure
            function of this, so the step alone is what resume actually needs; the
            ``scheduler`` block is written alongside it as an explicit record of the
            curve being followed and the rate last applied.
        tokens_seen: Training tokens consumed, for logging and the report.
        best_val_loss: Best validation loss observed, so a resumed run does not
            mistakenly overwrite a better checkpoint with a worse one.
        scaler: Optional ``torch.amp.GradScaler``; its scale factor is saved so that
            mixed-precision training resumes without re-discovering the scale.
        extra: Anything else worth recording, merged into the payload.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state(
            step,
            peak_lr=train_config.learning_rate,
            warmup_steps=train_config.warmup_steps,
            max_steps=train_config.max_steps,
            min_lr_ratio=train_config.min_lr_ratio,
        ),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        **(extra or {}),
    }

    # Write beside the target so the rename stays within one filesystem.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

    # A sidecar copy of the configs, so a checkpoint directory is readable without
    # loading torch. Guideline 6 asks for configs logged alongside checkpoints.
    (path.parent / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "train": asdict(train_config),
                "parameter_counts": model_config.count_parameters(),
                "scheduler": payload["scheduler"],
                "last_step": step,
                "tokens_seen": tokens_seen,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint, optionally restoring live objects in place.

    Args:
        path: Checkpoint file to read.
        model: If given, its parameters are replaced from the checkpoint.
        optimizer: If given, its state is replaced from the checkpoint.
        scaler: If given, its scale factor is restored.
        map_location: Device to materialise tensors on. Loading to CPU first and moving
            afterwards avoids a spike of GPU memory during load.
        restore_rng: Restore Python, NumPy and Torch RNG states for exact resume. Set
            False when loading a checkpoint purely for evaluation, where reseeding the
            global RNG would be an unwanted side effect.

    Returns:
        The full payload, including ``step``, ``tokens_seen``, ``best_val_loss`` and
        the embedded configs.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If it was written by an incompatible format version.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path} has checkpoint format {version}, this code expects "
            f"{CHECKPOINT_FORMAT_VERSION}"
        )

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    if restore_rng and "rng" in payload:
        rng = payload["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return payload


def model_config_from_checkpoint(path: str | Path) -> ModelConfig:
    """Read just the architecture out of a checkpoint.

    Useful for evaluation scripts, which must build a model of exactly the right shape
    before they can load weights into it.

    Args:
        path: Checkpoint file.

    Returns:
        The :class:`~lma.config.ModelConfig` the checkpoint was trained with.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ModelConfig(**payload["model_config"])


def prune_checkpoints(directory: str | Path, keep_last_n: int, pattern: str = "step-*.pt") -> list[Path]:
    """Delete old rolling checkpoints, newest first.

    Checkpoints for a 25M model are roughly 300 MB each once optimizer moments are
    included, so an unbounded rotation fills a Kaggle working directory quickly. The
    separately named best-validation checkpoint does not match ``pattern`` and is never
    considered here.

    Args:
        directory: Directory holding the rolling checkpoints.
        keep_last_n: How many of the most recent to retain.
        pattern: Glob identifying rolling checkpoints.

    Returns:
        The paths that were deleted.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    def step_of(p: Path) -> int:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else -1

    checkpoints = sorted(directory.glob(pattern), key=step_of)
    doomed = checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints
    for path in doomed:
        path.unlink(missing_ok=True)
    return doomed
