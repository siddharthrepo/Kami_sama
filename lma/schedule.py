"""Learning-rate schedule: linear warmup followed by cosine decay.

Warmup exists because a freshly initialised model has near-uniform attention and a
near-uniform output distribution, so the first gradients are large and poorly
conditioned. Applying the peak learning rate immediately tends to blow the loss up in
the first hundred steps. Ramping in from zero lets the second-moment estimates in AdamW
settle before large updates are taken.

Cosine decay then anneals toward a small floor rather than to zero. A floor keeps the
final steps doing useful work instead of standing still, and makes the schedule robust
to stopping slightly early — which matters when training runs against a session limit.

The schedule is a pure function of the step number, not stateful. That is deliberate:
resuming from a checkpoint only needs the step, so there is no scheduler state that can
be lost or restored inconsistently.
"""

from __future__ import annotations

import math


def learning_rate_at(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute the learning rate for a given optimiser step.

    Args:
        step: Zero-based optimiser step about to be taken.
        peak_lr: Learning rate reached at the end of warmup.
        warmup_steps: Steps spent ramping linearly from zero to ``peak_lr``.
        max_steps: Total planned steps; the cosine reaches its floor here.
        min_lr_ratio: Floor as a fraction of ``peak_lr``.

    Returns:
        The learning rate to set on every parameter group before stepping.
    """
    min_lr = peak_lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # +1 so that step 0 takes a small non-zero rate rather than none at all.
        return peak_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    decay_steps = max(1, max_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    # cos goes 1 -> -1 over [0, pi], so this coefficient goes 1 -> 0.
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coefficient * (peak_lr - min_lr)


def scheduler_state(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> dict:
    """Serialise the schedule so a checkpoint carries explicit scheduler state.

    The schedule itself is stateless — ``learning_rate_at`` is a pure function of the
    step — so nothing here is *needed* to resume correctly. It is written anyway for
    two reasons. First, the project spec lists scheduler state as a required checkpoint
    field, and a reader should not have to reason about purity to confirm it is present.
    Second, it makes a checkpoint self-describing: the exact curve a run was following
    can be reconstructed from the file alone, without the training config beside it,
    and ``last_lr`` records the rate actually in force when the checkpoint was written.

    Args:
        step: Optimiser steps completed. The next step's rate is derived from this.
        peak_lr: Learning rate reached at the end of warmup.
        warmup_steps: Steps spent ramping linearly from zero to ``peak_lr``.
        max_steps: Total planned steps; the cosine reaches its floor here.
        min_lr_ratio: Floor as a fraction of ``peak_lr``.

    Returns:
        A JSON-serialisable dict fully describing the schedule and its position on it.
    """
    return {
        "type": "linear_warmup_cosine_decay",
        "step": int(step),
        "peak_lr": float(peak_lr),
        "warmup_steps": int(warmup_steps),
        "max_steps": int(max_steps),
        "min_lr_ratio": float(min_lr_ratio),
        "min_lr": float(peak_lr * min_lr_ratio),
        "last_lr": learning_rate_at(
            step,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        ),
        "stateless": True,
    }


def apply_learning_rate(optimizer, learning_rate: float) -> float:
    """Set one learning rate across all parameter groups.

    Args:
        optimizer: Any ``torch.optim.Optimizer``.
        learning_rate: Value to write into every group.

    Returns:
        The learning rate that was applied, for convenient logging.
    """
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate
