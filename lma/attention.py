"""Quantitative analysis of what the attention heads learned.

Phase 2 asks for attention heatmaps plus two summary statistics per head and layer:
entropy and mean attention distance. Heatmaps show what one head does on one sentence;
these numbers say what it does in general, which is what makes a claim like "layer 0 is
positional, layer 6 is content-based" defensible rather than anecdotal.

**Entropy** measures how spread out a head's attention is. A head that puts all its mass
on one key has entropy 0; one that spreads evenly over its ``t+1`` permitted keys has
entropy ``ln(t+1)``. Raw entropy is therefore not comparable across query positions —
position 3 simply cannot be as diffuse as position 300 — so a length-normalised variant
is reported alongside it, dividing by ``ln(t+1)`` to give a value in ``[0, 1]``.

**Mean attention distance** is the average of ``t - j`` weighted by attention, i.e. how
far back a head looks. Small values mean local, syntax-like behaviour; large values mean
the head is retrieving something from far away. Contrasting this between Model H and
Model L is one of the more interesting comparisons available, because a model trained on
weaker data often fails to develop long-range heads at all.

Position 0 is excluded from every aggregate. Its only option is to attend to itself, so
it contributes a guaranteed entropy of 0 and distance of 0 to every head, diluting the
statistics with a constant that carries no information.
"""

from __future__ import annotations

import math

import torch


def attention_entropy(attn: torch.Tensor, normalise: bool = False) -> torch.Tensor:
    """Mean Shannon entropy of each head's attention distributions.

    Args:
        attn: Attention weights, shape ``(B, h, T, T)``, rows summing to 1 over the
            causally permitted keys.
        normalise: Divide each row's entropy by ``ln(t+1)``, the maximum achievable at
            that position. Makes positions comparable and puts the result in ``[0, 1]``.

    Returns:
        Entropy per head, shape ``(h,)``, in nats — averaged over the batch and over
        query positions 1 and above.
    """
    B, h, T, _ = attn.shape
    # 0 * log(0) is 0 in the limit; clamping keeps the log finite for masked entries,
    # which carry exactly zero weight and so contribute nothing to the sum.
    safe = attn.clamp_min(1e-12)
    row_entropy = -(attn * safe.log()).sum(dim=-1)  # (B, h, T)

    if normalise:
        positions = torch.arange(T, device=attn.device, dtype=attn.dtype)
        max_entropy = (positions + 1).log().clamp_min(1e-12)  # ln(t+1)
        row_entropy = row_entropy / max_entropy

    return row_entropy[:, :, 1:].mean(dim=(0, 2))


def mean_attention_distance(attn: torch.Tensor) -> torch.Tensor:
    """Average distance, in positions, that each head attends backwards.

    Args:
        attn: Attention weights, shape ``(B, h, T, T)``.

    Returns:
        Mean distance per head, shape ``(h,)``. A head attending only to the immediately
        preceding token scores near 1.0; one attending uniformly over its history scores
        near ``t/2`` averaged over positions.
    """
    B, h, T, _ = attn.shape
    positions = torch.arange(T, device=attn.device, dtype=attn.dtype)
    # distance[t, j] = t - j, non-negative on the causally allowed lower triangle.
    distance = positions.view(-1, 1) - positions.view(1, -1)
    weighted = (attn * distance).sum(dim=-1)  # (B, h, T)
    return weighted[:, :, 1:].mean(dim=(0, 2))


def attention_to_self(attn: torch.Tensor) -> torch.Tensor:
    """Share of attention each head places on the query position itself.

    A high value marks a head that is largely passing its own representation through,
    which in a pre-norm stack often means the head is close to inactive.

    Args:
        attn: Attention weights, shape ``(B, h, T, T)``.

    Returns:
        Mean diagonal weight per head, shape ``(h,)``.
    """
    diagonal = attn.diagonal(dim1=-2, dim2=-1)  # (B, h, T)
    return diagonal[:, :, 1:].mean(dim=(0, 2))


def attention_to_previous(attn: torch.Tensor) -> torch.Tensor:
    """Share of attention each head places on the immediately preceding token.

    Induction-like and positional heads concentrate here. Reported separately from the
    mean distance because a head can average a moderate distance either by consistently
    looking a few tokens back or by mixing very local and very distant attention.

    Args:
        attn: Attention weights, shape ``(B, h, T, T)``.

    Returns:
        Mean weight on offset ``-1`` per head, shape ``(h,)``.
    """
    previous = attn.diagonal(offset=-1, dim1=-2, dim2=-1)  # (B, h, T-1)
    return previous.mean(dim=(0, 2))


def summarise_layer(attn: torch.Tensor) -> dict[str, list[float]]:
    """Compute every per-head statistic for one layer.

    Args:
        attn: Attention weights from a single layer, shape ``(B, h, T, T)``.

    Returns:
        A dict of per-head lists: raw entropy, normalised entropy, mean distance,
        self-attention share and previous-token share.
    """
    return {
        "entropy_nats": attention_entropy(attn).tolist(),
        "entropy_normalised": attention_entropy(attn, normalise=True).tolist(),
        "mean_distance": mean_attention_distance(attn).tolist(),
        "self_share": attention_to_self(attn).tolist(),
        "previous_share": attention_to_previous(attn).tolist(),
    }


def summarise_model(attentions: list[torch.Tensor]) -> list[dict[str, list[float]]]:
    """Summarise every layer of one forward pass.

    Args:
        attentions: Per-layer attention tensors, as returned by
            ``GPT.forward(..., return_attention=True)``.

    Returns:
        One summary dict per layer, in layer order.
    """
    return [summarise_layer(layer) for layer in attentions]


def classify_head(mean_distance: float, normalised_entropy: float, context: int) -> str:
    """Give a head a short descriptive label from its statistics.

    A coarse heuristic, offered to make the discussion section concrete rather than to
    stand as a finding on its own. The thresholds are relative to the context length, so
    the labels mean the same thing regardless of sequence length.

    Args:
        mean_distance: Mean attention distance, in positions.
        normalised_entropy: Length-normalised entropy in ``[0, 1]``.
        context: Sequence length the statistics were measured over.

    Returns:
        One of ``"local"``, ``"diffuse"``, ``"long-range"`` or ``"mixed"``.
    """
    local = mean_distance < 0.05 * context
    diffuse = normalised_entropy > 0.7

    if local and not diffuse:
        return "local"
    if diffuse and not local:
        return "diffuse"
    if mean_distance > 0.25 * context:
        return "long-range"
    return "mixed"
