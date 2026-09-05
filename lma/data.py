"""Batching over a packed token stream.

Phase 1 produced text; ``scripts/pack_tokens.py`` turns that text into a flat array of
token ids stored as ``uint16``. Both vocabularies are 16,000, comfortably under 65,536,
so two bytes per token suffices — half what ``int32`` would cost, which matters because
the array is memory-mapped and the operating system's page cache is the only thing
standing between training and disk latency.

The array is one continuous stream of documents separated by EOS, not a padded batch of
documents. Packing this way wastes no compute on padding and means every position in
every batch carries a real gradient. The cost is that a training window can straddle a
document boundary; with EOS marking the seam the model learns to treat it as a reset,
which is the same convention GPT-style models are normally trained under.

Batch order is a pure function of ``(seed, micro_step)`` rather than a stateful shuffle.
That is what makes resume exact: a run restarted at step 9,000 draws precisely the
windows it would have drawn had it never stopped, so no data is silently re-shown.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


class TokenStream:
    """A memory-mapped ``uint16`` token array with its packing metadata.

    Attributes:
        path: The ``.bin`` file backing this stream.
        tokens: Read-only memmap of shape ``(n_tokens,)``.
        meta: Contents of the sidecar ``.meta.json`` written at packing time —
            document count, UTF-8 byte count, tokenizer fingerprint and so on.
    """

    def __init__(self, path: str | Path) -> None:
        """Open a packed split.

        Args:
            path: Path to the ``.bin`` file. Its ``.meta.json`` sidecar must sit
                beside it.

        Raises:
            FileNotFoundError: If either the array or its metadata is missing.
        """
        self.path = Path(path)
        meta_path = self.path.with_suffix(".meta.json")
        if not self.path.exists():
            raise FileNotFoundError(f"packed tokens not found: {self.path}")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata not found: {meta_path}. Re-run scripts/pack_tokens.py; the "
                "sidecar carries the token count and the UTF-8 byte count that "
                "bits-per-byte is computed against."
            )

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

        declared = self.meta.get("n_tokens")
        if declared is not None and declared != len(self.tokens):
            raise ValueError(
                f"{self.path} holds {len(self.tokens):,} tokens but its metadata "
                f"claims {declared:,}; the pack may have been interrupted"
            )

    def __len__(self) -> int:
        """Number of tokens in the stream."""
        return len(self.tokens)

    @property
    def utf8_bytes(self) -> int:
        """UTF-8 byte count of the source text, the denominator for bits-per-byte."""
        return int(self.meta["utf8_bytes"])

    def _window(self, offsets: np.ndarray, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice input/target pairs at the given start offsets.

        Args:
            offsets: Start positions, shape ``(batch,)``.
            seq_len: Window length.

        Returns:
            ``(x, y)`` int64 tensors of shape ``(batch, seq_len)``, where ``y`` is ``x``
            shifted one position left — the next-token targets.
        """
        # astype(int64) also copies out of the memmap, so the returned tensors do not
        # keep pages pinned after the batch is consumed.
        x = np.stack([self.tokens[o : o + seq_len] for o in offsets]).astype(np.int64)
        y = np.stack([self.tokens[o + 1 : o + 1 + seq_len] for o in offsets]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def random_batch(
        self, batch_size: int, seq_len: int, *, seed: int, micro_step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a batch of random windows, reproducibly.

        The windows are sampled with replacement from anywhere in the stream. At the
        scale used here — roughly 16,000 batches of 32 windows against a corpus of
        650M tokens — collisions are negligible, and sampling beats maintaining a
        shuffled index because it needs no state to resume.

        Args:
            batch_size: Windows per batch.
            seq_len: Tokens per window.
            seed: Run seed.
            micro_step: Globally increasing micro-batch counter. Together with ``seed``
                this fully determines the batch, so resuming reproduces the stream.

        Returns:
            ``(x, y)`` int64 tensors of shape ``(batch_size, seq_len)``.

        Raises:
            ValueError: If the stream is too short to yield even one window.
        """
        highest = len(self.tokens) - seq_len - 1
        if highest <= 0:
            raise ValueError(
                f"{self.path} holds {len(self.tokens):,} tokens, too few for a "
                f"{seq_len}-token window"
            )
        rng = np.random.default_rng([seed, micro_step])
        offsets = rng.integers(0, highest, size=batch_size)
        return self._window(offsets, seq_len)

    def sequential_batches(
        self, batch_size: int, seq_len: int, *, limit: int | None = None,
        spread: bool = False,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate non-overlapping windows deterministically, for evaluation.

        Validation and test loss must be comparable between evaluations and between
        the two models, so evaluation walks the stream deterministically rather than
        sampling it. Non-overlapping windows also mean every token is predicted exactly
        once, which is what makes the resulting cross-entropy a valid input to
        perplexity and to bits-per-byte.

        ``spread`` exists because a truncated walk is not a representative sample. The
        splits are not shuffled across sources, so the head of a validation file is one
        kind of text rather than a cross-section of it. Taking the first ``limit``
        batches therefore measures a corner of the split: on the Hindi validation set
        the first 10.7% scored 2.96 nats against 3.35 for the whole thing. With
        ``spread`` the same number of batches is drawn at evenly spaced offsets across
        the entire stream, so a cheap in-training estimate tracks the full-split figure
        instead of drifting from it. It stays fully deterministic, so successive
        evaluations remain comparable and checkpoint selection is unaffected by noise.

        Args:
            batch_size: Windows per batch.
            seq_len: Tokens per window.
            limit: Stop after this many batches. ``None`` walks the whole stream.
            spread: Space the batches evenly across the stream instead of taking the
                first ``limit``. Ignored when ``limit`` is ``None`` or already covers
                the stream.

        Yields:
            ``(x, y)`` int64 tensors. The final partial batch is dropped, so every
            yielded batch is full.
        """
        stride = seq_len * batch_size
        usable = len(self.tokens) - 1
        available = max(0, (usable - stride) // stride + 1)

        if limit is None or not spread or limit >= available:
            produced = 0
            for start in range(0, usable - stride + 1, stride):
                offsets = np.arange(start, start + stride, seq_len)
                yield self._window(offsets, seq_len)
                produced += 1
                if limit is not None and produced >= limit:
                    return
            return

        # Evenly spaced batch indices, first and last inclusive. Integer arithmetic
        # keeps this reproducible across platforms and numpy versions.
        for i in range(limit):
            index = (i * (available - 1)) // (limit - 1) if limit > 1 else 0
            start = index * stride
            offsets = np.arange(start, start + stride, seq_len)
            yield self._window(offsets, seq_len)


def to_device(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to the target device.

    Uses pinned memory and a non-blocking copy on CUDA, which lets the host-to-device
    transfer overlap with the previous step's compute.

    Args:
        batch: ``(x, y)`` CPU tensors.
        device: Destination device.

    Returns:
        The same pair, on ``device``.
    """
    x, y = batch
    if device.type == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)
