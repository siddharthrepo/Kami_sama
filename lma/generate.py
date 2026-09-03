"""Autoregressive sampling from a trained model.

Generation is the same forward pass used in training, run one position at a time: take
the logits at the last position, choose a token, append it, repeat. The correctness of
that loop rests on causality — the logits at position ``t`` must not depend on anything
after ``t``, or the model would be conditioning on tokens it has not emitted.
``tests/test_causal.py`` establishes that property directly.

Phase 2 asks for continuations under greedy decoding and at temperatures 0.5, 1.0 and
1.5. Those four settings trace a clear trade-off: greedy and low temperature maximise
per-token likelihood and tend to loop; high temperature buys diversity at the cost of
coherence. The diversity diagnostics in :mod:`lma.metrics` are what turn that
qualitative story into numbers.

No key/value cache is implemented. Caching would make generation roughly ``T`` times
cheaper by avoiding recomputation of past positions, but evaluation here generates a few
thousand short continuations, not a serving workload, and the uncached loop is the one
that visibly reuses the training forward pass — which is worth more than the speed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lma.model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Continue each prompt in a batch for up to ``max_new_tokens`` steps.

    Args:
        model: A trained model. Switched to ``eval()`` so dropout is inactive —
            sampling with dropout on would inject noise that has nothing to do with the
            distribution being measured.
        prompt: Prompt token ids, shape ``(B, T0)``. All rows must be the same length;
            pad or trim beforehand if they are not.
        max_new_tokens: Maximum tokens to append.
        temperature: Divides the logits before softmax. Below 1.0 sharpens the
            distribution toward the mode; above 1.0 flattens it. Ignored when
            ``greedy`` is True.
        top_k: If given, restrict sampling to the ``k`` highest-probability tokens. This
            truncates the long tail of near-zero probabilities that, summed over a
            16,000-token vocabulary, would otherwise contribute a noticeable share of
            the sampling mass.
        greedy: Take the argmax at every step instead of sampling.
        eos_id: If given, a sequence stops growing once it emits this token; the
            position is filled with ``eos_id`` thereafter and generation ends early when
            every row in the batch has finished.

    Returns:
        The prompt with its continuation appended, shape ``(B, T0 + n)`` where ``n <=
        max_new_tokens``.

    Raises:
        ValueError: If the prompt is already at or beyond the model's context length,
            leaving no room to generate.
    """
    was_training = model.training
    model.eval()

    context = model.config.max_seq_len
    if prompt.size(1) >= context:
        raise ValueError(
            f"prompt is {prompt.size(1)} tokens and the context is {context}; "
            "there is no room to generate"
        )

    ids = prompt
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)

    for _ in range(max_new_tokens):
        # Keep only the last `context` tokens: the positional embedding table has no
        # row beyond that, so a longer window cannot be scored at all.
        window = ids[:, -context:]
        logits, _, _ = model(window)
        logits = logits[:, -1, :]  # (B, vocab) - only the next-token distribution

        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                # Everything below the k-th largest logit becomes -inf, so softmax
                # assigns it exactly zero probability.
                threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        if eos_id is not None:
            # A finished row keeps emitting EOS rather than drifting on.
            nxt = torch.where(finished.unsqueeze(1), torch.full_like(nxt, eos_id), nxt)
            finished |= nxt.squeeze(1) == eos_id

        ids = torch.cat([ids, nxt], dim=1)

        if eos_id is not None and bool(finished.all()):
            break

    if was_training:
        model.train()
    return ids


@torch.no_grad()
def continuation_only(generated: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Strip the prompt off a generated batch.

    Args:
        generated: Output of :func:`generate`, shape ``(B, T0 + n)``.
        prompt_length: ``T0``.

    Returns:
        Just the newly generated tokens, shape ``(B, n)``.
    """
    return generated[:, prompt_length:]


def decode_batch(tokenizer, rows: torch.Tensor, eos_id: int | None = None) -> list[str]:
    """Decode token ids back to text, trimming at the first EOS.

    Args:
        tokenizer: A SentencePiece processor.
        rows: Token ids, shape ``(B, T)``.
        eos_id: If given, each row is cut at its first occurrence of this token, so the
            EOS padding added after a finished sequence does not appear in the text
            handed to the generation metrics.

    Returns:
        One decoded string per row.
    """
    out = []
    for row in rows.tolist():
        if eos_id is not None and eos_id in row:
            row = row[: row.index(eos_id)]
        out.append(tokenizer.decode(row))
    return out
