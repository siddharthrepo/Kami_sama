"""Evaluation metrics for language modelling and open-ended generation.

Three groups live here.

**Intrinsic.** Perplexity and bits-per-byte. Perplexity is the exponentiated
cross-entropy, so it is measured *per token* and is therefore not comparable across two
models whose tokenizers segment differently. Bits-per-byte normalises by UTF-8 bytes of
the original text instead, which is tokenizer-independent and is the number to trust
when putting Hindi and Nepali side by side. Their fertilities differ (1.31 against 1.42
on her corpora), so the perplexity comparison would otherwise flatter whichever model
happens to use more tokens per word.

**Overlap.** BLEU and chrF come from ``sacrebleu``; ROUGE-L is implemented here. The
reason for writing ROUGE-L by hand is that the usual ``rouge-score`` package applies
English stemming and English tokenization, which mangles Devanagari silently — it does
not error, it just scores badly for the wrong reason. ROUGE-L is only an LCS-based F1,
so implementing it directly is both short and correct for any script.

**Diversity.** Distinct-1/2 and repetition rate. These matter more than the overlap
metrics for open-ended continuation: a model stuck in a loop can still score reasonably
on chrF while producing text no human would accept, and the diversity numbers are what
expose that.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


def perplexity(cross_entropy_nats: float) -> float:
    """Convert mean cross-entropy to perplexity.

    Args:
        cross_entropy_nats: Mean negative log-likelihood per token, in nats.

    Returns:
        ``exp(cross_entropy)``, interpretable as the effective number of equally likely
        choices the model is deciding between at each position.
    """
    # Guard the exponential: an untrained or diverged model can produce a loss large
    # enough to overflow, and reporting inf tells the reader nothing useful.
    return math.exp(min(cross_entropy_nats, 700.0))


def bits_per_byte(total_nll_nats: float, total_utf8_bytes: int) -> float:
    """Compute bits-per-byte over a held-out corpus.

    BPB is the total negative log-likelihood of the text, converted from nats to bits
    and divided by the number of UTF-8 bytes that text occupies. Because the denominator
    is bytes rather than tokens, it is independent of how the tokenizer segments, which
    is what makes it the fair basis for comparing Model H against Model L.

    Args:
        total_nll_nats: Summed (not averaged) negative log-likelihood over every
            predicted token, in nats.
        total_utf8_bytes: UTF-8 byte length of the same text, as recorded at packing
            time in the ``.meta.json`` sidecar.

    Returns:
        Bits per byte. Lower is better.

    Raises:
        ValueError: If the byte count is not positive.
    """
    if total_utf8_bytes <= 0:
        raise ValueError("total_utf8_bytes must be positive")
    return total_nll_nats / math.log(2) / total_utf8_bytes


def lcs_length(a: Sequence, b: Sequence) -> int:
    """Length of the longest common subsequence of two sequences.

    Standard dynamic program in ``O(len(a) * len(b))`` time, but with ``O(min)`` memory:
    only the previous row of the table is needed to compute the next.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        Length of the longest subsequence common to both. A subsequence need not be
        contiguous, which is exactly what makes ROUGE-L tolerant of insertions.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            current[j] = previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: Sequence, reference: Sequence, beta: float = 1.0) -> dict[str, float]:
    """ROUGE-L for one prediction/reference pair.

    Args:
        prediction: Predicted tokens — whitespace words, not sub-word pieces. Scoring on
            tokenizer pieces would make the metric depend on the tokenizer and defeat
            the cross-model comparison.
        reference: Reference tokens, same convention.
        beta: Weight of recall relative to precision. ``1.0`` gives the balanced F1 that
            ROUGE-L conventionally reports.

    Returns:
        ``{"precision", "recall", "f1"}``, each in ``[0, 1]``.
    """
    if not prediction or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(prediction, reference)
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    if precision == 0.0 and recall == 0.0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    b2 = beta * beta
    f1 = (1 + b2) * precision * recall / (recall + b2 * precision)
    return {"precision": precision, "recall": recall, "f1": f1}


def corpus_rouge_l(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    """Mean ROUGE-L over a corpus, averaged per example.

    Averaging per example rather than pooling counts across the corpus is the
    conventional ROUGE-L aggregation, and it stops a handful of very long references
    from dominating the score.

    Args:
        predictions: Generated texts.
        references: Reference texts, aligned with ``predictions``.

    Returns:
        Mean precision, recall and F1.

    Raises:
        ValueError: If the two lists differ in length.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} predictions against {len(references)} references")
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for pred, ref in zip(predictions, references):
        scores = rouge_l(pred.split(), ref.split())
        for key in totals:
            totals[key] += scores[key]
    return {key: value / len(predictions) for key, value in totals.items()}


def _ngrams(tokens: Sequence, n: int) -> list[tuple]:
    """Return the list of ``n``-grams in a token sequence."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n_sequences(sequences: Iterable[Sequence], n: int) -> float:
    """Distinct-``n`` over pre-tokenized sequences.

    Computed across the whole set of generations rather than per example, so it captures
    a model that produces fluent but interchangeable outputs — high per-example variety,
    low variety overall.

    Args:
        sequences: Token sequences — words, or tokenizer pieces.
        n: n-gram order. Distinct-1 and Distinct-2 are the conventional pair.

    Returns:
        A value in ``[0, 1]``; higher means more varied. ``0.0`` when nothing long
        enough to form an n-gram was generated.
    """
    unique: set[tuple] = set()
    total = 0
    for sequence in sequences:
        grams = _ngrams(sequence, n)
        unique.update(grams)
        total += len(grams)
    return len(unique) / total if total else 0.0


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Distinct-``n`` over whitespace-separated words.

    Args:
        texts: Generated texts.
        n: n-gram order.

    Returns:
        A value in ``[0, 1]``; higher means more lexically varied.
    """
    return distinct_n_sequences((text.split() for text in texts), n)


def repetition_rate_sequences(sequences: Iterable[Sequence], n: int = 4) -> float:
    """Fraction of ``n``-grams that repeat within their own generation.

    This is the degenerate-loop detector. A model that falls into repeating a phrase
    scores near 1.0 here while its perplexity may look unremarkable, which is precisely
    the failure that intrinsic metrics miss.

    Args:
        sequences: Token sequences — words, or tokenizer pieces.
        n: n-gram order. 4 is the usual choice — long enough that natural repetition of
            common words does not dominate.

    Returns:
        Mean over generations of ``1 - unique/total`` n-grams. ``0.0`` means every
        n-gram in every generation was unique.
    """
    rates = []
    for sequence in sequences:
        grams = _ngrams(sequence, n)
        if not grams:
            continue
        rates.append(1.0 - len(set(grams)) / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def repetition_rate(texts: Iterable[str], n: int = 4) -> float:
    """Word-level repetition rate.

    **Use :func:`repetition_rate_sequences` on tokenizer pieces as well.** Splitting on
    whitespace is blind to the most common degenerate failure in Devanagari: a model
    looping on a single character emits something like ``"।।।।।।।।…"``, which contains no
    whitespace at all and therefore counts as one "word" with zero n-grams — scoring
    0.0, the same as perfectly varied text. This was observed in practice, not
    hypothesised. The word-level figure is retained because published numbers use it,
    but it must not be the only repetition figure reported.

    Args:
        texts: Generated texts.
        n: n-gram order.

    Returns:
        Mean over generations of ``1 - unique/total`` word n-grams.
    """
    return repetition_rate_sequences((text.split() for text in texts), n)


def token_frequency_stats(counter: Counter, vocab_size: int) -> dict[str, float]:
    """Summarise how much of a vocabulary a corpus actually exercises.

    Args:
        counter: Token id counts.
        vocab_size: Size of the vocabulary.

    Returns:
        Vocabulary utilisation, and the share of all tokens taken by the top 10, 100 and
        1,000 most frequent types — a compact numeric view of the Zipf curve.
    """
    total = sum(counter.values())
    if total == 0:
        return {"utilisation": 0.0, "top10_share": 0.0, "top100_share": 0.0, "top1000_share": 0.0}

    ordered = [count for _, count in counter.most_common()]
    return {
        "utilisation": len(counter) / vocab_size,
        "top10_share": sum(ordered[:10]) / total,
        "top100_share": sum(ordered[:100]) / total,
        "top1000_share": sum(ordered[:1000]) / total,
    }
