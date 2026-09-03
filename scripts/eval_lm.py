"""Intrinsic language-modelling evaluation: cross-entropy, perplexity, bits-per-byte.

Run per language, on that language's own held-out splits. The two models are never
evaluated on each other's data — they do not share a vocabulary, so the notion is not
even well defined.

Why both perplexity and bits-per-byte are reported, rather than just the familiar one:
perplexity is per *token*, so it depends on how the tokenizer segments. Hindi and Nepali
were tokenized with separate 16,000-piece vocabularies at different fertilities (1.31 and
1.42 tokens per word on her corpora), which means an identical modelling ability would
still produce different perplexities. Bits-per-byte divides the same total likelihood by
UTF-8 bytes of the original text instead, removing the tokenizer from the comparison. When
the report puts Model H beside Model L, BPB is the honest column.

Evaluation walks non-overlapping windows in a fixed order, so every token is predicted
exactly once and the result is reproducible run to run.

Usage::

    python -m scripts.eval_lm --lang hi --checkpoint hindi/checkpoints/best.pt
    python -m scripts.eval_lm --lang ne --splits validation test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.checkpoint import load_checkpoint  # noqa: E402
from lma.config import ModelConfig  # noqa: E402
from lma.data import TokenStream, to_device  # noqa: E402
from lma.metrics import bits_per_byte, perplexity  # noqa: E402
from lma.model import GPT  # noqa: E402

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}


@torch.no_grad()
def measure_split(
    model: GPT,
    stream: TokenStream,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Compute total and mean negative log-likelihood over a held-out split.

    The *sum* of the negative log-likelihood is accumulated, not a running mean of
    per-batch means. Those differ whenever batches contain different numbers of scored
    positions, and bits-per-byte needs the true total.

    Args:
        model: Trained model, switched to eval mode.
        stream: Held-out token stream.
        batch_size: Windows per forward pass.
        seq_len: Window length; should equal the model's context.
        device: Device to run on.
        max_batches: Stop early after this many batches. ``None`` walks the split.

    Returns:
        Cross-entropy in nats per token, perplexity, bits-per-byte, and the counts the
        figures were derived from.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    batches = 0
    started = time.time()

    for batch in stream.sequential_batches(batch_size, seq_len, limit=max_batches):
        x, y = to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            _, loss, _ = model(x, targets=y)
        scored = y.numel()
        total_nll += loss.item() * scored
        total_tokens += scored
        batches += 1

    if total_tokens == 0:
        raise ValueError(f"{stream.path} produced no full batches at seq_len={seq_len}")

    mean_nll = total_nll / total_tokens

    # Bits-per-byte must use the byte count of the text actually scored. When the whole
    # split is walked, that is every byte; when max_batches truncates it, the byte count
    # is scaled by the fraction of tokens covered. Scaling is exact only if bytes per
    # token are uniform across the split, so the full walk is the default.
    covered = total_tokens / len(stream)
    scored_bytes = stream.utf8_bytes * covered

    return {
        "cross_entropy_nats": mean_nll,
        "perplexity": perplexity(mean_nll),
        "bits_per_byte": bits_per_byte(total_nll, scored_bytes),
        "tokens_scored": total_tokens,
        "tokens_in_split": len(stream),
        "fraction_covered": covered,
        "utf8_bytes_scored": int(scored_bytes),
        "batches": batches,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    """Evaluate one model on its held-out splits and write the results as JSON."""
    parser = argparse.ArgumentParser(description="Perplexity and bits-per-byte for one model.")
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--checkpoint", default=None, help="default: <lang>/checkpoints/best.pt")
    parser.add_argument("--tokens-dir", default=None, help="default: <lang>/data/tokens")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=None, help="truncate for a quick check")
    parser.add_argument("--out", default=None, help="default: report/<lang dir>/lm_eval.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(LANGUAGE_DIRS[args.lang])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "best.pt"
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else root / "data" / "tokens"
    out_path = Path(args.out) if args.out else Path("report") / root.name / "lm_eval.json"

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    payload = load_checkpoint(checkpoint_path, map_location="cpu", restore_rng=False)
    model_config = ModelConfig(**payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model"])
    model.to(device)

    counts = model_config.count_parameters()
    print(
        f"language={args.lang}  checkpoint={checkpoint_path}\n"
        f"step={payload['step']:,}  tokens_seen={payload['tokens_seen']:,}\n"
        f"parameters={counts['total']:,}  vocab={model_config.vocab_size:,}  "
        f"context={model_config.max_seq_len}  device={device}\n",
        flush=True,
    )

    results = {}
    for split in args.splits:
        stream = TokenStream(tokens_dir / f"{split}.bin")
        result = measure_split(
            model, stream,
            batch_size=args.batch_size, seq_len=model_config.max_seq_len,
            device=device, max_batches=args.max_batches,
        )
        results[split] = result
        print(
            f"{split:>11}  loss {result['cross_entropy_nats']:.4f} nats  "
            f"PPL {result['perplexity']:.2f}  BPB {result['bits_per_byte']:.4f}  "
            f"({result['tokens_scored']:,} tokens, {result['seconds']:.0f}s)",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "language": args.lang,
                "checkpoint": str(checkpoint_path),
                "step": payload["step"],
                "tokens_seen": payload["tokens_seen"],
                "parameters": counts["total"],
                "vocab_size": model_config.vocab_size,
                "context": model_config.max_seq_len,
                "splits": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
