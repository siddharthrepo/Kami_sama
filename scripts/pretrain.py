"""Pretrain one monolingual decoder-only Transformer.

Run once per language, against that language's own packed corpus, tokenizer and
vocabulary. Nothing is shared between the two runs — no data, no weights, no
initialisation. Model L is never warm-started from Model H.

The loop is ordinary AdamW training with gradient accumulation, mixed precision and a
warmup-then-cosine learning rate. Three things in it are shaped by the environment
rather than by convention.

**It expects to be killed.** Kaggle terminates sessions at twelve hours and may
pre-empt sooner. A checkpoint is written every ``checkpoint_every`` steps, SIGINT and
SIGTERM are caught and turned into one final checkpoint before exit, and ``--resume``
picks up exactly where the previous session stopped — same step, same optimiser
moments, same RNG state, same batch order. The assignment makes this mandatory; here it
is also the only way the run finishes at all.

**Effective batch size is fixed in tokens, not sequences.** ``tokens_per_step``
determines the optimiser step, and gradient accumulation makes up whatever the GPU
cannot hold in one forward pass. Moving to a smaller GPU changes ``micro_batch_size``
and leaves the learning dynamics untouched.

**Validation is deterministic.** Evaluation walks non-overlapping windows in a fixed
order, so successive validation losses are comparable to each other and across the two
models. Sampling them randomly would add noise to exactly the number the run is steered
by.

Usage::

    python -m scripts.pretrain --lang hi --resume
    python -m scripts.pretrain --lang ne --max-minutes 690
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint  # noqa: E402
from lma.config import ModelConfig, TrainConfig  # noqa: E402
from lma.data import TokenStream, to_device  # noqa: E402
from lma.model import GPT  # noqa: E402
from lma.schedule import apply_learning_rate, learning_rate_at  # noqa: E402

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}

# Set by the signal handler; the loop checkpoints and exits at the next safe boundary.
_INTERRUPTED = False


def _handle_signal(signum, _frame) -> None:
    """Request a clean shutdown at the next step boundary.

    Exiting immediately would abandon the partially accumulated gradients and, worse,
    could interrupt a checkpoint write. Setting a flag lets the loop finish its step
    and save properly.

    Args:
        signum: Signal number, recorded in the log.
        _frame: Unused stack frame.
    """
    global _INTERRUPTED
    _INTERRUPTED = True
    print(f"\n[signal {signum}] finishing this step, then checkpointing and exiting", flush=True)


@torch.no_grad()
def evaluate(model: GPT, stream: TokenStream, config: TrainConfig, seq_len: int, device: torch.device) -> float:
    """Measure mean cross-entropy over a fixed slice of a held-out stream.

    Args:
        model: Model to evaluate. Restored to training mode before returning.
        stream: Held-out token stream.
        config: Training config, for ``eval_batches`` and ``micro_batch_size``.
        seq_len: Sequence length.
        device: Device to run on.

    Returns:
        Mean cross-entropy in nats per token.
    """
    model.eval()
    losses = []
    batches = stream.sequential_batches(config.micro_batch_size, seq_len, limit=config.eval_batches)
    for batch in batches:
        x, y = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=_autocast_dtype(config, device), enabled=device.type == "cuda"):
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def _autocast_dtype(config: TrainConfig, device: torch.device) -> torch.dtype:
    """Resolve the configured mixed-precision dtype, falling back where unsupported.

    Args:
        config: Training config carrying ``amp_dtype``.
        device: Target device.

    Returns:
        The dtype autocast should use.
    """
    if device.type != "cuda":
        return torch.float32
    requested = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = requested.get(config.amp_dtype, torch.float16)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        # Turing GPUs (Kaggle's T4) have no bfloat16 path; float16 with a loss scaler
        # is the fast option there.
        print("[amp] bfloat16 unsupported on this GPU, falling back to float16", flush=True)
        return torch.float16
    return dtype


def latest_checkpoint(directory: Path) -> Path | None:
    """Find the most recent rolling checkpoint in a directory.

    Args:
        directory: Checkpoint directory.

    Returns:
        The highest-numbered ``step-*.pt``, or None if there are none.
    """
    if not directory.exists():
        return None
    candidates = list(directory.glob("step-*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or -1))


def main() -> None:
    """Parse arguments, build or restore the run, and train."""
    parser = argparse.ArgumentParser(description="Pretrain one monolingual Transformer.")
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--model-config", default=None, help="default: <lang>/configs/model.json")
    parser.add_argument("--train-config", default=None, help="default: <lang>/configs/train.json")
    parser.add_argument("--tokens-dir", default=None, help="default: <lang>/data/tokens")
    parser.add_argument("--out-dir", default=None, help="default: <lang>/checkpoints")
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    parser.add_argument(
        "--max-minutes", type=float, default=None,
        help="stop and checkpoint after this long; set below the platform session limit",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default=None, help='"cuda", "cpu"; default: auto')
    args = parser.parse_args()

    root = Path(LANGUAGE_DIRS[args.lang])
    model_config = ModelConfig.from_json(args.model_config or root / "configs" / "model.json")
    train_config = TrainConfig.from_json(args.train_config or root / "configs" / "train.json")
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else root / "data" / "tokens"
    out_dir = Path(args.out_dir) if args.out_dir else root / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(train_config.seed)
    if device.type == "cuda":
        # TF32 costs a little precision on matmuls and buys a large speedup; at this
        # model scale the loss curve is indistinguishable.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_stream = TokenStream(tokens_dir / "train.bin")
    val_stream = TokenStream(tokens_dir / "validation.bin")

    # The one cross-check that catches a mismatched corpus: the tokenizer that packed
    # these tokens must be the one this vocabulary size came from.
    if "vocab_size" not in train_stream.meta:
        raise ValueError(
            f"{tokens_dir/'train.meta.json'} predates the vocab_size field; re-run "
            "scripts.pack_tokens so the corpus can be checked against this config"
        )
    packed_vocab = int(train_stream.meta["vocab_size"])
    if packed_vocab != model_config.vocab_size:
        raise ValueError(
            f"model config expects vocab_size={model_config.vocab_size} but the packed "
            f"corpus was built with {packed_vocab}"
        )
    if int(train_stream.tokens.max()) >= model_config.vocab_size:
        raise ValueError("packed corpus contains token ids outside the configured vocabulary")

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(train_config.weight_decay),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )
    amp_dtype = _autocast_dtype(train_config, device)
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype is torch.float16 and device.type == "cuda"))

    step, tokens_seen, best_val_loss = 0, 0, float("inf")
    if args.resume:
        checkpoint = latest_checkpoint(out_dir)
        if checkpoint is None:
            print(f"[resume] nothing in {out_dir}, starting fresh", flush=True)
        else:
            payload = load_checkpoint(checkpoint, model=model, optimizer=optimizer, scaler=scaler, map_location=device.type)
            step = payload["step"]
            tokens_seen = payload["tokens_seen"]
            best_val_loss = payload["best_val_loss"]
            print(f"[resume] {checkpoint} at step {step:,} ({tokens_seen:,} tokens seen)", flush=True)

    accum = train_config.grad_accum_steps(model_config.max_seq_len)
    counts = model_config.count_parameters()
    print(
        f"language={args.lang}  device={device}  amp={amp_dtype}\n"
        f"parameters={counts['total']:,} (non-embedding {counts['non_embedding']:,})\n"
        f"train={len(train_stream):,} tokens  validation={len(val_stream):,} tokens\n"
        f"steps={train_config.max_steps:,}  tokens/step={train_config.tokens_per_step:,}  "
        f"micro_batch={train_config.micro_batch_size}  grad_accum={accum}\n"
        f"planned tokens={train_config.max_steps * train_config.tokens_per_step:,} "
        f"({train_config.max_steps * train_config.tokens_per_step / len(train_stream):.2f} epochs)\n",
        flush=True,
    )
    model_config.to_json(out_dir / "model_config.json")
    train_config.to_json(out_dir / "train_config.json")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    log_path = out_dir / "train_log.jsonl"
    log = open(log_path, "a", encoding="utf-8")

    def record(entry: dict) -> None:
        """Append one JSON line to the training log and flush it."""
        log.write(json.dumps(entry) + "\n")
        log.flush()

    def checkpoint_now(tag: str) -> None:
        """Write a checkpoint under the given name and prune old rolling ones."""
        path = out_dir / (f"step-{step:06d}.pt" if tag == "rolling" else f"{tag}.pt")
        save_checkpoint(
            path, model=model, optimizer=optimizer, model_config=model_config,
            train_config=train_config, step=step, tokens_seen=tokens_seen,
            best_val_loss=best_val_loss, scaler=scaler,
            extra={"language": args.lang, "tokens_meta": train_stream.meta},
        )
        if tag == "rolling":
            prune_checkpoints(out_dir, train_config.keep_last_n)

    model.train()
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    started = time.time()
    window_start, window_tokens = time.time(), 0

    while step < train_config.max_steps and not _INTERRUPTED:
        learning_rate = apply_learning_rate(
            optimizer,
            learning_rate_at(
                step, peak_lr=train_config.learning_rate,
                warmup_steps=train_config.warmup_steps,
                max_steps=train_config.max_steps,
                min_lr_ratio=train_config.min_lr_ratio,
            ),
        )

        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for micro in range(accum):
            # Globally unique and reproducible: the same (step, micro) always draws the
            # same windows, which is what makes an interrupted run resume exactly.
            micro_step = step * accum + micro
            batch = train_stream.random_batch(
                train_config.micro_batch_size, model_config.max_seq_len,
                seed=train_config.seed, micro_step=micro_step,
            )
            x, y = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss, _ = model(x, targets=y)
                # Scale so the accumulated gradient equals the mean over the full
                # effective batch rather than the sum over micro-batches.
                loss = loss / accum
            scaler.scale(loss).backward()
            accumulated += loss.item()

        if train_config.grad_clip > 0:
            # Unscale first: clipping a scaled gradient would clip at the wrong norm.
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip).item()
        else:
            grad_norm = float("nan")

        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += train_config.tokens_per_step
        window_tokens += train_config.tokens_per_step

        if step % args.log_every == 0:
            elapsed = time.time() - window_start
            throughput = window_tokens / max(elapsed, 1e-9)
            remaining = (train_config.max_steps - step) * train_config.tokens_per_step / max(throughput, 1e-9)
            print(
                f"step {step:>6,}/{train_config.max_steps:,}  loss {accumulated:.4f}  "
                f"ppl {math.exp(min(accumulated, 20)):>8.1f}  lr {learning_rate:.2e}  "
                f"|g| {grad_norm:.2f}  {throughput/1e3:.1f}k tok/s  eta {remaining/3600:.1f}h",
                flush=True,
            )
            record({
                "step": step, "loss": accumulated, "lr": learning_rate,
                "grad_norm": grad_norm, "tokens_seen": tokens_seen,
                "tokens_per_second": throughput, "elapsed": time.time() - started,
            })
            window_start, window_tokens = time.time(), 0

        if step % train_config.eval_every == 0 or step == train_config.max_steps:
            val_loss = evaluate(model, val_stream, train_config, model_config.max_seq_len, device)
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            print(
                f"  validation  loss {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.2f}"
                f"{'  <- best' if improved else ''}",
                flush=True,
            )
            record({
                "step": step, "val_loss": val_loss,
                "val_perplexity": math.exp(min(val_loss, 20)),
                "tokens_seen": tokens_seen, "elapsed": time.time() - started,
            })
            if improved:
                checkpoint_now("best")
            window_start, window_tokens = time.time(), 0

        if step % train_config.checkpoint_every == 0:
            checkpoint_now("rolling")

        if deadline and time.time() > deadline:
            print(f"\n[time] reached --max-minutes at step {step:,}", flush=True)
            break

    checkpoint_now("rolling")
    record({"step": step, "event": "stopped", "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss, "elapsed": time.time() - started,
            "interrupted": _INTERRUPTED})
    log.close()

    hours = (time.time() - started) / 3600
    print(
        f"\nstopped at step {step:,}/{train_config.max_steps:,}  "
        f"{tokens_seen:,} tokens  best val loss {best_val_loss:.4f}  "
        f"({hours:.2f}h this session)\n"
        f"checkpoints in {out_dir}",
        flush=True,
    )
    if step < train_config.max_steps:
        print(f"resume with:  python -m scripts.pretrain --lang {args.lang} --resume", flush=True)


if __name__ == "__main__":
    main()
