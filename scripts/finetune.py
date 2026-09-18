"""Finetune one pretrained model on its own synthetic reasoning data.

Run once per language, starting from that language's own Phase 2 checkpoint. Nothing is
shared between the two runs: separate data, separate tokenizer, separate weights. Model L
is never initialised from Model H.

Four things differ from ``scripts/pretrain.py`` and each is deliberate.

**The loss is masked to the answer span.** A prompt averages about 110 characters and the
answer is a single word, so under an unmasked objective roughly nine tenths of the
gradient would go into learning to reproduce question templates — which nothing in
Phase 3 measures. Masking makes the training objective match the reported metric, exact
match on the answer. The unmasked loss is computed and logged alongside it, so the two
curves can be compared in the report and a masking bug would show up immediately rather
than hide.

**Batches are examples, not a token stream.** Pretraining draws random windows from a
flat ``.bin``; here each row is a self-contained question and must not bleed into its
neighbour. Sequences are right-padded to the longest in the batch. No attention mask is
needed for that padding: attention is causal, so a real token at position ``t`` can only
see positions ``<= t`` and never reaches the padding that follows it, and the padded
positions are excluded from the loss by :data:`~lma.model.IGNORE_INDEX`.

**It runs for several epochs, not a fraction of one.** The reasoning set is small and
deliberately repetitive in structure, so the model sees it more than once. That makes
overfitting possible here in a way it structurally was not during pretraining, which is
why validation is evaluated often and the best checkpoint is kept separately.

**The learning rate is an order of magnitude lower.** 1e-4 against pretraining's 6e-4.
The model already speaks the language; the job is to bend it towards a task, not to
relearn Hindi or Nepali from the reasoning corpus.

Usage::

    python -m scripts.finetune --lang hi
    python -m scripts.finetune --lang ne --epochs 3 --resume
"""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.checkpoint import load_checkpoint, prune_checkpoints, save_checkpoint  # noqa: E402
from lma.config import ModelConfig, TrainConfig  # noqa: E402
from lma.model import GPT, IGNORE_INDEX  # noqa: E402
from lma.schedule import apply_learning_rate, learning_rate_at  # noqa: E402

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}
TOKENIZER_NAMES = {"hi": "hi.model", "ne": "ne.model"}

# Set by the signal handler; the loop checkpoints and exits at the next safe boundary.
_INTERRUPTED = False


def _handle_signal(signum, _frame) -> None:
    """Request a clean shutdown at the next step boundary.

    Args:
        signum: Signal number, recorded in the log.
        _frame: Unused stack frame.
    """
    global _INTERRUPTED
    _INTERRUPTED = True
    print(f"\n[signal {signum}] finishing this step, then checkpointing and exiting",
          flush=True)


def encode_example(sp, prompt: str, answer: str, eos_id: int) -> tuple[list[int], int]:
    """Tokenise one example and locate where the answer begins.

    The prompt and the full text are encoded separately and the prompt is checked to be
    a prefix of the full encoding. It normally is, because the answer is preceded by a
    space and SentencePiece starts a new piece there — but a boundary merge would
    silently shift the mask onto the wrong tokens, so the prefix is verified rather than
    assumed. When it does not hold, the longest common prefix is used instead, which
    masks slightly more of the sequence and never less.

    Args:
        sp: Loaded SentencePieceProcessor.
        prompt: Question text ending with the answer marker.
        answer: Gold answer.
        eos_id: End-of-sequence id appended so the model learns to stop.

    Returns:
        ``(token_ids, answer_start)`` where ``answer_start`` indexes the first token of
        the answer within ``token_ids``.
    """
    prompt_ids = sp.encode(prompt)
    full_ids = sp.encode(f"{prompt} {answer}") + [eos_id]

    boundary = len(prompt_ids)
    if full_ids[:boundary] != prompt_ids:
        boundary = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            boundary += 1

    return full_ids, boundary


class ReasoningDataset:
    """Tokenised reasoning examples, batched with answer-only loss masks.

    Attributes:
        rows: One ``(token_ids, answer_start)`` pair per example.
        boundary_fallbacks: How many examples needed the common-prefix fallback in
            :func:`encode_example`. Reported so a tokenisation surprise is visible.
        truncated: How many examples exceeded the context and were dropped.
    """

    def __init__(self, path: Path, sp, eos_id: int, max_len: int) -> None:
        """Load and tokenise a JSONL split.

        Args:
            path: Split file written by ``scripts/make_reasoning_data.py``.
            sp: Loaded SentencePieceProcessor.
            eos_id: End-of-sequence id.
            max_len: Longest sequence to keep, in tokens.

        Raises:
            FileNotFoundError: If the split file is missing.
        """
        self.rows: list[tuple[list[int], int]] = []
        self.boundary_fallbacks = 0
        self.truncated = 0

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                ids, start = encode_example(sp, record["prompt"], record["answer"], eos_id)
                if len(ids) > max_len:
                    # Dropping is safer than truncating: a truncated example would lose
                    # its answer and train the model on an unanswerable prompt.
                    self.truncated += 1
                    continue
                prompt_ids = sp.encode(record["prompt"])
                if start != len(prompt_ids):
                    self.boundary_fallbacks += 1
                self.rows.append((ids, start))

    def __len__(self) -> int:
        """Number of usable examples."""
        return len(self.rows)

    def token_length_stats(self) -> dict:
        """Return min/mean/max sequence length, for the run log and the report."""
        lengths = [len(ids) for ids, _ in self.rows]
        return {
            "min": min(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "max": max(lengths),
        }

    def batches(self, batch_size: int, pad_id: int, rng: random.Random | None = None):
        """Yield padded batches.

        Args:
            batch_size: Examples per batch.
            pad_id: Token id used to fill inputs past the end of a sequence. Its value
                is irrelevant to the result — those positions are ignored by the loss
                and unreachable by causal attention — but it must be a valid id.
            rng: If given, the order is shuffled with it. Omit for deterministic
                evaluation order.

        Yields:
            ``(x, y_masked, y_full)`` int64 tensors of shape ``(B, T-1)``. ``y_masked``
            scores only answer tokens; ``y_full`` scores every real token.
        """
        order = list(range(len(self.rows)))
        if rng is not None:
            rng.shuffle(order)

        for start in range(0, len(order), batch_size):
            chunk = [self.rows[i] for i in order[start:start + batch_size]]
            width = max(len(ids) for ids, _ in chunk)

            x = torch.full((len(chunk), width - 1), pad_id, dtype=torch.long)
            y_masked = torch.full((len(chunk), width - 1), IGNORE_INDEX, dtype=torch.long)
            y_full = torch.full((len(chunk), width - 1), IGNORE_INDEX, dtype=torch.long)

            for row, (ids, answer_start) in enumerate(chunk):
                n = len(ids) - 1
                x[row, :n] = torch.tensor(ids[:-1], dtype=torch.long)
                y_full[row, :n] = torch.tensor(ids[1:], dtype=torch.long)
                # y[i] predicts ids[i + 1], so the first scored index is answer_start-1.
                first = max(answer_start - 1, 0)
                y_masked[row, first:n] = torch.tensor(ids[first + 1:], dtype=torch.long)

            yield x, y_masked, y_full


@torch.no_grad()
def evaluate(model: GPT, dataset: ReasoningDataset, batch_size: int, pad_id: int,
             device: torch.device, amp_dtype: torch.dtype, max_batches: int) -> dict:
    """Measure masked loss, unmasked loss and teacher-forced answer accuracy.

    Answer accuracy here is per-token and teacher-forced, which makes it a cheap proxy
    rather than the headline number — true exact match requires free generation and is
    computed by ``scripts/eval_reasoning.py``.

    Args:
        model: Model to evaluate. Restored to training mode before returning.
        dataset: Held-out split.
        batch_size: Examples per batch.
        pad_id: Padding id.
        device: Device to run on.
        amp_dtype: Autocast dtype.
        max_batches: Stop after this many batches.

    Returns:
        Dict with ``masked_loss``, ``full_loss`` and ``answer_token_accuracy``.
    """
    model.eval()
    masked_losses, full_losses = [], []
    correct = total = 0

    for i, (x, y_masked, y_full) in enumerate(dataset.batches(batch_size, pad_id)):
        if i >= max_batches:
            break
        x, y_masked, y_full = x.to(device), y_masked.to(device), y_full.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            logits, masked_loss, _ = model(x, targets=y_masked)
            # The unmasked loss is the same logits scored against more positions, so it
            # comes from one extra reduction rather than a second forward pass.
            full_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y_full.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

        masked_losses.append(masked_loss.item())
        full_losses.append(full_loss.item())

        scored = y_masked != IGNORE_INDEX
        predicted = logits.argmax(dim=-1)
        correct += (predicted[scored] == y_masked[scored]).sum().item()
        total += int(scored.sum().item())

    model.train()
    return {
        "masked_loss": sum(masked_losses) / max(1, len(masked_losses)),
        "full_loss": sum(full_losses) / max(1, len(full_losses)),
        "answer_token_accuracy": correct / max(1, total),
    }


def latest_checkpoint(directory: Path) -> Path | None:
    """Return the highest-numbered rolling checkpoint in a directory, or None."""
    found = sorted(directory.glob("step-*.pt"))
    return found[-1] if found else None


def build_train_config(args: argparse.Namespace, pretrain_config: TrainConfig,
                       steps_per_epoch: int) -> TrainConfig:
    """Derive the finetuning hyperparameters from the pretraining ones.

    Starting from the pretrained config rather than from defaults keeps every field the
    checkpoint format expects, and makes the diff between the two runs — learning rate,
    step count, warmup, evaluation cadence — explicit in one place.

    Args:
        args: Parsed arguments.
        pretrain_config: Config recovered from the pretrained checkpoint.
        steps_per_epoch: Optimiser steps in one pass over the training split.

    Returns:
        A TrainConfig for this finetuning run.
    """
    return replace(
        pretrain_config,
        learning_rate=args.learning_rate,
        max_steps=steps_per_epoch * args.epochs,
        warmup_steps=min(args.warmup_steps, max(1, steps_per_epoch // 2)),
        micro_batch_size=args.micro_batch_size,
        tokens_per_step=args.batch_size,  # examples per step; see the module docstring
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )


def run(args: argparse.Namespace) -> None:
    """Finetune one model end to end."""
    import sentencepiece as spm

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    lang_dir = Path(LANGUAGE_DIRS[args.lang])
    data_dir = Path(args.data_dir) if args.data_dir else lang_dir / "reasoning"
    out_dir = Path(args.out_dir) if args.out_dir else Path("checkpoints") / f"{lang_dir.name}-finetuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained = Path(args.pretrained) if args.pretrained else Path("checkpoints") / lang_dir.name / "best.pt"
    if not pretrained.exists():
        raise SystemExit(f"pretrained checkpoint not found: {pretrained}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # The tokenizer is the Phase 1 one, unchanged. The assignment requires the vocabulary
    # to stay fixed, and the pretrained embedding table is meaningless under any other.
    tokenizer_path = lang_dir / "tokenizer" / TOKENIZER_NAMES[args.lang]
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 3
    pad_id = eos_id

    print(f"language={args.lang}  device={device}", flush=True)
    print(f"  pretrained  {pretrained}", flush=True)
    print(f"  tokenizer   {tokenizer_path}  vocab={sp.get_piece_size():,}", flush=True)

    train_set = ReasoningDataset(data_dir / "train.jsonl", sp, eos_id, args.max_len)
    val_set = ReasoningDataset(data_dir / "validation.jsonl", sp, eos_id, args.max_len)
    print(f"  train       {len(train_set):,} examples  lengths {train_set.token_length_stats()}",
          flush=True)
    print(f"  validation  {len(val_set):,} examples", flush=True)
    if train_set.truncated or val_set.truncated:
        print(f"  WARNING dropped over-length examples: train={train_set.truncated} "
              f"val={val_set.truncated} (--max-len {args.max_len})", flush=True)
    if train_set.boundary_fallbacks:
        print(f"  note: {train_set.boundary_fallbacks} examples used the common-prefix "
              "answer boundary", flush=True)

    # Architecture and pretraining hyperparameters come from the checkpoint itself, so
    # the finetuned model is guaranteed to have the shape its weights were trained for.
    payload = load_checkpoint(pretrained, map_location="cpu")
    model_config = ModelConfig(**payload["model_config"])
    pretrain_config = TrainConfig(**payload["train_config"])

    if model_config.vocab_size != sp.get_piece_size():
        raise SystemExit(
            f"vocabulary mismatch: checkpoint expects {model_config.vocab_size:,} but "
            f"{tokenizer_path} has {sp.get_piece_size():,}"
        )

    model = GPT(model_config).to(device)
    model.load_state_dict(payload["model"])
    print(f"  parameters  {model.num_parameters():,}", flush=True)

    accum = max(1, args.batch_size // args.micro_batch_size)
    steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
    train_config = build_train_config(args, pretrain_config, steps_per_epoch)

    optimizer = torch.optim.AdamW(
        model.param_groups(train_config.weight_decay),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
    )
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    step, examples_seen, best_val = 0, 0, float("inf")
    if args.resume:
        resume_from = latest_checkpoint(out_dir)
        if resume_from is None:
            print(f"  [resume] nothing in {out_dir}, starting from the pretrained model",
                  flush=True)
        else:
            state = load_checkpoint(resume_from, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            step = state["step"]
            examples_seen = state["tokens_seen"]
            best_val = state["best_val_loss"]
            print(f"  [resume] {resume_from} at step {step:,}", flush=True)

    print(f"  steps/epoch {steps_per_epoch}  epochs {args.epochs}  "
          f"total {train_config.max_steps}  accum {accum}", flush=True)
    print(f"  lr {train_config.learning_rate:g}  warmup {train_config.warmup_steps}",
          flush=True)

    log_path = out_dir / "finetune_log.jsonl"
    log_file = open(log_path, "a", encoding="utf-8")

    def record(entry: dict) -> None:
        """Append one JSON line to the run log and flush it."""
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log_file.flush()

    record({"event": "start", "language": args.lang, "pretrained": str(pretrained),
            "train_examples": len(train_set), "steps_per_epoch": steps_per_epoch,
            "epochs": args.epochs, "config": {"learning_rate": train_config.learning_rate,
            "batch_size": args.batch_size, "max_len": args.max_len}})

    model.train()
    started = time.time()
    rng = random.Random(train_config.seed)
    epoch = step // steps_per_epoch

    while step < train_config.max_steps and not _INTERRUPTED:
        epoch += 1
        # A fresh shuffle each epoch, seeded from the run seed and the epoch number, so
        # the order is reproducible and a resumed run does not repeat the same order.
        epoch_rng = random.Random(train_config.seed + epoch)
        stream = train_set.batches(args.micro_batch_size, pad_id, epoch_rng)
        exhausted = False

        while not exhausted and step < train_config.max_steps and not _INTERRUPTED:
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
            accumulated_masked = accumulated_full = 0.0
            micro_done = 0

            for _ in range(accum):
                try:
                    x, y_masked, y_full = next(stream)
                except StopIteration:
                    exhausted = True
                    break

                x = x.to(device, non_blocking=True)
                y_masked = y_masked.to(device, non_blocking=True)
                y_full = y_full.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    logits, masked_loss, _ = model(x, targets=y_masked)
                    loss = masked_loss / accum
                    # Diagnostic only. Scored from the same logits, under no_grad, so it
                    # costs one reduction and cannot influence the update.
                    with torch.no_grad():
                        full_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), y_full.reshape(-1),
                            ignore_index=IGNORE_INDEX,
                        )
                scaler.scale(loss).backward()

                accumulated_masked += loss.item()
                accumulated_full += full_loss.item() / accum
                micro_done += 1
                examples_seen += x.size(0)

            if micro_done == 0:
                break

            if train_config.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip).item()
            else:
                grad_norm = float("nan")

            scaler.step(optimizer)
            scaler.update()
            step += 1

            if step % args.log_every == 0:
                print(f"step {step:>5,}/{train_config.max_steps:,}  epoch {epoch}  "
                      f"masked {accumulated_masked:.4f}  full {accumulated_full:.4f}  "
                      f"lr {learning_rate:.2e}  |g| {grad_norm:.3f}", flush=True)
                record({"step": step, "epoch": epoch, "masked_loss": accumulated_masked,
                        "full_loss": accumulated_full, "lr": learning_rate,
                        "grad_norm": grad_norm, "examples_seen": examples_seen,
                        "elapsed": time.time() - started})

            if step % train_config.eval_every == 0 or step == train_config.max_steps:
                metrics = evaluate(model, val_set, args.micro_batch_size, pad_id,
                                   device, amp_dtype, args.eval_batches)
                print(f"  [eval] step {step:,}  val masked {metrics['masked_loss']:.4f}  "
                      f"full {metrics['full_loss']:.4f}  "
                      f"answer-token acc {metrics['answer_token_accuracy']:.4f}", flush=True)
                record({"step": step, "event": "eval", **metrics})

                if metrics["masked_loss"] < best_val:
                    best_val = metrics["masked_loss"]
                    save_checkpoint(
                        out_dir / "best.pt", model=model, optimizer=optimizer,
                        model_config=model_config, train_config=train_config,
                        step=step, tokens_seen=examples_seen, best_val_loss=best_val,
                        scaler=scaler,
                        extra={"language": args.lang, "stage": "finetune",
                               "pretrained_from": str(pretrained),
                               "val_metrics": metrics},
                    )
                    print(f"  [checkpoint] new best ({best_val:.4f}) -> {out_dir/'best.pt'}",
                          flush=True)

            if step % train_config.checkpoint_every == 0:
                save_checkpoint(
                    out_dir / f"step-{step:06d}.pt", model=model, optimizer=optimizer,
                    model_config=model_config, train_config=train_config,
                    step=step, tokens_seen=examples_seen, best_val_loss=best_val,
                    scaler=scaler,
                    extra={"language": args.lang, "stage": "finetune",
                           "pretrained_from": str(pretrained)},
                )
                prune_checkpoints(out_dir, train_config.keep_last_n)

    save_checkpoint(
        out_dir / f"step-{step:06d}.pt", model=model, optimizer=optimizer,
        model_config=model_config, train_config=train_config,
        step=step, tokens_seen=examples_seen, best_val_loss=best_val, scaler=scaler,
        extra={"language": args.lang, "stage": "finetune",
               "pretrained_from": str(pretrained)},
    )
    record({"step": step, "event": "stopped", "examples_seen": examples_seen,
            "best_val_loss": best_val, "interrupted": _INTERRUPTED,
            "elapsed": time.time() - started})
    log_file.close()

    minutes = (time.time() - started) / 60
    print(f"\nfinished at step {step:,} after {minutes:.1f} min  "
          f"best val masked loss {best_val:.4f}", flush=True)
    print(f"  checkpoints {out_dir}", flush=True)
    if _INTERRUPTED:
        print(f"  resume with:  python -m scripts.finetune --lang {args.lang} --resume",
              flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune a pretrained model on synthetic reasoning data."
    )
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--pretrained", default=None,
                        help="default: checkpoints/<language>/best.pt")
    parser.add_argument("--data-dir", default=None,
                        help="default: <language>/reasoning")
    parser.add_argument("--out-dir", default=None,
                        help="default: checkpoints/<language>-finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64,
                        help="examples per optimiser step (default: 64)")
    parser.add_argument("--micro-batch-size", type=int, default=32,
                        help="examples per forward pass; a memory knob only")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="peak LR; an order below pretraining's 6e-4 (default: 1e-4)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-len", type=int, default=128,
                        help="longest example kept, in tokens (default: 128)")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="continue from the latest checkpoint in --out-dir")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
