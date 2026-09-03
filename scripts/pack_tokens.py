"""Encode the Phase 1 splits into flat uint16 token arrays for training.

This is the bridge between the two phases. Phase 1 left text in zstd-compressed JSONL
shards; pretraining wants a contiguous array of token ids it can memory-map and slice
without parsing anything. Doing that conversion once, offline, keeps the training loop
free of tokenizer calls and JSON decoding — the GPU should never wait on a CPU parsing
Devanagari.

Documents are encoded and concatenated with an EOS token between them, producing one
continuous stream per split. No padding is used, so no compute is spent on positions
that carry no gradient.

Three properties are worth calling out.

**Resumable.** Each input shard becomes a part file, written to a temporary name and
renamed into place on completion. Re-running skips finished parts, so an interrupted
pack costs only the shard it was working on. This is the same atomic-rename idiom the
Phase 1 scraper used.

**Parallel.** Shards are independent, so they are encoded across all cores. Encoding is
the CPU-bound step; a 650M-token corpus takes roughly an hour on twelve cores.

**Self-describing.** Each split gets a ``.meta.json`` sidecar recording the token count,
the document count, the UTF-8 byte count and a SHA-256 of the tokenizer that produced
it. The byte count is the denominator for bits-per-byte in Phase 2 evaluation, and the
tokenizer hash is what catches the one silent failure that matters here: packing a
corpus with a tokenizer that was trained on a different corpus.

Usage::

    python -m scripts.pack_tokens --config hindi/configs/dataset.json
    python -m scripts.pack_tokens --config nepali/configs/dataset.json --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.schema import read_shards  # noqa: E402

SPLITS = ("train", "validation", "test")

# uint16 addresses 0..65,535. Both vocabularies are 32,000, so this is safe and halves
# the on-disk size relative to int32 — 1.3 GB instead of 2.6 GB per language.
DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = np.iinfo(DTYPE).max + 1

# Documents encoded per SentencePiece call. Batching amortises the call overhead;
# beyond a few thousand the gain flattens while memory grows.
ENCODE_BATCH = 1_000

_TOKENIZER: spm.SentencePieceProcessor | None = None


def _init_worker(model_path: str) -> None:
    """Load the tokenizer once per worker process.

    SentencePiece processors are not picklable in a useful way, so each worker builds
    its own rather than receiving one per task.

    Args:
        model_path: Path to the ``.model`` file.
    """
    global _TOKENIZER
    _TOKENIZER = spm.SentencePieceProcessor(model_file=model_path)


def encode_shard(task: tuple[str, str, int]) -> dict:
    """Encode one input shard into a part file of token ids.

    Args:
        task: ``(shard_path, part_path, eos_id)``.

    Returns:
        A summary dict with the part path, token count, document count and UTF-8 byte
        count. Reads the existing summary instead of re-encoding when the part is
        already complete.
    """
    shard_path, part_path, eos_id = task
    part = Path(part_path)
    summary_path = part.with_suffix(".json")

    if part.exists() and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    assert _TOKENIZER is not None, "worker was not initialised with a tokenizer"

    ids: list[int] = []
    documents = 0
    utf8_bytes = 0
    pending: list[str] = []

    def flush() -> None:
        """Encode the buffered documents and append them to the stream."""
        nonlocal pending
        if not pending:
            return
        # num_threads=1 is essential, not an optimisation. SentencePiece's batch
        # encode defaults to spinning up its own thread pool sized to the machine, so
        # each pool worker would fan out across every core -- making --workers N mean
        # roughly N * n_cores of load rather than N cores. Pinning it to one thread is
        # what makes the --workers flag mean what it says.
        for encoded in _TOKENIZER.encode(pending, num_threads=1):
            ids.extend(encoded)
            # EOS marks the document boundary. The model learns it as a reset signal,
            # and generation uses it as a natural stopping point.
            ids.append(eos_id)
        pending = []

    for document in read_shards(Path(shard_path)):
        text = document.text
        if not text:
            continue
        documents += 1
        utf8_bytes += len(text.encode("utf-8"))
        pending.append(text)
        if len(pending) >= ENCODE_BATCH:
            flush()
    flush()

    array = np.asarray(ids, dtype=DTYPE)
    tmp = part.with_suffix(part.suffix + ".tmp")
    array.tofile(tmp)
    os.replace(tmp, part)

    summary = {
        "part": str(part),
        "source_shard": str(shard_path),
        "n_tokens": int(array.size),
        "n_documents": documents,
        "utf8_bytes": utf8_bytes,
    }
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    return summary


def concatenate_parts(parts: list[Path], destination: Path) -> int:
    """Join part files into one array, in order.

    Streamed in chunks rather than loaded whole, so peak memory stays flat regardless
    of corpus size. Written to a temporary name and renamed, so an interrupted
    concatenation cannot leave a half-written ``.bin`` that looks complete.

    Args:
        parts: Part files, in the order they should appear.
        destination: Final ``.bin`` path.

    Returns:
        Total tokens written.
    """
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    written = 0
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                while chunk := fh.read(1 << 24):  # 16 MiB
                    out.write(chunk)
                    written += len(chunk)
    os.replace(tmp, destination)
    return written // DTYPE().itemsize


def pack_split(
    split: str,
    splits_dir: Path,
    out_dir: Path,
    tokenizer_path: Path,
    vocab_size: int,
    eos_id: int,
    workers: int,
    keep_parts: bool,
) -> dict:
    """Encode every shard of one split and concatenate the result.

    Args:
        split: ``"train"``, ``"validation"`` or ``"test"``.
        splits_dir: Directory holding the per-split shard directories.
        out_dir: Where ``<split>.bin`` and its sidecar are written.
        tokenizer_path: SentencePiece model used for encoding.
        vocab_size: Size of that tokenizer's vocabulary, recorded in the metadata so
            training can refuse a corpus packed with a different tokenizer.
        eos_id: Token id appended after each document.
        workers: Parallel encoder processes.
        keep_parts: Retain the intermediate part files. Useful when re-packing
            repeatedly; wasteful otherwise, since parts duplicate the final array.

    Returns:
        The metadata dict written to ``<split>.meta.json``.

    Raises:
        FileNotFoundError: If the split directory holds no shards.
    """
    source = splits_dir / split
    shards = sorted(source.glob("*.jsonl.zst"))
    if not shards:
        raise FileNotFoundError(f"no shards under {source}")

    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (str(shard), str(parts_dir / f"{split}-{index:05d}.part"), eos_id)
        for index, shard in enumerate(shards)
    ]

    print(f"[{split}] {len(shards)} shards -> {workers} workers", flush=True)
    started = time.time()

    with Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path),)) as pool:
        summaries = []
        for done, summary in enumerate(pool.imap(encode_shard, tasks), start=1):
            summaries.append(summary)
            print(
                f"  [{split}] {done}/{len(tasks)} shards  "
                f"{sum(s['n_tokens'] for s in summaries):,} tokens",
                flush=True,
            )

    destination = out_dir / f"{split}.bin"
    total = concatenate_parts([Path(s["part"]) for s in summaries], destination)

    meta = {
        "split": split,
        "path": str(destination),
        "dtype": "uint16",
        "n_tokens": total,
        "n_documents": sum(s["n_documents"] for s in summaries),
        "utf8_bytes": sum(s["utf8_bytes"] for s in summaries),
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "source_shards": [Path(s["source_shard"]).name for s in summaries],
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "packed_at": time.time(),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out_dir / f"{split}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    expected = sum(s["n_tokens"] for s in summaries)
    if total != expected:
        raise ValueError(f"[{split}] wrote {total:,} tokens, expected {expected:,}")

    if not keep_parts:
        for summary in summaries:
            Path(summary["part"]).unlink(missing_ok=True)
            Path(summary["part"]).with_suffix(".json").unlink(missing_ok=True)

    size_gb = destination.stat().st_size / 1e9
    print(
        f"[{split}] {total:,} tokens from {meta['n_documents']:,} documents "
        f"-> {destination} ({size_gb:.2f} GB) in {meta['elapsed_seconds']:.0f}s\n",
        flush=True,
    )
    return meta


def main() -> None:
    """Parse arguments and pack every requested split."""
    parser = argparse.ArgumentParser(
        description="Encode Phase 1 splits into uint16 token arrays for pretraining."
    )
    parser.add_argument("--config", required=True, help="path to a dataset.json")
    parser.add_argument("--out-dir", default=None, help="default: <lang>/data/tokens")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
        help="encoder processes; defaults to cores minus two",
    )
    parser.add_argument(
        "--keep-parts", action="store_true",
        help="retain intermediate part files instead of deleting them",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    outputs = config["outputs"]
    splits_dir = Path(outputs["splits_dir"])
    tokenizer_dir = Path(outputs["tokenizer_dir"])

    models = sorted(p for p in tokenizer_dir.glob("*.model") if "-vocab" not in p.name)
    if len(models) != 1:
        raise FileNotFoundError(
            f"expected exactly one installed tokenizer in {tokenizer_dir}, found "
            f"{[p.name for p in models]}"
        )
    tokenizer_path = models[0]

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    vocab_size, eos_id = tokenizer.get_piece_size(), tokenizer.eos_id()
    if vocab_size > MAX_VOCAB_FOR_DTYPE:
        raise ValueError(
            f"vocabulary of {vocab_size:,} does not fit in uint16; switch DTYPE to "
            "uint32 and re-pack"
        )
    if eos_id < 0:
        raise ValueError(f"{tokenizer_path} defines no EOS token; cannot mark boundaries")

    out_dir = Path(args.out_dir) if args.out_dir else splits_dir.parent / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"language={config['lang']}  tokenizer={tokenizer_path}  "
        f"vocab={vocab_size:,}  eos={eos_id}  out={out_dir}\n",
        flush=True,
    )

    packed = {}
    for split in args.splits:
        packed[split] = pack_split(
            split, splits_dir, out_dir, tokenizer_path, vocab_size, eos_id,
            args.workers, args.keep_parts,
        )

    summary_path = out_dir / "pack_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "language": config["lang"],
                "tokenizer": str(tokenizer_path),
                "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                "vocab_size": vocab_size,
                "splits": packed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
