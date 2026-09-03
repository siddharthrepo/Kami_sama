"""Generation-quality evaluation: BLEU, chrF, ROUGE-L, and diversity diagnostics.

Held-out documents are cut at a fixed point: the first ``--prompt-tokens`` become the
prompt, the next ``--continuation-tokens`` become the reference. The model continues the
prompt under four decoding settings — greedy, and sampling at temperatures 0.5, 1.0 and
1.5 — and each continuation is scored against that reference.

A caveat the report should state plainly rather than bury: **these overlap metrics are
weak evidence for open-ended language modelling.** BLEU and ROUGE-L ask whether the model
reproduced *this particular* continuation, but a held-out news article has an enormous
space of fluent, correct continuations and only one is in the reference. A perfect model
would still score low. The numbers are reported because they are required and because
they are comparable *between* Model H and Model L under identical conditions — not
because a high score would mean the text is good.

chrF is the most informative of the three here. It works on character n-grams, so it is
insensitive to tokenizer segmentation and gives partial credit for correct morphology —
which matters a great deal in Devanagari, where Hindi and Nepali are heavily inflected
and a word-level metric scores a near-miss inflection the same as a wrong word.

The diversity diagnostics carry more signal than the overlap metrics. A model looping on
a phrase can still post a respectable chrF while producing text no reader would accept;
repetition rate and Distinct-1/2 are what expose that.

Usage::

    python -m scripts.eval_generation --lang hi --checkpoint hindi/checkpoints/best.pt
    python -m scripts.eval_generation --lang ne --num-prompts 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.schema import read_shards  # noqa: E402
from lma.checkpoint import load_checkpoint  # noqa: E402
from lma.config import ModelConfig  # noqa: E402
from lma.generate import decode_batch, generate  # noqa: E402
from lma.metrics import (  # noqa: E402
    corpus_rouge_l, distinct_n, distinct_n_sequences,
    repetition_rate, repetition_rate_sequences,
)
from lma.model import GPT  # noqa: E402

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}

# Decoding settings the assignment asks for. Greedy first, then increasing temperature.
DECODING = [
    {"name": "greedy", "greedy": True, "temperature": 0.0},
    {"name": "temp0.5", "greedy": False, "temperature": 0.5},
    {"name": "temp1.0", "greedy": False, "temperature": 1.0},
    {"name": "temp1.5", "greedy": False, "temperature": 1.5},
]

# sacrebleu's default "13a" tokenizer is tuned for Latin script and mishandles the
# Devanagari danda. "intl" applies Unicode-aware punctuation splitting, which segments
# both languages correctly. chrF needs no tokenizer at all - it works on characters.
BLEU_TOKENIZE = "intl"


def build_prompts(
    shard_dir: Path,
    tokenizer: spm.SentencePieceProcessor,
    *,
    prompt_tokens: int,
    continuation_tokens: int,
    limit: int,
) -> tuple[list[list[int]], list[str]]:
    """Cut held-out documents into (prompt, reference continuation) pairs.

    Only documents long enough to supply both halves are used, so every reference is a
    full ``continuation_tokens`` long and the metrics are not skewed by short references.

    Args:
        shard_dir: Directory of held-out ``.jsonl.zst`` shards.
        tokenizer: The language's SentencePiece model.
        prompt_tokens: Tokens of context given to the model.
        continuation_tokens: Tokens of reference held back for scoring.
        limit: Number of pairs to build.

    Returns:
        ``(prompt_ids, reference_texts)``.
    """
    needed = prompt_tokens + continuation_tokens
    prompts: list[list[int]] = []
    references: list[str] = []

    for document in read_shards(shard_dir):
        ids = tokenizer.encode(document.text)
        if len(ids) < needed:
            continue
        prompts.append(ids[:prompt_tokens])
        references.append(tokenizer.decode(ids[prompt_tokens:needed]))
        if len(prompts) >= limit:
            break

    return prompts, references


@torch.no_grad()
def run_decoding(
    model: GPT,
    tokenizer: spm.SentencePieceProcessor,
    prompts: list[list[int]],
    setting: dict,
    *,
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    top_k: int | None,
    seed: int,
) -> list[str]:
    """Generate one continuation per prompt under a single decoding setting.

    Args:
        model: Trained model.
        tokenizer: For decoding ids back to text.
        prompts: Prompt token id lists, all the same length.
        setting: One entry of :data:`DECODING`.
        max_new_tokens: Continuation length.
        batch_size: Prompts per generation call.
        device: Device to run on.
        top_k: Optional top-k truncation, applied only when sampling.
        seed: Reseeded per setting so sampling is reproducible and so the four settings
            are compared on identical random draws.

    Returns:
        One decoded continuation per prompt.
    """
    torch.manual_seed(seed)
    outputs: list[str] = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        batch = torch.tensor(chunk, dtype=torch.long, device=device)
        generated = generate(
            model, batch, max_new_tokens,
            temperature=setting["temperature"] or 1.0,
            top_k=None if setting["greedy"] else top_k,
            greedy=setting["greedy"],
            eos_id=tokenizer.eos_id(),
        )
        outputs.extend(decode_batch(tokenizer, generated[:, batch.size(1):], tokenizer.eos_id()))

    return outputs


def score(
    hypotheses: list[str], references: list[str],
    tokenizer: spm.SentencePieceProcessor | None = None,
) -> dict:
    """Score generated continuations against their references.

    Args:
        hypotheses: Generated texts.
        references: Reference continuations.
        tokenizer: When given, diversity is additionally measured over tokenizer
            pieces. This is not redundant with the word-level figures: a model that
            loops on a single character emits text containing no whitespace, which
            the word-level metrics see as one token and score as non-repetitive.
            The token-level figures are the ones that detect that failure.

    Returns:
        BLEU, chrF, chrF++, ROUGE-L, and diversity at both word and token level.
    """
    import sacrebleu

    # Metric objects rather than the corpus_* helpers: the reproducibility signature
    # (tokenizer, smoothing, version) hangs off the metric, not off the score it
    # returns, and that signature is what makes a reported BLEU comparable to anyone
    # else's.
    bleu_metric = sacrebleu.BLEU(tokenize=BLEU_TOKENIZE)
    bleu = bleu_metric.corpus_score(hypotheses, [references])
    chrf = sacrebleu.CHRF().corpus_score(hypotheses, [references])
    # word_order=2 turns chrF into chrF++, adding word bigrams to the character n-grams.
    chrfpp = sacrebleu.CHRF(word_order=2).corpus_score(hypotheses, [references])
    rouge = corpus_rouge_l(hypotheses, references)

    scores = {
        "bleu": bleu.score,
        # The spec asks for the n-gram order explicitly. sacrebleu's signature string
        # does not carry it, so record it as its own field rather than leaving a reader
        # to infer BLEU-4 from the length of the precisions list.
        "bleu_ngram_order": bleu_metric.max_ngram_order,
        "bleu_signature": str(bleu_metric.get_signature()),
        "bleu_precisions": bleu.precisions,
        "chrf": chrf.score,
        "chrf_pp": chrfpp.score,
        "rouge_l_f1": rouge["f1"],
        "rouge_l_precision": rouge["precision"],
        "rouge_l_recall": rouge["recall"],
        "distinct_1": distinct_n(hypotheses, 1),
        "distinct_2": distinct_n(hypotheses, 2),
        "repetition_4gram": repetition_rate(hypotheses, 4),
        "mean_words": sum(len(h.split()) for h in hypotheses) / max(1, len(hypotheses)),
    }

    if tokenizer is not None:
        pieces = [tokenizer.encode(h, out_type=str) for h in hypotheses]
        scores.update({
            "distinct_1_tokens": distinct_n_sequences(pieces, 1),
            "distinct_2_tokens": distinct_n_sequences(pieces, 2),
            "repetition_4gram_tokens": repetition_rate_sequences(pieces, 4),
            "mean_tokens": sum(len(p) for p in pieces) / max(1, len(pieces)),
        })

    return scores


def main() -> None:
    """Generate under every decoding setting, score, and write results plus samples."""
    parser = argparse.ArgumentParser(description="Generation quality for one model.")
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--checkpoint", default=None, help="default: <lang>/checkpoints/best.pt")
    parser.add_argument("--splits-dir", default=None, help="default: <lang>/data/splits")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--continuation-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=None, help="optional top-k when sampling")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-samples", type=int, default=10, help="qualitative samples to save")
    parser.add_argument("--out-dir", default=None, help="default: report/<lang dir>")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(LANGUAGE_DIRS[args.lang])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "best.pt"
    splits_dir = Path(args.splits_dir) if args.splits_dir else root / "data" / "splits"
    out_dir = Path(args.out_dir) if args.out_dir else Path("report") / root.name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    payload = load_checkpoint(checkpoint_path, map_location="cpu", restore_rng=False)
    model_config = ModelConfig(**payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    tokenizer_path = next(p for p in (root / "tokenizer").glob("*.model") if "-vocab" not in p.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    if args.prompt_tokens + args.continuation_tokens > model_config.max_seq_len:
        raise ValueError(
            f"prompt ({args.prompt_tokens}) + continuation ({args.continuation_tokens}) "
            f"exceeds the model's context of {model_config.max_seq_len}"
        )

    print(f"building {args.num_prompts} prompts from {splits_dir/'test'} ...", flush=True)
    prompts, references = build_prompts(
        splits_dir / "test", tokenizer,
        prompt_tokens=args.prompt_tokens,
        continuation_tokens=args.continuation_tokens,
        limit=args.num_prompts,
    )
    print(f"  built {len(prompts)} prompt/reference pairs\n", flush=True)

    results = {}
    samples = []
    for setting in DECODING:
        started = time.time()
        hypotheses = run_decoding(
            model, tokenizer, prompts, setting,
            max_new_tokens=args.continuation_tokens,
            batch_size=args.batch_size, device=device,
            top_k=args.top_k, seed=args.seed,
        )
        metrics = score(hypotheses, references, tokenizer)
        metrics["seconds"] = round(time.time() - started, 1)
        results[setting["name"]] = metrics

        print(
            f"{setting['name']:>9}  BLEU {metrics['bleu']:6.2f}  chrF {metrics['chrf']:6.2f}  "
            f"ROUGE-L {metrics['rouge_l_f1']:.4f}  "
            f"D-1 {metrics['distinct_1_tokens']:.4f}  D-2 {metrics['distinct_2_tokens']:.4f}  "
            f"rep-4 {metrics['repetition_4gram_tokens']:.4f}  (token level)",
            flush=True,
        )

        for i in range(min(args.num_samples, len(hypotheses))):
            samples.append({
                "setting": setting["name"],
                "prompt": tokenizer.decode(prompts[i]),
                "generated": hypotheses[i],
                "reference": references[i],
            })

    (out_dir / "generation_eval.json").write_text(
        json.dumps(
            {
                "language": args.lang,
                "checkpoint": str(checkpoint_path),
                "step": payload["step"],
                "num_prompts": len(prompts),
                "prompt_tokens": args.prompt_tokens,
                "continuation_tokens": args.continuation_tokens,
                "top_k": args.top_k,
                "seed": args.seed,
                "bleu_tokenizer": BLEU_TOKENIZE,
                "settings": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "generation_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {out_dir/'generation_eval.json'} and {out_dir/'generation_samples.json'}")


if __name__ == "__main__":
    main()
