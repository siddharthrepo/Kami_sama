"""Score the pretrained and finetuned models on the synthetic reasoning test set.

The assignment asks for pretrained versus finetuned accuracy on the same held-out set,
qualitative successes and failures, and a comparison between Model H and Model L. This
script produces all three for one language; run it per language and the Phase 3 tables
follow from the two JSON files.

Accuracy is **exact match on freely generated text**, not teacher-forced token accuracy.
The model is given the prompt, generates greedily until it emits end-of-sequence, and the
decoded continuation must equal the gold answer. That is the honest measure: teacher
forcing would show the model the correct answer tokens one position before it has to
predict them.

Three details are worth knowing.

**Prompts are bucketed by exact token length rather than padded.** The model uses learned
absolute positional embeddings, so left-padding would shift every real token to a
position it was never trained at, and right-padding cannot work for generation because
the continuation belongs at the end. Grouping equal-length prompts sidesteps both.

**Both strict and lenient accuracy are reported.** Hindi answers are stored in the direct
case (``डिब्बा``) while the prompt mentions them in the oblique (``डिब्बे``). A model
answering ``डिब्बे`` has identified the right entity and inflected it as the prompt did.
Strict scoring counts that wrong; lenient accepts it. The strict number is the headline
and the gap between the two is itself informative.

**First-word match is reported alongside exact match.** A model that has learned the
answer but not the habit of stopping emits ``सचिन, हितेश से लंबा है`` where the gold is
``सचिन``. Exact match scores that zero, correctly — but "cannot reason" and "will not
stop" are different findings, and the pretrained baseline is dominated by the second.
Scoring the first word separates them.

**A chance baseline is computed per example.** A three-entity superlative question is
right one time in three by luck, and an equality question one time in two. Reporting the
mean chance level alongside accuracy prevents a pretrained model's noise from being read
as partial competence.

Usage::

    python -m scripts.eval_reasoning --lang hi
    python -m scripts.eval_reasoning --lang ne --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.checkpoint import load_checkpoint  # noqa: E402
from lma.config import ModelConfig  # noqa: E402
from lma.generate import generate  # noqa: E402
from lma.model import GPT  # noqa: E402

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}
TOKENIZER_NAMES = {"hi": "hi.model", "ne": "ne.model"}

# Trailing punctuation a model may append to an otherwise correct answer.
TRAILING = " \t\n।.,!?\"'"


def normalise(text: str) -> str:
    """Strip whitespace and trailing punctuation from a generated answer.

    Args:
        text: Raw decoded continuation.

    Returns:
        The answer with surrounding whitespace and sentence punctuation removed.
    """
    return text.strip().strip(TRAILING).strip()


def oblique_variants(name: str) -> set[str]:
    """Return the surface forms that identify the same entity.

    Hindi masculine nouns ending in ``-ा`` appear as ``-े`` before a postposition, so a
    model that answers with the form it saw in the prompt has still picked the right
    entity. Both directions are generated because the gold answer is stored in the
    direct case and the prompt shows the oblique.

    Args:
        name: Gold answer.

    Returns:
        Every accepted surface form, including the original.
    """
    forms = {name}
    if name.endswith("ा"):
        forms.add(name[:-1] + "े")
    if name.endswith("े"):
        forms.add(name[:-1] + "ा")
    return forms


def chance_level(record: dict) -> float:
    """Probability of guessing this example correctly at random.

    Selection questions pick one of the named entities; equality questions are a coin
    flip. Averaged over the set this gives the floor any real result must clear.

    Args:
        record: One dataset row.

    Returns:
        Chance accuracy in [0, 1].
    """
    if record["template"] == "numeric_equality":
        return 0.5
    return 1.0 / max(1, record["n_entities"])


def load_model(path: Path, device: torch.device) -> tuple[GPT, dict]:
    """Load a checkpoint into a model in eval mode.

    Args:
        path: Checkpoint file.
        device: Device to place the model on.

    Returns:
        ``(model, payload)`` where payload is the raw checkpoint dict.

    Raises:
        SystemExit: If the checkpoint does not exist.
    """
    if not path.exists():
        raise SystemExit(f"checkpoint not found: {path}")
    payload = load_checkpoint(path, map_location="cpu")
    model = GPT(ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.no_grad()
def score_model(model: GPT, records: list[dict], sp, eos_id: int,
                device: torch.device, batch_size: int, max_new_tokens: int) -> list[dict]:
    """Generate an answer for every record and mark it right or wrong.

    Args:
        model: Model to evaluate.
        records: Dataset rows.
        sp: Loaded SentencePieceProcessor.
        eos_id: End-of-sequence id, used to stop generation early.
        device: Device to run on.
        batch_size: Prompts per generation call, within a length bucket.
        max_new_tokens: Generation budget. Answers are one to three tokens; the budget
            is larger so that a verbose wrong answer is seen in full rather than cut off
            and mistaken for a near miss.

    Returns:
        One result dict per record, in the original order.
    """
    # Bucket by exact prompt length so every row in a generation call is the same
    # length, which generate() requires and which avoids padding entirely.
    buckets: dict[int, list[int]] = defaultdict(list)
    encoded = [sp.encode(r["prompt"]) for r in records]
    for i, ids in enumerate(encoded):
        buckets[len(ids)].append(i)

    results: list[dict | None] = [None] * len(records)

    for length in sorted(buckets):
        indices = buckets[length]
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            prompt = torch.tensor([encoded[i] for i in chunk], dtype=torch.long,
                                  device=device)
            out = generate(model, prompt, max_new_tokens=max_new_tokens,
                           greedy=True, eos_id=eos_id)

            for row, index in enumerate(chunk):
                new_ids = out[row, length:].tolist()
                if eos_id in new_ids:
                    new_ids = new_ids[:new_ids.index(eos_id)]
                prediction = normalise(sp.decode(new_ids))
                # The leading word alone, so a correct-but-unterminated answer is
                # distinguishable from a wrong one.
                parts = prediction.split()
                first_word = normalise(parts[0]) if parts else ""

                gold = records[index]["answer"]
                accepted = oblique_variants(gold)
                results[index] = {
                    "example_id": records[index]["example_id"],
                    "template": records[index]["template"],
                    "attribute": records[index]["attribute"],
                    "n_entities": records[index]["n_entities"],
                    "hops": records[index]["hops"],
                    "prompt": records[index]["prompt"],
                    "gold": gold,
                    "prediction": prediction,
                    "first_word": first_word,
                    "strict": prediction == gold,
                    "lenient": prediction in accepted,
                    "first_word_match": first_word in accepted,
                    "chance": chance_level(records[index]),
                }

    return [r for r in results if r is not None]


def aggregate(results: list[dict]) -> dict:
    """Summarise accuracy overall and broken down by template, hops and attribute.

    Args:
        results: Per-example results from :func:`score_model`.

    Returns:
        A JSON-serialisable summary.
    """
    def rate(rows: list[dict], key: str) -> float:
        return round(sum(r[key] for r in rows) / max(1, len(rows)), 4)

    def group(field: str) -> dict:
        buckets: dict = defaultdict(list)
        for r in results:
            buckets[r[field]].append(r)
        return {
            str(k): {
                "n": len(v),
                "strict": rate(v, "strict"),
                "lenient": rate(v, "lenient"),
                "first_word": rate(v, "first_word_match"),
                "chance": round(sum(x["chance"] for x in v) / len(v), 4),
            }
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        }

    return {
        "examples": len(results),
        "exact_match_strict": rate(results, "strict"),
        "exact_match_lenient": rate(results, "lenient"),
        "first_word_match": rate(results, "first_word_match"),
        "chance_baseline": round(sum(r["chance"] for r in results) / max(1, len(results)), 4),
        "by_template": group("template"),
        "by_hops": group("hops"),
        "by_attribute": group("attribute"),
        "by_n_entities": group("n_entities"),
    }


def qualitative(results: list[dict], limit: int) -> dict:
    """Pick representative successes and failures for the report.

    Successes are drawn from the hardest examples available — the most reasoning hops —
    because a correct three-hop answer is evidence of something a correct one-hop answer
    is not. Failures are grouped by whether the model named an entity from the prompt at
    all, which separates a wrong comparison from a failure to engage with the format.

    Args:
        results: Per-example results.
        limit: How many of each kind to keep.

    Returns:
        A dict with ``successes``, ``wrong_entity`` and ``off_format`` lists.
    """
    def trim(r: dict) -> dict:
        return {k: r[k] for k in ("template", "hops", "prompt", "gold",
                                  "prediction", "first_word")}

    correct = sorted((r for r in results if r["lenient"]),
                     key=lambda r: -r["hops"])
    wrong = [r for r in results if not r["lenient"]]

    # Split the failures by whether the model even named an entity from the prompt.
    # Naming the wrong one is a failed comparison; naming nothing from the prompt is a
    # failure to engage with the question format at all.
    wrong_entity = [r for r in wrong if r["first_word"] and r["first_word"] in r["prompt"]]
    off_format = [r for r in wrong if not (r["first_word"] and r["first_word"] in r["prompt"])]

    return {
        "successes": [trim(r) for r in correct[:limit]],
        "wrong_entity": [trim(r) for r in wrong_entity[:limit]],
        "off_format": [trim(r) for r in off_format[:limit]],
    }


def run(args: argparse.Namespace) -> None:
    """Evaluate both checkpoints and write the comparison."""
    import sentencepiece as spm

    lang_dir = Path(LANGUAGE_DIRS[args.lang])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    sp = spm.SentencePieceProcessor(
        model_file=str(lang_dir / "tokenizer" / TOKENIZER_NAMES[args.lang]))
    eos_id = sp.eos_id() if sp.eos_id() >= 0 else 3

    data_dir = Path(args.data_dir) if args.data_dir else lang_dir / "reasoning"
    with open(data_dir / f"{args.split}.jsonl", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    if args.limit:
        records = records[:args.limit]

    checkpoints = {
        "pretrained": Path(args.pretrained) if args.pretrained
        else Path("checkpoints") / lang_dir.name / "best.pt",
        "finetuned": Path(args.finetuned) if args.finetuned
        else Path("checkpoints") / f"{lang_dir.name}-finetuned" / "best.pt",
    }

    print(f"language={args.lang}  split={args.split}  examples={len(records):,}  "
          f"device={device}", flush=True)

    report: dict = {
        "language": args.lang,
        "split": args.split,
        "examples": len(records),
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "models": {},
    }

    for stage, path in checkpoints.items():
        model, payload = load_model(path, device)
        started = time.time()
        results = score_model(model, records, sp, eos_id, device,
                              args.batch_size, args.max_new_tokens)
        elapsed = time.time() - started

        summary = aggregate(results)
        summary["checkpoint"] = str(path)
        summary["checkpoint_step"] = payload.get("step")
        summary["seconds"] = round(elapsed, 1)
        report["models"][stage] = summary
        report.setdefault("qualitative", {})[stage] = qualitative(results, args.examples)

        print(f"  {stage:11s} step {payload.get('step'):>6}  "
              f"exact {summary['exact_match_strict']:.4f} / "
              f"lenient {summary['exact_match_lenient']:.4f} / "
              f"first-word {summary['first_word_match']:.4f}  "
              f"(chance {summary['chance_baseline']:.4f})  {elapsed:.0f}s", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pre = report["models"]["pretrained"]["exact_match_strict"]
    post = report["models"]["finetuned"]["exact_match_strict"]
    report["improvement_strict"] = round(post - pre, 4)
    print(f"\n  improvement: {pre:.4f} -> {post:.4f}  "
          f"({post - pre:+.4f} absolute)", flush=True)

    out_dir = Path("report") / lang_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reasoning_eval.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"  written -> {out_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pretrained vs finetuned exact match on the reasoning test set."
    )
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--pretrained", default=None,
                        help="default: checkpoints/<language>/best.pt")
    parser.add_argument("--finetuned", default=None,
                        help="default: checkpoints/<language>-finetuned/best.pt")
    parser.add_argument("--data-dir", default=None, help="default: <language>/reasoning")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--examples", type=int, default=12,
                        help="qualitative examples of each kind to keep (default: 12)")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N examples, for a quick check")
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main() -> None:
    """Entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
