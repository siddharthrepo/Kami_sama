"""Attention heatmaps and per-head summary statistics.

Two deliverables come out of this script. The heatmaps show what individual heads do on
real sentences in the model's own language — an early layer, a late layer, and several
heads from each. The summary tables give entropy, mean attention distance,
self-attention share and previous-token share for every head in every layer, averaged
over many held-out batches, so statements about head specialisation rest on aggregates
rather than on one cherry-picked sentence.

Following the assignment's guideline that visualisation and computation stay in separate
functions, everything that computes lives in :mod:`lma.attention` and the functions here
named ``plot_*`` do nothing but draw. Every figure carries a title, axis labels and,
where more than one series is present, a legend.

A note on fonts: axis ticks are Devanagari tokens, which matplotlib renders as empty
boxes unless a Devanagari-capable font is configured. :func:`configure_devanagari_font`
selects one if available and otherwise falls back to numeric position labels, so the
figures stay readable on a machine without Indic fonts installed (Kaggle, for instance).

Usage::

    python -m scripts.attention_analysis --lang hi --checkpoint hindi/checkpoints/best.pt
    python -m scripts.attention_analysis --lang ne --num-batches 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a headless runner
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sentencepiece as spm  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.attention import classify_head, summarise_layer  # noqa: E402
from lma.checkpoint import load_checkpoint  # noqa: E402
from lma.config import ModelConfig  # noqa: E402
from lma.data import TokenStream, to_device  # noqa: E402
from lma.model import GPT  # noqa: E402

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}

DEVANAGARI_FONTS = (
    "Noto Sans Devanagari", "Noto Serif Devanagari", "Lohit Devanagari",
    "Nirmala UI", "Samyak Devanagari", "Kalimati", "FreeSerif",
)


def configure_devanagari_font() -> bool:
    """Point matplotlib at a font that can draw Devanagari.

    Returns:
        True if one was found and configured. When False, callers should label axes with
        positions instead of tokens — empty boxes convey nothing and look like a bug.
    """
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in DEVANAGARI_FONTS:
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return True
    print("  [font] no Devanagari font found; axes will use token positions", flush=True)
    return False


# ----------------------------------------------------------------- computation

@torch.no_grad()
def collect_statistics(
    model: GPT, stream: TokenStream, *, batch_size: int, seq_len: int,
    device: torch.device, num_batches: int,
) -> list[dict[str, list[float]]]:
    """Average the per-head statistics over several held-out batches.

    Args:
        model: Trained model in eval mode.
        stream: Held-out token stream.
        batch_size: Sequences per batch. Kept small — attention is ``(B, h, T, T)``, so
            memory grows quadratically in sequence length.
        seq_len: Sequence length.
        device: Device to run on.
        num_batches: Batches to average over.

    Returns:
        One dict of per-head statistic lists per layer.
    """
    totals: list[dict[str, np.ndarray]] = []
    seen = 0

    # spread=True for the same reason as in-training validation: the splits are not
    # shuffled across sources, so 20 batches from the head of the file describe one
    # kind of text rather than the split as a whole.
    for batch in stream.sequential_batches(batch_size, seq_len, limit=num_batches,
                                          spread=True):
        x, _ = to_device(batch, device)
        _, _, attentions = model(x, return_attention=True)

        for index, layer in enumerate(attentions):
            summary = {k: np.asarray(v) for k, v in summarise_layer(layer).items()}
            if index == len(totals):
                totals.append(summary)
            else:
                for key in totals[index]:
                    totals[index][key] = totals[index][key] + summary[key]
        seen += 1

    if seen == 0:
        raise ValueError("no batches available for attention statistics")
    return [{k: (v / seen).tolist() for k, v in layer.items()} for layer in totals]


@torch.no_grad()
def attention_for_sentence(
    model: GPT, tokenizer: spm.SentencePieceProcessor, text: str,
    *, device: torch.device, max_tokens: int,
) -> tuple[list[torch.Tensor], list[str]]:
    """Run one sentence through the model and return its attention plus token pieces.

    Args:
        model: Trained model.
        tokenizer: The language's SentencePiece model.
        text: A sentence in the model's language.
        device: Device to run on.
        max_tokens: Truncate to this many tokens so the heatmap stays legible.

    Returns:
        ``(attentions, pieces)`` — per-layer ``(1, h, T, T)`` tensors, and the token
        strings for axis labels with SentencePiece's word-boundary marker stripped.
    """
    ids = tokenizer.encode(text)[:max_tokens]
    pieces = [p.replace("▁", " ").strip() or "_" for p in tokenizer.encode(text, out_type=str)[:max_tokens]]
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, attentions = model(tensor, return_attention=True)
    return attentions, pieces


# ---------------------------------------------------------------- visualisation

def plot_head_heatmaps(
    attention: torch.Tensor, pieces: list[str], layer: int, heads: list[int],
    out_path: Path, language: str, use_tokens: bool,
) -> None:
    """Draw a grid of attention heatmaps, one panel per head.

    Args:
        attention: One layer's attention, shape ``(1, h, T, T)``.
        pieces: Token strings for the axis ticks.
        layer: Layer index, used in the title.
        heads: Head indices to draw.
        out_path: Destination PNG.
        language: Language name for the title.
        use_tokens: Label ticks with tokens; when False, use positions.
    """
    # Wrap into a grid rather than a single row. Seven panels side by side is a figure
    # five thousand pixels wide, which becomes illegible the moment it is embedded in a
    # report at page width; four columns keeps each panel readable.
    n = len(heads)
    n_cols = min(4, n)
    n_rows = -(-n // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 5.0 * n_rows),
                             squeeze=False)
    T = attention.shape[-1]

    flat = [ax for row in axes for ax in row]
    for ax in flat[n:]:
        ax.axis("off")

    for ax, head in zip(flat, heads):
        weights = attention[0, head].float().cpu().numpy()
        image = ax.imshow(weights, cmap="viridis", aspect="auto", vmin=0.0, origin="upper")
        ax.set_title(f"Layer {layer}, head {head}")
        ax.set_xlabel("Key position (attended to)")
        ax.set_ylabel("Query position (attending from)")

        if use_tokens and T <= 24:
            ax.set_xticks(range(T)); ax.set_xticklabels(pieces, rotation=90, fontsize=7)
            ax.set_yticks(range(T)); ax.set_yticklabels(pieces, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.046, label="Attention weight")

    fig.suptitle(f"{language}: causal self-attention, layer {layer}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_statistic_by_layer(
    statistics: list[dict[str, list[float]]], key: str, ylabel: str, title: str, out_path: Path,
) -> None:
    """Plot one per-head statistic across layers, one line per head.

    Args:
        statistics: Per-layer summaries from :func:`collect_statistics`.
        key: Which statistic to plot.
        ylabel: Y-axis label including units.
        title: Figure title.
        out_path: Destination PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    n_heads = len(statistics[0][key])
    layers = range(len(statistics))

    for head in range(n_heads):
        ax.plot(list(layers), [statistics[l][key][head] for l in layers],
                marker="o", label=f"head {head}")

    ax.set_title(title)
    ax.set_xlabel("Layer (0 = closest to the embeddings)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(layers))
    ax.grid(alpha=0.3)
    ax.legend(title="Attention head", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Produce heatmaps, statistic plots and a JSON summary for one model."""
    parser = argparse.ArgumentParser(description="Attention analysis for one model.")
    parser.add_argument("--lang", required=True, choices=sorted(LANGUAGE_DIRS))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokens-dir", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="layers to draw heatmaps for. Default: earliest, middle "
                             "and last. The spec asks for an early and a late layer; "
                             "the middle one is included because that is where the "
                             "sharply specialised heads tend to sit, and a figure that "
                             "skips it understates what the model learned.")
    parser.add_argument("--sentence", default=None, help="override the example sentence")
    parser.add_argument("--max-heatmap-tokens", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--heatmaps-only", action="store_true",
                        help="redraw only the example-sentence heatmaps. Needs the "
                             "checkpoint and tokenizer but not the token stream, so it "
                             "can regenerate figures locally after the fact -- for "
                             "instance when the training machine had no Devanagari font.")
    args = parser.parse_args()

    root = Path(LANGUAGE_DIRS[args.lang])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else root / "checkpoints" / "best.pt"
    tokens_dir = Path(args.tokens_dir) if args.tokens_dir else root / "data" / "tokens"
    out_dir = Path(args.out_dir) if args.out_dir else Path("report") / root.name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_tokens = configure_devanagari_font()

    payload = load_checkpoint(checkpoint_path, map_location="cpu", restore_rng=False)
    model_config = ModelConfig(**payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    tokenizer_path = next(p for p in (root / "tokenizer").glob("*.model") if "-vocab" not in p.name)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    language = "Hindi (Model H)" if args.lang == "hi" else "Nepali (Model L)"
    default_sentence = (
        "भारत की राजधानी नई दिल्ली है और यह देश का सबसे बड़ा शहर है।"
        if args.lang == "hi" else
        "नेपालको राजधानी काठमाडौं हो र यो देशको सबैभन्दा ठूलो सहर हो।"
    )
    sentence = args.sentence or default_sentence

    print(f"{language}  checkpoint={checkpoint_path}  step={payload['step']:,}\n", flush=True)

    # --- heatmaps on one example sentence, early layer and late layer
    attentions, pieces = attention_for_sentence(
        model, tokenizer, sentence, device=device, max_tokens=args.max_heatmap_tokens
    )
    layers = args.layers or sorted({0, model_config.n_layer // 2, model_config.n_layer - 1})
    for layer in layers:
        if not 0 <= layer < model_config.n_layer:
            raise SystemExit(f"layer {layer} out of range for a {model_config.n_layer}-layer model")
        path = out_dir / f"{args.lang}_attention_layer{layer}.png"
        plot_head_heatmaps(attentions[layer], pieces, layer, args.heads, path, language, use_tokens)
        print(f"  wrote {path}", flush=True)

    if args.heatmaps_only:
        # The statistics below need the held-out token stream, which is a multi-gigabyte
        # file that may not be present on the machine redrawing the figures. The existing
        # attention_stats.json already holds those numbers, so leave it untouched.
        if not use_tokens:
            print("\n  WARNING: still no Devanagari font - the axes will show positions "
                  "again. Install one (apt-get install fonts-indic) and clear "
                  "~/.cache/matplotlib/fontlist-*.json before re-running.")
        print("\nheatmaps only: statistics and attention_stats.json left unchanged")
        return

    # --- aggregate statistics over held-out batches
    stream = TokenStream(tokens_dir / f"{args.split}.bin")
    print(f"\naveraging attention statistics over {args.num_batches} batches ...", flush=True)
    statistics = collect_statistics(
        model, stream, batch_size=args.batch_size, seq_len=model_config.max_seq_len,
        device=device, num_batches=args.num_batches,
    )

    for key, ylabel, title in (
        ("entropy_nats", "Mean attention entropy (nats)",
         f"{language}: attention entropy by layer and head"),
        ("mean_distance", "Mean attention distance (positions)",
         f"{language}: how far back each head attends"),
        ("previous_share", "Mean weight on the previous token",
         f"{language}: previous-token attention by layer and head"),
    ):
        path = out_dir / f"{args.lang}_attention_{key}.png"
        plot_statistic_by_layer(statistics, key, ylabel, title, path)
        print(f"  wrote {path}", flush=True)

    # --- per-head labels and JSON summary
    print(f"\n{'layer':>5} {'head':>5} {'entropy':>9} {'norm':>6} {'distance':>9} "
          f"{'prev':>6} {'self':>6}  label", flush=True)
    labelled = []
    for layer, summary in enumerate(statistics):
        for head in range(len(summary["entropy_nats"])):
            label = classify_head(
                summary["mean_distance"][head],
                summary["entropy_normalised"][head],
                model_config.max_seq_len,
            )
            labelled.append({"layer": layer, "head": head, "label": label,
                             **{k: summary[k][head] for k in summary}})
            print(f"{layer:>5} {head:>5} {summary['entropy_nats'][head]:>9.3f} "
                  f"{summary['entropy_normalised'][head]:>6.3f} "
                  f"{summary['mean_distance'][head]:>9.2f} "
                  f"{summary['previous_share'][head]:>6.3f} "
                  f"{summary['self_share'][head]:>6.3f}  {label}", flush=True)

    out_json = Path("report") / root.name / "attention_stats.json"
    out_json.write_text(
        json.dumps({
            "language": args.lang, "checkpoint": str(checkpoint_path),
            "step": payload["step"], "split": args.split,
            "batches_averaged": args.num_batches, "context": model_config.max_seq_len,
            "example_sentence": sentence,
            "per_layer": statistics, "per_head": labelled,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
