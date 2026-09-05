"""Loss curves and learning-rate schedule from a training log.

``scripts/pretrain.py`` appends one JSON object per event to ``train_log.jsonl``. This
turns that into the figures the report needs. Reading the log rather than instrumenting
the trainer means curves can be redrawn at any time, including for a run that was
interrupted and resumed several times — the log is append-only across sessions.

Training and validation loss are drawn on one axis so the gap between them is visible;
that gap is what a reader looks at to judge overfitting. With roughly 0.7 of one epoch
over the corpus, no document is seen twice, so the two curves should track each other
closely — if validation diverges upward anyway, something is wrong with the data rather
than with the capacity.

Both models are plotted together where a comparison figure is requested, since the whole
point of the project is Model H against Model L.

Usage::

    python -m scripts.plot_training --lang hi
    python -m scripts.plot_training --compare
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LANGUAGE_DIRS = {"hi": "hindi", "ne": "nepali"}
LANGUAGE_NAMES = {"hi": "Hindi (Model H)", "ne": "Nepali (Model L)"}


def read_log(path: Path) -> dict[str, list]:
    """Parse a training log into parallel series.

    Args:
        path: Path to ``train_log.jsonl``.

    Returns:
        Series for training loss, validation loss, learning rate and throughput, each as
        ``(steps, values)`` pairs plus the tokens-seen axis.

    Raises:
        FileNotFoundError: If the log does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"no training log at {path}")

    series: dict[str, list] = {
        "train_steps": [], "train_loss": [], "train_tokens": [],
        "val_steps": [], "val_loss": [], "val_tokens": [],
        "lr_steps": [], "lr": [], "tps_steps": [], "tps": [],
    }

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line truncated by a hard kill; skip rather than fail
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            series["train_steps"].append(step)
            series["train_loss"].append(entry["loss"])
            series["train_tokens"].append(entry.get("tokens_seen", 0))
        if "val_loss" in entry:
            series["val_steps"].append(step)
            series["val_loss"].append(entry["val_loss"])
            series["val_tokens"].append(entry.get("tokens_seen", 0))
        if "lr" in entry:
            series["lr_steps"].append(step)
            series["lr"].append(entry["lr"])
        if "tokens_per_second" in entry:
            series["tps_steps"].append(step)
            series["tps"].append(entry["tokens_per_second"])

    return series


def plot_loss(series: dict[str, list], language: str, out_path: Path) -> None:
    """Draw training and validation loss, with a perplexity axis on the right.

    Args:
        series: Parsed log series.
        language: Display name for the title.
        out_path: Destination PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(series["train_steps"], series["train_loss"],
            linewidth=1.0, alpha=0.75, label="Training loss")
    if series["val_steps"]:
        ax.plot(series["val_steps"], series["val_loss"],
                marker="o", markersize=4, linewidth=1.6, label="Validation loss")

    ax.set_title(f"{language}: pretraining loss")
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Cross-entropy loss (nats per token)")
    ax.grid(alpha=0.3)
    ax.legend()

    # Second axis showing the same values as perplexity, which is what most readers
    # have intuition for. It is a relabelling of the left axis, not new data.
    right = ax.twinx()
    low, high = ax.get_ylim()
    right.set_ylim(low, high)
    ticks = ax.get_yticks()
    right.set_yticks(ticks)
    right.set_yticklabels([f"{math.exp(min(t, 20)):.0f}" for t in ticks])
    right.set_ylabel("Perplexity")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_schedule(series: dict[str, list], language: str, out_path: Path) -> None:
    """Draw the learning-rate schedule that was actually applied.

    Plotting the logged rate rather than recomputing it means the figure shows what the
    run did, including any resume that restarted the schedule incorrectly.

    Args:
        series: Parsed log series.
        language: Display name for the title.
        out_path: Destination PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(series["lr_steps"], series["lr"], linewidth=1.4, label="Learning rate")
    ax.set_title(f"{language}: learning-rate schedule (linear warmup, cosine decay)")
    ax.set_xlabel("Optimiser step")
    ax.set_ylabel("Learning rate")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(all_series: dict[str, dict], out_path: Path) -> None:
    """Overlay both models' validation loss against tokens seen.

    Tokens rather than steps on the x-axis, so the comparison is fair if the two runs
    ever use different batch sizes.

    Args:
        all_series: Parsed series keyed by language code.
        out_path: Destination PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for lang, series in all_series.items():
        if not series["val_steps"]:
            continue
        ax.plot([t / 1e6 for t in series["val_tokens"]], series["val_loss"],
                marker="o", markersize=4, linewidth=1.6, label=LANGUAGE_NAMES[lang])

    ax.set_title("Model H against Model L: validation loss")
    ax.set_xlabel("Training tokens seen (millions)")
    ax.set_ylabel("Validation cross-entropy (nats per token)")
    ax.grid(alpha=0.3)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Render loss and schedule figures for one or both models."""
    parser = argparse.ArgumentParser(description="Loss curves from a training log.")
    parser.add_argument("--lang", choices=sorted(LANGUAGE_DIRS), default=None)
    parser.add_argument("--compare", action="store_true", help="also draw the H-vs-L figure")
    parser.add_argument("--log", default=None, help="override the log path")
    args = parser.parse_args()

    languages = [args.lang] if args.lang else sorted(LANGUAGE_DIRS)
    all_series = {}

    for lang in languages:
        root = Path(LANGUAGE_DIRS[lang])
        log_path = Path(args.log) if args.log else Path("checkpoints") / root.name / "train_log.jsonl"
        try:
            series = read_log(log_path)
        except FileNotFoundError as exc:
            print(f"  skipping {lang}: {exc}")
            continue
        all_series[lang] = series

        out_dir = Path("report") / root.name / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_loss(series, LANGUAGE_NAMES[lang], out_dir / f"{lang}_loss.png")
        plot_schedule(series, LANGUAGE_NAMES[lang], out_dir / f"{lang}_lr_schedule.png")
        best = min(series["val_loss"]) if series["val_loss"] else float("nan")
        print(f"  {lang}: {len(series['train_steps'])} logged steps, "
              f"best validation loss {best:.4f} -> {out_dir}")

    if args.compare and len(all_series) > 1:
        out = Path("report") / "figures"
        out.mkdir(parents=True, exist_ok=True)
        plot_comparison(all_series, out / "validation_loss_comparison.png")
        print(f"  wrote {out/'validation_loss_comparison.png'}")


if __name__ == "__main__":
    main()
