"""Side-by-side comparison tables for Model H and Model L.

The assignment asks for every Phase 2 metric to be reported "for both models side by
side", and separately for a resource-level comparison write-up. Every artifact the
pipeline produces, however, is per-language: each Kaggle session trains and evaluates
one model and knows nothing about the other. Something has to join them, and doing it
by hand means copying roughly a hundred numbers into a report, which is how transcription
errors get in.

This script reads the JSON artifacts for both languages and emits the tables as Markdown,
ready to paste into ``report/phase2.md``. Every figure it prints is read from a file the
pipeline wrote, so the report cannot silently disagree with the artifacts.

It also pulls two Phase 1 numbers into the comparison -- corpus size and tokenizer
fertility -- because the specification names both as candidate explanations for the gap
between the two models, and neither appears in any Phase 2 artifact.

Usage
-----
    python -m scripts.compare_models                      # both languages
    python -m scripts.compare_models --only hi            # before the second run lands
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LANGUAGES = {"hi": "hindi", "ne": "nepali"}
LABELS = {"hi": "Model H (Hindi)", "ne": "Model L (Nepali)"}
DECODERS = ("greedy", "temp0.5", "temp1.0", "temp1.5")


def load_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it does not exist.

    Args:
        path: File to read.

    Returns:
        The parsed object, or None when the file is absent. Absence is normal while only
        one language has finished, so it is not an error.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gather(code: str, checkpoints_root: Path, report_root: Path) -> dict:
    """Collect every artifact for one language into a single dict.

    Args:
        code: Language code, ``hi`` or ``ne``.
        checkpoints_root: Directory holding ``<language>/config.json``.
        report_root: Directory holding ``<language>/*.json``.

    Returns:
        A dict of the loaded artifacts, with None for any that are missing.
    """
    name = LANGUAGES[code]
    return {
        "code": code,
        "label": LABELS[code],
        "checkpoint": load_json(checkpoints_root / name / "config.json"),
        "lm": load_json(report_root / name / "lm_eval.json"),
        "generation": load_json(report_root / name / "generation_eval.json"),
        "attention": load_json(report_root / name / "attention_stats.json"),
        # Phase 1 context, for the resource-level discussion.
        "tokenizer": load_json(report_root / name / "tokenizer_stats.json"),
        "splits": load_json(report_root / name / "split_stats.json"),
        # Phase 3.
        "reasoning": load_json(report_root / name / "reasoning_eval.json"),
        "reasoning_data": load_json(report_root / name / "reasoning_stats.json"),
        "attention_ft": load_json(report_root / name / "attention_stats_finetuned.json"),
    }


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table.

    Args:
        header: Column headings.
        rows: Row cells, already formatted as strings.

    Returns:
        The table as a Markdown string, with a trailing blank line.
    """
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def fertility(stats: dict | None) -> str:
    """Look up the fertility of the vocabulary that was actually selected.

    ``tokenizer_stats.json`` records one entry per candidate vocabulary size. The
    selected one is named separately, so the matching candidate has to be found rather
    than assumed to be first or last.

    Args:
        stats: Parsed ``tokenizer_stats.json``, or None.

    Returns:
        Fertility formatted to three decimals, or "--" when unavailable.
    """
    if not stats:
        return "--"
    chosen = stats.get("selected_vocab_size")
    for candidate in stats.get("candidates", []):
        if candidate.get("vocab_size") == chosen:
            return f"{candidate['fertility_tokens_per_word']:.3f}"
    return "--"


def chars_per_token(stats: dict | None) -> str:
    """Characters per token for the selected vocabulary, or "--"."""
    if not stats:
        return "--"
    chosen = stats.get("selected_vocab_size")
    for candidate in stats.get("candidates", []):
        if candidate.get("vocab_size") == chosen:
            return f"{candidate['chars_per_token']:.2f}"
    return "--"


def section_configuration(models: list[dict]) -> str:
    """Architecture and training configuration for each model.

    Included because the two models must be shown to be independent but comparable:
    identical hyperparameters mean any difference in the results is attributable to the
    corpus rather than to the setup.
    """
    fields = [
        ("Parameters (total)", lambda m: f"{m['checkpoint']['parameter_counts']['total']:,}"),
        ("Parameters (non-embedding)", lambda m: f"{m['checkpoint']['parameter_counts']['non_embedding']:,}"),
        ("Vocabulary", lambda m: f"{m['checkpoint']['model']['vocab_size']:,}"),
        ("Layers / heads / d_model", lambda m: f"{m['checkpoint']['model']['n_layer']} / "
                                               f"{m['checkpoint']['model']['n_head']} / "
                                               f"{m['checkpoint']['model']['d_model']}"),
        ("Context length", lambda m: f"{m['checkpoint']['model']['max_seq_len']}"),
        ("Optimiser steps", lambda m: f"{m['checkpoint']['last_step']:,}"),
        ("Training tokens seen", lambda m: f"{m['checkpoint']['tokens_seen']:,}"),
        ("Peak learning rate", lambda m: f"{m['checkpoint']['train']['learning_rate']:.1e}"),
    ]
    rows = []
    for name, get in fields:
        cells = [name]
        for m in models:
            cells.append(get(m) if m["checkpoint"] else "--")
        rows.append(cells)
    return table(["", *[m["label"] for m in models]], rows)


def section_corpus(models: list[dict]) -> str:
    """Phase 1 corpus and tokenizer figures, as context for the gap between the models."""
    rows = []
    for name, get in [
        ("Training documents",
         lambda m: f"{m['splits']['splits']['train']['documents']:,}" if m["splits"] else "--"),
        ("Training words",
         lambda m: f"{m['splits']['splits']['train']['words']:,}" if m["splits"] else "--"),
        ("Manual-collection share of words",
         lambda m: f"{100 * m['splits']['splits']['train']['manual_fraction_of_words']:.1f}%"
         if m["splits"] else "--"),
        ("Tokenizer fertility (tokens/word)", lambda m: fertility(m["tokenizer"])),
        ("Characters per token", lambda m: chars_per_token(m["tokenizer"])),
    ]:
        rows.append([name, *[get(m) for m in models]])
    return table(["", *[m["label"] for m in models]], rows)


def section_language_modelling(models: list[dict]) -> str:
    """Cross-entropy, perplexity and bits-per-byte on both held-out splits.

    Bits-per-byte is the figure to compare across languages: perplexity is per token and
    therefore depends on how aggressively each tokenizer segments its own language, so
    two models with different vocabularies are not directly comparable by perplexity.
    """
    rows = []
    for split in ("validation", "test"):
        for metric, key, fmt in (("Cross-entropy (nats)", "cross_entropy_nats", "{:.4f}"),
                                 ("Perplexity", "perplexity", "{:.2f}"),
                                 ("Bits per byte", "bits_per_byte", "{:.4f}")):
            cells = [f"{split.capitalize()} — {metric}"]
            for m in models:
                lm = m["lm"]
                cells.append(fmt.format(lm["splits"][split][key]) if lm else "--")
            rows.append(cells)
    return table(["", *[m["label"] for m in models]], rows)


def section_generation(models: list[dict]) -> str:
    """Generation quality and diversity, one block per decoding setting."""
    order = [
        ("BLEU", "bleu", "{:.2f}"),
        ("chrF", "chrf", "{:.2f}"),
        ("chrF++", "chrf_pp", "{:.2f}"),
        ("ROUGE-L (F1)", "rouge_l_f1", "{:.4f}"),
        ("Distinct-1 (tokens)", "distinct_1_tokens", "{:.4f}"),
        ("Distinct-2 (tokens)", "distinct_2_tokens", "{:.4f}"),
        ("Repetition, 4-gram (tokens)", "repetition_4gram_tokens", "{:.4f}"),
    ]
    blocks = []
    for decoder in DECODERS:
        rows = []
        for name, key, fmt in order:
            cells = [name]
            for m in models:
                gen = m["generation"]
                value = gen["settings"].get(decoder, {}).get(key) if gen else None
                cells.append(fmt.format(value) if value is not None else "--")
            rows.append(cells)
        blocks.append(f"**{decoder}**\n\n" + table(["", *[m["label"] for m in models]], rows))
    return "\n".join(blocks)


def section_attention(models: list[dict]) -> str:
    """Head-type census and per-layer means.

    The census answers the question the specification actually asks -- which heads look
    local and which look content-based -- in a form that can be compared between the two
    models at a glance.
    """
    labels = ["local", "mixed", "long-range", "diffuse"]
    rows = []
    for label in labels:
        cells = [f"Heads classified *{label}*"]
        for m in models:
            att = m["attention"]
            if not att:
                cells.append("--")
                continue
            counts = Counter(h["label"] for h in att["per_head"])
            cells.append(f"{counts.get(label, 0)} / {len(att['per_head'])}")
        rows.append(cells)

    for name, key, fmt in (("Mean entropy across heads (nats)", "entropy_nats", "{:.3f}"),
                           ("Mean attention distance (positions)", "mean_distance", "{:.2f}"),
                           ("Mean weight on previous token", "previous_share", "{:.4f}")):
        cells = [name]
        for m in models:
            att = m["attention"]
            if not att:
                cells.append("--")
                continue
            values = [h[key] for h in att["per_head"]]
            cells.append(fmt.format(sum(values) / len(values)))
        rows.append(cells)

    out = table(["", *[m["label"] for m in models]], rows)

    # The single most specialised head in each model, by lowest entropy. This is the
    # concrete claim worth making in the report, so surface it rather than leaving it
    # to be spotted by eye in a 49-row table.
    lines = ["\nMost specialised head in each model (lowest entropy):\n"]
    for m in models:
        att = m["attention"]
        if not att:
            lines.append(f"- {m['label']}: --")
            continue
        head = min(att["per_head"], key=lambda h: h["entropy_nats"])
        lines.append(
            f"- {m['label']}: layer {head['layer']}, head {head['head']} — "
            f"entropy {head['entropy_nats']:.3f} nats, "
            f"{100 * head['previous_share']:.1f}% of its mass on the previous token, "
            f"mean distance {head['mean_distance']:.2f}"
        )
    return out + "\n".join(lines) + "\n"


def section_reasoning(models: list[dict]) -> str:
    """Pretrained against finetuned exact match, with the chance floor beside it.

    The chance column is not decoration. A pretrained model that continues the prompt as
    prose scores zero on exact match however well it understands the question, and its
    first-word score can look like partial competence until it is read against the floor.
    """
    rows = []
    for label, stage, key in (
        ("Exact match — pretrained", "pretrained", "exact_match_strict"),
        ("Exact match — **finetuned**", "finetuned", "exact_match_strict"),
        ("Lenient match — finetuned", "finetuned", "exact_match_lenient"),
        ("First-word match — pretrained", "pretrained", "first_word_match"),
        ("First-word match — finetuned", "finetuned", "first_word_match"),
        ("Chance baseline", "finetuned", "chance_baseline"),
    ):
        cells = [label]
        for m in models:
            r = m["reasoning"]
            cells.append(f"{r['models'][stage][key]:.4f}" if r else "--")
        rows.append(cells)

    cells = ["Test examples"]
    for m in models:
        cells.append(f"{m['reasoning']['examples']:,}" if m["reasoning"] else "--")
    rows.append(cells)

    return table(["", *[m["label"] for m in models]], rows)


def section_reasoning_breakdown(models: list[dict]) -> str:
    """Finetuned accuracy per template family and per reasoning depth.

    Splitting by template is what separates the two regimes the models are actually in:
    symbolic comparison over named entities, which they learn, and numeric comparison,
    which they do not. Hop count alone cannot show that -- depth and template type are
    confounded by how the dataset is built, since every one-hop item is numeric.
    """
    first = next((m for m in models if m["reasoning"]), None)
    if first is None:
        return "_No reasoning evaluation found._\n"

    families = sorted(first["reasoning"]["models"]["finetuned"]["by_template"])
    rows = []
    for family in families:
        cells = [f"`{family}`"]
        chance = None
        for m in models:
            r = m["reasoning"]
            if not r:
                cells.append("--")
                continue
            entry = r["models"]["finetuned"]["by_template"][family]
            cells.append(f"{entry['strict']:.4f}")
            chance = entry["chance"]
        cells.append(f"{chance:.4f}" if chance is not None else "--")
        rows.append(cells)
    out = table(["Template", *[m["label"] for m in models], "chance"], rows)

    hops = sorted(first["reasoning"]["models"]["finetuned"]["by_hops"], key=int)
    rows = []
    for hop in hops:
        cells = [f"{hop} hop" + ("s" if hop != "1" else "")]
        chance = None
        for m in models:
            r = m["reasoning"]
            if not r:
                cells.append("--")
                continue
            entry = r["models"]["finetuned"]["by_hops"][hop]
            cells.append(f"{entry['strict']:.4f}")
            chance = entry["chance"]
        cells.append(f"{chance:.4f}" if chance is not None else "--")
        rows.append(cells)

    return (out + "\nBy reasoning depth. More hops scores *higher*, because depth and "
            "template type are confounded by construction: every one-hop item is a "
            "numeric comparison and every two- and three-hop item is a symbolic chain.\n\n"
            + table(["Depth", *[m["label"] for m in models], "chance"], rows))


def main() -> None:
    """Write the side-by-side comparison tables as Markdown."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"),
                        help="directory holding <language>/config.json")
    parser.add_argument("--report", type=Path, default=Path("report"),
                        help="directory holding <language>/*.json")
    parser.add_argument("--only", choices=sorted(LANGUAGES), default=None,
                        help="restrict to one language, for use before the second run lands")
    parser.add_argument("--out", type=Path, default=Path("report/comparison_tables.md"))
    args = parser.parse_args()

    codes = [args.only] if args.only else ["hi", "ne"]
    models = [gather(c, args.checkpoints, args.report) for c in codes]

    for m in models:
        missing = [k for k in ("checkpoint", "lm", "generation", "attention") if not m[k]]
        if missing:
            print(f"  note: {m['label']} is missing {', '.join(missing)} "
                  f"- those cells will read '--'")

    parts = [
        "## Model and training configuration\n", section_configuration(models),
        "\n## Corpus and tokenizer (Phase 1)\n", section_corpus(models),
        "\n## Intrinsic language-modelling metrics\n", section_language_modelling(models),
        "\n## Generation quality and diversity\n", section_generation(models),
        "\n## Attention summary\n", section_attention(models),
        "\n## Reasoning: pretrained against finetuned (Phase 3)\n", section_reasoning(models),
        "\n## Reasoning breakdown (Phase 3)\n", section_reasoning_breakdown(models),
    ]
    body = "".join(parts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "<!-- Generated by scripts/compare_models.py. Do not edit by hand: re-run the\n"
        "     script instead, so the report cannot drift from the artifacts. -->\n\n"
        + body, encoding="utf-8")
    print(body)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
