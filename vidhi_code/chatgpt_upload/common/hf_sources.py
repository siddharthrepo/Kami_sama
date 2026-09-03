"""Registry of public HuggingFace corpora for the downloaded share of each corpus.

These supply the ~80% of tokens we do not collect ourselves. Everything here is streamed
rather than downloaded whole: CC-100's Hindi split alone is around 21 GB, and we only
need a fraction of it, so pulling the entire archive to disk would be wasteful and would
not fit locally anyway.

Each entry names a dataset, the config/subset for our language, and which field holds the
text. Availability is *not* assumed — dataset ids and config names drift, and some are
gated behind an access agreement. The downloader probes each one and reports what it
could actually reach, rather than failing the whole run on a single bad id.

Ordering matters: entries are consumed top-down until the token target is met, so the
cleanest and most useful corpora are listed first.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HFSource:
    """One streamable HuggingFace corpus.

    Attributes:
        name: Short stable identifier, stored on every document as ``source``.
        dataset: HuggingFace dataset repository id.
        config: Config / subset name, usually the language code. None if not needed.
        data_dir: Subdirectory within the repository, used by datasets that place each
            language in its own folder under a shared config (Sangraha does this).
        split: Split to stream, almost always ``"train"``.
        text_field: Field of each record holding the raw text.
        gated: True if the dataset requires accepting terms and an auth token. Listed so
            the report can explain why a source was or was not used.
        notes: Provenance and licence detail for the dataset statistics report.
    """

    name: str
    dataset: str
    config: str | None
    data_dir: str | None = None
    text_field: str = "text"
    split: str = "train"
    gated: bool = False
    notes: str = ""


# --------------------------------------------------------------------------- Hindi

HINDI_HF_SOURCES = [
    HFSource(
        name="wikipedia-hi",
        dataset="wikimedia/wikipedia",
        config="20231101.hi",
        notes="Hindi Wikipedia. Small but very clean, edited prose. CC BY-SA.",
    ),
    HFSource(
        name="fineweb2-hi",
        dataset="HuggingFaceFW/fineweb-2",
        config="hin_Deva",
        notes="FineWeb-2 Hindi. Heavily filtered and deduplicated CommonCrawl; the "
              "best quality-per-token of the large web corpora. ODC-By.",
    ),
    HFSource(
        name="sangraha-hi",
        dataset="ai4bharat/sangraha",
        config=None,
        # Sangraha stores each language as parquet under <split>/<iso3>/, so the folder
        # is addressed directly rather than through a config name.
        data_dir="verified/hin",
        notes="AI4Bharat Sangraha, verified Hindi subset. Indic-specific cleaning "
              "pipeline, so it complements the generic web corpora.",
    ),
]

# -------------------------------------------------------------------------- Nepali

NEPALI_HF_SOURCES = [
    HFSource(
        name="wikipedia-ne",
        dataset="wikimedia/wikipedia",
        config="20231101.ne",
        notes="Nepali Wikipedia. Clean edited prose, but small.",
    ),
    HFSource(
        name="fineweb2-ne",
        dataset="HuggingFaceFW/fineweb-2",
        config="npi_Deva",
        notes="FineWeb-2 Nepali. Filtered CommonCrawl; the largest clean Nepali web "
              "source available. ODC-By.",
    ),
    HFSource(
        name="sangraha-ne",
        dataset="ai4bharat/sangraha",
        config=None,
        data_dir="verified/nep",
        notes="AI4Bharat Sangraha, verified Nepali subset.",
    ),
]

# CC-100 is deliberately absent. Its HuggingFace loader was script-based, and dataset
# scripts are no longer executed, so `load_dataset("cc100", "hi")` now fails outright.
# FineWeb-2 supersedes it anyway: it is larger, more aggressively filtered, and already
# deduplicated. Worth stating in the report, since CC-100 is the corpus a reader would
# otherwise expect to see cited.

HF_SOURCES: dict[str, list[HFSource]] = {
    "hi": HINDI_HF_SOURCES,
    "ne": NEPALI_HF_SOURCES,
}


def get_hf_sources(lang: str, only: list[str] | None = None) -> list[HFSource]:
    """Return the configured HuggingFace sources for a language.

    Args:
        lang: ``"hi"`` or ``"ne"``.
        only: Optional list of source names to restrict to.

    Raises:
        ValueError: If the language is unknown or a requested name is not registered.
    """
    if lang not in HF_SOURCES:
        raise ValueError(f"Unknown language {lang!r}; expected one of {list(HF_SOURCES)}")

    sources = HF_SOURCES[lang]
    if only:
        by_name = {s.name: s for s in sources}
        missing = set(only) - set(by_name)
        if missing:
            raise ValueError(
                f"Unknown HF source(s) for {lang}: {sorted(missing)}. "
                f"Available: {sorted(by_name)}"
            )
        sources = [by_name[n] for n in only]
    return sources
