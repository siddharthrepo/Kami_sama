# Monolingual Transformer LMs for Hindi and Nepali

Language Models and Agents — Monsoon 2026, Individual Project.

Two **completely independent** decoder-only Transformer language models built from
scratch: Hindi (Model H, higher-resource) and Nepali (Model L, from the permitted
lower-resource list). Separate corpora, separate tokenizers, separate vocabularies,
separate weights. Nothing is shared between them except library code.

**Phase 1 (data and tokenizers) is complete.** See [`report/phase1.md`](report/phase1.md)
for the full write-up.

## Why these two languages

Both use **Devanagari**. That is deliberate: it removes script as a confounding
variable, so any difference between Model H and Model L reflects the amount and quality
of available data rather than the writing system.

The cost of that choice is that language identification becomes real work — Hindi,
Nepali, Marathi, Bhojpuri and Maithili all share the script, so documents cannot be
separated by Unicode range alone. That cost is measurable: language filtering removed
7,832 documents from the Nepali corpus against 31 from the Hindi one.

## Phase 1 results

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Documents | 947,298 | 1,278,483 |
| Words | 535,501,269 | 506,993,066 |
| **Tokens** | **662,356,265** | **659,293,660** |
| **Manual tokens** | **185,075,090 (27.94%)** | **168,487,902 (25.56%)** |
| Vocabulary | 32,000 | 32,000 |
| Fertility (tokens/word) | 1.237 | 1.300 |
| Characters per token | 4.10 | 4.96 |
| Vocabulary utilisation | 99.6% | 99.7% |

Both corpora exceed the ~500M token target, and both exceed the required 20% manual
collection share. Token counts are **measured** by encoding the corpus with the trained
tokenizer, not estimated.

## Sources

**Manual (collected by us).** Scraped through WordPress REST APIs, which return complete
article bodies as structured JSON — far more efficient and far gentler on the sites than
page-by-page HTML scraping.

| Language | Site | Documents |
|---|---|---|
| Hindi | Jansatta | 266,296 |
| Hindi | The Wire Hindi | 37,480 |
| Nepali | Onlinekhabar | 354,793 (entire archive) |

eKantipur was deliberately excluded: its `robots.txt` carries the Cloudflare
content-signals preamble reserving rights against `ai-train` use.

**Downloaded (public corpora).**

| Language | Dataset | Subset |
|---|---|---|
| Hindi | `HuggingFaceFW/fineweb-2` | `hin_Deva` |
| Hindi | `wikimedia/wikipedia` | `20231101.hi` |
| Nepali | `HuggingFaceFW/fineweb-2` | `npi_Deva` |
| Nepali | `wikimedia/wikipedia` | `20231101.ne` |

CC-100 is not used — its HuggingFace loader is script-based and no longer executes.
AI4Bharat Sangraha is configured but was never reached, as the token budget filled first.

## Repository layout

```
common/          shared library code — schema, fetching, cleaning, language ID
scripts/         command-line entry points
hindi/           Model H: config, tokenizer, corpus (gitignored)
nepali/          Model L: config, tokenizer, corpus (gitignored)
report/          phase reports, statistics JSON, figures
```

Corpora and tokenizer training samples are **not committed** — see the Drive links below.

## Reproduction

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**1. Collect the manual share.** Sharded by month, one file per window, resumable —
re-running the same command skips completed windows.

```bash
.venv/bin/python -m scripts.scrape --lang hi --out data/manual/hi \
  --sources jansatta thewirehindi --limit 300000 --workers 8 --delay 0.25

.venv/bin/python -m scripts.scrape --lang ne --out data/manual/ne \
  --sources onlinekhabar --limit 400000 --workers 8 --delay 0.25
```

**2. Download the public share.** Streamed and filtered on the fly; add `--probe` to
check source availability without downloading.

```bash
.venv/bin/python -m scripts.download --lang hi --out data/downloaded/hi --max-tokens 450000000
.venv/bin/python -m scripts.download --lang ne --out data/downloaded/ne --max-tokens 450000000
```

**3. Clean.** Merges both halves, normalises Unicode, filters by language, deduplicates.
Manual shards are read first so they win deduplication ties.

```bash
.venv/bin/python -m scripts.clean --config hindi/configs/dataset.json
.venv/bin/python -m scripts.clean --config nepali/configs/dataset.json
```

**4. Split** into train/validation/test (98/1/1, document-level, seed 1337).

```bash
.venv/bin/python -m scripts.split --config hindi/configs/dataset.json
.venv/bin/python -m scripts.split --config nepali/configs/dataset.json
```

**5. Train tokenizers.** Trains four candidate vocabulary sizes, evaluates each on
held-out validation text, and installs the selected one.

```bash
.venv/bin/python -m scripts.train_tokenizer --config hindi/configs/dataset.json \
  --vocab-sizes 8000 16000 32000 48000 --sample-lines 2000000

.venv/bin/python -m scripts.train_tokenizer --config nepali/configs/dataset.json \
  --vocab-sizes 8000 16000 32000 48000 --sample-lines 2000000
```

**6. Measure the corpus** and render the report figures.

```bash
.venv/bin/python -m scripts.corpus_stats --config hindi/configs/dataset.json
.venv/bin/python -m scripts.corpus_stats --config nepali/configs/dataset.json
```

Total runtime on a 12-core CPU machine: roughly 4 hours, dominated by collection.
No GPU is required for Phase 1.

### Packaging artifacts for Google Drive

Bundles the large artifacts into archives with checksums, and generates the rows for the
link table below.

```bash
.venv/bin/python -m scripts.package_for_drive --out drive_upload
```

## Third-party models used for data cleaning

Two pretrained components are used, both for **data preparation only**. Neither is a
language model nor a tokenizer, so neither falls under the assignment's prohibition:

- **fastText `lid.176`** (`common/langid.py`) — language identification, to separate
  Hindi from Nepali, Marathi and Bhojpuri. Downloaded automatically on first use.
- **trafilatura** (`scripts/scrape.py`, `common/wordpress.py`) — selects the article body
  within an HTML page.

The tokenizers in `hindi/tokenizer/` and `nepali/tokenizer/` are trained from scratch by
`scripts/train_tokenizer.py`.

## Google Drive links

Produced by `scripts/package_for_drive.py` (default `--profile full`, ~11.1 GB total).
Set each file to "Anyone with the link can view" so graders do not have to request access.

_Links to be added before submission._

| Archive | Contents | Size | Link |
|---|---|---|---|
| `hindi-manual-corpus.tar` | Hindi scraped shards — Jansatta, The Wire Hindi | 345 MB | _pending_ |
| `nepali-manual-corpus.tar` | Nepali scraped shards — Onlinekhabar | 381 MB | _pending_ |
| `hindi-downloaded-corpus.tar` | Hindi FineWeb-2 + Wikipedia shards | 930 MB | _pending_ |
| `nepali-downloaded-corpus.tar` | Nepali FineWeb-2 + Wikipedia shards | 1.2 GB | _pending_ |
| `hindi-clean-corpus.tar` | Hindi corpus after cleaning, before splitting | 1.3 GB | _pending_ |
| `nepali-clean-corpus.tar` | Nepali corpus after cleaning, before splitting | 1.5 GB | _pending_ |
| `hindi-splits.tar` | Hindi train/validation/test — input to Phase 2 | 1.3 GB | _pending_ |
| `nepali-splits.tar` | Nepali train/validation/test — input to Phase 2 | 1.5 GB | _pending_ |
| `tokenizers.tar` | Both tokenizers, all four candidate vocab sizes, training samples | 2.4 GB | _pending_ |
| `reports-and-logs.tar` | Statistics JSON, figures, phase report, all run logs | 5 MB | _pending_ |

The trained tokenizers (`hi.model`, `hi.vocab`, `ne.model`, `ne.vocab` — 3.3 MB total) are
small enough to be committed directly and are in this repository; the Drive copy is a
backup that also carries the intermediate vocabulary sizes and the SentencePiece training
samples.
