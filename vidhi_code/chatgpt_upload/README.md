# Monolingual Transformer LMs for Hindi and Nepali

**Language Models and Agents — Monsoon 2026, Individual Project**

Two **completely independent** decoder-only Transformer language models, trained
from scratch for:

- **Model H — Hindi:** higher-resource language
- **Model L — Nepali:** lower-resource language from the permitted list

The project is designed so that the two models have separate corpora, tokenizers,
vocabularies, configurations, checkpoints, and training artifacts. Shared code is
limited to reusable data-processing infrastructure.

> **Current status:** Phase 1 — corpus construction and tokenizer development —
> is complete.

See [`report/`](report/) for the Phase 1 report, statistics, and figures.


## Why these two languages

Both use **Devanagari**. That is deliberate: it removes script as a confounding
variable, so any difference between Model H and Model L reflects the amount and quality
of available data rather than the writing system.

The cost of that choice is that language identification becomes real work — Hindi,
Nepali, Marathi, Bhojpuri and Maithili all share the script. Therefore, Unicode/script detection alone cannot reliably identify the language.

The pipeline consequently performs explicit language identification after
Unicode normalization.

## Phase 1 results

| | Hindi (Model H) | Nepali (Model L) |
|---|---|---|
| Documents | 1,063,900 | 1,331,446 |
| Words | 587,959,968 | 527,408,637 |
| **Tokens** | **768,048,386** | **746,342,222** |
| **Manual tokens** | **214,243,269 (27.89%)** | **163,321,796 (21.88%)** |
| Vocabulary | 16,000 | 16,000 |
| Fertility (tokens/word) | 1.381 | 1.259 |
| Characters per token | 4.59 | 3.99 |
| Vocabulary utilisation | 97.9% | 94.8% |

Both corpora exceed the ~500M token target, and both exceed the required 20% manual
collection share. Token counts are **measured** by encoding the corpus with the trained
tokenizer, not estimated.

## Data Construction

**Manual Data**

Manual data is collected directly from public websites rather than being taken
from a pre-existing packaged dataset.

The scraper is resumable and maintains collection state so interrupted runs can
continue without restarting the entire process.

Configured Hindi sources include:

- Jansatta
- The Wire Hindi
- Amar Ujala (Tried but rejected)
- Navbharat Times (Tried but rejected)


Configured Nepali sources include:

- Onlinekhabar
- Setopati
- Ratopati

The actual contribution of each source is recorded in the corpus statistics.

**Public Data**

The downloaded portion is collected through streaming rather than requiring
the complete source datasets to be downloaded locally first.

The configured public sources include:

Hindi
- Hindi Wikipedia
- FineWeb-2 Hindi (Requirement Satisfied)
- AI4Bharat Sangraha verified Hindi


Nepali
- Nepali Wikipedia
- FineWeb-2 Nepali (Requirement Satisfied)
- AI4Bharat Sangraha verified Nepali

The download pipeline filters records while streaming and stops when the
configured token budget is reached.

## Cleaning Pipeline

The cleaning pipeline is applied independently to Hindi and Nepali.

Raw documents
      │
      ▼
Unicode normalization
      │
      ▼
Language identification
      │
      ▼
Text extraction / quality filtering
      │
      ▼
Boilerplate removal
      │
      ▼
Exact deduplication
      │
      ▼
Near-duplicate filtering
      │
      ▼
Clean corpus

## Dataset Splitting

The final corpus is divided at the document level:

| --- | --- |
| Train |  96% |
| Validation  | 2% |
| Test | 2% |

with:

seed = 1337

The split is deterministic and hash-based.

Document-level splitting prevents pieces of the same document from appearing
in multiple splits.

The tokenizer is trained using training data only. Validation and test data
are therefore not used to learn the tokenizer vocabulary.

## Repository layout

```
common/          shared library code — schema, fetching, cleaning, language ID
scripts/         command-line entry points
hindi/           Model H: config, tokenizer, data (gitignored)
nepali/          Model L: config, tokenizer, data (gitignored)
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
  --sources jansatta thewirehindi --limit 300000 --workers 4 --delay 0.25

.venv/bin/python -m scripts.scrape --lang ne --out data/manual/ne \
  --sources onlinekhabar --limit 400000 --workers 4 --delay 0.25
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

**4. Split** into train/validation/test (96/2/2, document-level, seed 1337).

```bash
.venv/bin/python -m scripts.split --config hindi/configs/dataset.json
.venv/bin/python -m scripts.split --config nepali/configs/dataset.json
```

**5. Train tokenizers.** Trains four candidate vocabulary sizes, evaluates each on
held-out validation text, and installs the selected one.

```bash
.venv/bin/python -m scripts.train_tokenizer --config hindi/configs/dataset.json \
  --vocab-sizes 8000 12000 16000 24000 --sample-lines 2000000

.venv/bin/python -m scripts.train_tokenizer --config nepali/configs/dataset.json \
  --vocab-sizes 8000 12000 16000 24000 --sample-lines 2000000
```

**6. Measure the corpus** and render the report figures.

```bash
.venv/bin/python -m scripts.corpus_stats --config hindi/configs/dataset.json
.venv/bin/python -m scripts.corpus_stats --config nepali/configs/dataset.json
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

## Google Drive link

https://drive.google.com/drive/folders/1fLdmz5Bd_M-yUM6KCAKtrvndxCNGjtHf?usp=drive_link

Artifacts
  - raw_data
  -  hindi_clean
  - nepali_clean
  - Tokenized_corpora
  - logs


The trained tokenizers (`hi.model`, `hi.vocab`, `ne.model`, `ne.vocab` — 3.3 MB total) are
small enough to be committed directly and are in this repository; the Drive copy is a
backup that also carries the intermediate vocabulary sizes and the SentencePiece training
samples.