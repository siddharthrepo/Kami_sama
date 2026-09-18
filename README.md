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
| Documents | 1,063,900 | 1,331,446 |
| Words | 587,959,968 | 527,408,637 |
| **Tokens** | **768,048,386** | **746,342,222** |
| **Manual tokens** | **214,243,269 (27.89%)** | **163,321,796 (21.88%)** |
| Vocabulary | 16,000 | 16,000 |
| Fertility (tokens/word) | 1.306 | 1.415 |
| Characters per token | 3.87 | 4.56 |
| Vocabulary utilisation | 94.8% | 97.9% |

> Vocabulary is **16,000** for both languages, fixed in Phase 1 by the ~25M parameter
> budget: under weight tying the embedding is `vocab_size x d_model`, so at
> `d_model = 448` the largest affordable table is 17,566 pieces. See
> `report/phase1.md` §6.2 and `report/phase2.md` §0.

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
  --vocab-sizes 8000 12000 16000 24000 --sample-lines 2000000

.venv/bin/python -m scripts.train_tokenizer --config nepali/configs/dataset.json \
  --vocab-sizes 8000 12000 16000 24000 --sample-lines 2000000
```

**6. Measure the corpus** and render the report figures.

```bash
.venv/bin/python -m scripts.corpus_stats --config hindi/configs/dataset.json
.venv/bin/python -m scripts.corpus_stats --config nepali/configs/dataset.json
```

Total runtime on a 12-core CPU machine: roughly 4 hours, dominated by collection.
No GPU is required for Phase 1.

### Phase 2 — pretraining and evaluation

**7. Pack tokens** into flat `uint16` arrays (local, CPU, minutes per language).

```bash
.venv/bin/python -m scripts.pack_tokens --config hindi/configs/dataset.json
.venv/bin/python -m scripts.pack_tokens --config nepali/configs/dataset.json
```

**8. Pretrain** (GPU, ~3.5 h per language). `--resume` continues from the last checkpoint.

```bash
.venv/bin/python -m scripts.pretrain --lang hi --resume
.venv/bin/python -m scripts.pretrain --lang ne --resume
```

**9. Evaluate** — perplexity/BPB, generation metrics, attention.

```bash
.venv/bin/python -m scripts.eval_lm --lang hi --splits validation test
.venv/bin/python -m scripts.eval_generation --lang hi --num-prompts 200
.venv/bin/python -m scripts.attention_analysis --lang hi --heads 0 1 2 3 4 5 6
.venv/bin/python -m scripts.plot_training --compare
.venv/bin/python -m scripts.compare_models
```

`kaggle/phase2_pretrain.ipynb` runs steps 8-9 for one language per session.

### Phase 3 — reasoning finetuning and analysis

**10. Generate the reasoning corpora** (local, CPU, seconds). Verifies that no entity
name and no prompt is shared between splits, and refuses to write if either check fails.

```bash
.venv/bin/python -m scripts.make_reasoning_data --lang hi
.venv/bin/python -m scripts.make_reasoning_data --lang ne
```

**11. Finetune** from each language's own pretrained checkpoint (GPU, ~20 min each). The
tokenizer and vocabulary stay fixed; the loss is masked to the answer span.

```bash
.venv/bin/python -m scripts.finetune --lang hi
.venv/bin/python -m scripts.finetune --lang ne
```

**12. Score** pretrained against finetuned on the held-out reasoning test split.

```bash
.venv/bin/python -m scripts.eval_reasoning --lang hi
.venv/bin/python -m scripts.eval_reasoning --lang ne
```

**13. Attention, pretrained against finetuned**, on the same comparative-reasoning prompt
so the two panels are directly comparable.

```bash
.venv/bin/python -m scripts.attention_analysis --lang hi \
  --checkpoint checkpoints/hindi/best.pt --sentence "<a chain_pair prompt>" \
  --layers 0 6 --heatmaps-only --out-dir report/hindi/figures-reasoning-pretrained

.venv/bin/python -m scripts.attention_analysis --lang hi \
  --checkpoint checkpoints/hindi-finetuned/best.pt --sentence "<the same prompt>" \
  --layers 0 6 --heatmaps-only --out-dir report/hindi/figures-reasoning-finetuned
```

`kaggle/phase3_finetune.ipynb` runs steps 11-13 for **both** languages in one session and
packages the results.

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

## Phase 2 results

Two independent ~24.3M-parameter decoder-only Transformers, 16,000 vocabulary, 7 layers
x 7 heads x 448 dimensions, 512 context, trained on 524,288,000 tokens each
(0.71 / 0.73 epochs -- no document seen twice).

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Parameters | 24,298,176 | 24,298,176 |
| Validation perplexity | **28.43** | **45.23** |
| Validation bits per byte | **0.4930** | **0.4527** |
| Test perplexity | 28.23 | 45.09 |
| Test bits per byte | 0.4916 | 0.4527 |
| Bits per character | 1.2099 | 1.1991 |
| BLEU-4 @ temperature 0.5 | 3.78 | 2.21 |
| chrF @ temperature 1.0 | 22.42 | 25.10 |

Perplexity and bits-per-byte disagree about which model is better, because Nepali tokens
carry more text (12.15 bytes/token against 9.80). Normalised to bits per character the
two models are 0.9% apart. Full analysis in **[`report/phase2.md`](report/phase2.md)**.

---

## Phase 3 results

Each model finetuned on its own synthetic comparative-reasoning corpus — 24,000 / 3,000 /
3,000 examples per language, eight template families, generated by
`scripts/make_reasoning_data.py`. No finetuning data or weights shared between languages.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Exact match, pretrained | 0.0000 | 0.0000 |
| **Exact match, finetuned** | **0.5293** | **0.4553** |
| Chance baseline | 0.3438 | 0.3438 |
| `chain_pair` (multi-hop) | 0.8373 | 0.8600 |
| `numeric_superlative_most` | 0.0912 | 0.0349 |

Symbolic comparison is learned well; numeric comparison is not, falling below the chance
floor because a 16,000-piece BPE vocabulary carries no representation of numeric
magnitude. The pretrained baseline is zero because those models continue the prompt as
prose rather than answering — their first-word match (0.2527 / 0.1423) sits *below*
chance, so they are not reasoning at reduced accuracy but not engaging at all.

Finetuning left attention essentially unchanged: per-layer entropy, mean distance and
previous-token share move in the third decimal. Full analysis, including the four
consolidating questions, in **[`report/phase3.md`](report/phase3.md)**.

---

## Google Drive links

All large artifacts live in one shared Drive folder, organised by pipeline stage. Set each
folder to **"Anyone with the link can view"** so graders do not have to request access.

| Drive folder | Contents | Phase | Link |
|---|---|---|---|
| `raw_data/` | Scraped shards (Jansatta, The Wire Hindi, Onlinekhabar) and downloaded shards (FineWeb-2, Wikipedia), both languages | 1 | _pending_ |
| `hindi_clean/` | Hindi corpus after Unicode normalisation, language filtering and deduplication | 1 | _pending_ |
| `nepali_clean/` | Nepali corpus, same stage | 1 | _pending_ |
| `Hindi_Split/` | Hindi train / validation / test, 98/1/1 at document level | 1 | _pending_ |
| `Nepali_Split/` | Nepali train / validation / test | 1 | _pending_ |
| `Tokenized_corpora/` | Packed `uint16` token arrays — the direct input to pretraining | 2 | _pending_ |
| `logs/` | Collection, cleaning and run logs | 1–3 | _pending_ |
| **`checkpoints/`** | **All four model checkpoints — see below** | **2–3** | **_pending_** |

`checkpoints/` contains one folder per model, each with `best.pt`, its rolling step
checkpoints and its training log:

| Subfolder | Model | Stage | Link |
|---|---|---|---|
| `checkpoints/hindi/` | Model H | pretrained (Phase 2) | _pending_ |
| `checkpoints/nepali/` | Model L | pretrained (Phase 2) | _pending_ |
| `checkpoints/hindi-finetuned/` | Model H | reasoning-finetuned (Phase 3) | _pending_ |
| `checkpoints/nepali-finetuned/` | Model L | reasoning-finetuned (Phase 3) | _pending_ |

Every checkpoint carries model weights, optimiser state, scheduler state, training step
and configuration, plus the four RNG streams — so training resumes bit-identically from
any of them.

**Not on Drive, and deliberately so:** everything under `report/` — statistics, figures,
attention heatmaps and the three phase reports — is committed to this repository. The
assignment grades what is on the branch, not what is behind a link.

The trained tokenizers (`hi.model`, `hi.vocab`, `ne.model`, `ne.vocab` — 3.3 MB total) are
small enough to be committed directly and are in this repository; the Drive copy is a
backup that also carries the intermediate vocabulary sizes and the SentencePiece training
samples.
