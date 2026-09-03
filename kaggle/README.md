# Running Phase 2 on Kaggle

This is the operator's guide for `phase2_pretrain.ipynb` — the notebook that pretrains and
evaluates both models. It assumes you have never seen this repository before.

You run the notebook **twice**: once for Hindi (Model H) and once for Nepali (Model L).
Each run takes about **four hours** of GPU time and produces one tarball containing a
trained checkpoint, its training log, and every evaluation artifact.

The two models share **no** data, tokenizer, vocabulary or weights. Only the library code
is shared, which the assignment permits.

---

## 0. What you need before starting

- A Kaggle account with GPU quota (30 GPU-hours/week; two runs cost about 8).
- The Phase 1 artifacts and this repository's code, uploaded as Kaggle Datasets (§1).
- Roughly 25 GB of Kaggle Dataset storage for the packed corpora.

Nothing is downloaded from the internet except three pip packages and a font, so the
notebook works with Kaggle's internet toggle on. It **does** need internet on — see §2.

---

## 1. Kaggle Datasets to create

The notebook looks for its inputs by **searching** `/kaggle/input` recursively, not by
fixed paths, so you can name the datasets whatever you like. There is exactly one rule:

> **A language's data files must have `hindi` or `nepal` in their path, and never both.**

That substring is how the notebook decides which language a file belongs to. Kaggle
derives the mount directory from the dataset title, so in practice the title supplies it:
`hindi/data/tokens/` mounts as `/kaggle/input/hindi-data-tokens/` and matches.

The dangerous case is a *single* dataset holding both languages under a name like
`hindi-nepali-corpus`. Every path in it then matches both languages, and Nepali would
silently resolve to the Hindi files — training Model L on Hindi data, which the
vocabulary check downstream cannot catch because both vocabularies are the same size.
The notebook now detects this and refuses to continue:

```
LANGUAGE CONTAMINATION: the same source file was selected for both models:
  /kaggle/input/hindi-nepali-corpus/train.bin
  ...
Cause: a dataset path contains both 'hindi' and 'nepal'.
```

**Code datasets need no particular name.** They are identified by the modules they
contain — `lma/` is whichever directory holds `model.py`, `checkpoint.py` and
`schedule.py` — so renaming them is safe. Cell 6 prints which directory it picked for
each package.

### Data datasets

| Contents | Source in this repo | Notes |
|---|---|---|
| `train.bin`, `train.meta.json` | `<lang>/data/tokens/` | ~1.5 GB (Hindi), the packed uint16 token stream |
| `validation.bin`, `validation.meta.json` | `<lang>/data/tokens/` | |
| `test.bin`, `test.meta.json` | `<lang>/data/tokens/` | |
| `hi.model` / `ne.model` | `<lang>/tokenizer/` | SentencePiece model, needed to decode generations |
| `dataset.json`, `model.json`, `train.json` | `<lang>/configs/` | |
| `hi-test-*.jsonl.zst` / `ne-test-*.jsonl.zst` | `<lang>/data/splits/test/` | raw held-out text, used as generation prompts |

Upload one dataset per language, with `hindi` or `nepali` in the dataset title so the
substring rule is satisfied automatically.

### Code datasets

Three separate uploads, one per package. The dataset names below are the ones currently
in use; any name works, but they must stay **three separate datasets**.

| Upload this directory | Currently named | Identified by |
|---|---|---|
| `scripts/` | `evaluation_scripts` | `pretrain.py` + `eval_generation.py` + `plot_training.py` |
| `common/` | `common/` | `langid.py` + `normalize.py` + `sources.py` |
| `lma/` | `large_model_agent` | `model.py` + `checkpoint.py` + `schedule.py` |

They must not be merged into one dataset: both `scripts/` and `common/` contain a file
called `clean.py`, and flattening them would silently overwrite one with the other. The
identifying signatures above are chosen to avoid that collision.

Only `*.py` files are copied out of these, so `__pycache__` and stray files are harmless.

> **Updating code later.** Kaggle pins the dataset *version* a notebook is attached to.
> Uploading a new version does **not** reach a notebook on its own — you must open the
> notebook's **Data** panel and refresh the dataset to the new version. Skipping this is
> the most common way to spend four hours training with code you thought you had replaced.

---

## 2. Notebook settings

Open `phase2_pretrain.ipynb` in the Kaggle editor and set, in the right-hand panel:

- **Accelerator → GPU T4 x2.** Do this *first*; changing it restarts the session.
- **Internet → On.** Cell 8 pip-installs `sentencepiece`, `sacrebleu` and `zstandard`,
  and installs a Devanagari font.
- **Persistence → Files only** (optional, but it lets `/kaggle/working` survive between
  interactive sessions so training can resume).

### Only one GPU gets used, and that is correct

Kaggle offers T4 **x2**, but this notebook trains on a single GPU. A 24M-parameter model
at batch 32 does not saturate one T4, so splitting it across two would add
synchronisation overhead for no gain. Select the x2 accelerator anyway — it is the same
quota cost, and the second card is simply idle.

### Run it as a committed job, not an interactive session

Use **Save Version → Save & Run All (Commit)** rather than leaving the interactive editor
open. A committed run:

- executes headless for up to 12 hours and survives you closing the browser,
- **persists `/kaggle/working` as a downloadable output.**

An interactive session that times out or drops its connection loses everything in
`/kaggle/working`, which means losing the entire training run. This has already cost us
one four-hour run.

---

## 3. The control panel (cell 2)

Everything you would normally want to change lives in one cell.

```python
LANGUAGE      = "hi"      # "hi" = Hindi (Model H), "ne" = Nepali (Model L)

RUN_PROBE     = True      # 50-step throughput measurement before committing to a full run
RUN_TRAINING  = True      # the pretraining run itself
RUN_EVAL      = True      # perplexity / BPB / generation / attention, after training

MAX_MINUTES   = 660       # stop and checkpoint before Kaggle's 12-hour session limit
MAX_STEPS     = None      # None = use train.json (16,000)
MICRO_BATCH   = None      # None = use train.json (32). Drop to 16 or 8 on a CUDA OOM.
AMP_DTYPE     = None      # None = train.json ("float16"). Set "float32" if loss goes nan.

EVAL_PROMPTS  = 200       # held-out prompts for the generation metrics
```

**`LANGUAGE` is the only one you normally touch.** Set it to `"hi"`, run everything,
download the output, then set it to `"ne"` and run everything again.

`MICRO_BATCH` is safe to lower. Gradient accumulation keeps the tokens-per-step budget
fixed at 32,768, so halving the micro-batch doubles the accumulation count and the
learning dynamics are **unchanged** — only speed changes.

---

## 4. What each cell does

| Cell | Section | What it does | Time |
|---|---|---|---|
| 2 | Control panel | the settings above | instant |
| 4 | Environment | confirms a GPU is attached, prints disk space | instant |
| 6 | Assemble tree | symlinks data from `/kaggle/input`, copies the `.py` files, builds `/kaggle/working/vidhi` | ~10 s |
| 8 | Dependencies | pip installs; installs a Devanagari font and clears matplotlib's font cache | ~1 min |
| 10 | Sanity checks | parameter count, **causal-mask check**, corpus/vocabulary agreement | ~20 s |
| 12 | Throughput probe | 50 steps, measures tokens/second, estimates wall clock | ~2 min |
| 14 | **Pretraining** | the actual run: 16,000 steps | **~3.7 h** |
| 16 | Evaluation | perplexity/BPB, generation metrics, attention analysis, loss curves | ~15 min |
| 18 | Package | writes `phase2-<lang>-results.tar.gz` to the Output tab | ~2 min |

### Three lines to check before walking away

**Cell 8** must not say `NONE`:

```
Devanagari font: Noto Sans Devanagari
```

If it prints `NONE`, matplotlib will draw every Hindi/Nepali character as an empty box and
the attention heatmaps lose the example sentence they are supposed to show.

**Cell 10** must report exact zero drift:

```
causal mask     past drift 0.0 (exact), future reacts by 3.412
```

This is the most important assertion in the notebook. It edits the second half of a
sequence and checks that the logits in the first half do not move *at all*. A masking bug
is silent — the loss still falls, the model still trains, but it has been allowed to read
the future and every number downstream is meaningless. `0.0` exactly, not "small":
a masked position's softmax weight is `exp(-inf)`, which is precisely zero.

**Cell 14** must say what you expect about resuming:

```
[resume] nothing in /kaggle/working/checkpoints/hindi, starting fresh
```

If it says it is resuming and you wanted a clean run, delete
`/kaggle/working/checkpoints/<lang>` first. Never resume across a code change — you would
end up with a checkpoint whose first half was written by different code from its second.

---

## 5. Resuming after a dead session

`--resume` makes cell 14 idempotent. **Re-run the whole notebook with the same `LANGUAGE`**
and training continues from the last checkpoint at the exact step, optimiser moments, RNG
state and batch order. Nothing is repeated and nothing is lost.

This works because the batch order is a pure function of `(seed, micro_step)` rather than
of an iterator's position, and because the learning-rate schedule is a pure function of the
step number. Resume was verified bit-exact: all 31 weight tensors identical between an
uninterrupted run and a resumed one.

`MAX_MINUTES = 660` exists so the run stops cleanly and writes a final checkpoint at
11 hours, rather than being killed mid-write at Kaggle's 12-hour limit.

---

## 6. What comes out

Cell 18 writes `/kaggle/working/phase2-<lang>-results.tar.gz`, downloadable from the
notebook's **Output** tab:

```
checkpoints/<lang>/best.pt                        ~280 MB  lowest-validation-loss weights
checkpoints/<lang>/step-*.pt                      periodic checkpoints
checkpoints/<lang>/train_log.jsonl                one JSON object per logged step
checkpoints/<lang>/config.json                    model + training config, readable without torch
report/<lang>/lm_eval.json                        val/test cross-entropy, perplexity, BPB
report/<lang>/generation_eval.json                BLEU / chrF / chrF++ / ROUGE-L / distinct-n
report/<lang>/generation_samples.json             qualitative continuations, all four decoders
report/<lang>/attention_stats.json                per-head entropy and mean attention distance
report/<lang>/figures/*.png                       loss curve, LR schedule, attention heatmaps
```

Every checkpoint contains model weights, optimizer state, scheduler state, the training
step, and the full configuration — the five things the assignment requires.

**Download the tarball before starting the other language.** A new committed run replaces
the previous output.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `!! NO GPU ATTACHED !!` | accelerator not set | Settings → Accelerator → GPU T4 x2, re-run |
| `MISSING INPUTS: ...` | a dataset is absent, or its path lacks `hindi`/`nepal` | rename the dataset or re-upload; the message names each missing file |
| `LANGUAGE CONTAMINATION: ...` | one dataset path contains both `hindi` and `nepal` | split the languages into separate datasets, or rename so each path names one language |
| `code for lma/ (no input directory holds ...)` | a code dataset is missing or partial | re-upload that package; the message lists the modules it looked for |
| `corpus/config vocabulary mismatch` | `.bin` files packed with a different tokenizer than `model.json` expects | re-pack, or fix `vocab_size` in `model.json` |
| `CUDA out of memory` | another process, or an unlucky allocation | set `MICRO_BATCH = 16`; dynamics are unchanged |
| loss becomes `nan` | fp16 underflow | set `AMP_DTYPE = "float32"`, ~30% slower |
| `Devanagari font: NONE` | apt-get failed, probably internet off | turn internet on and re-run cell 8 |
| code changes did not take effect | dataset version pinned | Data panel → refresh the dataset to its newest version, restart |
| everything vanished after the session ended | interactive session, not committed | use Save Version → Save & Run All (Commit) |

---

## 8. After both runs

The remaining work is local and needs no GPU:

1. Extract both tarballs into the repository root.
2. `python -m scripts.plot_training --compare` — the Hindi-vs-Nepali loss comparison.
   This can only be produced locally, because each Kaggle session sees one language.
3. Write `report/phase2.md` with the tables, figures and analysis.
4. Upload both `best.pt` files to Google Drive and put the links in the top-level README.
   Checkpoints must not be committed to git; `.gitignore` already blocks `*.pt`.
5. Commit to the `phase-2` branch.
