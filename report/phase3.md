# Phase 3 — Reasoning Finetuning, Attention Analysis, and Final Report

This report consolidates the whole project. Phases 1 and 2 are summarised in §1 with the
figures needed to follow the argument; the full treatments remain in `report/phase1.md`
and `report/phase2.md`. §2–§6 are Phase 3 proper. §7 answers the four questions the
assignment asks the final report to address.

Every number here was measured from a file in this repository. The paths are given so
each one can be checked.

---

## 0. What Phase 3 found

Two models, identical in architecture, differing only in the corpus they were trained on,
were finetuned on synthetic comparative-reasoning data in their own languages.

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Exact match, pretrained | 0.0000 | 0.0000 |
| **Exact match, finetuned** | **0.5293** | **0.4553** |
| Chance baseline | 0.3438 | 0.3438 |
| Test examples | 3,000 | 3,000 |

Source: `report/<lang>/reasoning_eval.json`.

Three findings, in order of how much they matter.

**Both models learned transitive reasoning, and the higher-resource model learned it
better.** Model H clears chance by 18.6 points, Model L by 11.2. The resource-tier
ordering that Phase 2 established for language modelling reappears in reasoning, which is
not obvious in advance: reasoning could have been a capability so far out of reach for a
24M-parameter model that corpus quality made no difference to it.

**Symbolic comparison works; numeric comparison does not.** Chained inequalities over
named entities reach 0.84 and 0.86. Comparisons over stated numbers fall to 0.09 — *below*
the 0.29 chance floor for those templates, which means systematic error rather than
guessing. §5 argues this is a tokenisation limit, not a reasoning limit.

**Finetuning changed what the models emit, not how they read.** Per-head attention
entropy, mean distance and previous-token share move in the third decimal place between
the pretrained and finetuned checkpoints (§6). The layer arc Phase 2 identified survives
intact.

---

## 1. The project, consolidated

### 1.1 What was built

```
Phase 1   two corpora, two tokenizers            report/phase1.md
   |      768.0M / 746.3M tokens, 16,000 pieces each
   v
Phase 2   two decoder-only Transformers          report/phase2.md
   |      24,298,176 parameters each, trained from scratch
   v
Phase 3   two finetuned models + evaluation      this document
          reasoning exact match 0.5293 / 0.4553
```

The two models share no data, no tokenizer, no vocabulary and no weights. Only the
library code is common, which the assignment permits.

### 1.2 Phase 1 — the corpora

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Documents | 1,063,900 | 1,331,446 |
| Tokens | 768,048,386 | 746,342,222 |
| Manual share of tokens | 27.89% | 21.88% |
| Training split | 738,559,188 | 717,728,934 |
| Vocabulary | 16,000 | 16,000 |
| Fertility (tokens/word) | 1.259 | 1.381 |
| Characters per token | 3.992 | 4.586 |
| Byte-fallback rate | ~1.33% | ~2.17% |

Both corpora exceed the assignment's ~500M target. The manual share clears the required
20% in both cases, with less margin for Nepali — a consequence of how much less Nepali
text is reachable, which §7.1 returns to.

The vocabulary of 16,000 was not a free choice. Under weight tying the embedding matrix
is `vocab_size × d_model` and is counted once, so vocabulary trades directly against
depth. At `d_model = 448` the rest of the model costs 17,130,176 parameters, leaving
7,869,824 of the ~25M budget for the table — a ceiling of `7,869,824 ÷ 448 = 17,566`
pieces. 16,000 was the largest candidate that fit.

### 1.3 Phase 2 — the models

Identical architecture for both, so that the corpus is the only variable:

| | Value |
|---|---|
| Layers / heads / `d_model` / `d_ff` | 7 / 7 / 448 / 1792 |
| Context | 512 |
| Parameters | 24,298,176 |
| Training tokens | 524,288,000 |
| Epochs | 0.710 (H), 0.730 (L) |

Held-out results:

| | Model H | Model L |
|---|---|---|
| Test cross-entropy (nats) | 3.3402 | 3.8087 |
| Test perplexity | **28.23** | **45.09** |
| Test bits-per-byte | 0.4916 | **0.4527** |
| **Bits per character** | **1.2099** | 1.1991 |

The headline Phase 2 finding is that perplexity and bits-per-byte disagree about which
model is better, and that neither is a fair cross-language comparison on its own. A
Nepali token carries 12.146 bytes against Hindi's 9.796, so per-token perplexity
penalises Nepali for doing more work per prediction; bits-per-byte over-corrects, because
Devanagari costs ~2.45 UTF-8 bytes per character in Hindi and ~2.65 in Nepali. Bits per
character is the honest unit, and by it the two models are within 0.9% of each other.

That matters for Phase 3 because it sets the expectation: at the *language-modelling*
level these two models are nearly equally good. Any large gap in reasoning is therefore
about something other than raw modelling quality.

---

## 2. The synthetic reasoning dataset

Built programmatically by `scripts/make_reasoning_data.py` with the language packs in
`common/reasoning.py`. No existing benchmark was downloaded.

### 2.1 Construction

Nothing is labelled after the fact. Each example begins from a ground-truth structure —
an ordering, or a set of numbers — and the question and the answer are both rendered from
that same structure. The label cannot disagree with the prompt because neither exists
independently of it.

Two entity pools per language, because they take different grammar:

- **60 person names** — `राम`, `सचिन`, `हितेश` … / `हरि`, `बिकास`, `गोपाल` …
- **40 object names** — `किताब`, `डिब्बा`, `हथौड़ा` … / `किताब`, `बाकस`, `डोको` …

Two kinds of attribute. **Comparison** attributes (height, age, weight, speed) are stated
as relations: *"सचिन, हितेश से लंबा है।"* **Numeric** attributes (price, age, weight,
length) are stated as measured values: *"डिब्बे की कीमत 1009 रुपये है।"*

### 2.2 The eight template families

| Family | Structure | Asked |
|---|---|---|
| `chain_superlative_3` / `_4` | chain `A>B>C` | largest / smallest |
| `chain_pair_3` / `_4` | chain `A>B>C` | relation between **A and C** |
| `numeric_pair` | two values | which is greater |
| `numeric_superlative_3` / `_4` | three or four values | highest / lowest |
| `numeric_equality` | two values | equal or not |

`chain_pair` is the multi-hop case the assignment calls out. It draws **only
non-adjacent** positions from the chain, so no single premise names both entities and the
answer cannot be read off one sentence.

### 2.3 Size and variety

Per language: **24,000 train / 3,000 validation / 3,000 test**.

Template mix in the training split (`report/<lang>/reasoning_stats.json`):

| Template | Count |
|---|---|
| `chain_pair` (3- and 4-entity) | 6,000 |
| `chain_superlative_least` | 3,025 |
| `chain_superlative_most` | 2,975 |
| `numeric_superlative_least` | 3,003 |
| `numeric_superlative_most` | 2,997 |
| `numeric_equality` | 3,000 |
| `numeric_pair` | 3,000 |

Reasoning depth: **12,000 one-hop, 8,022 two-hop, 3,978 three-hop**. Attributes are spread
across age (5,990), weight (6,007), speed (3,082), length (2,990), height (2,979) and
price (2,952).

Equality answers are balanced **1,519 / 1,481** yes/no, so a model that always answers one
way scores 50% on that template and no better.

### 2.4 Avoiding train–test leakage

Two mechanisms, both verified rather than asserted.

**Disjoint entity pools.** The pools are partitioned *before* any example is generated, so
a name that appears in test never appears in train:

| | train | validation | test |
|---|---|---|---|
| Person names | 36 | 12 | 12 |
| Object names | 24 | 8 | 8 |

**Prompt deduplication.** Every prompt is hashed; duplicates are rejected within and
across splits.

`scripts/make_reasoning_data.py` computes every pairwise overlap — names and prompt
hashes, all split pairs — writes them into `reasoning_stats.json`, and exits non-zero if
any is non-zero. All measured **zero**.

The distinct-answer counts confirm the partition held: **62 in train** (36 persons + 24
objects + 2 yes/no tokens) and **22 in validation and test** (12 + 8 + 2), exactly the
pool arithmetic.

### 2.5 Language correctness

The assignment requires natural phrasing in each language, not English templates with
substituted names. Three things this demanded:

**Questions are written per attribute, not assembled from a noun.** Hindi interrogatives
agree with gender — `किसकी कीमत` but `किसका वज़न`. A builder that stored only the noun
would produce ungrammatical output for half the attributes.

**Oblique case.** Hindi masculine nouns ending in `-ा` take `-े` before a postposition, so
it is `डिब्बे की कीमत`, never `डिब्बा की कीमत`. Fourteen of the forty Hindi object names
are affected. Answers stay in the direct form (`डिब्बा`), which is correct for a standalone
answer — and is why §4 reports a lenient score that accepts either.

**Nepali is structurally different, not translated.** It marks the comparative with
`भन्दा` and the superlative with `सबैभन्दा`, and uses Devanagari digits (`२४`), matching
Phase 1's normalisation which deliberately preserves them.

---

## 3. Finetuning protocol

`scripts/finetune.py`, run once per language from that language's own Phase 2 checkpoint.
Nothing is shared: Model L is never initialised from Model H.

### 3.1 Hyperparameters

| Setting | Value | Against pretraining |
|---|---|---|
| Learning rate | 1e-4 | 6e-4 |
| Warmup steps | 100 | 500 |
| Schedule | linear warmup, cosine decay to 10% | same |
| Examples per optimiser step | 64 | — |
| Micro-batch | 32 | 32 |
| Epochs | 3 (1,125 steps) | 0.71 |
| Max sequence length | 128 tokens | 512 |
| Weight decay / betas / grad clip | 0.1 / (0.9, 0.95) / 1.0 | same |
| Seed | 1337 | same |
| Mixed precision | float16 | same |

Recorded in `report/<lang>/finetune_log.jsonl` (start record) and in the `train_config`
block of every checkpoint.

The learning rate is an order of magnitude below pretraining's. The model already speaks
the language; the job is to bend it towards a task, not to relearn Hindi or Nepali from
24,000 short questions.

### 3.2 The loss is masked to the answer

A prompt averages about 110 characters and the answer is a single word. Measured over a
batch of eight examples, an unmasked objective scores 276 token positions where the
answer occupies 19 — so roughly 93% of the gradient would go into learning to reproduce
question templates, which nothing in Phase 3 measures.

The loss therefore scores only the answer span and the end-of-sequence token; prompt
positions are set to `IGNORE_INDEX`. The unmasked loss is computed from the same logits
and logged alongside, as a diagnostic.

That diagnostic earns its place. Over the run the two curves move in **opposite
directions**:

| | step 25 | step 100 | step 1125 |
|---|---|---|---|
| Validation masked loss (H) | — | **1.1404** | 2.1151 |
| Validation unmasked loss (H) | — | 4.0537 | 5.5044 |

The model gets better at producing answers and steadily worse at reproducing question
text. That is exactly what answer-masking is supposed to do, and seeing it confirms the
mask is applied to the span intended rather than to an off-by-one neighbour.

### 3.3 Checkpointing, and what the curves show

Checkpoints carry model weights, optimiser state, scheduler state, training step and
configuration — the same resume-capable format as pretraining, plus all four RNG streams.

Validation was evaluated every 100 steps. It bottoms out early:

| step | H masked | L masked |
|---|---|---|
| 100 | **1.1404** | 1.4293 |
| 200 | 1.6241 | **1.4050** |
| 500 | 2.3875 | 2.2641 |
| 1125 | 2.1151 | 2.0494 |

**Best validation lands at step 100 for Model H and step 200 for Model L, out of 1,125.**
Three epochs is more than this task needs; the model fits the reasoning corpus within a
few hundred examples and then degrades.

This is stated plainly rather than hidden because it is the one place where the Phase 2
guarantee does not carry over. Pretraining saw under one epoch, so no document was ever
seen twice and overfitting was structurally impossible. Finetuning repeats a small corpus
and overfitting is therefore possible — and it happened. What prevented it from reaching
the reported results is the best-checkpoint mechanism: `best.pt` tracks validation loss
and is only overwritten on improvement, so the evaluated model is the step-100 one, not
the step-1125 one. Early stopping here is load-bearing, not decoration.

---

## 4. Pretrained versus finetuned

`scripts/eval_reasoning.py`, greedy decoding, up to 8 new tokens, on the full 3,000-example
test split. Accuracy is exact match on **freely generated** text — the model sees only the
prompt. Teacher forcing would show it the answer tokens one position before it must
predict them.

### 4.1 Headline

| | pretrained | finetuned | chance |
|---|---|---|---|
| **Model H (Hindi)** | 0.0000 | **0.5293** | 0.3438 |
| **Model L (Nepali)** | 0.0000 | **0.4553** | 0.3438 |

Three metrics are reported, because exact match alone mis-describes the baseline.

| | strict | lenient | first-word |
|---|---|---|---|
| H pretrained | 0.0000 | 0.0000 | 0.2527 |
| H finetuned | 0.5293 | 0.5303 | 0.5303 |
| L pretrained | 0.0000 | 0.0000 | 0.1423 |
| L finetuned | 0.4553 | 0.4553 | 0.4553 |

**Strict** requires the generated text to equal the gold answer. **Lenient** also accepts
the oblique form (`डिब्बे` for `डिब्बा`), since a model that inflects the entity as the
prompt did has still identified it. **First-word** scores only the leading word.

The strict and first-word columns coincide for both finetuned models, which says
something useful on its own: after finetuning the models emit the answer and stop. The
gap between them for the pretrained models says the opposite.

### 4.2 The pretrained baseline is zero for a specific reason

A pretrained score of exactly 0.0000 could mean the model cannot reason, or that it
cannot produce an answer in the required form. The qualitative output settles it:

```
prompt: … विनोद और ओमकार में किसकी उम्र अधिक है?  उत्तर:
output: हां, यह एक बहुत ही अजीब बात

prompt: … देव र नबिन मध्ये को छिटो छ?  उत्तर:
output: - यो पनि एक सुन्दर र सुन्दर
```

The pretrained models continue the text as prose. They have never seen the question
format, never learned to stop, and run to the token budget. Exact match is therefore zero
by construction.

But first-word match is **0.2527** for Hindi and **0.1423** for Nepali, against a chance
level of **0.3438** — both *below* chance. The pretrained models are not reasoning at
reduced accuracy; they are not engaging with the task at all. Without the chance baseline,
0.2527 could easily be misread as partial competence.

### 4.3 Where the accuracy actually is

| Template | Model H | Model L | chance |
|---|---|---|---|
| `chain_pair` (multi-hop) | **0.8373** | **0.8600** | 0.2917 |
| `chain_superlative_most` | 0.7667 | 0.6590 | 0.2932 |
| `chain_superlative_least` | 0.7306 | 0.5111 | 0.2900 |
| `numeric_equality` | 0.5093 | 0.5147 | 0.5000 |
| `numeric_pair` | 0.3627 | 0.1493 | 0.5000 |
| `numeric_superlative_least` | 0.0981 | 0.0477 | 0.2920 |
| `numeric_superlative_most` | 0.0912 | 0.0349 | 0.2913 |

The split is stark and it is not about difficulty in the usual sense. The **symbolic**
templates — chains of stated relations between named entities — are answered well. The
**numeric** templates, which require comparing magnitudes, collapse.

By reasoning depth:

| | Model H | Model L | chance |
|---|---|---|---|
| 1 hop | 0.2653 | 0.1867 | 0.3958 |
| 2 hops | **0.8619** | **0.7835** | 0.3112 |
| 3 hops | 0.6472 | 0.5971 | 0.2500 |

Read alone this table looks broken: more hops should be harder, and here more hops scores
better. The resolution is that hop count and template type are confounded by
construction — every one-hop item is a numeric template and every two- and three-hop item
is a symbolic chain. The hop table is therefore the same finding as the template table,
not an independent one. The genuine difficulty effect is visible only *within* the
symbolic families, where 3-hop (0.6472) does sit below 2-hop (0.8619), as expected.

---

## 5. Error analysis

### 5.1 What success looks like

The strongest results are on the hardest available items — three-hop chains where the two
entities in the question never appear in the same premise:

```
प्रश्न: हितेश, निखिल से तेज़ है। कपिल, उदय से तेज़ है। उदय, हितेश से तेज़ है।
        निखिल और कपिल में कौन तेज़ है?
उत्तर: कपिल          ✓  (chain: कपिल > उदय > हितेश > निखिल)

प्रश्न: छत्र, सागर भन्दा जेठो छ। सन्तोष, छत्र भन्दा जेठो छ। रवि, सन्तोष भन्दा जेठो छ।
        रवि र सागर मध्ये को जेठो छ?
उत्तर: रवि           ✓  (chain: रवि > सन्तोष > छत्र > सागर)
```

Note that the premises are presented out of chain order — the generator shuffles them, so
"the first name mentioned" is not a usable shortcut. Answering requires assembling the
ordering from scattered pairwise facts and then reading off a relation that no premise
states.

### 5.2 What failure looks like

**Numeric comparison, the dominant failure.**

```
प्रश्न: विनोद की उम्र 79 साल है। ओमकार की उम्र 49 साल है।
        विनोद और ओमकार में किसकी उम्र अधिक है?
उत्तर: ओमकार         ✗  (gold विनोद — 79 > 49)

प्रश्न: कपिल की उम्र 16 साल है। सचिन की उम्र 57 साल है। हितेश की उम्र 56 साल है।
        सबसे अधिक उम्र किसकी है?
उत्तर: हितेश         ✗  (gold सचिन — 57 > 56 > 16)
```

The second is diagnostic. The model picks `हितेश` at 56 over `सचिन` at 57 — the right
*kind* of answer, an entity with a large value, but it cannot separate two numbers that
differ by one. It has learned the format and the shape of the task; it has not learned
magnitude.

**Why:** a 16,000-piece BPE vocabulary has no notion of numeric order. `57` and `56` are
token sequences that happen to share a prefix; nothing in 524M tokens of news and
encyclopedia text teaches an embedding table that one denotes a larger quantity than the
other. Sub-chance performance on `numeric_superlative` follows: the model is not guessing
uniformly, it is systematically choosing on some surface feature — plausibly token
frequency or digit length — that correlates with the wrong answer.

`numeric_equality` sits at 0.5093 and 0.5147 against a 0.5 chance floor, i.e. **no signal
at all**, which is the same story in its cleanest form: asked whether two numbers are
equal, the models cannot tell.

**Symbolic failures are ordinary.** Where chains fail it is usually the `least` direction:

```
प्रश्न: उदय, अजय से भारी है। विनोद, ओमकार से भारी है। ओमकार, उदय से भारी है।
        सबसे हल्का कौन है?
उत्तर: ओमकार         ✗  (gold अजय)
```

`chain_superlative_least` trails `chain_superlative_most` in both models — by 3.6 points
in Hindi and 14.8 in Nepali. Finding the *minimum* requires traversing the assembled chain
to its far end, whereas the maximum is the entity that appears on the left of a premise
and never on the right. The asymmetry is consistent with the models having learned a
positional heuristic that happens to work for one direction.

### 5.3 The attribute view corroborates it

| Attribute | Kind | Model H | Model L |
|---|---|---|---|
| height | symbolic | 0.7877 | 0.7519 |
| speed | symbolic | 0.7797 | 0.7159 |
| weight | mixed | 0.5415 | 0.4690 |
| age | mixed | 0.5195 | 0.4470 |
| price | numeric only | 0.2891 | 0.1698 |
| length | numeric only | 0.2715 | 0.1880 |

`height` and `speed` appear only in comparison templates and score highest. `price` and
`length` appear only in numeric templates and score lowest. `weight` and `age` appear in
both and land in between. The gradient tracks the symbolic/numeric split cleanly, which is
what you would expect if the split is the real variable.

---

## 6. Attention analysis, pretrained against finetuned

`scripts/attention_analysis.py` — the Phase 2 toolkit, unmodified — rerun on the finetuned
checkpoints. Heatmaps for an early layer (0) and a late layer (6) were produced for both
stages on the *same* comparative-reasoning prompt, so the pair is directly comparable.

Figures: `report/<lang>/figures-reasoning-pretrained/` and `figures-reasoning-finetuned/`.
Per-head statistics: `report/<lang>/attention_stats.json` (pretrained, from Phase 2) and
`attention_stats_finetuned.json`.

### 6.1 The measured change is very small

| Layer | H entropy (norm.) | H distance | H prev-share |
|---|---|---|---|
| 0 | 0.911 → 0.911 | 105.23 → 105.14 | 0.0188 → 0.0188 |
| 1 | 0.791 → 0.797 | 89.63 → 91.92 | 0.0412 → 0.0402 |
| 2 | 0.501 → 0.501 | 23.56 → 23.23 | 0.1820 → 0.1758 |
| 3 | 0.330 → 0.337 | 10.23 → 10.01 | 0.3619 → 0.3554 |
| 4 | 0.489 → 0.499 | 64.62 → 64.27 | 0.1425 → 0.1392 |
| 5 | 0.567 → 0.578 | 66.32 → 69.10 | 0.0574 → 0.0565 |
| 6 | 0.708 → 0.712 | 86.00 → 89.14 | 0.0230 → 0.0230 |

| Layer | L entropy (norm.) | L distance | L prev-share |
|---|---|---|---|
| 0 | 0.920 → 0.920 | 102.89 → 102.47 | 0.0184 → 0.0183 |
| 1 | 0.832 → 0.837 | 97.57 → 100.23 | 0.0311 → 0.0307 |
| 2 | 0.531 → 0.528 | 43.11 → 45.79 | 0.1593 → 0.1503 |
| 3 | 0.477 → 0.501 | 19.78 → **23.80** | 0.2016 → 0.1874 |
| 4 | 0.398 → 0.408 | 27.10 → 27.82 | 0.3036 → 0.2961 |
| 5 | 0.516 → 0.532 | 73.29 → 72.63 | 0.0658 → 0.0668 |
| 6 | 0.690 → 0.688 | 84.65 → 87.80 | 0.0219 → 0.0220 |

**Finetuning did not restructure attention.** The largest single movement anywhere is
Model L's layer 3, where mean distance rises from 19.78 to 23.80 positions and normalised
entropy from 0.477 to 0.501 — attention becoming slightly broader and less local. Every
other cell moves in the second or third decimal.

The layer arc Phase 2 identified — diffuse survey in layers 0–1, sharp focus at layer 3
(H) or 4 (L), broadening again through layers 5–6 — survives unchanged. So does head
specialisation: the previous-token detectors stay previous-token detectors, with
prev-share shifting by at most 0.014.

### 6.2 Reading the heatmaps

The layer-6 panels on a `chain_pair` prompt show the structure Phase 2 described. Position
0 (`प्रश्न`) receives disproportionate weight across every head — the attention-sink
behaviour a softmax exhibits when a head has nothing specific to attend to and must place
its mass somewhere. Below the diagonal the heads pick out short-range structure; the
strict upper-triangular emptiness is the causal mask, visible directly.

Comparing the pretrained and finetuned panels for the same prompt, the differences are
not apparent by eye. This is consistent with the statistics rather than a failure to
look: a change of 0.007 in normalised entropy is not something a colour scale resolves.

### 6.3 Why this is the expected result, not a null one

The finetuned checkpoint is 100 optimiser steps past the pretrained one, at a learning
rate of 1e-4 — one sixth of pretraining's peak — over 6,400 examples averaging 36 tokens.
That is a very small perturbation applied to a model that had already consumed 524 million
tokens.

The loss curves say the same thing from the other side. Masked loss fell sharply while
unmasked loss *rose* (§3.2): the model changed its output distribution at the answer
position without materially changing how it reads the prompt. A task learned by adjusting
the final mapping rather than by reorganising intermediate representations is precisely a
task that leaves attention statistics alone.

The honest conclusion for §3.2's question — did finetuning change local versus long-range
attention or head specialisation? — is **no, measurably not**, and the mechanism above
explains why.

---

## 7. The four questions

### 7.1 How did data scale and quality differ between Model H and Model L?

**Scale barely differed; quality and composition did.**

| | Model H | Model L |
|---|---|---|
| Tokens collected | 768,048,386 | 746,342,222 |
| Documents | 1,063,900 | 1,331,446 |
| Manual share of tokens | 27.89% | 21.88% |

The token counts are within 3% of each other, which understates the difference in
difficulty. Reaching ~746M Nepali tokens required 25% *more documents* than Hindi, because
Nepali documents are shorter and sparser. The manual share tells the same story: Hindi
comfortably exceeded the 20% requirement at 27.89%, while Nepali reached 21.88% with
little margin, and getting there took proportionally more scraping of a smaller pool of
sources.

The practical consequence is not "less data" but **less redundancy per token**. Both
corpora hit the target; the Nepali one is assembled from thinner material.

---

### 7.2 How do language-modeling and reasoning results compare across the two resource tiers?

| | Model H | Model L | gap |
|---|---|---|---|
| Test perplexity | 28.23 | 45.09 | −60% for L |
| Test bits-per-byte | 0.4916 | 0.4527 | +8% for L |
| **Test bits per character** | **1.2099** | **1.1991** | **+0.9% for L** |
| **Reasoning exact match** | **0.5293** | **0.4553** | **−14% for L** |

The two levels disagree, and the disagreement is the interesting part.

**At the language-modelling level the two models are nearly identical.** Once normalised
to bits per character — the only unit that is fair across two different tokenizers and two
different scripts — they are within 0.9%. Perplexity's 60% gap is almost entirely a
tokenizer artefact: a Nepali token carries 4.586 characters against Hindi's 3.992, so each
Nepali prediction is a harder problem and per-token loss is correspondingly higher.

**At the reasoning level a real gap appears.** Model H leads by 7.4 points absolute, 14%
relative, and the lead is consistent across almost every template. On the two symbolic
superlative families it is wide: 0.7306 against 0.5111 for `chain_superlative_least`.

So the resource tier does not show up in how well each model predicts its own language —
it shows up in what each model can be taught to *do* with that language. Reasoning is the
more sensitive probe of corpus quality, which is not obvious in advance and is the
clearest single result of this project.

---

### 7.3 What tokenizer / corpus factors most affected the lower-resource model?

Three, in descending order of measured impact.

**1. Fertility, and what it costs in effective context.** Nepali's tokenizer needs 1.381
tokens per word against Hindi's 1.259 — 9.7% more. With a fixed 512-token context and a
fixed 524M-token training budget, Model L therefore saw *fewer words* than Model H for the
same compute. The same effect compresses the reasoning prompts: an average Nepali example
is 38.0 tokens against Hindi's 35.5.

**2. Byte-fallback rate.** Nepali falls back to raw bytes on ~2.17% of tokens against
Hindi's ~1.33% — 63% more often. Every byte-fallback token is a piece the tokenizer failed
to learn, spending context on something carrying almost no semantic content. This is a
direct consequence of a thinner corpus: fewer occurrences of rare forms means fewer merges
learned for them.

**3. Corpus composition rather than corpus size.** Model L's corpus needed 25% more
documents to reach a similar token count, which means shorter documents and less long-range
context per document. For a model whose entire ability to chain inequalities depends on
holding several facts at once, that is the factor most likely to explain the reasoning gap
— and it is consistent with Model L's weakness being concentrated in exactly the templates
that require traversing an assembled ordering (`chain_superlative_least`, 0.5111 against
Hindi's 0.7306).

Notably, **vocabulary size is not on this list**. Both models use 16,000 pieces, fixed in
Phase 1 by the parameter budget. Holding it constant is what allows the comparison above
to attribute differences to the corpus rather than to the tokenizer's capacity.

---

### 7.4 What evidence explains the observed differences?

Four pieces, each independently checkable.

**The architecture is identical, so the corpus is the only variable.** Both models are
7 layers × 7 heads × 448 dimensions, 24,298,176 parameters, 16,000-piece vocabulary,
512-token context, trained for the same 16,000 steps on the same 524,288,000 tokens with
the same seed. `report/<lang>/checkpoint_config.json` records this for both. Any measured
difference has to come from the data.

**Normalising for script removes most of the apparent language-modelling gap.** Perplexity
says Model L is 60% worse; bits per character says 0.9% better. The intermediate quantity
— 4.586 versus 3.992 characters per token — fully accounts for the discrepancy. This is
the evidence that the perplexity gap is a measurement artefact and not a quality gap.

**The reasoning gap survives that normalisation.** Exact match is computed on generated
text against a gold string; it has no tokenizer-dependence at all. Model H's 7.4-point lead
is therefore a real difference in capability, not a unit problem. It is the one gap that
cannot be normalised away.

**The failure modes are shared, which localises the cause.** Both models fail on numeric
comparison in the same way and at similar magnitude (0.09 and 0.03 on numeric
superlatives, both below chance). Both succeed on symbolic chains. If the Nepali corpus
were simply *worse*, one would expect degradation spread across all templates; instead the
profile is the same shape, shifted down. That pattern points at a shared architectural
limit — no representation of numeric magnitude in a 16k BPE vocabulary — with a corpus
effect layered on top of it, rather than at a single cause.

---

## 8. Limitations

Stated because they bound what the results above support.

**Finetuning overfits, and early stopping is doing real work.** Best validation arrives at
step 100 of 1,125. The reported numbers come from that checkpoint. A run configured for
one epoch rather than three would be a cleaner match to the observed optimum.

**Both models are compute-limited, not data-limited.** Phase 2's validation loss was still
falling at step 16,000 — 0.048 nats over the final 4,000 steps for Model H. Neither model
is converged; the reasoning results are for undertrained models.

**The reasoning set is synthetic and narrow.** Eight template families over two entity
pools. High accuracy on it is evidence of transitive comparison over a controlled
vocabulary, not of general reasoning.

**The chance baseline is per-template, not per-model.** It assumes uniform guessing among
the named entities, which is the right floor for the selection templates but treats the
equality template's two-way choice identically to a three-way one.

---

## 9. Reproduction

```bash
# 1. Generate the reasoning corpora (local, CPU, seconds)
python -m scripts.make_reasoning_data --lang hi
python -m scripts.make_reasoning_data --lang ne

# 2. Finetune from each language's own pretrained checkpoint (GPU, ~20 min each)
python -m scripts.finetune --lang hi
python -m scripts.finetune --lang ne

# 3. Pretrained vs finetuned exact match (GPU, ~5 min each)
python -m scripts.eval_reasoning --lang hi
python -m scripts.eval_reasoning --lang ne

# 4. Attention, on a comparative-reasoning prompt, both stages
python -m scripts.attention_analysis --lang hi --checkpoint checkpoints/hindi/best.pt \
    --sentence "<a chain_pair prompt>" --layers 0 6 --heatmaps-only \
    --out-dir report/hindi/figures-reasoning-pretrained
python -m scripts.attention_analysis --lang hi \
    --checkpoint checkpoints/hindi-finetuned/best.pt \
    --sentence "<the same prompt>" --layers 0 6 --heatmaps-only \
    --out-dir report/hindi/figures-reasoning-finetuned

# 5. Verify
python -m pytest tests/ -v
```

`kaggle/phase3_finetune.ipynb` runs steps 2–4 for both languages in one session and
packages the results. `kaggle/README.md` is the operator's guide.

Determinism: the same data, seed 1337, code and GPU reproduce these numbers bit-for-bit.
This was verified in Phase 2, where two people's independent runs matched at every logged
step and on every evaluation metric.

### Artefacts

| Path | Contents |
|---|---|
| `<lang>/reasoning/*.jsonl` | The generated corpora, 24k/3k/3k |
| `report/<lang>/reasoning_stats.json` | Size, template variety, leakage verification |
| `report/<lang>/reasoning_eval.json` | Pretrained vs finetuned, all breakdowns, qualitative examples |
| `report/<lang>/finetune_log.jsonl` | Every logged step and evaluation |
| `report/<lang>/attention_stats_finetuned.json` | Per-layer and per-head, finetuned |
| `report/<lang>/figures-reasoning-{pretrained,finetuned}/` | Layer 0 and 6 heatmaps, same prompt |
| `report/<lang>/figures-finetuned/` | Entropy, distance and previous-share plots |

Google Drive links for the finetuned checkpoints are in the top-level `README.md`.
