# How This Works — Phase 2, A Complete Walkthrough

This document explains everything built in Phase 2, assuming you have read
`how-it-works.md` (Phase 1) and nothing else. Every number in it was measured from the
files on disk, and every code reference points at a real line you can open.

---

## 0. What are we actually doing?

Phase 1 went to the market and chopped the vegetables. **Phase 2 cooks the meal.**

We build a **robot that writes Hindi**, and a completely separate robot that writes
Nepali. Each one is shown half a billion tokens of its own language, one token at a
time, and asked the same question over and over:

> Here are the words so far. What is the next one?

It guesses. We tell it the right answer. It nudges its internal numbers a little. Then we
do that 16,000 times, on 32,768 tokens each time.

That is the whole idea. There is no other trick. Everything below is detail about *how*
you make that loop work, and *how you check* that what came out actually learned
something.

**What "done" looks like:** two trained models, each about 24 million numbers, that can
continue a piece of text in their own language — plus a pile of measurements proving they
work and showing what they learned.

Here is what we ended up with:

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Parameters | 24,298,176 | 24,298,176 |
| Tokens seen in training | 524,288,000 | 524,288,000 |
| Passes over the corpus ("epochs") | 0.710 | 0.730 |
| Test perplexity | **28.23** | **45.09** |
| Test bits-per-byte | **0.4916** | **0.4527** |

Look at that last pair of rows and notice they **disagree about which model is better**.
Hindi wins on perplexity; Nepali wins on bits-per-byte. That is not a bug, and explaining
it is the single most interesting thing in this phase. We get to it in §8.

---

## 1. Words you need to know

Phase 1's vocabulary still applies (token, tokenizer, corpus, train/validation/test).
These are the new ones.

**Parameter**
One number inside the model that gets adjusted during training. Our models have
24,298,176 of them. "Training" means "slowly changing these 24 million numbers until the
model's guesses get good."

**Model / network**
The whole pile of parameters, plus the fixed recipe for combining them with input to
produce an output.

**Transformer**
The particular recipe we use. Invented in 2017; it's what GPT, Claude and every other
modern language model are built from. Its key idea is **attention** (below).

**Decoder-only**
Our flavour of transformer. It reads left to right and only ever predicts the *next*
token. It is never allowed to peek at what comes after. (The other flavours, used for
translation, can see the whole input at once. We don't need that.)

**Attention**
The mechanism that lets the model look back at earlier words and decide which ones matter
right now. When predicting the word after *"भारत की राजधानी नई"*, a good model should be
paying attention to *"राजधानी"*. Attention is the machinery that makes that possible, and
§5 builds it from scratch.

**Head**
Attention doesn't look back just once — it looks back several times in parallel, each
"head" free to care about something different. We have 7 heads per layer and 7 layers, so
**49 heads** in total. §10 shows what each of them ended up doing.

**Layer / block**
One round of (attention, then a small amount of ordinary arithmetic). We stack 7 of them.
Deeper stacks can build more complicated ideas, because each layer works on the output of
the last.

**Embedding**
A lookup table turning a token number into a list of 448 numbers. Token 582 becomes a
particular row of 448 numbers; token 1601 becomes a different row. Those rows start random
and get trained, so after training, similar words end up with similar rows.

**d_model (we use 448)**
How many numbers represent one token as it flows through the network. Bigger = more room
to express things, and more parameters.

**Context length (we use 512)**
The most tokens the model can look back over at once. Token 600 cannot see token 1. This
is a hard architectural wall, not a preference — see §5.2.

**Loss (cross-entropy)**
A single number saying how wrong the model's guesses were. Lower is better. Measured in
**nats**. If the model were guessing uniformly at random among 16,000 tokens, the loss
would be ln(16000) ≈ 9.68. Ours ends at 3.34.

**Perplexity (PPL)**
Just `e^loss`, reported because it has a nice interpretation: *"the model is as confused
as if it were choosing uniformly among this many options."* Loss 3.34 → perplexity 28.4,
so our Hindi model is about as unsure as someone picking between 28 equally likely words.

**Bits-per-byte (BPB)**
The same information, divided by how many *bytes* of raw text were covered instead of how
many tokens. This matters enormously and §8 is devoted to it.

**Epoch**
One complete pass over the training data. We do **less than one** — 0.71 for Hindi. That
sounds broken but is exactly right; §7.4 explains.

**Checkpoint**
A saved snapshot of the model mid-training, so a crash doesn't destroy hours of work.

---

## 2. The big picture

```
   Phase 1 output                                  ┌──────────────────────┐
   (splits + tokenizer)                            │  hi.model  (16,000)  │
            │                                      └──────────┬───────────┘
            ▼                                                 │
   ┌─────────────────────┐                                    │
   │  STEP 1  PACK       │ ◄──────────────────────────────────┘
   │  text → numbers     │  scripts/pack_tokens.py
   └──────────┬──────────┘  738,559,188 uint16 numbers in one flat file
              │
              ▼
   ┌─────────────────────┐
   │  STEP 2  BUILD      │  lma/model.py
   │  the transformer    │  24,298,176 parameters, from nn.Linear upward
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  STEP 3  TRAIN      │  scripts/pretrain.py
   │  16,000 steps       │  ~3.5 hours on one Kaggle T4
   └──────────┬──────────┘
              │
              ├──────────────┬──────────────┬──────────────┐
              ▼              ▼              ▼              ▼
      ┌──────────────┐ ┌───────────┐ ┌───────────┐ ┌──────────────┐
STEP 4│  MEASURE     │ │ GENERATE  │ │  LOOK     │ │  DRAW        │
      │  PPL / BPB   │ │ + score   │ │  INSIDE   │ │  the curves  │
      └──────────────┘ └───────────┘ └───────────┘ └──────────────┘
       eval_lm.py    eval_generation.py  attention_   plot_training.py
                                         analysis.py  compare_models.py
```

Which script does what, and how long it actually took:

| Step | Script | Where | Time |
|---|---|---|---|
| 1. Pack tokens | `scripts/pack_tokens.py` | laptop (CPU) | minutes per language |
| 2. — | `lma/model.py` | (library, not run directly) | — |
| 3. Pretrain | `scripts/pretrain.py` | Kaggle T4 GPU | ~3.5 h per language |
| 4. LM metrics | `scripts/eval_lm.py` | Kaggle T4 | ~4 min per language |
| 5. Generation | `scripts/eval_generation.py` | Kaggle T4 | ~5 min per language |
| 6. Attention | `scripts/attention_analysis.py` | Kaggle T4 | ~2 min per language |
| 7. Plots | `scripts/plot_training.py` | laptop | seconds |
| 8. Comparison | `scripts/compare_models.py` | laptop | seconds |

Everything from step 3 onward runs on Kaggle because a laptop CPU would take weeks.
Steps 1, 7 and 8 run locally because they are cheap.

---

## 3. STEP 1 — Packing text into numbers

### The problem this solves

Phase 1 left us with compressed text files: `hi-train-00000.jsonl.zst` and 200-odd
friends, each holding thousands of documents as *text*.

The GPU cannot use that. Three separate problems:

1. **It's text, not numbers.** The tokenizer has to run over every document.
2. **It's compressed.** Decompressing repeatedly during training would waste all our GPU
   time on the CPU.
3. **It's in pieces.** Training wants to grab a random window of 512 tokens instantly,
   from anywhere in 738 million. Hunting through 200 files for that is hopeless.

### How we solved it

Do all the work **once**, up front, and write the result as one flat wall of numbers.

```
hi-train-00000.jsonl.zst          ┐
hi-train-00001.jsonl.zst          │   tokenize      ┌────────────────────────┐
        ...                       ├──  every    ──► │      train.bin         │
hi-train-00203.jsonl.zst          │   document      │  1,477,118,376 bytes   │
                                  ┘                 │  738,559,188 numbers   │
                                                    └────────────────────────┘
```

Each token number is stored as **uint16** — an unsigned 16-bit integer, exactly 2 bytes,
holding any value from 0 to 65,535. Our vocabulary is 16,000, so every token id fits
comfortably. That is why the file is exactly `738,559,188 × 2 = 1,477,118,376` bytes.

If we had chosen a 32-bit integer the file would be double the size for zero benefit.
If Phase 1 had chosen a vocabulary above 65,536, uint16 would silently overflow and the
data would be quietly corrupted — one more small reason a modest vocabulary is easier to
live with.

Documents are separated by a special **end-of-sentence token** (id 3), so the model can
learn that documents end and a fresh one begins.

### The code

`scripts/pack_tokens.py` (345 lines). The interesting part is how it avoids two failure
modes.

**Failure mode 1 — running out of memory.** You cannot tokenize 738 million tokens in RAM.
So the script writes *parts*: it processes one shard at a time into
`hindi/data/tokens/parts/`, then concatenates the parts at the end.

**Failure mode 2 — a half-written file that looks finished.** If the machine dies during
concatenation you would be left with a `train.bin` that opens fine and is silently
truncated. Training would then run on partial data and you would never know.

The fix is the same trick Phase 1 used for shards — write to a temporary name, rename at
the very end (`scripts/pack_tokens.py:154`). A rename is atomic on Linux: either the whole
file exists under the final name or it doesn't. There is no in-between state.

### What comes out

```
hindi/data/tokens/
├── train.bin              1,477,118,376 bytes    738,559,188 tokens
├── train.meta.json        counts and provenance
├── validation.bin            30,719,548 bytes     15,359,774 tokens
├── validation.meta.json
├── test.bin                  30,386,648 bytes     15,193,324 tokens
├── test.meta.json
└── pack_summary.json      everything, including the tokenizer's SHA-256
```

Measured, both languages:

| | Hindi | Nepali |
|---|---|---|
| Train tokens | 738,559,188 | 717,728,934 |
| Train documents | 1,021,431 | 1,278,292 |
| Validation tokens | 15,359,774 | 14,825,389 |
| Test tokens | 15,193,324 | 15,119,345 |

### The tricky bit — the tokenizer's fingerprint is recorded

`pack_summary.json` stores the SHA-256 of the tokenizer file that produced these numbers:

```json
"tokenizer_sha256": "15bb291a2e55261f7b24017d76c5bbc07d5659a3edfde07f118a04defbe21450"
```

This matters more than it looks. Token id 582 means `▁भारत` **only under one specific
tokenizer**. Retrain the tokenizer with a different random seed and id 582 becomes some
other word — but the `.bin` file looks exactly the same. You would train happily on
scrambled data and get a mysteriously bad model.

Recording the hash makes that mistake detectable instead of invisible.

---

## 4. STEP 2 — Deciding the shape of the model

Before writing any model code you must choose: how many layers, how wide, how many heads?
The assignment fixes the budget at **~25 million parameters** and leaves the rest to you.

### Where the parameters actually go

Everything is decided by four numbers. Ours:

```
vocab_size = 16,000     n_layer = 7     d_model = 448     d_ff = 1,792
```

And here is every parameter in the model, derived rather than asserted:

| Piece | Formula | Count |
|---|---|---|
| Token embedding | 16,000 × 448 | 7,168,000 |
| Positional embedding | 512 × 448 | 229,376 |
| — per block — | | |
| LayerNorm ×2 | 2 × (448 × 2) | 1,792 |
| QKV projection | 448 × 1,344 + 1,344 | 603,456 |
| Output projection | 448 × 448 + 448 | 201,152 |
| Feed-forward up | 448 × 1,792 + 1,792 | 804,608 |
| Feed-forward down | 1,792 × 448 + 448 | 803,264 |
| **one block total** | | **2,414,272** |
| All 7 blocks | 2,414,272 × 7 | 16,899,904 |
| Final LayerNorm | 448 × 2 | 896 |
| Output head | *tied — see §6.3* | **0** |
| **TOTAL** | | **24,298,176** |

You can check every line of that by hand, and it matches `report/hindi/checkpoint_config.json`
exactly. Being able to reproduce this table on a whiteboard is worth more in an evaluation
than any other single fact about the project.

### The trade-off that decides everything

Notice the token embedding: **7,168,000 parameters, 29.5% of the entire model**, spent
purely on a lookup table. That is the cost of the vocabulary, and it is why Phase 1's
choice of 16,000 pieces was the decision that fixed this model's shape.

The arithmetic is unforgiving. Everything *except* the embedding costs 17,130,176. From a
25,000,000 budget that leaves 7,869,824 for the table, and each vocabulary entry costs
`d_model = 448` parameters:

```
7,869,824 ÷ 448 = 17,566 pieces — the largest vocabulary we could afford
```

A 32,000-piece vocabulary would need 14,336,000 parameters for the embedding alone, and
the model would blow the budget before a single transformer block existed. That is the
whole reason Phase 1 chose 16,000 despite 24,000 scoring slightly better on fertility.

### Why 7 layers of width 448

We had roughly 24M to spend on the non-embedding part and several ways to spend it:

| Shape | Total params | Aspect ratio (d/n_layer) |
|---|---|---|
| 384 wide × 10 deep | 24,092,928 | 38.4 |
| **448 wide × 7 deep** | **24,298,176** | **64.0** |
| 512 wide × 5 deep | 24,218,624 | 102.4 |

All three fit. We chose the middle one for three reasons:

1. **Aspect ratio 64.0 matches GPT-2 small** (768/12), which is a well-tested point in
   this design space rather than a guess.
2. **512 × 5 is too shallow.** Five blocks give very little room to compose ideas — each
   layer can only build on the one below it, and five storeys isn't much of a building.
3. **448 ÷ 7 heads = 64 exactly.** Head width 64 is the size GPU attention kernels are
   tuned for. 384 × 10 would also have worked but costs more sequential layers per step
   on a single T4, for no measured gain at this scale.

Recorded with reasoning in `hindi/configs/model.json` under `_depth_vs_width`.

---

## 5. STEP 3 — Building the transformer

This is the heart of the phase. The assignment forbids `nn.Transformer`, HuggingFace
model classes, and any pre-built attention block. We may use `nn.Linear`, `nn.Embedding`,
`nn.LayerNorm` and `nn.Dropout` — nothing higher level.

So everything below is built by hand in `lma/model.py` (396 lines).

We deliberately also avoid `F.scaled_dot_product_attention`, PyTorch's built-in fast
attention. It would be faster, but the whole point of this phase is to be able to explain
every tensor operation, and a single opaque call defeats that.

### 5.1 From token numbers to vectors

Two lookup tables, added together.

```
input:  [582, 277, 1601, ...]              token ids, one per position

token embedding    row 582   → [0.12, -0.44, ..., 0.03]   448 numbers
positional emb     row 0     → [0.01,  0.22, ..., -0.1]   448 numbers
                               ─────────────────────────
                    added     → [0.13, -0.22, ..., -0.07]  448 numbers
```

**Why add a position at all?** Because attention has no built-in sense of order. To the
raw attention mechanism, *"कुत्ते ने आदमी को काटा"* and *"आदमी ने कुत्ते को काटा"* are the same
bag of words. Position must be injected explicitly or word order simply doesn't exist.

**Why learned absolute positions?** There is a literal table with 512 rows, one per
position, and those numbers are trained like any other. We chose it over sinusoidal or
rotary encodings for two reasons:

- Every operation is explainable end to end — no trigonometry to hand-wave through.
- It makes the context ceiling **honest**. There is no row 512, so the model *cannot* run
  on a longer sequence; it raises an error. Sinusoidal and rotary encodings would happily
  extrapolate to position 900 and quietly produce garbage. A loud failure beats a silent
  one.

### 5.2 Attention, with real shapes

This is the part worth reading slowly. `lma/model.py:84-152`.

Take a batch of 32 sequences, each 512 tokens long, each token now 448 numbers wide.

**Step 1 — make three copies with different jobs.** One `nn.Linear` produces all three at
once:

```
x                  (32, 512, 448)
  → self.qkv(x)    (32, 512, 1344)          1344 = 3 × 448
  → .split(448)    three tensors, each (32, 512, 448)
```

They are called **query**, **key** and **value**. The metaphor: each position emits a
query ("what am I looking for?"), advertises a key ("what do I have?"), and holds a value
("what I will hand over if you pick me").

**Step 2 — split into 7 heads.**

```
(32, 512, 448)
  → .view(32, 512, 7, 64)        split 448 into 7 heads of 64
  → .transpose(1, 2)             (32, 7, 512, 64)
```

That transpose matters: it brings the head axis forward so each head's `(512, 64)` matrix
is contiguous, and the matrix multiplies below operate per head automatically.

**Step 3 — every position scores every position.**

```
q                    (32, 7, 512,  64)
k.transpose(-2,-1)   (32, 7,  64, 512)
q @ k^T              (32, 7, 512, 512)     ← the attention scores
```

That `512 × 512` grid, for each of 7 heads and 32 sequences, is the object everything
else is about. Entry `(t, s)` is "how much does position *t* care about position *s*?"

**Step 4 — divide by √64 = 8.** Without this, the dot products grow with head width, the
softmax gets fed huge numbers, and it saturates — meaning almost all the weight lands on
one position and the gradients vanish. Scaling by `1/√dk` keeps the scores in a sane
range whatever the head width. It's one multiply and skipping it breaks training.

**Step 5 — the causal mask** (§5.3, below).

**Step 6 — softmax.** Turns each row of scores into probabilities summing to 1. Row *t*
is now a proper distribution over positions 0…*t*.

**Step 7 — weighted sum of values, then reassemble.**

```
attn @ v             (32, 7, 512, 512) @ (32, 7, 512, 64) → (32, 7, 512, 64)
  → .transpose(1,2)  (32, 512, 7, 64)
  → .contiguous()    ← required: transpose only shuffles strides, view needs real layout
  → .view(32,512,448) heads concatenated back
  → self.proj(...)   final mix across heads
```

### 5.3 The causal mask — why the model can't cheat

If position 5 could attend to position 6, predicting position 6 would be trivial: just
copy it. The model would score a perfect loss during training and be completely useless.

So we forbid it, with an upper-triangular mask (`lma/model.py:128`):

```
        key:  0    1    2    3
query 0     [ ok   ✗    ✗    ✗  ]
      1     [ ok   ok   ✗    ✗  ]
      2     [ ok   ok   ok   ✗  ]
      3     [ ok   ok   ok   ok ]
```

The `✗` entries are set to **−∞ before the softmax**. Since `e^−∞ = 0`, they come out as
exactly zero afterward — not "very small", exactly zero. That is what makes it an
*additive* mask: add −∞, and the softmax does the rest.

Two details in the code that are easy to get wrong:

**The softmax runs in float32.** Under mixed precision the scores arrive as float16, where
exponentiating a large number overflows to infinity and the result becomes NaN. The code
promotes to float32 for the softmax only, then converts back (`lma/model.py:140`).

**No row can be entirely masked.** Row 0 has exactly one allowed position — itself. If a
row were fully −∞ the softmax would divide by zero and produce NaN. Because the diagonal
is always allowed, this can't happen.

**We test this rather than assume it.** `tests/test_causal.py` contains five tests,
including the exact experiment the assignment asks for:

```python
def test_future_token_cannot_change_past_logits()   # change token t+1 → logits at t identical
def test_prefix_gives_same_logits_as_full_sequence()
def test_the_prefix_check_has_teeth()               # sanity: the test can actually fail
def test_attention_mass_is_confined_to_the_past()
def test_shuffling_positions_changes_output()
```

The third one deserves a mention: a test that always passes is worse than no test. It
deliberately breaks the mask and confirms the check *does* catch it.

### 5.4 The feed-forward network

After attention has mixed information *between* positions, each position does some private
thinking on its own:

```
448  →  1792  →  GELU  →  448
```

Widen by 4×, apply a non-linearity, narrow back. Without the non-linearity, two stacked
linear layers collapse into one linear layer and the extra depth buys nothing at all.

GELU rather than ReLU: it's smooth near zero, which behaves better with gradient descent,
and it's what GPT-2 used.

### 5.5 One block, and why pre-norm

```
        x ─────────────────────┐
        │                      │
    LayerNorm                  │
        │                      │
    Attention                  │
        │                      │
        └──────── + ───────────┘      ← residual: add the input back
                  │
                  ├───────────┐
              LayerNorm       │
                  │           │
             FeedForward      │
                  │           │
                  └──── + ────┘
                        │
                        ▼
```

Two things happen here that both matter enormously.

**Residual connections.** Each sublayer's output is *added to* its input rather than
replacing it. This creates an unbroken path from the embeddings all the way to the output.
Gradients flow back along that path without being repeatedly multiplied down, which is
what allows deep stacks to train at all.

**Pre-norm.** The LayerNorm sits *inside* the branch, before attention — not after the
addition. Post-norm (the original 2017 design) puts a normalisation directly on the
residual path, so gradients must traverse 7 of them on the way back. At this depth
post-norm needs a much more carefully tuned warmup to avoid diverging early. Pre-norm just
works, which is why essentially every modern model uses it.

Because the last block's output never gets normalised under pre-norm, a **final
LayerNorm** follows the stack (896 parameters).

### 5.6 The output head, and the free 7 million parameters

The last step turns 448 numbers back into a score for each of 16,000 tokens. That is a
`448 × 16,000` matrix — 7,168,000 parameters.

We don't allocate it. We reuse the input embedding, transposed. This is **weight tying**,
and the reason it makes sense is that both matrices are answering the same question in
opposite directions:

- the embedding maps *token → meaning*
- the output head maps *meaning → token*

The row for `भारत` in one is a natural fit for the column for `भारत` in the other.

The saving is enormous at our scale — 7,168,000 parameters, **29.5% of the entire model**,
for free. `lm_head` shows as `0` in the parameter table for exactly this reason.
`tests/test_model.py:47` verifies the two really are one shared tensor and not an
accidental copy.

---

## 6. STEP 4 — Training

`scripts/pretrain.py` (366 lines). This is the loop that does the actual learning.

### The loop, in words

```
repeat 16,000 times:
    grab 32,768 random tokens from train.bin
    ask the model to predict each next token
    measure how wrong it was            → loss
    compute which direction every parameter should move  → backward()
    nudge all 24 million parameters a little             → optimizer.step()
    every 500 steps:   check the validation loss
    every 1,000 steps: save a checkpoint
```

### The hyperparameters, and why

From `hindi/configs/train.json`:

| Setting | Value | Why |
|---|---|---|
| `tokens_per_step` | 32,768 | The quantity actually held fixed |
| `micro_batch_size` | 32 | Only a memory knob — see below |
| `max_steps` | 16,000 | 16,000 × 32,768 = 524,288,000 tokens |
| `learning_rate` | 6e-4 | Conservative; we get one run, no budget for a restart |
| `warmup_steps` | 500 | Start tiny, ramp up |
| `min_lr_ratio` | 0.1 | Decay to 10% of peak, not to zero |
| `weight_decay` | 0.1 | Mild pull toward zero, discourages over-reliance on any one weight |
| `beta2` | 0.95 | Slightly faster-adapting than the 0.999 default; standard for LMs |
| `grad_clip` | 1.0 | Rescale any oversized update — one bad batch can't wreck the run |
| `seed` | 1337 | Same seed + same data + same GPU = bit-identical run |
| `amp_dtype` | float16 | Roughly 2× faster on a T4 |

### Gradient accumulation — how to have a big batch on a small GPU

We want 32,768 tokens per step. At 512 tokens per sequence that's 64 sequences. A T4
cannot hold 64 sequences of this model in memory at once.

So we do 2 passes of 32 sequences, adding up the gradients, and only *then* update:

```
micro-batch 1 (32 seqs) → backward()   gradients accumulate
micro-batch 2 (32 seqs) → backward()   gradients accumulate
                          optimizer.step()   ← one update, from all 64
```

Mathematically identical to one batch of 64. The key design point: **`tokens_per_step` is
what stays fixed**, and accumulation is derived from it. Move to a smaller GPU, halve
`micro_batch_size`, and accumulation automatically doubles — the learning dynamics are
completely unchanged. That's a property worth having, because it means the run is
reproducible on different hardware.

### The learning-rate schedule

```
6e-4 ┤          ╭──────╮
     │        ╭─╯       ╰──╮
     │      ╭─╯             ╰───╮
     │    ╭─╯                    ╰────╮
6e-5 ┤ ╭──╯                            ╰────────
     └─┴────┴──────────────────────────────────
       0   500                            16,000
       warmup           cosine decay
```

**Warmup (steps 0–500).** At initialisation the parameters are random and the gradients
are large and meaningless. Taking full-size steps immediately can throw the model into a
region it never recovers from. So we start near zero and ramp up.

**Cosine decay (500–16,000).** Big steps early to travel far; small steps late to settle
precisely. Decaying to 10% of peak rather than exactly zero keeps a little learning
happening right to the end.

The schedule is **stateless** (`lma/schedule.py`) — the learning rate is computed from the
step number by a formula, not accumulated. That means resuming from a checkpoint gives
exactly the right learning rate with no extra state to save or corrupt.

### Checkpointing — the mandatory part

The assignment is explicit that a run which can't resume loses marks. Kaggle terminates
sessions at twelve hours and can pre-empt sooner, so this is not hypothetical.

Every checkpoint holds:

```python
{
  'model':      ...,   # the 24 million parameters
  'optimizer':  ...,   # AdamW's momentum — dropping this causes a visible loss spike
  'scheduler':  ...,   # where we are in the schedule
  'scaler':     ...,   # mixed-precision state
  'step':       16000,
  'tokens_seen': 524288000,
  'best_val_loss': 3.3498,
  'model_config': ..., 'train_config': ...,
  'rng':        ...,   # Python, NumPy, CPU and CUDA random streams
  'language':   'hi',
}
```

The `rng` entry is the one most people forget. Without it, a resumed run draws a different
sequence of random batches than an uninterrupted one, and the two diverge. With it, a run
interrupted at step 9,000 and resumed produces *bit-identical* output to one that never
stopped. That's the difference between "it restarts" and "it resumes".

The script also catches `SIGTERM`, so a pre-emption warning turns into one final
checkpoint rather than lost work.

### What it looked like

```
step     50   loss 8.415   ← barely better than random (ln 16000 = 9.68)
step    100   loss 7.259
step    500   loss 5.34    ← warmup ends
step  4,000   loss 3.72
step  8,000   loss 3.52
step 16,000   loss 3.35    ← still falling
```

Roughly 39,200 tokens/second on a T4; about 3.5 hours per language.

---

## 7. STEP 5 — Did it work? The language-model metrics

`scripts/eval_lm.py` (186 lines). Run on validation and test, for both models.

### The results

| | Model H (Hindi) | Model L (Nepali) |
|---|---|---|
| Validation loss | 3.3476 | 3.8117 |
| Validation perplexity | 28.43 | 45.23 |
| Validation BPB | 0.4930 | 0.4527 |
| **Test loss** | **3.3402** | **3.8087** |
| **Test perplexity** | **28.23** | **45.09** |
| **Test BPB** | **0.4916** | **0.4527** |
| Tokens scored | 15,187,968 | 15,114,240 |

### 7.1 Why "test is slightly better than validation" is good news

Look carefully: test loss (3.3402) is **lower** than validation loss (3.3476). Both
languages. That is the opposite of what a model that memorised its training data would do.

The usual worry with any trained model is **overfitting** — memorising the training set
instead of learning the language. The signature is: training loss keeps dropping while
validation loss starts *rising*. We see nothing of the kind.

### 7.2 Why training loss looks *higher* than test loss

A confusing thing you'll notice in the logs: the reported training loss is often above the
test loss. That looks backwards. Two reasons, both mundane:

1. **Dropout is on during training, off during evaluation.** Training deliberately
   handicaps the model by zeroing 10% of activations. Evaluation doesn't.
2. **The training figure is a running average** over recent steps, so it lags behind the
   model's current ability by hundreds of steps of improvement.

### 7.3 Why overfitting was structurally impossible

Beyond the measurements, there's an argument from the setup itself:

```
tokens available:  738,559,188
tokens consumed:   524,288,000
                   ───────────
                   0.710 epochs
```

**No document was ever seen twice.** You cannot memorise what you've seen once. This is a
much stronger guarantee than "the validation curve looked fine".

### 7.4 Why less than one epoch is the right answer, not a shortfall

This surprises people, so it's worth stating plainly.

The Chinchilla scaling result says that for a fixed compute budget, the best loss comes
from roughly **20 tokens per parameter**. For our 24,298,176 parameters that's ~486M
tokens. We used 524M — **21.6 tokens per parameter**, essentially exactly optimal.

This is also why the assignment asks for ~25M parameters *and* ~500M tokens: those two
numbers are the same decision.

And fresh tokens beat repeated tokens. A gradient computed on unseen text carries new
information; a second pass over the same text mostly re-confirms what the weights already
hold. Given a fixed number of steps, spending them on new data wins.

For the record, validation loss was **still falling** at the final step — 0.048 nats of
improvement over the last 4,000 steps for Hindi, 0.053 for Nepali. No plateau. The models
are limited by GPU hours, not by data.

### 7.5 A useful diagnostic to know

If someone reports a suspiciously *low* perplexity, the usual culprits are:

- **padding counted in the loss** — the model trivially predicts pad tokens and the
  average collapses (our packing has no padding at all, so this can't happen here);
- **val/test built before deduplication**, so documents leak across splits;
- **boilerplate not stripped** — nav bars and cookie notices repeating across thousands of
  pages are nearly free to predict;
- **evaluating on the training split** by mistake.

---

## 8. The two metrics disagree — and that's the finding

This is the most interesting result in Phase 2, so it gets its own section.

```
                 perplexity        bits-per-byte
Hindi   (H)         28.23   ✓ better      0.4916
Nepali  (L)         45.09              0.4527   ✓ better
```

Hindi wins by one measure and loses by the other. Both numbers are correct. The
explanation is entirely about **what each metric divides by**.

### Perplexity divides by tokens

`PPL = e^(loss per token)`. A token is whatever your tokenizer says it is. Ours pack
different amounts of text per token in the two languages:

| | Hindi | Nepali |
|---|---|---|
| Characters per token | 3.992 | 4.586 |
| Bytes per token | 9.796 | 12.146 |

A Nepali token carries **24% more bytes** than a Hindi one. Predicting it is a harder job.
Of course the per-token loss is higher — each guess is worth more.

### Bits-per-byte divides by bytes

BPB normalises by raw UTF-8 bytes, which is tokenizer-independent. By that measure Nepali
looks better — because each Nepali token covers more ground, and the harder-per-token
prediction is spread over more bytes.

### But BPB has its own trap

Devanagari costs about **2.5 UTF-8 bytes per character** (measured: 2.454 for our Hindi
split, 2.649 for Nepali), while Latin script costs 1. So
BPB for any Devanagari model is divided by a bigger number than an English model's would
be, making it look artificially good. Comparing our 0.49 BPB against an English model's
0.49 BPB would be meaningless.

### The honest number is bits-per-character

| | Hindi | Nepali |
|---|---|---|
| **Bits per character** | **1.2099** | **1.1991** |

Nepali still edges ahead, but by 0.9% rather than the 12% BPB implies. That is the honest
comparison, and it's the number to quote.

### What to take from this

Three lessons worth stating out loud:

1. **Never compare perplexity across different tokenizers.** A "Hindi PPL 18" found online
   and our 28.23 may describe models of very different quality, in either direction. The
   PDF asks for BPB alongside PPL for exactly this reason.
2. **BPB is better but not neutral** — it's biased by how the script encodes into UTF-8.
3. **Bits-per-character is the cross-script-honest unit** at this level of analysis.

### Is 1.21 bits/char plausible?

Sanity check, because "is my result reasonable" is a fair question:

| Reference | bits/char |
|---|---|
| Uniform over 16,000 tokens (no learning) | ~3.5 |
| **Ours, 24M params, 0.71 epochs** | **1.21** |
| Large modern LMs on well-resourced languages | ~0.6–0.9 |

Comfortably below any trivial baseline, comfortably above what heavily-trained large
models reach. Exactly where a 24M-parameter model on 524M tokens should land.

---

## 9. STEP 6 — Can it actually write?

`scripts/eval_generation.py` (324 lines). Take 200 held-out prefixes of 64 tokens, ask
each model to continue for 128 tokens, and compare against what really followed.

Four decoding settings:

- **greedy** — always take the single most likely token
- **temperature 0.5** — sample, but heavily favour likely tokens
- **temperature 1.0** — sample from the model's actual distribution
- **temperature 1.5** — flatten the distribution, take more risks

### Hindi

| Setting | BLEU | chrF++ | ROUGE-L | distinct-1 | distinct-2 | 4-gram repetition |
|---|---|---|---|---|---|---|
| greedy | 2.76 | 16.10 | 0.1312 | 0.0567 | 0.1423 | **0.7576** |
| temp 0.5 | **3.78** | 19.84 | **0.1426** | 0.1007 | 0.3301 | 0.3621 |
| temp 1.0 | 1.90 | **20.02** | 0.1217 | 0.2732 | 0.7758 | 0.0037 |
| temp 1.5 | 0.36 | 16.16 | 0.0631 | 0.5375 | 0.9786 | 0.0000 |

### Nepali

| Setting | BLEU | chrF++ | ROUGE-L | distinct-1 | distinct-2 | 4-gram repetition |
|---|---|---|---|---|---|---|
| greedy | 1.60 | 15.76 | 0.0947 | 0.0785 | 0.1433 | **0.7893** |
| temp 0.5 | **2.21** | 19.41 | **0.1107** | 0.1717 | 0.4149 | 0.3263 |
| temp 1.0 | 1.31 | **21.13** | 0.0962 | 0.4161 | 0.8680 | 0.0076 |
| temp 1.5 | 0.19 | 19.29 | 0.0467 | 0.5838 | 0.9813 | 0.0000 |

### Reading the table

**Greedy decoding loops.** 76% of Hindi 4-grams are repeats. Always taking the most likely
token drives the model into cycles: it produces a phrase, that phrase makes itself likely
again, and it never escapes. This is a well-known property of greedy decoding, not a
defect in this model.

**Temperature 1.5 is incoherent.** 98% distinct bigrams sounds impressive until you realise
it means almost nothing repeats — including things that *should*. BLEU collapses to 0.36.

**The metrics disagree about the best setting, and that's informative.** BLEU and ROUGE-L
prefer temperature 0.5; chrF++ prefers 1.0. That's not noise — they measure different
things. BLEU wants exact n-gram matches, so it rewards playing safe. chrF++ works at
character level and is more forgiving of a different-but-fluent continuation.

### Why these metrics are weak here, and saying so is part of the task

The assignment asks us to explain whether each metric is informative. Honestly:

**They are all fairly weak for open-ended generation**, because there are thousands of
valid continuations of any prefix and we compare against exactly one. A perfectly fluent
continuation that happens to differ from the reference scores near zero on BLEU. Our
BLEU of 2.76 says almost nothing about fluency.

**chrF++ is the least bad** of the three for Devanagari. It operates on character n-grams,
so it gives partial credit for correct morphology even when the word is different —
important for a language with rich inflection.

**The diversity diagnostics are more informative than the overlap metrics.** Repetition
rate and distinct-n directly measure a real failure mode (looping) without needing a
reference at all. They are the numbers actually worth reporting for this task.

---

## 10. STEP 7 — Looking inside the attention

`scripts/attention_analysis.py` (358 lines). We built multi-head attention ourselves, so
we should check what it learned. 49 heads per model.

### Three measurements per head

**Entropy** — how spread out the attention is. Low = focused on one place. High = smeared
across everything. We report it both raw (nats) and **normalised** by `ln(t+1)`, because
position 500 can attend to 500 places and position 5 cannot; without normalising, later
positions look diffuse purely for arithmetic reasons.

**Mean attention distance** — on average, how far back does this head look?

**Previous-token share** — how much weight lands on exactly the token before. A head with
90% here is doing one very specific job.

### The layer arc

| Layer | Hindi norm. entropy | Hindi mean distance | Nepali norm. entropy | Nepali mean distance |
|---|---|---|---|---|
| 0 | 0.911 | 105.2 | 0.920 | 102.9 |
| 1 | 0.791 | 89.6 | 0.832 | 97.6 |
| 2 | 0.501 | 23.6 | 0.531 | 43.1 |
| 3 | **0.330** | **10.2** | 0.477 | 19.8 |
| 4 | 0.489 | 64.6 | **0.398** | **27.1** |
| 5 | 0.567 | 66.3 | 0.516 | 73.3 |
| 6 | 0.708 | 86.0 | 0.690 | 84.7 |

Both models learned the same three-act structure without being told to:

```
  entropy
   0.9  ┤ ●                                    ●        broad survey
        │   ●                              ●
   0.6  ┤                              ●
        │       ●                  ●
   0.3  ┤           ●   ●                              sharp, local
        └───┴───┴───┴───┴───┴───┴───┴
          L0  L1  L2  L3  L4  L5  L6
```

1. **Layers 0–1 survey broadly.** Nearly maximum entropy, average distance ~100 tokens.
   Gathering general context.
2. **Layers 3–4 focus sharply.** This is where the model does its most specific work.
   Hindi's minimum is at layer 3, Nepali's at layer 4.
3. **Layers 5–6 broaden again.** Assembling toward the final prediction.

That both models found this independently, from different corpora, is a real result.

### The specialists

| | Model H | Model L |
|---|---|---|
| Sharpest head | **L3H3** — 0.163 nats | **L4H1** — 0.112 nats |
| Its previous-token share | 88.0% | 95.6% |
| Its mean distance | 1.37 | 1.73 |
| Longest-range head | L1H1 — distance 179.4 | L1H0 — distance 171.0 |

Model L's head L4H1 puts **95.6% of its attention on exactly the previous token**. It has
become a dedicated previous-token detector. Nobody designed this; it emerged because
knowing the immediately preceding token is enormously useful for predicting the next one.

### The head census

Classifying all 49 heads by distance and entropy:

| Type | Model H | Model L |
|---|---|---|
| diffuse | 17 | 17 |
| mixed | 16 | 14 |
| local | 15 | 16 |
| long-range | 1 | 2 |

Strikingly similar distributions from two independent runs on two different languages.

### Attention sinks

Position 0 receives far more attention than its content justifies, especially in the
diffuse early layers. This is a known transformer phenomenon: softmax rows must sum to 1,
so when a head has nothing in particular to look at, it needs somewhere to dump the
weight. Position 0 is always available. It functions as a "no-op" target.

---

## 11. Every file, explained

### The model library — `lma/`

| File | Lines | What it does |
|---|---|---|
| `model.py` | 396 | `GPT`, `Block`, `CausalSelfAttention`, `FeedForward` — the whole architecture |
| `config.py` | 238 | `ModelConfig` / `TrainConfig` dataclasses, JSON loading, validation |
| `data.py` | 208 | Reads `.bin` files via memmap; random and sequential batching |
| `checkpoint.py` | 238 | Save/load with all mandated fields plus RNG state |
| `schedule.py` | 118 | Stateless warmup + cosine decay |
| `metrics.py` | 267 | BLEU, chrF++, ROUGE-L, distinct-n, repetition |
| `attention.py` | 167 | Entropy, mean distance, previous/self share, head classification |
| `generate.py` | 148 | Greedy and temperature sampling |

### Programs you run — `scripts/`

| Script | What it does |
|---|---|
| `pack_tokens.py` | Phase 1 splits → flat uint16 `.bin` arrays |
| `pretrain.py` | The training loop |
| `eval_lm.py` | Cross-entropy, perplexity, BPB |
| `eval_generation.py` | Samples continuations, scores them |
| `attention_analysis.py` | Heatmaps and per-head statistics |
| `plot_training.py` | Loss curves, LR schedule, H-vs-L comparison |
| `compare_models.py` | Builds `report/comparison_tables.md` |

### Tests — `tests/`

`test_model.py` (8 tests) checks parameter counts against hand-derived arithmetic, weight
tying, forward shapes, that attention rows are valid causal distributions, and that
over-long sequences are rejected.

`test_causal.py` (5 tests) is entirely about the mask, including a test verifying the other
tests can actually fail.

### Configs and results

| Path | Contents |
|---|---|
| `<lang>/configs/model.json` | Architecture, with reasoning in `_`-prefixed keys |
| `<lang>/configs/train.json` | Hyperparameters, same convention |
| `report/<lang>/lm_eval.json` | Perplexity / BPB, both splits |
| `report/<lang>/generation_eval.json` | All four decoding settings |
| `report/<lang>/attention_stats.json` | Per-layer and all 49 per-head numbers |
| `report/<lang>/train_log.jsonl` | Every logged step |
| `report/<lang>/checkpoint_config.json` | Config + the parameter-count table |

---

## 12. The clever bits

### 12.1 `tokens_per_step` is fixed; batch size is derived

Most training scripts make batch size the primary setting. Ours makes *tokens per step*
primary and derives accumulation from it. The consequence: you can move to a GPU with half
the memory and the learning dynamics don't change at all. Reproducibility across hardware
comes free.

### 12.2 The schedule has no state

The learning rate is a pure function of the step number. Nothing accumulates, so nothing
can drift or be lost when resuming. Compare with PyTorch's stateful schedulers, where
forgetting to save the scheduler silently restarts the schedule.

### 12.3 RNG state lives in the checkpoint

Saving the four random streams is what makes a resumed run *bit-identical* to an
uninterrupted one, rather than merely similar. This is the difference between restarting
and resuming, and it's also what let us verify two people's runs matched exactly.

### 12.4 Attention is captured before dropout

`lma/model.py:142` grabs the attention matrix immediately after softmax, *before* dropout
is applied. If we captured after, the analysis would be looking at a randomly thinned
sample rather than the real distribution. (In eval mode dropout is off so it makes no
difference — but the code shouldn't depend on being called in the right mode.)

### 12.5 The uint16 choice is load-bearing

Storing tokens as uint16 halves the data files and, more importantly, forces you to notice
that vocabularies above 65,536 would silently corrupt. Small decisions like this are where
data bugs hide.

---

## 13. How to run all of it

### Step 1 — Pack tokens (laptop, minutes per language)

```bash
python -m scripts.pack_tokens --lang hi
python -m scripts.pack_tokens --lang ne
```

Produces `<lang>/data/tokens/{train,validation,test}.bin` plus metadata. Needs ~3 GB free.

### Step 2 — Pretrain (Kaggle T4, ~3.5 h per language)

```bash
python -m scripts.pretrain --lang hi
python -m scripts.pretrain --lang ne
```

If the session dies:

```bash
python -m scripts.pretrain --lang hi --resume
```

### Step 3 — Evaluate (Kaggle, ~11 min per language)

```bash
python -m scripts.eval_lm         --lang hi --split validation --split test
python -m scripts.eval_generation --lang hi
python -m scripts.attention_analysis --lang hi
```

### Step 4 — Plots and tables (laptop, seconds)

```bash
python -m scripts.plot_training --lang hi
python -m scripts.plot_training --lang ne
python -m scripts.plot_training --compare
python -m scripts.compare_models
```

### Step 5 — Verify

```bash
python -m pytest tests/ -v
```

13 tests, all passing. Worth running before submitting anything.

---

## 14. Things that went wrong

### The vocabulary nearly broke the parameter budget

Phase 1's original sweep considered up to 32,000 pieces. At `d_model = 448` that
embedding alone is 14,336,000 parameters — the model would have exceeded 25M before a
single transformer block existed. The candidate at 24,000 was still over budget at
27,882,176.

The fix was to derive the ceiling instead of guessing it: `7,869,824 ÷ 448 = 17,566`
affordable pieces, so 16,000 was the largest candidate that fit. Worth remembering that
Phase 1 and Phase 2 are not independent — a Phase 1 decision can make Phase 2 impossible.

### A plotting bug that only showed up locally

`scripts/plot_training.py` defaulted to looking for the training log at
`<lang>/checkpoints/train_log.jsonl` instead of `checkpoints/<lang>/train_log.jsonl`. On
Kaggle the notebook passes `--log` explicitly, so the bug was invisible there and only
appeared when running `--compare` on a laptop.

The lesson: a default path that's always overridden is a default path that's never tested.

### The training logs were gitignored

`checkpoints/` is correctly excluded from git — 300 MB files don't belong there. But
`train_log.jsonl` lives in that directory and the assignment requires logs *in the repo*.
They had to be copied out to `report/<lang>/`. Easy to miss, and it would have cost marks
under "training logs and loss curves".

### A test that validated the wrong model

`test_production_config_is_within_budget` was asserting the budget against a 384-wide,
6-layer configuration totalling 24,906,624 parameters — an architecture that was never
trained. It passed, which is the worst kind of failure: a green test guarding nothing.
It now checks the real 24,298,176.

### Stale comments outlived the code

Several docstrings in `lma/config.py` still said "32,000-piece vocabulary" long after the
decision changed to 16,000, and the dataclass defaults still held `n_head=6, d_model=384`.
The trained model was correct — the JSON configs drove everything — but anyone reading the
source would have been misled.

---

## 15. Questions people ask

**Why not use `nn.Transformer`?**
The assignment forbids it, and the reason is sound: the point of this phase is being able
to explain every tensor operation. A single opaque call defeats that. We also avoid
`F.scaled_dot_product_attention` for the same reason, even though it's faster.

**Why is the training loss higher than the test loss?**
Dropout is active during training and off during evaluation, and the training figure is a
running average that lags. See §7.2.

**Isn't 0.71 epochs too few?**
No — it's compute-optimal. 21.6 tokens per parameter, essentially exactly what scaling
laws recommend. See §7.4.

**Why do Hindi and Nepali have identical architectures?**
Because both Phase 1 vocabularies came out at 16,000, nothing forced them apart. Holding
the shape constant makes the *corpus* the only variable in the H-vs-L comparison, which is
the comparison the project is about.

**Which model is better?**
Wrong question, and knowing why is the point. Hindi wins on perplexity, Nepali on BPB, and
on the honest cross-script measure — bits per character — they're within 0.9% of each
other. See §8.

**Can the two models be compared directly at all?**
Only through tokenizer-independent measures. They have separate vocabularies, so a token
means something different in each. This is exactly why the assignment requires BPB.

**Why is BLEU so low?**
Because BLEU compares against one reference continuation when thousands are valid. A
fluent continuation that differs from the reference scores near zero. The diversity
diagnostics are more informative here. See §9.

**What would you do differently?**
Train longer. Validation loss was still falling at step 16,000 with no sign of a plateau —
the models are limited by GPU hours, not by data or architecture.

---

## 16. Where to look next

- `report/phase2.md` — the formal report, with the full analysis
- `report/comparison_tables.md` — Model H vs Model L side by side
- `lma/model.py:84` — the attention forward pass, the single most important function
- `tests/test_causal.py` — proof the model can't see the future
- `hindi/configs/model.json` — every architecture decision, with its reasoning
- `how-it-works.md` — Phase 1, if you haven't read it
