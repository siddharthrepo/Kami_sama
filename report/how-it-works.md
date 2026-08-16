# How This Works — A Complete Walkthrough

This document explains everything built in Phase 1, assuming you know nothing about
language models, web scraping, or this project. Every number in it was measured from the
files on disk, and every code reference points at a real line you can open.

---

## 0. What are we actually doing?

Later in this project (Phase 2) we will build a **robot that writes Hindi**, and a second
robot that writes Nepali. To learn, a robot like that needs to read an enormous amount of
text — hundreds of millions of words.

**Phase 1 does not build any robot.** Phase 1 only prepares the reading material.

Think of it like cooking. Phase 2 is the meal. Phase 1 is going to the market, washing the
vegetables, chopping them, and laying them out. That's all we did — but for 2.2 million
documents.

**What "done" looks like:** two big piles of clean text (one Hindi, one Nepali) plus two
"chopping machines" that cut text into pieces a robot can swallow.

Here is what we ended up with:

| | Hindi | Nepali |
|---|---|---|
| Documents collected | 947,298 | 1,278,483 |
| Tokens (see §1) | 662,356,265 | 659,293,660 |
| Text we collected ourselves | 27.94% | 25.56% |
| Chopping machine ("tokenizer") | 32,000 pieces | 32,000 pieces |

---

## 1. Words you need to know

Read this section once and the rest of the document will make sense.

**Document**
One article. One news story, or one Wikipedia page. In this project a document is a blob
of text plus some labels about where it came from.

**Corpus**
A big collection of documents. We built two: a Hindi corpus and a Nepali corpus.

**Token**
A piece of a word. Robots cannot read whole words — they read numbered pieces. The Hindi
word `भारत` is one token. The rarer word `मौलिक` is also one token, but a made-up word
would be broken into several. "Token" is just the unit a robot counts in.

**Tokenizer**
The machine that cuts text into tokens and gives each one a number.

```
"भारत में शिक्षा"  →  ['▁भारत', '▁में', '▁शिक्षा']  →  [582, 277, 1601]
```

That `▁` symbol marks where a space was, so the text can be rebuilt exactly.

**Vocabulary**
The complete list of pieces a tokenizer knows. Ours has 32,000 entries per language. It is
literally a list — you can open `hindi/tokenizer/hi.vocab` in a text editor and read it.

**Fertility**
How many tokens it takes to write one word, on average. Lower is better (fewer pieces =
more efficient). Hindi came out at **1.237**, Nepali at **1.300**.

**Shard**
One file holding a chunk of documents. Instead of one gigantic file we write many smaller
ones. If a program crashes, we lose one shard instead of everything.

**Scraping**
Writing a program that visits websites and copies the text out, automatically.

**Train / validation / test**
Three piles of the same corpus. **Train** is what the robot studies (98%). **Validation**
is a small pile used to check progress (1%). **Test** is the final exam, kept sealed (1%).
Splitting them prevents cheating — you can't test someone on questions they studied.

**Deduplication ("dedup")**
Throwing away copies. The same news article often appears in several places; keeping all
copies would make the robot over-learn that one article.

---

## 2. The big picture

```
        ┌──────────────────────┐         ┌───────────────────────────┐
STEP 1  │  scrape websites     │  STEP 2 │  download public datasets │
        │  (we collect it)     │         │  (someone else collected) │
        └──────────┬───────────┘         └────────────┬──────────────┘
                   │                                  │
                   └───────────────┬──────────────────┘
                                   ▼
                          ┌──────────────────┐
                   STEP 3 │      CLEAN       │  fix letters, remove wrong
                          │                  │  language, remove copies
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                   STEP 4 │   SPLIT 98/1/1   │  study / check / exam
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                   STEP 5 │ BUILD TOKENIZER  │  learn how to chop words
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                   STEP 6 │  COUNT + DRAW    │  how many tokens? make charts
                          └──────────────────┘
```

Which script does what, and how long it actually took:

| Step | Script | Time taken | Output size |
|---|---|---|---|
| 1. Scrape | `scripts/scrape.py` | ~2 hours | 726 MB (343 shards) |
| 2. Download | `scripts/download.py` | ~35 min | 2.1 GB (85 shards) |
| 3. Clean | `scripts/clean.py` | ~30 min | 2.8 GB (90 shards) |
| 4. Split | `scripts/split.py` | ~10 min | 2.8 GB |
| 5. Tokenizer | `scripts/train_tokenizer.py` | ~10 min | 3.3 MB |
| 6. Count | `scripts/corpus_stats.py` | ~30 min | 8 charts + JSON |

---

## 3. STEP 1 — Collecting text ourselves

### The problem this solves

The assignment has a rule: **at least 20% of the text must be collected by us**, not
downloaded ready-made. The point is to learn what real, messy text collection is like.

But websites don't hand you their articles. You face three obstacles:

1. You don't know the addresses of the articles.
2. Each page is mostly junk — menus, adverts, "related stories", footers. The article is a
   small part of it.
3. There are hundreds of thousands of them, so clicking is not an option.

### How we solved it

**First attempt: sitemaps.** Most websites publish a file listing all their page
addresses, meant for Google. Ask for it once, get thousands of addresses.

This mostly **failed**, and the failures are worth knowing:

- **Amar Ujala** — its old articles are paywalled stubs. The page says "watch a video
  advertisement to keep reading" and the real text never appears.
- **Navbharat Times** — its sitemap lists only section pages ("Sports", "Business"), not
  articles. We tested 40 pages; all 40 were correctly rejected as having no article text.
- **Setopati** — its sitemap lists only categories, tags and author pages.

**Second attempt: the WordPress API.** This worked, and it is enormously better.

Many news sites run on WordPress, which exposes a machine-readable door at
`/wp-json/wp/v2/posts`. Ask it a question and it returns **100 complete articles as
structured data**, with the body already separated from the page furniture.

```
Sitemap way:      1 request  →  1 messy web page  →  guess where the article is
WordPress way:    1 request  →  100 clean articles, body already separated
```

For 150,000 articles that is **1,500 requests instead of 150,000** — a hundred times
fewer. It is also far gentler on the website, which matters when you're taking a lot.

### The code

**`common/sources.py`** — the address book. A list of which sites to visit.

```python
Source(
    name="jansatta",
    lang="hi",
    home="https://www.jansatta.com",
    api="wordpress",              # ← use the fast door
    archive_start="2014-01-01",
)
```

The `Source` class is at `common/sources.py:40`.

**`common/fetch.py`** — the polite downloader (`PoliteFetcher`, line 35). It does three
things so we don't get banned or behave badly:

- reads each site's `robots.txt` and obeys it
- waits a fixed gap between requests to the same site (0.25 seconds)
- retries when a request fails, then gives up rather than hammering

**`common/wordpress.py`** — talks to the WordPress door. The important idea is at
`month_windows()` (line 169): instead of walking the archive from newest to oldest in one
long chain, we chop the archive into **one window per calendar month** and fetch eight
windows at the same time.

Why? Because a chain is slow by nature — you can't ask for the next page until the current
one answers. Splitting into months breaks the chain into eight independent chains.

```
worker 1:  2011-01 ──► 2011-09 ──► 2012-05 ...
worker 2:  2011-02 ──► 2011-10 ──► 2012-06 ...      all 8 at once
   ...
worker 8:  2011-08 ──► 2012-04 ...
```

**`scripts/scrape.py`** — the boss that runs everything. `extract_document()` (line 172)
takes a raw web page and returns clean text, or nothing if the page isn't good enough.

### What comes out

One file per month, written into `data/manual/hi/`:

```
shard-jansatta-2026-08.jsonl.zst
shard-jansatta-2026-07.jsonl.zst
...
```

Inside, one document per line. Here is a real one (shortened):

```json
{
  "text": "बिहार की सियासत में कुछ नाम वक्त गुजर जाने के बाद भी जहन में...",
  "lang": "hi",
  "source_type": "manual",
  "source": "jansatta",
  "url": "https://www.jansatta.com/jansatta-special/assembly-election-...",
  "title": "मुख्यमंत्री: एक मेडल, एक बगावत और एक नया नेता...",
  "fetched_at": 1786869364.1986604,
  "doc_id": "b116f576aaef19cf"
}
```

The `.zst` ending means the file is compressed, about 4× smaller than plain text.

**Real results:**

| Site | Documents | Shards |
|---|---|---|
| Jansatta (Hindi) | 266,296 | 186 files total for Hindi |
| The Wire Hindi | 37,480 | |
| Onlinekhabar (Nepali) | 354,793 | 157 files |

345 MB for Hindi, 381 MB for Nepali.

### The tricky bit — how we survive a crash

Scraping took hours. Computers get closed, connections drop. So each shard is written to a
**temporary name** first, and only **renamed** to its final name when the month is
completely finished.

```
shard-jansatta-2015-03.jsonl.zst.tmp    ← still being written
        ↓ rename, only on success
shard-jansatta-2015-03.jsonl.zst        ← finished, guaranteed complete
```

On Linux, renaming is *instantaneous and cannot half-happen*. So a file with the final
name is always a complete file. If the program dies, a `.tmp` file is left behind and the
next run deletes it and redoes that month.

That means **the existence of the file is the record that the month is done.** No separate
"progress" file to get out of sync. This is `AtomicJsonlWriter` at `common/schema.py:174`.

We used it: the Hindi scrape ran in two sessions and the second one skipped 37 already-done
months automatically.

### One source we deliberately did not use

**eKantipur**, Nepal's biggest newspaper, was excluded. Its `robots.txt` contains a section
about "content signals" whose entire purpose is to say *don't use this to train AI*. No
formal rule was broken by reading it, but the intent is obvious, so we left it alone. It is
recorded in `common/sources.py` with the reason written in a comment.

---

## 4. STEP 2 — Getting text other people collected

### The problem this solves

We need ~500 million tokens per language. Scraping alone would take weeks. The other ~75%
comes from public research datasets.

### How we solved it

**HuggingFace** is a website where researchers publish enormous text collections for free.
We used two:

| Dataset | What it is | Which part |
|---|---|---|
| **FineWeb-2** | a cleaned, filtered crawl of the whole web | `hin_Deva` (Hindi), `npi_Deva` (Nepali) |
| **Wikipedia** | the encyclopedia | `20231101.hi`, `20231101.ne` |

Those subset names decode as: `hin` = Hindi, `Deva` = Devanagari script, `20231101` = the
1 November 2023 snapshot.

The critical technique is **streaming**. FineWeb-2's Hindi part is 34.4 GB — far more than
we need and more than fits comfortably on the laptop. Streaming means we pull one document
at a time, decide immediately whether to keep it, and throw away the rest. Nothing large is
ever stored.

```
HuggingFace  ──one document at a time──►  keep it?  ──yes──►  write to shard
                                              │
                                              └──no───►  discard, never stored
```

That is why 34 GB of source data produced only 930 MB on disk.

### The code

**`common/hf_sources.py`** — the list of datasets (`HFSource`, line 23).

**`scripts/download.py`** — the downloader.

- `open_stream()` (line 69) opens the tap
- `collect_source()` (line 128) reads documents one at a time, filters them, and writes
  shards of 20,000 documents (`DOCS_PER_SHARD`, line 54)
- `save_progress()` (line 64) records how many documents we've consumed, so a resumed run
  skips them rather than starting over

Progress is stored in a small file, `data/downloaded/hi/download-progress.json`:

```json
{
  "wikipedia-hi": {"consumed": 163093, "kept": 44170, "words": 37755661},
  "fineweb2-hi":  {"consumed": 505184, "kept": 413426, "words": 240022639}
}
```

We used this three times — every time we needed more data, the same command picked up
exactly where it stopped.

### What comes out

Identical format to Step 1. **The only difference is one field:**

```json
{"text": "...", "source_type": "downloaded", "source": "fineweb2-hi"}
                               ^^^^^^^^^^^^
```

That single word is what later proves the 20% requirement. More on this in §10.

**Real results:** 930 MB Hindi (35 shards), 1.2 GB Nepali (50 shards).

### A dataset that no longer works

**CC-100** is the corpus most people cite for this kind of work. It cannot be used any
more: HuggingFace used to run a small program to unpack it, and that mechanism was removed
for security reasons. Trying to load it fails with
`Dataset scripts are no longer supported`. FineWeb-2 replaces it and is better anyway —
bigger and already deduplicated.

---

## 5. STEP 3 — Cleaning

### The problem this solves

The two piles are now merged, and they are dirty in four different ways:

1. **The same letter typed differently.** Devanagari lets you write क़ either as one
   character or as क plus a dot. They look identical but are different data. The robot
   would waste effort learning both.
2. **Wrong language.** Hindi and Nepali use the *same alphabet*. Public datasets sort text
   by alphabet, so Hindi leaks into the Nepali pile.
3. **Copies.** The same article appears in several sources.
4. **Leftover junk** that survived extraction.

### How we solved it — five filters in a fixed order

```
document
   │
   ├─1─► fix the letters            (normalise)
   ├─2─► remove English-only lines  (leftover adverts)
   ├─3─► check the language         (is this really Nepali?)
   ├─4─► is it long enough?         (≥120 words of real prose)
   └─5─► have we seen it before?    (two kinds of duplicate check)
   │
   ▼
keep it
```

**The order is not arbitrary.** Filter 2 removes text, which can push a document below the
length limit in filter 4 — so filter 4 must come after. And filter 5 hashes the text, so it
must come after everything that changes the text.

### The code

The whole thing lives in **`scripts/clean.py`**, in `process_batch()` (line 133). Each
filter delegates to a small module:

| Filter | Called at | Lives in |
|---|---|---|
| 1. Fix letters | `clean.py:144` | `common/normalize.py` → `normalize_text()` (line 68) |
| 2. English lines | `clean.py:145` | `common/clean.py` → `drop_foreign_lines()` (line 97) |
| 3. Language check | `clean.py:150` | `common/langid.py` → `predict_batch()` (line 99) |
| 4. Length gate | `clean.py:159` | `common/clean.py` → `is_good_document()` (line 169) |
| 5. Duplicates | `clean.py:168,174` | `scripts/clean.py` → `text_hash()`, `prefix_hash()` |

#### Filter 1 — fixing the letters

`normalize_text()` applies "NFC" normalisation, which forces one standard spelling for
characters that have two encodings. It also deletes invisible characters (yes, text
contains invisible characters — they control how letters join) and standardises curly
quotes into straight ones.

It **deliberately keeps** the danda `।`, which is the Devanagari full stop, and the
Devanagari digits ०–९, because Nepali writes dates with them and changing them would
change the language.

#### Filter 3 — the language check, and why it's the interesting one

We used **fastText**, a small language-detector made by Facebook, at
`common/langid.py:58`. Give it text, it says "this is Nepali, 98% sure".

We did not simply trust it. We **tested it against known answers**: our scraped Nepali came
from a Nepali newspaper, so it is definitely Nepali.

| Test | Correctly accepted |
|---|---|
| 3,000 Hindi documents we scraped | **100.0%** |
| 3,000 Nepali documents we scraped | **99.9%** |

Then we ran it on everything. **The result is the most interesting number in this project:**

| | Documents thrown out as wrong-language |
|---|---|
| Hindi corpus | **31** |
| Nepali corpus | **7,832** |

**252 times more.** The Nepali pile was full of Hindi. This is the price of the two
languages sharing an alphabet — and no simple letter-checking rule could ever have caught
it, because the letters are identical.

**Speed trick:** the language check is the only slow part, so it is called **once per 2,000
documents** instead of once per document (`BATCH_SIZE`, `scripts/clean.py:45`). Look at
`process_batch()` — the `predict_batch()` call sits *outside* the per-document loop. That
single decision is most of the reason cleaning ran at 585–720 documents per second.

#### Filter 5 — two kinds of duplicate

**Exact duplicate:** we compute a "fingerprint" (a SHA-1 hash) of the whole document. Same
fingerprint = same document.

**Near duplicate:** the same article republished elsewhere often has different boilerplate
but the same opening paragraph. So we take a second fingerprint of just the **first 300
characters** with spaces removed (`PREFIX_CHARS`, `scripts/clean.py:50`).

This is a simplification. A proper method called MinHash would catch more, but it is far
slower, and FineWeb-2 already removed its own duplicates, so our main target is overlap
between our scraped pile and the downloaded pile. This limitation is written down in
`report/phase1.md` §8 rather than hidden.

### What comes out

| | Hindi | Nepali |
|---|---|---|
| Documents in | 960,600 | 1,327,685 |
| Failed length gate | 7,159 | 1,983 |
| Wrong language | 31 | 7,832 |
| Exact duplicates | 2,172 | 23,087 |
| Near duplicates | 3,940 | 16,300 |
| **Kept** | **947,298 (98.6%)** | **1,278,483 (96.3%)** |
| English lines removed | 421,407 | 62,683 |
| Speed | 585 docs/sec | 720 docs/sec |

### The tricky bit — English words are kept, English lines are removed

This confuses people, so here it is explicitly.

Real Hindi journalism contains English. `नया iPhone लॉन्च हुआ` is a perfectly normal Hindi
sentence. We measured it: **61.9% of Hindi documents contain some English**, against only
**2.9% of Nepali documents** (Nepali writes foreign names in Devanagari instead).

If we deleted English words, that sentence would become `नया लॉन्च हुआ` — broken, and no
longer representative of how Hindi is actually written.

But *whole lines* of English inside a Hindi article are not language at all — they are
leftover advert widgets:

```
Best Affordable 108MP Camera Smartphones
KODAK 32 inches Special Edition Series HD Ready Smart LED TV 32SE5001BL
Karizma XMR vs Yamaha R15 V4 comparison
```

So the rule is: **drop a line if more than 80% of its letters are English and it has at
least four words.** Individual English words inside Hindi sentences survive. The four-word
minimum protects short mentions like a product name.

Measured cost: 0.076% of our scraped Hindi words, 1.44% of downloaded Hindi words. The
downloaded data loses 19× more, which tells you our own collection was cleaner.

---

## 6. STEP 4 — Splitting into three piles

### The problem this solves

If you test a student on the exact questions they revised, the score is meaningless. Same
for a robot. So part of the text must be locked away and never studied.

### How we solved it

98% train, 1% validation, 1% test.

**Why so little held back?** Every document in the test pile is one the robot never learns
from — held-out data is a pure cost. 1% is about 6.5 million tokens, which is already far
more than enough to measure performance reliably. Holding back 10% (as is common in other
kinds of machine learning) would waste ~58 million training tokens for no extra accuracy.

**How documents are assigned.** Not by shuffling — the corpus is too big to hold in memory.
Instead each document's fingerprint is turned into a number between 0 and 1:

```
under 0.98  → train
0.98–0.99   → validation
over 0.99   → test
```

This is `assign_split()` at `scripts/split.py:46`. Because it's based on the document's own
fingerprint plus a fixed seed (1337), **the same document always lands in the same pile**,
every time you run it, on any machine. Reproducible without storing anything.

We verified it on 200,000 fake ids: 98.005% / 0.984% / 1.011%. Correct.

### The tricky bit — splitting whole documents, never sentences

If half an article went to train and half to test, the robot would be tested on text whose
context it had already memorised. Every later measurement would be flattering and wrong.

So splitting happens at the **document** level. One article goes entirely into one pile.

### What comes out

| | Hindi | | Nepali | |
|---|---|---|---|---|
| Pile | Documents | Tokens | Documents | Tokens |
| train | 928,487 | 649,135,502 | 1,252,767 | 645,993,506 |
| validation | 9,300 | 6,547,980 | 12,962 | 6,742,672 |
| test | 9,511 | 6,672,783 | 12,754 | 6,557,482 |

---

## 7. STEP 5 — Building the tokenizer

### The problem this solves

A robot cannot read letters or words — it reads numbers. Something must decide how to cut
text into pieces and number them. That decision matters: cut too finely and sentences
become enormously long; cut too coarsely and you need a vocabulary of millions.

### How BPE works — a worked example

The method is called **Byte Pair Encoding**. It is simple enough to explain completely.

Start with individual characters. Then repeat one rule 32,000 times:

> *Find the two neighbouring pieces that appear together most often, and glue them into
> one new piece.*

```
Start:      भ  ा  र  त        (four separate characters)

Round 1:    "भ"+"ा" is everywhere in Hindi  →  glue  →  भा  र  त
Round 2:    "भा"+"र" is very common         →  glue  →  भार  त
Round 3:    "भार"+"त" is very common        →  glue  →  भारत
...
Round 32000: stop.
```

Nobody tells it which words matter. It just counts. Words that appear constantly get glued
into single pieces early; rare words stay in fragments.

You can see the result directly:

```
'भारत'    → ['▁भारत']     id 582     ← extremely common, glued early
'क्रिकेट'  → ['▁क्रिकेट']   id 1570    ← common
'मौलिक'   → ['▁मौलिक']    id 9584    ← rarer, glued later
'iPhone'  → ['▁iPhone']   id 9208    ← English, but common enough in Hindi news
```

**Low id = glued early = common.** The numbering is a frequency ranking.

### The code

**`scripts/train_tokenizer.py`**:

- `build_training_sample()` (line 48) — pulls 2,000,000 random lines from the **train pile
  only** into a plain text file
- `train_one()` (line 96) — runs BPE at one vocabulary size
- `evaluate()` (line 124) — tests the result on the **validation pile**
- `run()` (line 185) — does all four sizes, compares, installs the winner

**Why sample instead of using everything?** SentencePiece (the library) must hold its
training text in memory. 649 million tokens will not fit. Two million lines is far past the
point where the statistics stop changing.

**Why train pile only?** If the tokenizer saw validation text, it would build pieces
tailored to the very text used to judge it. That's marking your own homework.

### Choosing the size — we tested four

The assignment requires choosing the size using evidence, not habit. So we trained 8,000 /
16,000 / 32,000 / 48,000 and measured each on held-out text.

**Hindi**

| Vocabulary | Fertility (lower=better) | Vocabulary actually used |
|---|---|---|
| 8,000 | 1.388 | 96.5% |
| 16,000 | 1.272 | 95.8% |
| **32,000** | **1.205** | **89.0%** |
| 48,000 | 1.181 | 78.6% |

**Nepali**

| Vocabulary | Fertility | Vocabulary actually used |
|---|---|---|
| 8,000 | 1.552 | 97.6% |
| 16,000 | 1.385 | 98.0% |
| **32,000** | **1.272** | **96.2%** |
| 48,000 | 1.226 | 91.5% |

Both chose **32,000**, but for slightly different reasons.

For **Hindi**, 48,000 is barely better (1.181 vs 1.205) and leaves 21% of the vocabulary
unused — clearly wasteful.

For **Nepali**, 48,000 genuinely was better and still 91.5% used. It was rejected for a
different reason: **the robot has a fixed size budget.** In Phase 2 the model may have
about 25 million parameters, and the vocabulary lookup table costs
`vocabulary size × 384`:

```
32,000 vocabulary  →  12.3M for the lookup table,  12.7M left for the actual model
48,000 vocabulary  →  18.4M for the lookup table,   6.6M left for the actual model
```

Paying a quarter of the entire robot for a 3.6% better chopping machine is a bad trade. So
the selection code (`train_tokenizer.py:240`) first discards sizes above a cap, *then*
picks the best of what remains.

### What comes out

```
hindi/tokenizer/hi.model     790,782 bytes   ← the machine
hindi/tokenizer/hi.vocab     745,703 bytes   ← the readable list of 32,000 pieces
nepali/tokenizer/ne.model    907,119 bytes
nepali/tokenizer/ne.vocab    862,038 bytes
```

### The tricky bit — the two tokenizers are genuinely separate

The assignment requires the two languages to share nothing. Here is the proof. The word
`अधिकार` ("right") exists in both languages:

```
Hindi tokenizer  → 1127
Nepali tokenizer → 1472
```

Different numbers, because each vocabulary was built independently from its own corpus.
They are not translations of each other and cannot be mixed.

Also note `नेपालमा` ("in Nepal") is a **single** Nepali token — the `-मा` ending is glued
on. Nepali packs more meaning into each word, which is exactly why it needs more tokens per
word overall (1.300 vs 1.237).

---

## 8. STEP 6 — Counting and drawing

### The problem this solves

Up to this point every token count was a **guess** — we multiplied word counts by an
assumed number. Now the tokenizer exists, so we can finally count for real.

This matters more than it sounds. Our original guess was 1.8 tokens per word. The true
value is 1.237. Every earlier figure was **45% too high**. See §12.

### How we solved it

Run every single document through the tokenizer and count the pieces. Crucially, put each
document's tokens into the right bucket based on the label it has carried since Step 1:

```
document (tagged "manual", "jansatta")
   ↓ tokenizer
   637 tokens
   ↓
manual bucket      += 637
jansatta bucket    += 637
```

The tokenizer never touches the labels, so provenance survives.

### The code

**`scripts/corpus_stats.py`**:

- `measure_split()` (line 58) — encodes every document, counting by source
- `plot_source_breakdown()` (line 132), `plot_zipf()` (line 156),
  `plot_length_histogram()` (line 177), `plot_fertility_vs_vocab()` (line 194) — the four
  charts

Note that the drawing functions are entirely separate from the counting function. That's a
requirement of the assignment ("keep visualization and computation logic in separate
functions") and it also means you can redraw charts without recounting.

### What comes out — the final numbers

| | Hindi | Nepali |
|---|---|---|
| Documents | 947,298 | 1,278,483 |
| Words | 535,501,269 | 506,993,066 |
| **Tokens** | **662,356,265** | **659,293,660** |
| **Manual tokens** | **185,075,090 (27.94%)** | **168,487,902 (25.56%)** |
| Target ~500M | ✅ +32% | ✅ +32% |
| Requirement ≥20% manual | ✅ | ✅ |
| Fertility | 1.237 | 1.300 |
| Characters per token | 4.10 | 4.96 |
| Unknown tokens | 0.000% | 0.000% |
| Vocabulary used | 99.6% | 99.7% |

**Zero unknown tokens** is not luck — the tokenizer has "byte fallback" switched on, so
anything it doesn't recognise is stored as raw bytes rather than being lost. Nothing is
ever unrepresentable.

**99.6% vocabulary used** means almost every one of the 32,000 pieces earns its place. Only
one token in the entire Hindi corpus appeared exactly once.

Plus eight charts in `report/hindi/figures/` and `report/nepali/figures/`.

---

## 9. Every file, explained

### Shared library — `common/`

| File | What it is |
|---|---|
| `schema.py` | Defines what a document is, and reads/writes compressed shard files. Contains `AtomicJsonlWriter`, the crash-safety mechanism. |
| `fetch.py` | Polite web downloader — obeys `robots.txt`, waits between requests, retries failures. |
| `sources.py` | The address book of websites to scrape, plus why each was chosen. |
| `wordpress.py` | Talks to the WordPress API and splits archives into monthly windows. |
| `hf_sources.py` | The list of HuggingFace datasets to download. |
| `normalize.py` | Fixes Devanagari letters into one standard form. |
| `clean.py` | Quality checks — length, script, English-line removal. |
| `langid.py` | Wraps the fastText language detector. |

### Programs you run — `scripts/`

| File | What it does |
|---|---|
| `scrape.py` | Step 1 — collect text from websites |
| `download.py` | Step 2 — stream text from HuggingFace |
| `clean.py` | Step 3 — merge, fix, filter, deduplicate |
| `split.py` | Step 4 — divide into train/validation/test |
| `train_tokenizer.py` | Step 5 — build the tokenizer |
| `corpus_stats.py` | Step 6 — measure and draw |

### Settings and results

| File | What it holds |
|---|---|
| `hindi/configs/dataset.json` | Every setting for Hindi, with reasons written in |
| `nepali/configs/dataset.json` | Same for Nepali |
| `report/phase1.md` | The formal report |
| `report/*/**.json` | Machine-readable statistics from each stage |
| `report/*/figures/*.png` | The eight charts |

---

## 10. The four clever bits

These are the decisions that are easy to miss and expensive to get wrong.

### 10.1 Manual data is read first — and that protects the 20%

The rule says at least 20% of tokens must be ours. Here's the trap:

FineWeb-2 crawled the web. We scraped news sites. **The same article is often in both.**
When deduplication finds two copies it keeps one — and if it kept the downloaded copy, our
manual share would shrink every time, silently, with no error message.

The fix is one line of ordering (`scripts/clean.py:303`): the manual folder is read
**before** the downloaded folder. Deduplication keeps whichever it saw first. So our copy
always wins.

### 10.2 Renaming files is how we survive crashes

Explained in §3. A file under its final name is always complete, because renaming on Linux
cannot half-happen. So the file's *existence* is the record that its work finished — no
separate progress file that could disagree with reality.

This is why every long job could be stopped and restarted freely, which we did many times.

### 10.3 A document is never split across piles

Explained in §6. Whole documents only, so the exam stays sealed.

### 10.4 The provenance label is stamped at birth

Every document has carried `source_type` since the moment it was first written, at
`common/schema.py:32`.

This cannot be added later. Look at a cleaned Hindi paragraph — there is nothing in the
text that says whether it came from our scraper or from FineWeb-2. If we hadn't recorded it
at collection time, the 20% requirement would be unprovable.

That's why the very first thing built in this project was the document format, before a
single line of collection code.

---

## 11. How to run all of it

### Setup, from nothing

```bash
git clone <repository>
cd vidhi_project

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Requires Python 3.12 and about 8 GB of free disk. No GPU. Everything below was run on a
12-core laptop.

### Step 1 — Collect text ourselves (~2 hours)

```bash
mkdir -p data/manual/hi data/manual/ne logs

nohup .venv/bin/python -m scripts.scrape \
  --lang hi --out data/manual/hi \
  --sources jansatta thewirehindi \
  --limit 300000 --workers 8 --delay 0.25 \
  > logs/hi-scrape.log 2>&1 &

nohup .venv/bin/python -m scripts.scrape \
  --lang ne --out data/manual/ne \
  --sources onlinekhabar \
  --limit 400000 --workers 8 --delay 0.25 \
  > logs/ne-scrape.log 2>&1 &
```

- `nohup ... &` runs it in the background so closing the terminal doesn't kill it
- `--workers 8` fetches eight months at once
- `--delay 0.25` waits a quarter-second between requests to one site, so we stay polite
- `--limit` is approximate: workers finish their current month before stopping

Watch it: `tail -f logs/hi-scrape.log`
**Expect:** ~726 MB, 343 files. **If it dies, run the identical command** — finished months
are skipped.

### Step 2 — Download public text (~35 minutes)

First check the sources are reachable (downloads almost nothing):

```bash
.venv/bin/python -m scripts.download --lang hi --out data/downloaded/hi --probe
```

Then:

```bash
nohup .venv/bin/python -m scripts.download \
  --lang hi --out data/downloaded/hi --max-tokens 450000000 \
  > logs/hi-download.log 2>&1 &

nohup .venv/bin/python -m scripts.download \
  --lang ne --out data/downloaded/ne --max-tokens 450000000 \
  > logs/ne-download.log 2>&1 &
```

**Expect:** ~2.1 GB. Also resumable — re-running continues from the recorded position.

### Step 3 — Clean (~30 minutes)

⚠️ **If you have run this before, delete the old output first.** Otherwise new shards are
added alongside the old ones and every document appears twice:

```bash
rm -rf hindi/data/clean nepali/data/clean
```

```bash
nohup .venv/bin/python -m scripts.clean --config hindi/configs/dataset.json  > logs/hi-clean.log 2>&1 &
nohup .venv/bin/python -m scripts.clean --config nepali/configs/dataset.json > logs/ne-clean.log 2>&1 &
```

On first run this downloads the fastText language model (~1 MB) automatically.

### Step 4 — Split (~10 minutes)

```bash
rm -rf hindi/data/splits nepali/data/splits    # if re-running

.venv/bin/python -m scripts.split --config hindi/configs/dataset.json
.venv/bin/python -m scripts.split --config nepali/configs/dataset.json
```

### Step 5 — Build the tokenizers (~10 minutes)

```bash
.venv/bin/python -m scripts.train_tokenizer \
  --config hindi/configs/dataset.json \
  --vocab-sizes 8000 16000 32000 48000 --sample-lines 2000000

.venv/bin/python -m scripts.train_tokenizer \
  --config nepali/configs/dataset.json \
  --vocab-sizes 8000 16000 32000 48000 --sample-lines 2000000
```

Run these **one after another, not at the same time** — each holds its training text in
memory and two together can exhaust 15 GB of RAM.

SentencePiece prints thousands of lines. The useful ones:

```bash
grep -E "fertility=|selected" logs/hi-tok.log
```

### Step 6 — Measure and draw (~30 minutes)

```bash
.venv/bin/python -m scripts.corpus_stats --config hindi/configs/dataset.json
.venv/bin/python -m scripts.corpus_stats --config nepali/configs/dataset.json
```

Prints the final verdict:

```
=== Hindi final corpus ===
  documents        : 947,298
  TOKENS           : 662,356,265 (662.4M)  OK
  manual tokens    : 185,075,090 (27.94%)  OK
```

### Checking on things at any time

```bash
ps -eo pid,etime,cmd --no-headers | grep "[s]cripts\."   # what's running
tail -f logs/hi-clean.log                                 # follow a job
du -sh data/ hindi/data nepali/data                       # disk used
df -h /                                                   # disk free
```

---

## 12. Things that went wrong

Recorded honestly, because they explain why the project looks the way it does.

### The token estimate was wrong by 45%

For most of the project we estimated tokens as `words × 1.8`. When the first real tokenizer
was trained, the true value turned out to be **1.237**.

Everything we had believed was inflated. What looked like 658M tokens was really 453M —
**below** the 500M requirement, not comfortably above it.

**How it was caught:** by training a trial tokenizer partway through rather than waiting
until the end.
**How it was fixed:** collected more data, corrected the constant to a deliberately
pessimistic 1.15 (`scripts/download.py:50`), and made the final count a real measurement.

**Lesson:** an estimate you never check is a number you don't have.

### Three websites didn't work the way we expected

Amar Ujala (paywalled stubs), Navbharat Times (sitemap with no articles), Setopati (sitemap
with no articles). All discovered by testing on 40–50 pages before committing, which cost
minutes instead of hours. This led to the WordPress API approach, which was 100× more
efficient.

### The Nepali manual share nearly dropped below 20%

After adding more downloaded data, Nepali's manual share fell to **20.59%** — barely above
the requirement. Adding public data *dilutes* the manual fraction, which is obvious in
hindsight and wasn't planned for.

Fixed by scraping the rest of the Onlinekhabar archive, which took it to **25.56%**.

### The tokenizer picked a size that would break Phase 2

The selection code optimised only for chopping quality, and chose 48,000 for Nepali. Correct
by its own rule — but it knew nothing about the Phase 2 parameter budget, where a 48,000
vocabulary would consume three-quarters of the model.

Fixed by adding a size cap before the quality comparison (`train_tokenizer.py:240`). All
four sizes were already trained and saved, so nothing had to be redone.

### One thing we would do differently

`corpus_stats.py` runs on one CPU core when it could easily use twelve — each shard could
be counted independently. It cost about 30 minutes where 4 would have done. It's noted in
`report/phase1.md` §8 and should be fixed before Phase 2, where tokenizing runs repeatedly.

---

## 13. Questions people ask

**"Why not count tokens while collecting, instead of at the end?"**
Because a token only exists once a tokenizer exists, and the tokenizer must be trained on
the finished corpus. Counting earlier means counting with *someone else's* tokenizer, which
answers a different question. (A common mistake is to count with a downloaded multilingual
tokenizer — this both breaks the assignment's rules and gives a number that won't match your
own tokenizer, typically by 30–40%.)

**"Why is there English in a Hindi vocabulary?"**
Because there is English in Hindi text. We measured it: 61.9% of Hindi documents contain
some. The Hindi vocabulary spends 3,141 of its 32,000 slots (9.8%) on English fragments;
Nepali spends only 383 (1.2%). That difference is a real property of how the two languages
are written, not a bug.

**"Why is the test pile only 1%?"**
Because 1% is ~6.5 million tokens, already far more than enough to measure reliably, and
every held-out document is one the robot never learns from. The 80/10/10 split people
remember from other machine learning is for small labelled datasets.

**"Did we train a model?"**
**No.** Phase 1 has no neural network at all. The tokenizer is a counting algorithm, not a
model. The robot is built in Phase 2.

**"Do we need Kaggle or a GPU?"**
Not for Phase 1 — all of it runs on a normal laptop CPU. Phase 2 needs a GPU because
training a neural network is a different kind of work.

**"What are fastText and trafilatura, and are they allowed?"**
fastText detects which language a document is in; trafilatura finds the article inside a
web page. The assignment forbids pretrained *language models* and pretrained *tokenizers*.
Neither of these is either — they are cleaning tools, like the compression library. Both are
declared openly in the README and the report rather than left to be discovered.

---

## 14. Where to look next

| You want | Read |
|---|---|
| The formal Phase 1 report | `report/phase1.md` |
| Exact numbers, machine-readable | `report/hindi/corpus_stats.json` |
| Every setting, with reasons | `hindi/configs/dataset.json` |
| Commands and Drive links | `README.md` |
| The charts | `report/*/figures/` |
