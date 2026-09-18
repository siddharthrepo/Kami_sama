"""Language packs and example generators for the Phase 3 synthetic reasoning data.

The assignment requires a finetuning corpus that is **built programmatically**, in each
model's own language, covering comparative and transitive reasoning. Generating it
ourselves rather than downloading a benchmark is what makes the ground-truth labels
trustworthy: every answer here is computed from the same numbers that produced the
question, so a label cannot silently disagree with its prompt.

Three design decisions carry most of the weight.

**Every question string is written out per language, not assembled from parts.** It is
tempting to store a noun ("कीमत") and build questions around it, but Hindi postpositions
and interrogatives agree with the noun's gender — ``किसकी कीमत`` but ``किसका वज़न`` — and
a generic builder gets that wrong. Storing the finished sentence per attribute costs a
few more strings and removes a whole class of ungrammatical output.

**Statement order is shuffled.** If the premises always appeared in chain order, "the
first name mentioned" would be a perfect shortcut for the superlative questions and the
model could score well without doing any comparison at all.

**Entity pools are disjoint across splits.** Names used in test never appear in train,
so a model cannot succeed by memorising that a particular name tends to be the tallest.
This is the primary leakage control; exact-prompt deduplication is the secondary one.

Numerals follow Phase 1's normalisation, which deliberately preserves Devanagari digits
(see ``common/normalize.py``). Nepali text conventionally uses them, so ``ne`` defaults
to Devanagari digits and ``hi`` to ASCII, matching how each language's corpus is written.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Callable, Iterator

# Marker that separates the question from the answer. Both languages use the same two
# words, which keeps the finetuning format identical across models even though every
# other string differs.
PROMPT_PREFIX = "प्रश्न:"
ANSWER_PREFIX = "उत्तर:"

DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


@dataclass(frozen=True)
class ComparisonAttribute:
    """A qualitative attribute compared without numbers, e.g. height.

    Used for the transitive-chain templates, where premises are relations between people
    ("A is taller than B") rather than measured values.

    Attributes:
        key: Stable identifier recorded in the dataset, e.g. ``"height"``.
        statement: Premise pattern with ``{a}`` and ``{b}``; ``a`` is the greater one.
        q_most: Question asking which entity is greatest.
        q_least: Question asking which entity is least.
        q_pair: Question comparing two named entities, with ``{a}`` and ``{b}``.
    """

    key: str
    statement: str
    q_most: str
    q_least: str
    q_pair: str


@dataclass(frozen=True)
class NumericAttribute:
    """An attribute stated as a measured value, e.g. price in rupees.

    Attributes:
        key: Stable identifier recorded in the dataset, e.g. ``"price"``.
        subject: ``"person"`` or ``"object"``, selecting which entity pool to draw from.
        statement: Value pattern with ``{e}`` (entity) and ``{v}`` (value).
        q_most: Question asking whose value is highest.
        q_least: Question asking whose value is lowest.
        q_pair: Question asking which of ``{a}`` and ``{b}`` has the higher value.
        q_equal: Yes/no question asking whether ``{a}`` and ``{b}`` are equal.
        low: Inclusive lower bound for generated values.
        high: Inclusive upper bound for generated values.
    """

    key: str
    subject: str
    statement: str
    q_most: str
    q_least: str
    q_pair: str
    q_equal: str
    low: int
    high: int


@dataclass(frozen=True)
class LanguagePack:
    """Everything language-specific needed to render an example.

    Attributes:
        code: ``"hi"`` or ``"ne"``.
        comparison_attributes: Attributes for the qualitative chain templates.
        numeric_attributes: Attributes for the value-based templates.
        person_names: Pool of person names, partitioned across splits.
        object_names: Pool of object names, partitioned across splits.
        yes: Affirmative answer token for equality questions.
        no: Negative answer token for equality questions.
        devanagari_digits: Whether to render numbers in Devanagari digits.
        oblique_aa_ending: Whether nouns ending in -ा take the oblique -े before a
            postposition. True for Hindi, False for Nepali.
    """

    code: str
    comparison_attributes: tuple[ComparisonAttribute, ...]
    numeric_attributes: tuple[NumericAttribute, ...]
    person_names: tuple[str, ...]
    object_names: tuple[str, ...]
    yes: str
    no: str
    devanagari_digits: bool
    oblique_aa_ending: bool

    def number(self, value: int) -> str:
        """Render an integer in this language's conventional digits."""
        text = str(value)
        return text.translate(DEVANAGARI_DIGITS) if self.devanagari_digits else text

    def oblique(self, name: str) -> str:
        """Return the form a noun takes directly before a postposition.

        Hindi masculine nouns ending in ``-ा`` shift to ``-े`` in the oblique case, so
        it is ``डिब्बे की कीमत``, never ``डिब्बा की कीमत``. Every name in the pools that
        ends in ``-ा`` is masculine, so the rule can be applied by ending alone; a
        feminine ``-ा`` noun would need an exception list.

        Nepali does not inflect the stem — ``बाकसको`` attaches the postposition
        directly — so this is the identity function there.

        Args:
            name: Entity name in its direct (citation) form.

        Returns:
            The oblique form, or the name unchanged where the language has no such rule.
        """
        if self.oblique_aa_ending and name.endswith("ा"):
            return name[:-1] + "े"
        return name


# --------------------------------------------------------------------------------------
# Hindi
# --------------------------------------------------------------------------------------
# Comparative adjectives are masculine throughout, which is why the person pools below
# contain only masculine given names. Mixing in feminine names would require लंबी/बड़ी
# and a second set of statement patterns for no gain in reasoning difficulty.

HINDI = LanguagePack(
    code="hi",
    comparison_attributes=(
        ComparisonAttribute(
            key="height",
            statement="{a}, {b} से लंबा है।",
            q_most="सबसे लंबा कौन है?",
            q_least="सबसे छोटा कौन है?",
            q_pair="{a} और {b} में कौन लंबा है?",
        ),
        ComparisonAttribute(
            key="age",
            statement="{a}, {b} से बड़ा है।",
            q_most="सबसे बड़ा कौन है?",
            q_least="सबसे छोटा कौन है?",
            q_pair="{a} और {b} में कौन बड़ा है?",
        ),
        ComparisonAttribute(
            key="weight",
            statement="{a}, {b} से भारी है।",
            q_most="सबसे भारी कौन है?",
            q_least="सबसे हल्का कौन है?",
            q_pair="{a} और {b} में कौन भारी है?",
        ),
        ComparisonAttribute(
            key="speed",
            statement="{a}, {b} से तेज़ है।",
            q_most="सबसे तेज़ कौन है?",
            q_least="सबसे धीमा कौन है?",
            q_pair="{a} और {b} में कौन तेज़ है?",
        ),
    ),
    numeric_attributes=(
        # कीमत, उम्र and लंबाई are feminine -> की / किसकी. वज़न is masculine -> का / किसका.
        NumericAttribute(
            key="price",
            subject="object",
            statement="{e} की कीमत {v} रुपये है।",
            q_most="सबसे अधिक कीमत किसकी है?",
            q_least="सबसे कम कीमत किसकी है?",
            q_pair="{a} और {b} में किसकी कीमत अधिक है?",
            q_equal="क्या {a} और {b} की कीमत बराबर है?",
            low=20,
            high=4000,
        ),
        NumericAttribute(
            key="age",
            subject="person",
            statement="{e} की उम्र {v} साल है।",
            q_most="सबसे अधिक उम्र किसकी है?",
            q_least="सबसे कम उम्र किसकी है?",
            q_pair="{a} और {b} में किसकी उम्र अधिक है?",
            q_equal="क्या {a} और {b} की उम्र बराबर है?",
            low=6,
            high=80,
        ),
        NumericAttribute(
            key="weight",
            subject="object",
            statement="{e} का वज़न {v} किलो है।",
            q_most="सबसे अधिक वज़न किसका है?",
            q_least="सबसे कम वज़न किसका है?",
            q_pair="{a} और {b} में किसका वज़न अधिक है?",
            q_equal="क्या {a} और {b} का वज़न बराबर है?",
            low=1,
            high=95,
        ),
        NumericAttribute(
            key="length",
            subject="object",
            statement="{e} की लंबाई {v} सेंटीमीटर है।",
            q_most="सबसे अधिक लंबाई किसकी है?",
            q_least="सबसे कम लंबाई किसकी है?",
            q_pair="{a} और {b} में किसकी लंबाई अधिक है?",
            q_equal="क्या {a} और {b} की लंबाई बराबर है?",
            low=5,
            high=300,
        ),
    ),
    person_names=(
        "राम", "श्याम", "मोहन", "सुरेश", "राजेश", "अनिल", "विकास", "दीपक",
        "मनोज", "संजय", "अजय", "विजय", "राहुल", "अमित", "प्रवीण", "नरेश",
        "गोपाल", "हरि", "किशोर", "लोकेश", "महेश", "नितिन", "पंकज", "रमेश",
        "सचिन", "तरुण", "उमेश", "वरुण", "यश", "आशीष", "भरत", "चेतन",
        "धीरज", "गौरव", "हितेश", "जतिन", "कपिल", "ललित", "मुकेश", "निखिल",
        "ओमकार", "परेश", "रोहित", "सुनील", "तुषार", "उदय", "विनोद", "योगेश",
        "अरुण", "बलराम", "चंदन", "दिनेश", "गिरीश", "हेमंत", "इंद्र", "जगदीश",
        "कमल", "मदन", "नवीन", "प्रमोद",
    ),
    object_names=(
        "किताब", "कलम", "बैग", "कुर्सी", "मेज़", "घड़ी", "जूता", "कमीज़",
        "चश्मा", "छाता", "ताला", "पंखा", "कंबल", "तकिया", "बोतल", "कटोरी",
        "थाली", "चम्मच", "बाल्टी", "झाड़ू", "दर्पण", "कंघी", "तौलिया", "रुमाल",
        "टोपी", "बेल्ट", "मोजा", "दस्ताना", "पर्दा", "चादर", "गमला", "डिब्बा",
        "टोकरी", "सीढ़ी", "हथौड़ा", "पेचकस", "रस्सी", "कैंची", "सुई", "धागा",
    ),
    yes="हाँ",
    no="नहीं",
    devanagari_digits=False,
    oblique_aa_ending=True,
)


# --------------------------------------------------------------------------------------
# Nepali
# --------------------------------------------------------------------------------------
# Nepali marks the comparative with भन्दा and the superlative with सबैभन्दा, so the
# question forms differ structurally from Hindi rather than being a word-for-word map.

NEPALI = LanguagePack(
    code="ne",
    comparison_attributes=(
        ComparisonAttribute(
            key="height",
            statement="{a}, {b} भन्दा अग्लो छ।",
            q_most="सबैभन्दा अग्लो को हो?",
            q_least="सबैभन्दा होचो को हो?",
            q_pair="{a} र {b} मध्ये को अग्लो छ?",
        ),
        ComparisonAttribute(
            key="age",
            statement="{a}, {b} भन्दा जेठो छ।",
            q_most="सबैभन्दा जेठो को हो?",
            q_least="सबैभन्दा कान्छो को हो?",
            q_pair="{a} र {b} मध्ये को जेठो छ?",
        ),
        ComparisonAttribute(
            key="weight",
            statement="{a}, {b} भन्दा गह्रौं छ।",
            q_most="सबैभन्दा गह्रौं को हो?",
            q_least="सबैभन्दा हलुका को हो?",
            q_pair="{a} र {b} मध्ये को गह्रौं छ?",
        ),
        ComparisonAttribute(
            key="speed",
            statement="{a}, {b} भन्दा छिटो छ।",
            q_most="सबैभन्दा छिटो को हो?",
            q_least="सबैभन्दा ढिलो को हो?",
            q_pair="{a} र {b} मध्ये को छिटो छ?",
        ),
    ),
    numeric_attributes=(
        NumericAttribute(
            key="price",
            subject="object",
            statement="{e}को मूल्य {v} रुपैयाँ हो।",
            q_most="सबैभन्दा बढी मूल्य कसको हो?",
            q_least="सबैभन्दा कम मूल्य कसको हो?",
            q_pair="{a} र {b} मध्ये कसको मूल्य बढी छ?",
            q_equal="के {a} र {b}को मूल्य बराबर छ?",
            low=20,
            high=4000,
        ),
        NumericAttribute(
            key="age",
            subject="person",
            statement="{e}को उमेर {v} वर्ष हो।",
            q_most="सबैभन्दा बढी उमेर कसको हो?",
            q_least="सबैभन्दा कम उमेर कसको हो?",
            q_pair="{a} र {b} मध्ये कसको उमेर बढी छ?",
            q_equal="के {a} र {b}को उमेर बराबर छ?",
            low=6,
            high=80,
        ),
        NumericAttribute(
            key="weight",
            subject="object",
            statement="{e}को तौल {v} किलो हो।",
            q_most="सबैभन्दा बढी तौल कसको हो?",
            q_least="सबैभन्दा कम तौल कसको हो?",
            q_pair="{a} र {b} मध्ये कसको तौल बढी छ?",
            q_equal="के {a} र {b}को तौल बराबर छ?",
            low=1,
            high=95,
        ),
        NumericAttribute(
            key="length",
            subject="object",
            statement="{e}को लम्बाइ {v} सेन्टिमिटर हो।",
            q_most="सबैभन्दा बढी लम्बाइ कसको हो?",
            q_least="सबैभन्दा कम लम्बाइ कसको हो?",
            q_pair="{a} र {b} मध्ये कसको लम्बाइ बढी छ?",
            q_equal="के {a} र {b}को लम्बाइ बराबर छ?",
            low=5,
            high=300,
        ),
    ),
    person_names=(
        "राम", "श्याम", "हरि", "गोपाल", "विनोद", "सुनिल", "दीपक", "प्रकाश",
        "नरेश", "बिकास", "सागर", "अनिल", "राजु", "कृष्ण", "मनोज", "सुरेश",
        "रमेश", "गणेश", "दिनेश", "महेश", "उमेश", "नबिन", "पवन", "सन्तोष",
        "भीम", "चन्द्र", "धन", "गोविन्द", "हेम", "इन्द्र", "जीवन", "कमल",
        "लक्ष्मण", "मदन", "नारायण", "ओम", "पदम", "रवि", "शंकर", "तुलसी",
        "उद्धव", "विष्णु", "यज्ञ", "अर्जुन", "बद्री", "छत्र", "देव", "गंगा",
        "हिमाल", "जनक", "केशव", "लोक", "मुकुन्द", "निर्मल", "प्रेम", "रुद्र",
        "सरोज", "तेज", "उपेन्द्र", "युवराज",
    ),
    object_names=(
        "किताब", "कलम", "झोला", "कुर्सी", "टेबल", "घडी", "जुत्ता", "कमिज",
        "चस्मा", "छाता", "ताल्चा", "पंखा", "कम्बल", "सिरानी", "बोतल", "कचौरा",
        "थाल", "चम्चा", "बाल्टिन", "कुचो", "ऐना", "काइँयो", "तौलिया", "रुमाल",
        "टोपी", "पेटी", "मोजा", "पन्जा", "पर्दा", "तन्ना", "गमला", "बाकस",
        "डोको", "भर्याङ", "मार्तोल", "पेचकिला", "डोरी", "कैंची", "सियो", "धागो",
    ),
    yes="हो",
    no="होइन",
    devanagari_digits=True,
    oblique_aa_ending=False,
)


LANGUAGE_PACKS: dict[str, LanguagePack] = {"hi": HINDI, "ne": NEPALI}


@dataclass
class Example:
    """One generated reasoning item.

    Attributes:
        example_id: Stable hash of the prompt, used for deduplication.
        template: Which generator produced it, e.g. ``"chain_pair"``.
        attribute: Attribute key, e.g. ``"height"``.
        n_entities: How many entities the premises mention.
        hops: Reasoning steps needed. 1 for a direct comparison, 2+ when the answer
            requires chaining premises that never mention both entities together.
        prompt: Everything the model sees, ending with the answer marker.
        answer: The gold continuation, without a leading space.
        entities: Entity names used, in the order the premises were generated.
    """

    example_id: str
    template: str
    attribute: str
    n_entities: int
    hops: int
    prompt: str
    answer: str
    entities: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The full sequence finetuning trains on: prompt, a space, then the answer.

        Derived rather than stored. Writing it to disk alongside its two components
        would inflate the dataset by a third and create a third place for the same
        string to disagree with itself.
        """
        return f"{self.prompt} {self.answer}"

    def to_dict(self) -> dict:
        """Return a JSON-serialisable view. ``text`` is derived, so it is not stored."""
        return {
            "example_id": self.example_id,
            "template": self.template,
            "attribute": self.attribute,
            "n_entities": self.n_entities,
            "hops": self.hops,
            "prompt": self.prompt,
            "answer": self.answer,
            "entities": self.entities,
        }


def _assemble(sentences: list[str], question: str, answer: str, rng: random.Random,
              **meta) -> Example:
    """Shuffle premises, build the prompt, and package the result.

    Shuffling is the point: in chain order the first-mentioned entity is always the
    greatest, which would let the model answer superlative questions by position alone.

    Args:
        sentences: Premise sentences, in chain order.
        question: The question sentence.
        answer: Gold answer string.
        rng: Seeded random source.
        **meta: Passed through to the Example (template, attribute, n_entities, hops,
            entities).

    Returns:
        A fully built Example.
    """
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    body = " ".join(shuffled + [question])
    prompt = f"{PROMPT_PREFIX} {body}\n{ANSWER_PREFIX}"
    return Example(
        example_id=hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16],
        prompt=prompt,
        answer=answer,
        **meta,
    )


def gen_chain_superlative(pack: LanguagePack, names: list[str], rng: random.Random,
                          n_entities: int = 3) -> Example:
    """Transitive chain, asking for the greatest or least entity.

    Builds ``A > B > C`` as pairwise premises and asks which is largest or smallest.
    Answering needs every premise, because no single one names both extremes.
    """
    attr = rng.choice(pack.comparison_attributes)
    chain = rng.sample(names, n_entities)
    sentences = [attr.statement.format(a=chain[i], b=pack.oblique(chain[i + 1]))
                 for i in range(n_entities - 1)]

    if rng.random() < 0.5:
        question, answer, which = attr.q_most, chain[0], "most"
    else:
        question, answer, which = attr.q_least, chain[-1], "least"

    return _assemble(
        sentences, question, answer, rng,
        template=f"chain_superlative_{which}",
        attribute=attr.key,
        n_entities=n_entities,
        hops=n_entities - 1,
        entities=chain,
    )


def gen_chain_pair(pack: LanguagePack, names: list[str], rng: random.Random,
                   n_entities: int = 3) -> Example:
    """Transitive chain, asking to relate two entities no premise mentions together.

    This is the multi-hop case the assignment calls out: given ``A > B`` and ``B > C``,
    decide the relation between ``A`` and ``C``. The pair is always non-adjacent in the
    chain, so the answer cannot be read off a single premise.
    """
    attr = rng.choice(pack.comparison_attributes)
    chain = rng.sample(names, n_entities)
    sentences = [attr.statement.format(a=chain[i], b=pack.oblique(chain[i + 1]))
                 for i in range(n_entities - 1)]

    # Non-adjacent indices only: adjacent ones are answerable from one premise.
    candidates = [(i, j) for i in range(n_entities) for j in range(i + 2, n_entities)]
    i, j = rng.choice(candidates)
    first, second = chain[i], chain[j]

    # Present the pair in random order so the answer is not always the first named.
    if rng.random() < 0.5:
        question = attr.q_pair.format(a=pack.oblique(first), b=pack.oblique(second))
    else:
        question = attr.q_pair.format(a=pack.oblique(second), b=pack.oblique(first))

    return _assemble(
        sentences, question, chain[i], rng,
        template="chain_pair",
        attribute=attr.key,
        n_entities=n_entities,
        hops=j - i,
        entities=chain,
    )


def gen_numeric_pair(pack: LanguagePack, persons: list[str], objects: list[str],
                     rng: random.Random) -> Example:
    """Two entities with stated values; ask which value is higher."""
    attr = rng.choice(pack.numeric_attributes)
    pool = persons if attr.subject == "person" else objects
    a, b = rng.sample(pool, 2)

    va = rng.randint(attr.low, attr.high)
    vb = rng.randint(attr.low, attr.high)
    while vb == va:  # a tie has no single correct answer for this question
        vb = rng.randint(attr.low, attr.high)

    sentences = [attr.statement.format(e=pack.oblique(a), v=pack.number(va)),
                 attr.statement.format(e=pack.oblique(b), v=pack.number(vb))]
    question = attr.q_pair.format(a=pack.oblique(a), b=pack.oblique(b))

    return _assemble(
        sentences, question, a if va > vb else b, rng,
        template="numeric_pair",
        attribute=attr.key,
        n_entities=2,
        hops=1,
        entities=[a, b],
    )


def gen_numeric_superlative(pack: LanguagePack, persons: list[str], objects: list[str],
                            rng: random.Random, n_entities: int = 3) -> Example:
    """Several entities with stated values; ask for the highest or lowest."""
    attr = rng.choice(pack.numeric_attributes)
    pool = persons if attr.subject == "person" else objects
    chosen = rng.sample(pool, n_entities)

    # Distinct values, so the superlative is unambiguous. sample() raises if the range
    # is too narrow, which is the right behaviour: silently emitting fewer values than
    # entities would produce a question whose premises do not cover every name.
    values = rng.sample(range(attr.low, attr.high + 1), n_entities)

    sentences = [attr.statement.format(e=pack.oblique(e), v=pack.number(v))
                 for e, v in zip(chosen, values)]

    if rng.random() < 0.5:
        question, target, which = attr.q_most, max(values), "most"
    else:
        question, target, which = attr.q_least, min(values), "least"
    answer = chosen[values.index(target)]

    return _assemble(
        sentences, question, answer, rng,
        template=f"numeric_superlative_{which}",
        attribute=attr.key,
        n_entities=n_entities,
        hops=1,
        entities=chosen,
    )


def gen_numeric_equality(pack: LanguagePack, persons: list[str], objects: list[str],
                         rng: random.Random) -> Example:
    """Two entities with stated values; ask whether they are equal.

    Equal and unequal cases are produced with equal probability, so the label is
    balanced and a model that always answers "no" scores 50%, not better.
    """
    attr = rng.choice(pack.numeric_attributes)
    pool = persons if attr.subject == "person" else objects
    a, b = rng.sample(pool, 2)

    va = rng.randint(attr.low, attr.high)
    if rng.random() < 0.5:
        vb, answer = va, pack.yes
    else:
        vb = rng.randint(attr.low, attr.high)
        while vb == va:
            vb = rng.randint(attr.low, attr.high)
        answer = pack.no

    sentences = [attr.statement.format(e=pack.oblique(a), v=pack.number(va)),
                 attr.statement.format(e=pack.oblique(b), v=pack.number(vb))]
    question = attr.q_equal.format(a=pack.oblique(a), b=pack.oblique(b))

    return _assemble(
        sentences, question, answer, rng,
        template="numeric_equality",
        attribute=attr.key,
        n_entities=2,
        hops=1,
        entities=[a, b],
    )


# Generator name -> callable. The driver samples from this table, so adding a template
# family is a one-line change here rather than an edit to the generation loop.
GENERATORS: dict[str, Callable] = {
    "chain_superlative_3": lambda p, pe, ob, r: gen_chain_superlative(p, pe, r, 3),
    "chain_superlative_4": lambda p, pe, ob, r: gen_chain_superlative(p, pe, r, 4),
    "chain_pair_3": lambda p, pe, ob, r: gen_chain_pair(p, pe, r, 3),
    "chain_pair_4": lambda p, pe, ob, r: gen_chain_pair(p, pe, r, 4),
    "numeric_pair": lambda p, pe, ob, r: gen_numeric_pair(p, pe, ob, r),
    "numeric_superlative_3": lambda p, pe, ob, r: gen_numeric_superlative(p, pe, ob, r, 3),
    "numeric_superlative_4": lambda p, pe, ob, r: gen_numeric_superlative(p, pe, ob, r, 4),
    "numeric_equality": lambda p, pe, ob, r: gen_numeric_equality(p, pe, ob, r),
}


def partition_pool(pool: tuple[str, ...], fractions: dict[str, float],
                   rng: random.Random) -> dict[str, list[str]]:
    """Split an entity pool into disjoint per-split lists.

    Disjointness is the leakage control: a name that appears in test never appears in
    train, so the model cannot have memorised anything about that particular name.

    Args:
        pool: All available names.
        fractions: Split name -> share of the pool, summing to 1.0.
        rng: Seeded random source, so the partition is reproducible.

    Returns:
        Split name -> list of names, with no name in more than one split.

    Raises:
        ValueError: If any split would receive fewer than four names, which is too few
            to build a four-entity question.
    """
    shuffled = list(pool)
    rng.shuffle(shuffled)

    out: dict[str, list[str]] = {}
    start = 0
    names = list(fractions)
    for i, split in enumerate(names):
        if i == len(names) - 1:
            out[split] = shuffled[start:]
        else:
            take = int(round(len(pool) * fractions[split]))
            out[split] = shuffled[start:start + take]
            start += take

    for split, got in out.items():
        if len(got) < 4:
            raise ValueError(
                f"split {split!r} received only {len(got)} names; at least 4 are needed "
                "to build a four-entity question. Enlarge the pool or change fractions."
            )
    return out


def generate_split(pack: LanguagePack, persons: list[str], objects: list[str],
                   count: int, rng: random.Random,
                   max_attempts_per_item: int = 200) -> Iterator[Example]:
    """Yield ``count`` unique examples, cycling through the generator families evenly.

    Uniqueness is enforced on the prompt hash. Because a split's entity pool is finite,
    the same question can be drawn twice; retrying keeps the dataset free of exact
    duplicates, which would otherwise inflate both training weight and test accuracy.

    Args:
        pack: Language pack to render with.
        persons: Person names available to this split.
        objects: Object names available to this split.
        count: How many examples to produce.
        rng: Seeded random source.
        max_attempts_per_item: Give up after this many consecutive duplicate draws.

    Yields:
        Unique Examples.

    Raises:
        RuntimeError: If the pool is too small to reach ``count`` unique examples.
    """
    families = sorted(GENERATORS)
    seen: set[str] = set()
    produced = 0

    while produced < count:
        name = families[produced % len(families)]
        make = GENERATORS[name]

        for _ in range(max_attempts_per_item):
            example = make(pack, persons, objects, rng)
            if example.example_id not in seen:
                seen.add(example.example_id)
                yield example
                produced += 1
                break
        else:
            raise RuntimeError(
                f"could not find a new unique example for family {name!r} after "
                f"{max_attempts_per_item} attempts ({produced} of {count} produced). "
                "The entity pool for this split is too small for the requested size."
            )
