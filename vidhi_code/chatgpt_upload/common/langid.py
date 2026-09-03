"""Language identification for Devanagari text.

This is the step the whole corpus depends on. Hindi, Nepali, Marathi, Bhojpuri, Maithili
and Sanskrit all use Devanagari, so the cheap script-ratio check used during collection
cannot tell them apart — it only rejects English.

That matters because the large web corpora are organised by **script**, not language.
FineWeb-2's ``npi_Deva`` split is "Nepali-ish Devanagari text", which in practice contains
a meaningful amount of Hindi, and ``hin_Deva`` contains Marathi and Bhojpuri. Training
Model L on Hindi-contaminated Nepali would quietly undermine the entire comparison the
project is built around.

We use fastText's ``lid.176`` classifier, which covers 176 languages including ``hi``,
``ne``, ``mr``, ``bh`` and ``sa``. The compressed ``.ftz`` build is ~1 MB rather than
126 MB, at a small accuracy cost — worth it to keep the repository self-contained and the
download quick.

Accuracy on our own data is measurable rather than assumed: the scraped corpora are
known-language by construction (Nepali news sites publish Nepali), so running the
classifier over them gives an honest error rate to quote in the report.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

# Compressed fastText language-ID model. ~1 MB, 176 languages.
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
DEFAULT_MODEL_PATH = Path(os.environ.get("LANGID_MODEL_DIR", "models")) / "lid.176.ftz"

# Characters of each document actually classified. Language identity is obvious from a
# few hundred words, and truncating makes the pass dramatically faster across a million
# documents.
CLASSIFY_CHARS = 2000


def ensure_model(path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the fastText language-ID model if it is not already present.

    Args:
        path: Where the model should live.

    Returns:
        Path to the model file.
    """
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading language-ID model to {path} ...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"Downloaded {path.stat().st_size/1e6:.1f} MB", flush=True)
    return path


class LanguageIdentifier:
    """Wraps fastText ``lid.176`` for document-level language identification.

    The model is loaded once and reused. Prediction is done on a truncated prefix of each
    document, with newlines flattened, because fastText treats a newline as a document
    boundary and warns when it finds one.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """Load the classifier, downloading the model on first use.

        Args:
            model_path: Optional explicit path to a fastText ``.ftz``/``.bin`` model.
        """
        import fasttext

        path = ensure_model(Path(model_path) if model_path else DEFAULT_MODEL_PATH)
        # fastText prints a deprecation notice on load; silence it rather than have it
        # interleave with pipeline progress output.
        fasttext.FastText.eprint = lambda *args, **kwargs: None
        self.model = fasttext.load_model(str(path))

    def predict(self, text: str) -> tuple[str, float]:
        """Return the most likely language code and its confidence.

        Args:
            text: Document text.

        Returns:
            A ``(lang_code, confidence)`` pair, e.g. ``("ne", 0.98)``. Returns
            ``("unknown", 0.0)`` for text too short to classify.
        """
        sample = text[:CLASSIFY_CHARS].replace("\n", " ").strip()
        if len(sample) < 20:
            return "unknown", 0.0

        labels, scores = self.model.predict(sample, k=1)
        if not labels:
            return "unknown", 0.0
        return labels[0].replace("__label__", ""), float(scores[0])

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        """Classify many documents in one call, which is markedly faster than looping.

        Args:
            texts: Document texts.

        Returns:
            One ``(lang_code, confidence)`` pair per input, in order.
        """
        samples = [t[:CLASSIFY_CHARS].replace("\n", " ").strip() for t in texts]
        # fastText rejects empty strings; substitute a placeholder and mark it after.
        cleaned = [s if len(s) >= 20 else "x" for s in samples]

        labels, scores = self.model.predict(cleaned, k=1)
        results: list[tuple[str, float]] = []
        for original, label, score in zip(samples, labels, scores):
            if len(original) < 20:
                results.append(("unknown", 0.0))
            else:
                results.append((label[0].replace("__label__", ""), float(score[0])))
        return results

    def accepts(self, text: str, lang: str, min_confidence: float = 0.5) -> bool:
        """Decide whether a document is confidently in the target language.

        Args:
            text: Document text.
            lang: Expected language code, ``"hi"`` or ``"ne"``.
            min_confidence: Minimum classifier confidence required.

        Returns:
            True if the document should be kept.
        """
        predicted, confidence = self.predict(text)
        return predicted == lang and confidence >= min_confidence
