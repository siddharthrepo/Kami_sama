"""Phase 2: decoder-only Transformer language models for Hindi and Nepali.

Library code is shared between the two languages; *data* is not. Model H (Hindi) and
Model L (Nepali) have separate corpora, separate tokenizers, separate vocabularies and
separate weights — nothing crosses between them except the classes in this package.
Per-language configuration, checkpoints, logs and evaluation output live under
``hindi/`` and ``nepali/`` respectively.

This mirrors the arrangement Phase 1 already used for ``common/``.
"""

from lma.config import ModelConfig, TrainConfig

__all__ = ["ModelConfig", "TrainConfig"]
