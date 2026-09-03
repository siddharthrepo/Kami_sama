"""Model and training configuration for the Phase 2 Transformers.

Two independent models are configured through this module — Hindi (Model H) and Nepali
(Model L) — but the *shape* of both is deliberately identical. Phase 1 settled both
languages on a 32,000-piece vocabulary, so nothing forces the architectures apart, and
holding them constant means any difference in perplexity, generation quality or
attention behaviour is attributable to the corpora rather than to the model. That is
precisely the comparison this project exists to make.

The budget is approximately 25M trainable parameters per model.
:meth:`ModelConfig.count_parameters` derives that count analytically from the
hyperparameters alone, without instantiating anything, so a config can be checked for
budget compliance before a GPU is ever allocated. ``tests/test_model.py`` asserts that
this analytic count equals the real ``nn.Module`` total; that assertion is what stops
the two from silently drifting apart as the architecture is edited.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture of one decoder-only (GPT-style) Transformer.

    The defaults describe the configuration chosen for this project: 384 model
    dimensions, 7 layers and 6 heads, which lands at 24,906,624 parameters against a
    32,000 vocabulary — just under the ~25M target.

    Attributes:
        vocab_size: Size of this language's vocabulary. Read from the Phase 1
            SentencePiece model rather than hardcoded, so the config cannot disagree
            with the tokenizer that produced the training data.
        n_layer: Number of stacked Transformer blocks.
        n_head: Number of attention heads. Must divide ``d_model``.
        d_model: Model dimension, written ``d`` throughout the attention code.
        d_ff: Inner width of the position-wise feed-forward network. Conventionally
            ``4 * d_model``, which is what the default expresses.
        max_seq_len: Context length, and simultaneously the number of rows in the
            learned positional embedding table. This is a hard ceiling: the model
            cannot be evaluated on a longer sequence than this, because there is no
            positional vector to look up for position ``max_seq_len`` or beyond.
        dropout: Dropout on sublayer outputs, inside the residual stream.
        attn_dropout: Dropout applied to the attention probabilities after softmax.
        embd_dropout: Dropout applied to the summed token+position embeddings.
        bias: Whether ``nn.Linear`` layers carry a bias term. LayerNorm always does.
        tie_weights: Share one matrix between the input embedding and the output
            projection. With a 32,000 vocabulary and ``d_model`` 384 this saves
            12,288,000 parameters — 33% of what the untied model would cost — and
            spends the savings on depth instead.
        init_std: Standard deviation of the normal initialiser. Output projections
            inside blocks are additionally scaled by ``1/sqrt(2 * n_layer)``; see
            :meth:`lma.model.GPT._init_weights`.
    """

    vocab_size: int
    n_layer: int = 7
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    max_seq_len: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1
    embd_dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model={self.d_model} is not divisible by n_head={self.n_head}; "
                "every head must get an equal slice of the model dimension"
            )
        if self.vocab_size <= 0 or self.max_seq_len <= 0 or self.n_layer <= 0:
            raise ValueError("vocab_size, max_seq_len and n_layer must all be positive")

    @property
    def d_head(self) -> int:
        """Dimension each attention head works in, ``d_model // n_head``."""
        return self.d_model // self.n_head

    def count_parameters(self) -> dict[str, int]:
        """Derive the trainable parameter count analytically.

        Computed from the hyperparameters alone — no tensors are allocated — so a
        candidate architecture can be sized against the ~25M budget instantly.

        Returns:
            A breakdown by component plus two summary keys. ``total`` is every
            trainable parameter; ``non_embedding`` excludes the token and positional
            tables, which is the figure usually quoted when comparing model capacity,
            since embeddings scale with vocabulary rather than with depth.
        """
        d = self.d_model
        bias = int(self.bias)

        def linear(fan_in: int, fan_out: int) -> int:
            return fan_in * fan_out + fan_out * bias

        # LayerNorm carries a weight and a bias vector regardless of self.bias.
        norm = 2 * d

        # Q, K and V are produced by one fused projection, hence 3 * d out-features.
        attention = linear(d, 3 * d) + linear(d, d)
        feed_forward = linear(d, self.d_ff) + linear(self.d_ff, d)
        block = norm + attention + norm + feed_forward

        parts = {
            "token_embedding": self.vocab_size * d,
            "positional_embedding": self.max_seq_len * d,
            "blocks": self.n_layer * block,
            "final_norm": norm,
            # A tied head reuses the token embedding matrix and adds nothing.
            "lm_head": 0 if self.tie_weights else linear(d, self.vocab_size),
        }
        parts["total"] = sum(parts.values())
        parts["non_embedding"] = (
            parts["total"] - parts["token_embedding"] - parts["positional_embedding"]
        )
        return parts

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """Load a config, ignoring any keys that are documentation rather than fields.

        Config files carry explanatory keys such as ``notes`` and ``_comment`` for the
        benefit of a human reader. Those are dropped here so that annotating a config
        can never change how a model is built.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def to_json(self, path: str | Path) -> None:
        """Write this config to disk, with the derived parameter counts alongside.

        The counts are informational — :meth:`from_json` ignores them on the way back
        in — but writing them means every checkpoint directory carries a record of how
        large the model was, which is what the assignment's reproducibility guideline
        asks for.
        """
        payload = asdict(self)
        payload["_parameter_counts"] = self.count_parameters()
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class TrainConfig:
    """Pretraining hyperparameters for one model.

    Attributes:
        tokens_per_step: Optimiser step size measured in tokens, held constant so the
            schedule is independent of whatever micro-batch a given GPU can fit.
            Gradient accumulation makes up the difference; see :attr:`micro_batch_size`.
        micro_batch_size: Sequences per forward pass. Purely a memory knob — lower it
            on a smaller GPU and ``grad_accum_steps`` rises to compensate, leaving the
            effective batch and therefore the learning dynamics unchanged.
        max_steps: Total optimiser steps. At the default of 16,000 steps and 32,768
            tokens per step the model sees ~524M tokens, which is roughly one pass over
            a 500M-token corpus and close to the compute-optimal ratio of ~20 tokens
            per parameter for a 25M-parameter model.
        learning_rate: Peak learning rate, reached at the end of warmup.
        min_lr_ratio: Floor of the cosine decay, as a fraction of the peak.
        warmup_steps: Linear ramp from zero. Warmup matters most in the first few
            hundred steps, when the randomly initialised attention logits are near
            uniform and large updates destabilise training.
        weight_decay: Applied to matrices only. Biases, LayerNorm parameters and
            embeddings are excluded; decaying them is known to hurt.
        beta1: AdamW first-moment decay.
        beta2: AdamW second-moment decay.
        grad_clip: Global gradient-norm clip, ``0`` to disable.
        eval_every: Steps between validation passes.
        eval_batches: Validation batches per pass. Enough for a stable estimate
            without stalling training.
        checkpoint_every: Steps between checkpoint writes. Kaggle sessions terminate
            at 12 hours and can be pre-empted sooner, so this bounds the work lost to
            an interruption.
        keep_last_n: How many rolling checkpoints to retain on disk, ignoring the
            separately preserved best-validation checkpoint.
        seed: Seed for parameter initialisation and batch order.
        amp_dtype: ``"float16"``, ``"bfloat16"`` or ``"float32"``. Kaggle's T4 is
            Turing-class and has no usable bfloat16 path, so float16 with a gradient
            scaler is the fast option there.
    """

    tokens_per_step: int = 32_768
    micro_batch_size: int = 32
    max_steps: int = 16_000
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 100
    checkpoint_every: int = 1_000
    keep_last_n: int = 2
    seed: int = 1337
    amp_dtype: str = "float16"

    def grad_accum_steps(self, max_seq_len: int) -> int:
        """Micro-batches to accumulate before stepping the optimiser.

        Args:
            max_seq_len: Sequence length, from the :class:`ModelConfig`.

        Returns:
            ``tokens_per_step / (micro_batch_size * max_seq_len)``, at least 1.

        Raises:
            ValueError: If the micro-batch does not divide the target step size, which
                would make the effective batch differ from the configured one and
                quietly change the learning dynamics.
        """
        per_micro = self.micro_batch_size * max_seq_len
        if self.tokens_per_step % per_micro != 0:
            raise ValueError(
                f"tokens_per_step={self.tokens_per_step} is not divisible by "
                f"micro_batch_size * max_seq_len = {per_micro}"
            )
        return max(1, self.tokens_per_step // per_micro)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        """Load a training config, ignoring documentation-only keys."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in fields})

    def to_json(self, path: str | Path) -> None:
        """Write this training config to disk."""
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
