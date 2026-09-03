"""Decoder-only (GPT-style) Transformer, built from primitive PyTorch layers.

Everything here is written out by hand: the query/key/value projections, the reshape
into heads, the scaled dot product, the causal mask, the softmax and the recombination.
No ``nn.Transformer*`` module, no pre-built attention block, and deliberately no
``F.scaled_dot_product_attention`` either — that function fuses masking and softmax into
one opaque kernel, which would be both against the spirit of the assignment and useless
for Phase 2's attention analysis, since it does not hand the attention weights back.

The cost of writing attention out explicitly is materialising the (B, h, T, T) score
matrix. At the configured context length of 512 that is affordable, and it buys two
things the fused path cannot give: every intermediate is inspectable, and
``forward(..., return_attention=True)`` returns per-layer attention distributions for
the heatmaps, entropy and mean-attention-distance metrics.

Shape convention used throughout::

    B  batch size
    T  sequence length (<= config.max_seq_len)
    d  model dimension  (config.d_model)
    h  number of heads  (config.n_head)
    dk dimension per head (d // h)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lma.config import ModelConfig

# Positions marked with this label contribute nothing to the loss. Packed training
# batches are dense and never use it, but evaluation over a final partial sequence does.
IGNORE_INDEX = -100


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, implemented from first principles.

    The forward pass follows the standard formulation::

        Attention(Q, K, V) = softmax(Q K^T / sqrt(dk) + M) V

    where ``M`` is additive: zero on positions a query is allowed to attend to and
    negative infinity on future positions, so that after the softmax those positions
    receive exactly zero weight. Position ``t`` therefore sees only positions ``<= t``.

    The ``1/sqrt(dk)`` scaling matters because the dot product of two independent
    ``dk``-dimensional vectors with unit-variance components has variance ``dk``.
    Without the correction, logits grow with head width, the softmax saturates, and
    gradients through it vanish — training stalls at initialisation.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the projections and the causal mask.

        Args:
            config: Architecture configuration. ``d_model`` must divide by ``n_head``,
                which :class:`~lma.config.ModelConfig` enforces on construction.
        """
        super().__init__()
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_head
        self.scale = 1.0 / math.sqrt(self.d_head)

        # One fused matrix produces Q, K and V together: a single (d -> 3d) matmul is
        # markedly faster than three (d -> d) matmuls, and is mathematically identical.
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Lower-triangular ones, shape (1, 1, T, T): position i may attend to j <= i.
        # Registered as a buffer so it moves with .to(device) and is saved with the
        # module, but is not a trainable parameter.
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply causal self-attention.

        Args:
            x: Input activations, shape ``(B, T, d)``.
            return_attention: If True, also return the post-softmax attention
                distributions. They are captured *before* attention dropout so that
                the analysis sees the true distribution rather than a thinned sample;
                in ``eval()`` mode dropout is inactive and the two coincide anyway.

        Returns:
            ``(output, attention)`` where ``output`` has shape ``(B, T, d)`` and
            ``attention`` has shape ``(B, h, T, T)`` or is None. Row ``t`` of the
            attention matrix sums to 1 over keys ``0..t``.

        Raises:
            ValueError: If ``T`` exceeds the context length the mask was built for.
        """
        B, T, d = x.shape
        if T > self.causal_mask.size(-1):
            raise ValueError(
                f"sequence length {T} exceeds the model's context of "
                f"{self.causal_mask.size(-1)}; the causal mask and the positional "
                "embedding table are both sized for the configured maximum"
            )

        # (B, T, d) -> (B, T, 3d) -> three tensors of (B, T, d).
        q, k, v = self.qkv(x).split(self.d_model, dim=2)

        # Split the model dimension across heads and bring the head axis forward, so
        # each head's (T, dk) matrix is contiguous in the last two dimensions and the
        # batched matmuls below operate per head:
        #   (B, T, d) -> (B, T, h, dk) -> (B, h, T, dk)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Scaled dot product: (B, h, T, dk) @ (B, h, dk, T) -> (B, h, T, T).
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Additive causal mask. masked_fill with -inf is the additive form of M: the
        # masked entries become -inf before the softmax and exactly 0 after it.
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax is computed in float32 when the scores arrive in reduced precision.
        # Under autocast the scores are float16, where exponentiating a large-magnitude
        # logit overflows; promoting first avoids that. The promotion is conditional
        # rather than an unconditional .float() so that a float64 model -- used in the
        # numerical-precision controls in tests/test_causal.py -- is not silently
        # downcast to float32 here.
        #
        # Row t of the mask always leaves at least position 0 unmasked, so no row is
        # entirely -inf and the softmax cannot produce NaN.
        reduced = scores.dtype in (torch.float16, torch.bfloat16)
        attn = F.softmax(scores, dim=-1, dtype=torch.float32 if reduced else scores.dtype)
        attn = attn.to(q.dtype)
        captured = attn if return_attention else None

        attn = self.attn_dropout(attn)

        # (B, h, T, T) @ (B, h, T, dk) -> (B, h, T, dk), then concatenate the heads
        # back into the model dimension: (B, h, T, dk) -> (B, T, h, dk) -> (B, T, d).
        # .contiguous() is required because .transpose() only permutes strides, and
        # .view() needs a contiguous buffer.
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)

        return self.resid_dropout(self.proj(y)), captured


class FeedForward(nn.Module):
    """Position-wise feed-forward network: expand, apply GELU, project back.

    Applied identically and independently at every position. Attention moves
    information *between* positions; this is where per-position transformation happens.
    The inner dimension is wider than the model dimension (4x by default), which is
    where most of a Transformer's parameters and most of its capacity sit.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the two linear layers.

        Args:
            config: Architecture configuration; ``d_ff`` sets the inner width.
        """
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, d)`` to ``(B, T, d)`` through the wider inner dimension."""
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """One pre-norm Transformer block.

    Pre-norm places the LayerNorm *inside* each residual branch::

        x = x + Attention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))

    rather than normalising the sum. The consequence is that the residual stream is an
    unbroken identity path from the embeddings to the final norm, so gradients reach
    early layers without passing through any normalisation. Post-norm stacks of this
    depth typically need a carefully tuned warmup to train at all; pre-norm is
    forgiving, which is worth more than the small final-quality edge post-norm can
    show when it does converge.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the two sublayers and their norms."""
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run both sublayers with residual connections.

        Args:
            x: Activations, shape ``(B, T, d)``.
            return_attention: Forwarded to the attention sublayer.

        Returns:
            ``(output, attention)``; ``output`` keeps shape ``(B, T, d)``.
        """
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class GPT(nn.Module):
    """A decoder-only Transformer language model.

    Forward pass, in order: token embedding + learned positional embedding, embedding
    dropout, ``n_layer`` pre-norm blocks, a final LayerNorm, and a linear projection to
    vocabulary logits. Trained with the causal language-modelling objective —
    cross-entropy between the logits at position ``t`` and the token at ``t+1``.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Build the model and initialise its weights.

        Args:
            config: Architecture configuration.
        """
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Learned absolute positions. Self-attention is permutation invariant — with no
        # position signal the model would see a bag of tokens — so position is injected
        # here by addition. A learned table has one row per position, which is what
        # makes max_seq_len a hard ceiling: there is simply no vector to look up beyond
        # it. Sinusoidal or rotary schemes extrapolate further; learned embeddings were
        # chosen for being the simplest thing that is fully understood end to end, and
        # for making that ceiling explicit rather than implicit.
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.embd_dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            # One matrix serves both roles: row v of the embedding is the vector for
            # token v on the way in, and the direction scored for token v on the way
            # out. Saves vocab_size * d_model parameters -- 12.29M of a ~25M budget
            # here -- and couples the two representations, which usually helps a model
            # this small. Assigning the Parameter makes them the same object, so the
            # gradients from both uses accumulate into it.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # Residual-branch scaling. Each layer adds its output into the residual stream,
        # so the stream's variance grows with depth. Shrinking the projections that
        # write into it by 1/sqrt(2 * n_layer) -- two such projections per block --
        # keeps activations near unit scale at initialisation.
        residual_std = config.init_std / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise one submodule.

        Linear and embedding weights get ``N(0, init_std)``; biases start at zero.
        LayerNorm keeps PyTorch's default unit weight and zero bias.

        Args:
            module: Submodule visited by ``nn.Module.apply``.
        """
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Run the model over a batch of token ids.

        Args:
            idx: Token ids, shape ``(B, T)``, values in ``[0, vocab_size)``.
            targets: Optional next-token labels, shape ``(B, T)``. Callers are
                responsible for the shift — the training loader supplies ``x = seq[:-1]``
                and ``y = seq[1:]`` — so position ``t`` of ``targets`` is the token the
                model should predict from ``idx[:t+1]``. Positions labelled
                :data:`IGNORE_INDEX` are excluded from the loss.
            return_attention: If True, collect the attention distribution from every
                layer. Costs ``n_layer * B * h * T * T`` values held at once, so it is
                intended for analysis on small batches, not for training.

        Returns:
            ``(logits, loss, attentions)``. ``logits`` has shape ``(B, T, vocab_size)``;
            ``loss`` is a scalar mean cross-entropy or None when no targets were given;
            ``attentions`` is a list of ``n_layer`` tensors of shape ``(B, h, T, T)``
            or None.

        Raises:
            ValueError: If ``T`` exceeds ``config.max_seq_len``.
        """
        B, T = idx.shape
        if T > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {T} exceeds context {self.config.max_seq_len}: the "
                "positional embedding table has no row for positions beyond it"
            )

        pos = torch.arange(T, device=idx.device)
        # (B, T, d) from the token table, plus (T, d) from the position table, which
        # broadcasts across the batch: every sequence uses the same position vectors.
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        attentions: list[torch.Tensor] | None = [] if return_attention else None
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attention)
            if attentions is not None:
                attentions.append(attn)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten batch and time together; cross_entropy averages over every
            # position that is not IGNORE_INDEX.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

        return logits, loss, attentions

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count trainable parameters as PyTorch sees them.

        Args:
            non_embedding: If True, exclude the token and positional embedding tables.

        Returns:
            The parameter count. With weight tying the shared matrix is counted once,
            because ``nn.Module.parameters()`` deduplicates by identity — which is why
            this agrees with :meth:`lma.config.ModelConfig.count_parameters`.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            total -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return total

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Split parameters into decayed and undecayed groups for AdamW.

        Weight decay is applied to the matrices that perform linear mixing, and
        withheld from biases, LayerNorm gains and embeddings. Decaying a LayerNorm gain
        pulls it toward zero and suppresses the activations it is meant to rescale;
        decaying embeddings penalises rare tokens hardest, since they receive gradient
        least often, which is exactly backwards.

        Args:
            weight_decay: Coefficient for the decayed group.

        Returns:
            Two parameter-group dicts ready to pass to ``torch.optim.AdamW``.
        """
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or name.endswith("emb.weight"):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
