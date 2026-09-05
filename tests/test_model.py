"""Structural tests for the Transformer: parameter budget, shapes, weight tying.

The parameter-count test is the important one. ``ModelConfig.count_parameters`` is what
the report quotes and what the architecture was sized against, but it is arithmetic on
hyperparameters, not a measurement of the model. If the two ever disagree, the reported
figure is fiction. This asserts they agree.

Runs under pytest, or directly with ``python tests/test_model.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.config import ModelConfig, TrainConfig  # noqa: E402
from lma.model import GPT  # noqa: E402

# A small configuration, so the tests run in seconds on CPU. Structural properties do
# not depend on scale; the production budget is checked separately below.
TINY = ModelConfig(vocab_size=257, n_layer=3, n_head=4, d_model=64, d_ff=256, max_seq_len=32)


def test_analytic_parameter_count_matches_model() -> None:
    """The config's arithmetic must equal what PyTorch actually allocates."""
    for config in (TINY, ModelConfig(vocab_size=16_000)):
        model = GPT(config)
        predicted = config.count_parameters()
        assert model.num_parameters() == predicted["total"], (
            f"analytic total {predicted['total']:,} != actual "
            f"{model.num_parameters():,} for {config}"
        )
        assert model.num_parameters(non_embedding=True) == predicted["non_embedding"]


def test_production_config_is_within_budget() -> None:
    """The shipped architecture must sit under the assignment's ~25M target."""
    config = ModelConfig(vocab_size=16_000)
    total = config.count_parameters()["total"]
    assert 24_000_000 <= total <= 25_000_000, f"{total:,} parameters is off budget"


def test_weight_tying_shares_one_tensor() -> None:
    """A tied head must be the *same* tensor, not a copy that drifts apart."""
    model = GPT(TINY)
    assert model.lm_head.weight is model.tok_emb.weight

    untied = GPT(ModelConfig(**{**TINY.__dict__, "tie_weights": False}))
    assert untied.lm_head.weight is not untied.tok_emb.weight
    # Untying costs exactly one vocab x d_model matrix.
    assert (
        untied.num_parameters() - GPT(TINY).num_parameters()
        == TINY.vocab_size * TINY.d_model
    )


def test_forward_shapes_and_loss() -> None:
    """Logits are (B, T, vocab); loss is a finite scalar."""
    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, loss, attentions = model(idx, targets=idx)

    assert logits.shape == (2, 16, TINY.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert attentions is None

    # An untrained model over V tokens should sit near ln(V) nats.
    import math

    assert abs(loss.item() - math.log(TINY.vocab_size)) < 1.0


def test_attention_weights_are_causal_distributions() -> None:
    """Returned attention must be row-stochastic and strictly lower-triangular."""
    model = GPT(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 12))
    _, _, attentions = model(idx, return_attention=True)

    assert attentions is not None and len(attentions) == TINY.n_layer
    for layer, attn in enumerate(attentions):
        assert attn.shape == (2, TINY.n_head, 12, 12), f"layer {layer}: {attn.shape}"
        # Every query row is a probability distribution.
        assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5)
        # Nothing above the diagonal: no query attends to a future key.
        upper = torch.triu(torch.ones(12, 12, dtype=torch.bool), diagonal=1)
        assert attn.masked_select(upper).abs().max() == 0.0, f"layer {layer} leaks forward"


def test_rejects_sequences_longer_than_context() -> None:
    """Exceeding the positional table must raise, not silently truncate or crash."""
    model = GPT(TINY).eval()
    too_long = torch.randint(0, TINY.vocab_size, (1, TINY.max_seq_len + 1))
    try:
        model(too_long)
    except ValueError as exc:
        assert "context" in str(exc)
    else:
        raise AssertionError("expected ValueError for over-length sequence")


def test_param_groups_partition_every_parameter() -> None:
    """The AdamW decay split must cover each parameter exactly once."""
    model = GPT(TINY)
    groups = model.param_groups(weight_decay=0.1)
    seen = [p for g in groups for p in g["params"]]

    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(seen) == len(expected), "a parameter is duplicated or missing"
    assert {id(p) for p in seen} == expected

    # Embeddings, biases and LayerNorm gains belong to the undecayed group.
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert id(model.tok_emb.weight) in no_decay_ids
    assert id(model.pos_emb.weight) in no_decay_ids
    assert id(model.ln_f.weight) in no_decay_ids


def test_grad_accum_rejects_indivisible_batch() -> None:
    """A micro-batch that does not divide the token budget must be refused."""
    assert TrainConfig().grad_accum_steps(512) == 2
    try:
        TrainConfig(tokens_per_step=1000, micro_batch_size=32).grad_accum_steps(512)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for indivisible step size")


if __name__ == "__main__":
    torch.manual_seed(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    config = ModelConfig(vocab_size=16_000)
    print()
    for key, value in config.count_parameters().items():
        print(f"  {key:22s} {value:>12,}")
