"""Empirical proof that the model cannot see the future.

The assignment asks for this specifically: *"verify empirically that the model cannot
see the future — for example, show that changing token t+1 does not change the logits
at position t."* A causal-masking bug is silent. The model still trains, the loss still
falls, and the result is a model that has been allowed to cheat — its perplexity is
meaningless and every downstream number is contaminated. Nothing else in the pipeline
would catch it, so it is checked here directly.

**Two different strictnesses are used, for a reason.**

:func:`test_future_token_cannot_change_past_logits` demands *bit-exact* equality. It
perturbs tokens inside a sequence of fixed length, so both runs execute identical tensor
shapes and therefore identical kernels and reduction orders. A masked position's softmax
weight is ``exp(-inf)``, which is precisely 0.0, so its value vector contributes exactly
0.0 to the weighted sum. Nothing may move at all, and any tolerance here would also
tolerate a genuine leak.

:func:`test_prefix_gives_same_logits_as_full_sequence` compares runs of *different*
sequence length, which select different BLAS kernels with different summation orders.
Measured drift there is 1.5-2.7e-7 against a float32 epsilon of 1.19e-7 — round-off, and
confirmed as such: at length 20, where the shapes coincide, drift is exactly 0.0, and
recomputing the same model in float64 drops drift to 3e-16, float64's own epsilon. A real
leak is five orders of magnitude larger; :func:`test_the_prefix_check_has_teeth` measures
one deliberately, at 1.9e-2 to 3.5e-1, to prove the bound below actually discriminates.

Runs under pytest, or directly with ``python tests/test_causal.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lma.config import ModelConfig  # noqa: E402
from lma.model import GPT  # noqa: E402

TINY = ModelConfig(
    vocab_size=257, n_layer=3, n_head=4, d_model=64, d_ff=256, max_seq_len=32,
    dropout=0.0, attn_dropout=0.0, embd_dropout=0.0,
)

# Round-off ceiling for comparisons across differing tensor shapes. Two orders of
# magnitude above the observed 2.7e-7, three below the smallest genuine leak measured.
ROUNDOFF_TOLERANCE = 1e-5


def _model() -> GPT:
    """Build a deterministic model in eval mode, with dropout disabled."""
    torch.manual_seed(1337)
    return GPT(TINY).eval()


@torch.no_grad()
def test_future_token_cannot_change_past_logits() -> None:
    """Editing token t+1 must leave every logit at positions <= t bit-identical."""
    model = _model()
    B, T = 4, 24
    idx = torch.randint(0, TINY.vocab_size, (B, T))

    baseline, _, _ = model(idx)

    for t in (0, 5, 11, T - 2):
        edited = idx.clone()
        # Replace everything from t+1 onwards, guaranteed to differ from the originals
        # so the test cannot pass by accident.
        edited[:, t + 1:] = (edited[:, t + 1:] + 1) % TINY.vocab_size
        assert not torch.equal(idx, edited), "the perturbation changed nothing"

        perturbed, _, _ = model(edited)

        past = slice(0, t + 1)
        drift = (baseline[:, past] - perturbed[:, past]).abs().max().item()
        # Exact: the shapes match, so only a genuine leak could move anything.
        assert drift == 0.0, (
            f"logits at positions 0..{t} moved by {drift:.3e} when token {t + 1} "
            "changed - the causal mask leaks"
        )

        # The converse, so the test is not vacuous: it must also fail on a model that
        # ignores its input entirely. Positions at and after the edit must react.
        if t + 1 < T:
            reaction = (baseline[:, t + 1:] - perturbed[:, t + 1:]).abs().max().item()
            assert reaction > 0.0, "logits after the edit did not react at all"


@torch.no_grad()
def test_prefix_gives_same_logits_as_full_sequence() -> None:
    """Scoring a prefix alone must match scoring it inside a longer sequence.

    This is the property that makes autoregressive generation valid: the logits used to
    sample token t+1 must not depend on tokens the model has not emitted yet. Compared
    at round-off tolerance rather than exactly, because the two runs use different
    sequence lengths — see the module docstring.
    """
    model = _model()
    idx = torch.randint(0, TINY.vocab_size, (2, 20))

    full, _, _ = model(idx)
    for k in (1, 7, 19, 20):
        prefix, _, _ = model(idx[:, :k])
        drift = (full[:, :k] - prefix).abs().max().item()
        assert drift < ROUNDOFF_TOLERANCE, (
            f"prefix of length {k} scored differently by {drift:.3e}, above the "
            f"{ROUNDOFF_TOLERANCE:.0e} round-off bound"
        )
    # At full length the shapes coincide, so this one must be exact.
    identical, _, _ = model(idx)
    assert (full - identical).abs().max().item() == 0.0


@torch.no_grad()
def test_the_prefix_check_has_teeth() -> None:
    """A model with the causal mask removed must fail the prefix check loudly.

    Without this, a tolerance of 1e-5 is just an assertion that some number is small.
    Here the mask is deliberately disabled and the resulting drift measured, confirming
    that a real leak sits far above the bound the previous test applies.
    """
    model = _model()
    idx = torch.randint(0, TINY.vocab_size, (2, 20))

    for block in model.blocks:
        block.attn.causal_mask.fill_(1.0)  # every position may attend everywhere

    leaky_full, _, _ = model(idx)
    worst = 0.0
    for k in (1, 7, 19):
        leaky_prefix, _, _ = model(idx[:, :k])
        worst = max(worst, (leaky_full[:, :k] - leaky_prefix).abs().max().item())

    assert worst > 100 * ROUNDOFF_TOLERANCE, (
        f"an unmasked model drifted only {worst:.3e}; the round-off bound would not "
        "distinguish it from a correct model, so the test above proves nothing"
    )


@torch.no_grad()
def test_attention_mass_is_confined_to_the_past() -> None:
    """Every attention row must place all of its mass on positions <= the query."""
    model = _model()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    _, _, attentions = model(idx, return_attention=True)

    future = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    for layer, attn in enumerate(attentions):
        leaked = attn.masked_select(future).abs().max().item()
        assert leaked == 0.0, f"layer {layer} puts {leaked:.3e} weight on the future"
        # Row t spreads exactly 1.0 over its t+1 permitted keys.
        assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-6)


@torch.no_grad()
def test_shuffling_positions_changes_output() -> None:
    """Positional information must actually reach the model.

    Without positional embeddings, self-attention is permutation invariant and a
    shuffled sequence would score identically at the shuffled positions. This guards
    the embedding addition in ``GPT.forward``, and is the control the optional
    no-positional-embedding ablation would be measured against.
    """
    model = _model()
    idx = torch.randint(0, TINY.vocab_size, (1, 12))
    shuffled = idx[:, torch.randperm(12)]

    a, _, _ = model(idx)
    b, _, _ = model(shuffled)
    assert (a - b).abs().max().item() > 0.0, "output is permutation invariant"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
