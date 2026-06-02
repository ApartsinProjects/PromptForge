"""Unit tests for the Manifold-Entropy critic.

The critic is deterministic (no LLM call). Tests use the TF-IDF backend
to stay fully offline; the sentence-transformer backend is exercised by
the smoke test only when the model is available locally.
"""
from __future__ import annotations

import pytest

from synsmith.critics.manifold_entropy import (
    ManifoldEntropy,
    ManifoldEntropyConfig,
)
from synsmith.schema import SyntheticSample


def _make(text: str, idx: int) -> SyntheticSample:
    return SyntheticSample(
        sample_id=f"s{idx}",
        text=text,
        requested_attributes={"intent": "general_question"},
        generated_attributes={"intent": "general_question"},
        prompt_version=1,
        iteration=0,
    )


def test_homogeneous_batch_low_effective_rank():
    """Same text repeated should produce effective_rank close to 1."""
    batch = [_make("Please reset my password.", i) for i in range(8)]
    critic = ManifoldEntropy(ManifoldEntropyConfig(use_embeddings=False))
    result = critic.score(batch)
    assert result.n_samples == 8
    assert result.effective_rank < 1.5  # Near-rank-1 batch.
    assert result.manifold_entropy < 0.5  # Low entropy.


def test_diverse_batch_high_effective_rank():
    """A genuinely diverse batch should have effective_rank > 4."""
    batch = [
        _make("Please reset my password to access my account.", 0),
        _make("I want a full refund for the broken shipment.", 1),
        _make("This is unacceptable, no one ever responds!", 2),
        _make("Hi, what are your customer support hours?", 3),
        _make("My card payment was rejected at checkout.", 4),
        _make("Can you confirm whether my address is on file?", 5),
        _make("The product arrived damaged and missing parts.", 6),
        _make("Where can I find the shipping policy details?", 7),
    ]
    critic = ManifoldEntropy(ManifoldEntropyConfig(use_embeddings=False))
    result = critic.score(batch)
    assert result.n_samples == 8
    assert result.effective_rank > 4.0
    assert result.eigenvalue_decay > 0.1


def test_empty_batch_returns_zero():
    """No samples, return a sane zero-shaped result."""
    critic = ManifoldEntropy()
    result = critic.score([])
    assert result.n_samples == 0
    assert result.manifold_entropy == 0.0
    assert result.effective_rank == 1.0


def test_single_sample_returns_zero():
    """One sample, no pair to compute, return zero."""
    critic = ManifoldEntropy()
    result = critic.score([_make("only one", 0)])
    assert result.n_samples == 1
    assert result.manifold_entropy == 0.0


def test_render_for_prompt_complaints_on_collapse():
    """A collapsed batch should produce a non-empty complaint string."""
    batch = [_make("Please reset my password.", i) for i in range(8)]
    critic = ManifoldEntropy(ManifoldEntropyConfig(use_embeddings=False))
    result = critic.score(batch)
    text = critic.render_for_prompt(result)
    assert text  # Non-empty complaint.
    assert "single direction" in text or "effective rank" in text


def test_render_for_prompt_quiet_on_diverse():
    """A diverse batch should produce no complaint."""
    batch = [
        _make("Please reset my password.", 0),
        _make("I want a full refund.", 1),
        _make("This is unacceptable!", 2),
        _make("What are your hours?", 3),
        _make("My card was rejected.", 4),
        _make("Can you confirm my address?", 5),
        _make("The product is damaged.", 6),
        _make("Where is the shipping policy?", 7),
    ]
    critic = ManifoldEntropy(ManifoldEntropyConfig(use_embeddings=False))
    result = critic.score(batch)
    text = critic.render_for_prompt(result)
    # Either empty or a mild remark; we just assert no "single direction" alarm.
    assert "single direction" not in text
