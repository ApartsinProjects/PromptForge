"""Manifold-Entropy critic.

The Manifold-Entropy critic measures the geometric "spread" of the synthetic
batch on a fixed sentence-transformer embedding manifold. It generalizes the
discriminator into a feature embedder and reports the entropy of the
sample-similarity-kernel eigenvalues, a quantity that drops sharply when the
generator emits paraphrastic clusters around a small set of anchors.

The metric is a continuous reformulation of the Coverage Hole Finder's role
on the diversity axis: where Coverage Hole Finder asks "which REAL exemplars
is the synthetic batch failing to cover?", Manifold-Entropy asks "is the
synthetic batch itself spread thinly enough to plausibly cover anything?".
The two are complementary; both are deterministic (no LLM call) and cheap.

Inspired by the manifold-entropy reformulation of GAN discriminators in
arXiv:2208.12055, with the discriminator replaced by a fixed sentence-
transformer encoder (we treat the encoder as the "shared" feature space the
adversarial training would otherwise have to learn).

API:

    from synsmith.critics.manifold_entropy import (
        ManifoldEntropy, ManifoldEntropyConfig, ManifoldEntropyResult
    )

    critic = ManifoldEntropy(ManifoldEntropyConfig(use_embeddings=True))
    result = critic.score(batch)
    print(result.manifold_entropy, result.effective_rank)

Returns:

    manifold_entropy : float
        Shannon entropy (nats) of the normalized eigenvalue spectrum of the
        sample-similarity kernel. Higher means the batch occupies more of
        the embedding manifold.
    effective_rank : float
        exp(manifold_entropy), the participation-ratio equivalent of "how
        many distinct directions the batch spans". Comparable to Vendi
        score (which is exp(entropy) of the Gram-matrix eigenvalues
        divided by n); Vendi uses the trace-normalization, while we use
        the per-sample-normalized similarity kernel so the scale lives in
        [1, n] regardless of embedding norm.
    eigenvalue_decay : float
        Ratio of the second eigenvalue to the first. Close to 1.0 means
        the batch is genuinely high-rank; close to 0.0 means the batch is
        dominated by a single direction (collapse).
    n_samples : int
        Batch size; 0 if the batch was empty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from synsmith.schema import SyntheticSample


@dataclass
class ManifoldEntropyConfig:
    use_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    ridge: float = 1e-6


class ManifoldEntropyResult(BaseModel):
    manifold_entropy: float
    effective_rank: float
    eigenvalue_decay: float
    n_samples: int


class ManifoldEntropy:
    """Sentence-transformer-backed manifold-entropy critic.

    Loads the encoder lazily on first call so importing this module is
    cheap; pass `use_embeddings=False` to fall back to TF-IDF (1-2 grams)
    when sentence-transformers is not installed or a fully deterministic
    no-network result is required.
    """

    def __init__(self, config: ManifoldEntropyConfig | None = None) -> None:
        self.config = config or ManifoldEntropyConfig()
        self._embedder = None

    def _featurize(self, texts: list[str]) -> np.ndarray:
        if self.config.use_embeddings:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer  # noqa: WPS433

                self._embedder = SentenceTransformer(
                    self.config.embedding_model
                )
            return self._embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
        # TF-IDF fallback: bigrams, balanced norm.
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(texts).toarray()
        # Row-normalize so the kernel scale is comparable.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def score(self, batch: list[SyntheticSample]) -> ManifoldEntropyResult:
        if len(batch) < 2:
            return ManifoldEntropyResult(
                manifold_entropy=0.0,
                effective_rank=1.0,
                eigenvalue_decay=0.0,
                n_samples=len(batch),
            )
        texts = [s.text for s in batch]
        X = self._featurize(texts)
        n = X.shape[0]
        # Sample-similarity kernel (n x n). Ridge keeps the spectrum well-
        # conditioned when batches contain near-duplicates.
        K = X @ X.T + self.config.ridge * np.eye(n)
        # Normalize so eigenvalues sum to 1 (probability simplex).
        eigvals = np.linalg.eigvalsh(K)
        eigvals = eigvals[eigvals > 1e-12]
        if eigvals.size == 0:
            return ManifoldEntropyResult(
                manifold_entropy=0.0,
                effective_rank=1.0,
                eigenvalue_decay=0.0,
                n_samples=n,
            )
        eigvals = np.sort(eigvals)[::-1]  # descending
        p = eigvals / eigvals.sum()
        entropy = float(-(p * np.log(p)).sum())
        decay = float(p[1] / p[0]) if p.size > 1 else 0.0
        return ManifoldEntropyResult(
            manifold_entropy=entropy,
            effective_rank=float(np.exp(entropy)),
            eigenvalue_decay=decay,
            n_samples=n,
        )

    def render_for_prompt(self, result: ManifoldEntropyResult) -> str:
        """Render a structured-feedback line the updater can consume."""
        if result.n_samples < 2:
            return ""
        if result.eigenvalue_decay < 0.1:
            return (
                "The synthetic batch's embedding distribution is dominated by "
                "a single direction (eigenvalue_decay = "
                f"{result.eigenvalue_decay:.2f}). Vary the topical center, "
                "the discourse function, and the sentence-shape distribution "
                "across samples."
            )
        if result.effective_rank < 0.5 * result.n_samples:
            return (
                "The synthetic batch's effective rank "
                f"({result.effective_rank:.1f} of {result.n_samples}) is "
                "well below the sample count, meaning many samples lie close "
                "to one another on the embedding manifold. Spread the "
                "samples by varying surface form and topic."
            )
        return ""
