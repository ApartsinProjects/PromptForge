"""Mode-Seeking Diversity score (MSGAN analog).

MSGAN's identity: if two different latent codes produce similar outputs,
the generator has collapsed. For AttrForge, the "latent code" is the
target attribute vector, and the "output" is the synthetic text.

For every pair of samples in the batch:

    ms(i, j) = embedding_distance(text_i, text_j) / max(1, hamming(target_i, target_j))

A healthy generator preserves a ratio close to (or above) the same ratio
computed on the real set when projected through the same encoder. A
collapsing generator drives the ratio toward zero: changing the
attributes does not change the text.

This is a fully deterministic, model-free (no LLM judge) signal that
catches the subtle failure mode where coverage looks fine but the
generator stopped *listening* to the attribute vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from attrforge.schema import SyntheticSample


@dataclass
class ModeSeekingConfig:
    use_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    max_pairs: int = 2000


class ModeSeekingResult(BaseModel):
    mode_seeking_ratio: float
    n_pairs: int
    text_distance_mean: float
    target_distance_mean: float
    attribute_sensitivity: dict[str, float] = {}


class ModeSeeking:
    """Compute MSGAN-style attribute-responsiveness ratio for a batch."""

    def __init__(self, config: ModeSeekingConfig | None = None) -> None:
        self.config = config or ModeSeekingConfig()
        self._embedder = None

    def score(self, batch: list[SyntheticSample]) -> ModeSeekingResult:
        if len(batch) < 2:
            return ModeSeekingResult(
                mode_seeking_ratio=0.0,
                n_pairs=0,
                text_distance_mean=0.0,
                target_distance_mean=0.0,
            )

        texts = [s.text for s in batch]
        targets = [s.requested_attributes for s in batch]

        emb = self._encode(texts)
        # cosine distance = 1 - cosine similarity
        sim = emb @ emb.T
        np.fill_diagonal(sim, 1.0)
        n = len(texts)

        pair_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_idx.append((i, j))
        if len(pair_idx) > self.config.max_pairs:
            rng = np.random.default_rng(0)
            chosen = rng.choice(len(pair_idx), size=self.config.max_pairs, replace=False)
            pair_idx = [pair_idx[k] for k in chosen]

        text_dists = []
        target_dists = []
        for i, j in pair_idx:
            text_dists.append(float(1.0 - sim[i, j]))
            target_dists.append(self._hamming(targets[i], targets[j]))

        text_arr = np.asarray(text_dists)
        targ_arr = np.asarray(target_dists)
        nonzero = targ_arr > 0
        if nonzero.sum() == 0:
            ratio = 0.0
        else:
            ratio = float((text_arr[nonzero] / targ_arr[nonzero]).mean())

        sensitivity = self._per_attribute_sensitivity(batch, emb)

        return ModeSeekingResult(
            mode_seeking_ratio=ratio,
            n_pairs=len(pair_idx),
            text_distance_mean=float(text_arr.mean()),
            target_distance_mean=float(targ_arr.mean()),
            attribute_sensitivity=sensitivity,
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.config.use_embeddings:
            try:
                if self._embedder is None:
                    from sentence_transformers import SentenceTransformer

                    self._embedder = SentenceTransformer(self.config.embedding_model)
                return np.asarray(
                    self._embedder.encode(texts, normalize_embeddings=True)
                )
            except Exception:
                pass
        # TF-IDF fallback, L2-normalized
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts).toarray()
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return vec / norm

    @staticmethod
    def _hamming(a: dict, b: dict) -> int:
        keys = set(a) | set(b)
        return sum(1 for k in keys if a.get(k) != b.get(k))

    def _per_attribute_sensitivity(
        self, batch: list[SyntheticSample], emb: np.ndarray
    ) -> dict[str, float]:
        """For each attribute, mean text-distance between pairs that differ in only that attribute."""
        targets = [s.requested_attributes for s in batch]
        all_attrs = set().union(*targets) if targets else set()
        per_attr: dict[str, list[float]] = {a: [] for a in all_attrs}
        n = len(batch)
        for i in range(n):
            for j in range(i + 1, n):
                diff = [a for a in all_attrs if targets[i].get(a) != targets[j].get(a)]
                if len(diff) == 1:
                    dist = float(1.0 - (emb[i] @ emb[j]))
                    per_attr[diff[0]].append(dist)
        return {a: (float(np.mean(v)) if v else 0.0) for a, v in per_attr.items()}
