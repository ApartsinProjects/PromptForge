"""Coverage Hole Finder: density-ratio coverage signal (vanilla GAN derivation).

A trained vanilla GAN discriminator implicitly estimates the density ratio
``p_real(x) / (p_real(x) + p_synth(x))``. We construct that estimator
explicitly with a logistic regression on TF-IDF features: train it to
classify real vs synthetic, then for each *real* sample compute the
predicted probability of being real. Real samples the classifier most
confidently calls real are the modes of the real distribution that the
synthetic distribution has not covered.

The top-K most-uncovered real exemplars become *few-shot hints* in the
next generator prompt: "Here are real examples whose style and content
the current synthetic batch is failing to reproduce. Generate more like
these."

This converts a fuzzy "missing modes" feedback into a concrete exemplar
set that the prompt updater can ground its rewrite in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from synsmith.schema import RealExample, SyntheticSample


@dataclass
class CoverageHoleConfig:
    top_k: int = 5
    min_real: int = 5
    min_synth: int = 5
    ngram_range: tuple[int, int] = (1, 2)


class CoverageHole(BaseModel):
    text: str
    label: str | None = None
    p_real: float = Field(..., description="Classifier's probability that this is real.")


class CoverageHoleResult(BaseModel):
    holes: list[CoverageHole] = Field(default_factory=list)
    classifier_auroc: float = 0.5
    notes: str = ""


class CoverageHoleFinder:
    """Density-ratio coverage analysis with a tiny LR classifier."""

    def __init__(self, config: CoverageHoleConfig | None = None) -> None:
        self.config = config or CoverageHoleConfig()

    def find(
        self,
        real: list[RealExample],
        synthetic: list[SyntheticSample],
    ) -> CoverageHoleResult:
        if len(real) < self.config.min_real or len(synthetic) < self.config.min_synth:
            return CoverageHoleResult(
                holes=[], classifier_auroc=0.5, notes="not enough samples to fit"
            )

        real_texts = [r.text for r in real]
        synth_texts = [s.text for s in synthetic]
        all_texts = real_texts + synth_texts
        y = np.concatenate(
            [np.ones(len(real_texts)), np.zeros(len(synth_texts))]
        )

        vec = TfidfVectorizer(ngram_range=self.config.ngram_range, min_df=1)
        X = vec.fit_transform(all_texts)

        try:
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(X, y)
        except Exception as exc:
            return CoverageHoleResult(holes=[], classifier_auroc=0.5, notes=str(exc))

        p_real_all = clf.predict_proba(X)[:, 1]
        real_p = p_real_all[: len(real_texts)]

        # AUROC ~ how well the classifier separates real from synthetic.
        auroc = self._auroc(y, p_real_all)

        order = np.argsort(-real_p)[: self.config.top_k]
        holes = [
            CoverageHole(
                text=real[int(i)].text,
                label=real[int(i)].label,
                p_real=float(real_p[int(i)]),
            )
            for i in order
        ]
        return CoverageHoleResult(holes=holes, classifier_auroc=auroc)

    @staticmethod
    def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Pairwise AUROC; ties broken at 0.5. Avoids importing sklearn.metrics."""
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        wins = (pos[:, None] > neg[None, :]).sum()
        ties = (pos[:, None] == neg[None, :]).sum()
        total = len(pos) * len(neg)
        return float((wins + 0.5 * ties) / total)

    def render_for_prompt(self, result: CoverageHoleResult) -> str:
        if not result.holes:
            return ""
        bullets = "\n".join(
            f"- (label={h.label or '?'}, p_real={h.p_real:.2f}) {h.text}"
            for h in result.holes
        )
        return (
            "The synthetic distribution is failing to cover real examples "
            "like these. The next batch should produce more in this style:\n"
            + bullets
        )
