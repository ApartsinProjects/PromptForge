"""Downstream evaluator: trains a classifier on synthetic data, tests on held-out real.

This is the project description's RQ4 protocol made executable. For each
baseline run, we train the same classifier on its produced synthetic data
and evaluate on a fixed held-out real test set. The training-data factor
is the only thing that changes across conditions, so accuracy and macro-F1
differences attribute to the *quality of the synthetic data*, not the
model.

Default classifier: TF-IDF features (1-2 grams) + logistic regression.
Chosen because:

1. It is interpretable: low-capacity, no representation drift, easy to
   confirm that gains do not come from feature memorization.
2. It fits in ~1 second on the dataset sizes here, so it can be run inside
   the loop as a continuous RQ4 signal, not just at the end.
3. It exposes a clean per-class F1 breakdown which the project
   description's "rare/hard subset robustness" objective needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from synsmith.schema import RealExample, SyntheticSample


@dataclass
class DownstreamConfig:
    ngram_range: tuple[int, int] = (1, 2)
    C: float = 1.0
    class_weight: str | None = "balanced"
    min_df: int = 1
    seed: int = 17


class DownstreamResult(BaseModel):
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    per_class_support: dict[str, int] = Field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    n_train: int
    n_test: int
    label_set: list[str] = Field(default_factory=list)
    notes: str = ""


class DownstreamEvaluator:
    """Train on synthetic, test on real. RQ4 protocol."""

    def __init__(self, config: DownstreamConfig | None = None) -> None:
        self.config = config or DownstreamConfig()

    def evaluate(
        self,
        synthetic: Iterable[SyntheticSample],
        test_real: Iterable[RealExample],
        *,
        label_attribute: str = "intent",
    ) -> DownstreamResult:
        synth = [
            s
            for s in synthetic
            if s.requested_attributes.get(label_attribute) and s.text
        ]
        test = [t for t in test_real if t.label and t.text]

        if not synth or not test:
            return DownstreamResult(
                accuracy=0.0,
                macro_f1=0.0,
                n_train=len(synth),
                n_test=len(test),
                notes="not enough samples to train or test",
            )

        labels = sorted({s.requested_attributes[label_attribute] for s in synth} |
                       {t.label for t in test})

        X_train_text = [s.text for s in synth]
        y_train = [s.requested_attributes[label_attribute] for s in synth]
        X_test_text = [t.text for t in test]
        y_test = [t.label for t in test]

        vec = TfidfVectorizer(
            ngram_range=self.config.ngram_range, min_df=self.config.min_df
        )
        X_train = vec.fit_transform(X_train_text)
        X_test = vec.transform(X_test_text)

        try:
            clf = LogisticRegression(
                max_iter=2000,
                C=self.config.C,
                class_weight=self.config.class_weight,
                random_state=self.config.seed,
            )
            clf.fit(X_train, y_train)
        except Exception as exc:
            return DownstreamResult(
                accuracy=0.0,
                macro_f1=0.0,
                n_train=len(synth),
                n_test=len(test),
                label_set=labels,
                notes=f"classifier failed to fit: {exc}",
            )

        pred = clf.predict(X_test)
        return self._score(np.asarray(y_test), pred, labels, len(synth), len(test))

    def _score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: list[str],
        n_train: int,
        n_test: int,
    ) -> DownstreamResult:
        per_class_f1: dict[str, float] = {}
        per_class_support: dict[str, int] = {}
        for lbl in labels:
            tp = int(((y_pred == lbl) & (y_true == lbl)).sum())
            fp = int(((y_pred == lbl) & (y_true != lbl)).sum())
            fn = int(((y_pred != lbl) & (y_true == lbl)).sum())
            support = int((y_true == lbl).sum())
            per_class_support[lbl] = support
            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_class_f1[lbl] = f1

        macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
        accuracy = float((y_pred == y_true).mean())

        confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in labels} for a in labels}
        for t, p in zip(y_true.tolist(), y_pred.tolist()):
            if t in confusion and p in confusion[t]:
                confusion[t][p] += 1

        return DownstreamResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            per_class_f1=per_class_f1,
            per_class_support=per_class_support,
            confusion=confusion,
            n_train=n_train,
            n_test=n_test,
            label_set=labels,
        )
