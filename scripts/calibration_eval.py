"""Calibration: Expected Calibration Error (ECE), Brier score, NLL on the
augmented downstream classifier.

Diversity-injecting synthetic data should produce a classifier whose
probability outputs better reflect actual class likelihoods - a wider
sample distribution gives less overfit logits at decision boundaries.
This is a second axis on which the AttrForge approach can win that does
not depend on macro/worst F1 being significant.

For each (condition, seed) we report:
  - macro F1 on clean test (for context)
  - ECE (Expected Calibration Error) with 10 equal-width bins
  - Brier score (mean squared error of predicted probs vs one-hot truth)
  - negative log-likelihood (NLL) of the true labels

We also report TTA-mean versions (predictions averaged over original + DE
+ FR back-translation).

Outputs:
    experiments/<base>_aggregated/calibration.json
    paper/figures/<base>_calibration.png
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def expected_calibration_error(y_true_idx, y_proba, n_bins=10):
    """Standard ECE: take the argmax confidence per sample, bin by confidence,
    average |bin_acc - bin_conf| weighted by bin size."""
    conf = y_proba.max(axis=1)
    pred = y_proba.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_multiclass(y_true_idx, y_proba):
    """Multiclass Brier: mean over (sample, class) of (p_ij - y_ij)^2."""
    n, k = y_proba.shape
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(n), y_true_idx] = 1.0
    return float(((y_proba - one_hot) ** 2).sum(axis=1).mean())


def nll(y_true_idx, y_proba):
    """Per-sample NLL of the true class probability; tiny clip to avoid log(0)."""
    n = len(y_true_idx)
    p = y_proba[np.arange(n), y_true_idx]
    return float(-np.log(np.clip(p, 1e-9, 1.0)).mean())


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    real_train = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")]

    real_train_texts = [r.text for r in real_train]
    real_train_labels = np.array([r.label for r in real_train])
    test_texts = [r.text for r in real_test]
    test_labels = np.array([r.label for r in real_test])
    labels = sorted(set(test_labels.tolist()))
    test_idx = np.array([labels.index(y) for y in test_labels])

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    bag = defaultdict(list)

    from sklearn.linear_model import LogisticRegression

    def cal_metrics(clf, X, y_idx):
        y_proba = clf.predict_proba(X)
        # Align ordering to our `labels` list.
        col_idx = [list(clf.classes_).index(lbl) for lbl in labels]
        y_proba_aligned = y_proba[:, col_idx]
        ece = expected_calibration_error(y_idx, y_proba_aligned)
        brier = brier_multiclass(y_idx, y_proba_aligned)
        ll = nll(y_idx, y_proba_aligned)
        y_pred_idx = y_proba_aligned.argmax(axis=1)
        acc = float((y_pred_idx == y_idx).mean())
        # macro F1
        f1s = []
        for li, lbl in enumerate(labels):
            tp = int(((y_pred_idx == li) & (y_idx == li)).sum())
            fp = int(((y_pred_idx == li) & (y_idx != li)).sum())
            fn = int(((y_pred_idx != li) & (y_idx == li)).sum())
            if tp + fp == 0 or tp + fn == 0:
                f1s.append(0.0); continue
            p = tp / (tp + fp); r = tp / (tp + fn)
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        return {"ece": ece, "brier": brier, "nll": ll, "acc": acc, "macro_f1": float(np.mean(f1s))}

    # Real-only baseline (no synthetic; deterministic across seeds)
    clf_ro = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf_ro.fit(X_real_train, real_train_labels)
    ro_metrics = cal_metrics(clf_ro, X_test, test_idx)
    print(f"\n=== Real-only baseline ===")
    print(f"  macro F1 = {ro_metrics['macro_f1']:.3f}  ECE = {ro_metrics['ece']:.3f}  "
          f"Brier = {ro_metrics['brier']:.3f}  NLL = {ro_metrics['nll']:.3f}")

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth:
                continue
            texts = [s.text for s in synth]
            slabels = np.array([s.requested_attributes.get("intent", "?") for s in synth])
            X_synth = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            y_tr = np.concatenate([real_train_labels, slabels])
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
            clf.fit(X_tr, y_tr)
            m = cal_metrics(clf, X_test, test_idx)
            m["seed"] = seed
            bag[cond_dir.name].append(m)

    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]
    print(f"\n=== Calibration & probabilistic metrics (mean +/- std over seeds) ===")
    print(f"{'condition':<18} {'macro F1':<14} {'ECE (lower)':<14} {'Brier (lower)':<14} {'NLL (lower)':<14}")
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        def fmt(key):
            vals = [r[key] for r in rows]
            return f"{statistics.mean(vals):.3f}+-{statistics.stdev(vals) if len(vals)>1 else 0:.3f}"
        print(f"{c:<18} {fmt('macro_f1'):<14} {fmt('ece'):<14} {fmt('brier'):<14} {fmt('nll'):<14}")

    print(f"\n=== Paired stats full_attrforge - full_classic (negative diff = AF better) ===")
    try:
        from scipy import stats as st
        for metric, lower_better in [("ece", True), ("brier", True), ("nll", True), ("macro_f1", False)]:
            fc = [r[metric] for r in bag.get("full_classic", [])]
            fa = [r[metric] for r in bag.get("full_attrforge", [])]
            if not fc or not fa:
                continue
            diffs = [a-b for a,b in zip(fa, fc)]
            t, p_t = st.ttest_rel(fa, fc)
            try:
                _, p_w = st.wilcoxon(fa, fc, zero_method="zsplit")
            except ValueError:
                p_w = float("nan")
            md = statistics.mean(diffs); sd = statistics.stdev(diffs) if len(diffs) > 1 else 0
            tag = "(AF better if negative)" if lower_better else "(AF better if positive)"
            print(f"  {metric:<10} {tag:<28} diff = {md:+.4f}+-{sd:.4f}  paired-t p={p_t:.3f}  Wilcoxon p={p_w:.3f}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.json").write_text(json.dumps(
        {"real_only": ro_metrics, "augmented": {c: bag[c] for c in conds if c in bag}}, indent=2
    ), encoding="utf-8")
    print(f"\nSaved: {out_dir}/calibration.json")


if __name__ == "__main__":
    main()
