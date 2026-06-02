"""Test-time augmentation (TTA) via back-translation: a new comparison condition.

For every test item, generate two back-translation paraphrases (already
cached in real_test_paraphrased.jsonl as DE-pivot and FR-pivot variants).
At inference time, predict on the original item AND the two paraphrases,
then aggregate the three predictions into one per item (we evaluate two
aggregation rules: majority vote and softmax-averaged logits).

If AttrForge's higher lexical diversity produces a classifier whose
predictions are stable across paraphrases (the surface-invariance result
in robustness.json), AttrForge should gain MORE from TTA than full_classic
(whose keyword-driven predictions disagree across paraphrases, so TTA
introduces noise).

Reports per condition:
  - macro F1 single (the v1 augmentation number, included for comparison)
  - macro F1 TTA majority-vote
  - macro F1 TTA mean-logits
  - worst-class F1 for each
  - paired-t for full_attrforge vs full_classic on TTA-mean F1

Outputs:
  experiments/<base>_aggregated/tta.json
  paper/figures/<base>_tta.png
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def predict_with_proba(clf, X):
    """Return (y_pred, y_proba). y_proba is (n, k) with columns aligned to clf.classes_."""
    return clf.predict(X), clf.predict_proba(X)


def aggregate_majority(per_item_classes):
    """Take majority vote over a list of [orig_pred, para1_pred, para2_pred] strings."""
    return [Counter(votes).most_common(1)[0][0] for votes in per_item_classes]


def aggregate_mean_logits(per_item_proba_lists, classes):
    """Average per-class probability across [orig_proba, para1_proba, para2_proba]
    and take the argmax to produce one prediction per item."""
    out = []
    for per_item in per_item_proba_lists:
        mean = np.mean(per_item, axis=0)
        out.append(classes[int(np.argmax(mean))])
    return out


def macro_and_worst_f1(y_true, y_pred, labels):
    f1s = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_true == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_true != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_true == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s[lbl] = 0.0; continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s[lbl] = 2 * p * r / (p + r) if (p + r) else 0.0
    return float(np.mean(list(f1s.values()))), float(min(f1s.values())), f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    real_train = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")]
    para_test = [json.loads(line) for line in (REPO / "experiments/_splits/real_test_paraphrased.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    # Group paraphrases by origin_idx so we get exactly the DE and FR versions for each test item.
    para_by_idx: dict[int, list[dict]] = defaultdict(list)
    for r in para_test:
        para_by_idx[r["_origin_idx"]].append(r)
    n_items = len(real_test)
    para_per_item = [para_by_idx[i] for i in range(n_items)]
    n_paraphrases = max(len(v) for v in para_per_item)

    print(f"Loaded {n_items} clean test items + {n_paraphrases} paraphrases each.")

    real_train_texts = [r.text for r in real_train]
    real_train_labels = np.array([r.label for r in real_train])
    clean_texts = [r.text for r in real_test]
    clean_labels = np.array([r.label for r in real_test])
    labels = sorted(set(clean_labels.tolist()))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_clean = enc.encode(clean_texts, normalize_embeddings=True, show_progress_bar=False)
    # Per-item paraphrase encodings: shape (n_items, n_paraphrases, dim)
    X_para_per_item = [
        enc.encode([p["text"] for p in pp], normalize_embeddings=True, show_progress_bar=False)
        if pp else np.empty((0, X_clean.shape[1]))
        for pp in para_per_item
    ]

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # bag[condition] -> list per seed with macro/worst for single, majority, mean
    bag = defaultdict(list)

    from sklearn.linear_model import LogisticRegression

    def eval_single_and_tta(X_tr, y_tr):
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
        clf.fit(X_tr, y_tr)
        classes = list(clf.classes_)
        # Single prediction on clean test
        y_pred_single, y_proba_clean = predict_with_proba(clf, X_clean)
        single_macro, single_worst, _ = macro_and_worst_f1(clean_labels, y_pred_single, labels)

        # TTA: gather paraphrase predictions per item
        per_item_classes: list[list[str]] = []
        per_item_proba: list[list[np.ndarray]] = []
        for i in range(n_items):
            X_par = X_para_per_item[i]
            if len(X_par) == 0:
                per_item_classes.append([y_pred_single[i]])
                per_item_proba.append([y_proba_clean[i]])
                continue
            y_par, p_par = predict_with_proba(clf, X_par)
            classes_for_item = [y_pred_single[i]] + list(y_par)
            probas_for_item = [y_proba_clean[i]] + list(p_par)
            per_item_classes.append(classes_for_item)
            per_item_proba.append(probas_for_item)

        y_pred_maj = np.array(aggregate_majority(per_item_classes))
        y_pred_mean = np.array(aggregate_mean_logits(per_item_proba, classes))
        maj_macro, maj_worst, _ = macro_and_worst_f1(clean_labels, y_pred_maj, labels)
        mean_macro, mean_worst, _ = macro_and_worst_f1(clean_labels, y_pred_mean, labels)
        return {
            "single_macro": single_macro, "single_worst": single_worst,
            "maj_macro": maj_macro,       "maj_worst": maj_worst,
            "mean_macro": mean_macro,     "mean_worst": mean_worst,
        }

    # Real-only (no synthetic): only one classifier (no per-seed variance because
    # all real_train items used regardless of seed)
    print("\n=== Real-only baseline ===")
    r = eval_single_and_tta(X_real_train, real_train_labels)
    print(f"  real-only: single macro={r['single_macro']:.3f} -> majority macro={r['maj_macro']:.3f} -> mean macro={r['mean_macro']:.3f}")
    print(f"             single worst={r['single_worst']:.3f} -> majority worst={r['maj_worst']:.3f} -> mean worst={r['mean_worst']:.3f}")
    real_only_result = r

    print("\n=== Augmentation conditions ===")
    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth:
                continue
            synth_texts = [s.text for s in synth]
            synth_labels = np.array([s.requested_attributes.get("intent", "?") for s in synth])
            X_synth = enc.encode(synth_texts, normalize_embeddings=True, show_progress_bar=False)
            X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            y_tr = np.concatenate([real_train_labels, synth_labels])
            res = eval_single_and_tta(X_tr, y_tr)
            res["seed"] = seed
            bag[cond_dir.name].append(res)

    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]

    print(f"\n=== Summary (mean +/- std over seeds) ===")
    print(f"{'condition':<18} {'single_macro':<14} {'tta_majority':<14} {'tta_mean':<14} {'mean - single':<14}")
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        s = [r["single_macro"] for r in rows]
        j = [r["maj_macro"]    for r in rows]
        m = [r["mean_macro"]   for r in rows]
        gain = [a-b for a,b in zip(m, s)]
        def fmt(v): return f"{statistics.mean(v):.3f}+-{statistics.stdev(v) if len(v)>1 else 0:.3f}"
        print(f"{c:<18} {fmt(s):<14} {fmt(j):<14} {fmt(m):<14} {fmt(gain):<14}")

    print(f"\n=== Worst-class summary (mean +/- std over seeds) ===")
    print(f"{'condition':<18} {'single_worst':<14} {'tta_majority':<14} {'tta_mean':<14} {'gain':<14}")
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        s = [r["single_worst"] for r in rows]
        j = [r["maj_worst"]    for r in rows]
        m = [r["mean_worst"]   for r in rows]
        gain = [a-b for a,b in zip(m, s)]
        def fmt(v): return f"{statistics.mean(v):.3f}+-{statistics.stdev(v) if len(v)>1 else 0:.3f}"
        print(f"{c:<18} {fmt(s):<14} {fmt(j):<14} {fmt(m):<14} {fmt(gain):<14}")

    print(f"\n=== Paired stats full_attrforge - full_classic on each TTA metric ===")
    try:
        from scipy import stats as st
        for metric in ("single_macro", "maj_macro", "mean_macro",
                       "single_worst", "maj_worst", "mean_worst"):
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
            print(f"  {metric:<18} FA-FC diff={md:+.3f}+-{sd:.3f}  paired-t p={p_t:.3f}  Wilcoxon p={p_w:.3f}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "real_only": real_only_result,
        "augmented": {c: bag[c] for c in conds if c in bag},
    }
    (out_dir / "tta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/tta.json")

    # Plot: single vs TTA-mean for each condition (macro and worst)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    iter_conds = [c for c in conds if c in bag]
    xs = list(range(len(iter_conds)))

    for ax, (key, title) in zip(axes, [("macro", "Macro F1"), ("worst", "Worst-class F1")]):
        single = [statistics.mean([r[f"single_{key}"] for r in bag[c]]) for c in iter_conds]
        mean_  = [statistics.mean([r[f"mean_{key}"]   for r in bag[c]]) for c in iter_conds]
        single_sd = [statistics.stdev([r[f"single_{key}"] for r in bag[c]]) if len(bag[c]) > 1 else 0 for c in iter_conds]
        mean_sd   = [statistics.stdev([r[f"mean_{key}"]   for r in bag[c]]) if len(bag[c]) > 1 else 0 for c in iter_conds]
        width = 0.38
        ax.bar([x - width/2 for x in xs], single, width, yerr=single_sd, capsize=3,
               color="#3a6ea5", label="single prediction")
        ax.bar([x + width/2 for x in xs], mean_, width, yerr=mean_sd, capsize=3,
               color="#c0392b", label="TTA: mean over original + DE + FR back-translation")
        ax.set_xticks(xs)
        ax.set_xticklabels(iter_conds, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title}: single prediction vs test-time augmentation (TTA)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Test-time augmentation (TTA) via back-translation: per-condition gain over single prediction "
                 "(mean +/- std, 5 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{args.base}_tta.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_tta.png", dpi=160, bbox_inches="tight")
    print(f"Saved figure: {fig_dir}/{args.base}_tta.png")


if __name__ == "__main__":
    main()
