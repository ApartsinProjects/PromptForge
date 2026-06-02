"""Banking77 augmentation evaluation: cross-task replication of v1's
scarce-real protocol on the 10-class card/payment subset.

Same pipeline as scripts/scarce_real_eval.py, parameterized for Banking77:
  - train split:  experiments/_splits/banking77_real_train.jsonl  (300 items, 30/class)
  - test split:   experiments/_splits/banking77_real_test.jsonl   (~440 items, 40-50/class)
  - synthetic:    experiments/<base>_seed<seed>/<condition>/.../iter_*/samples.jsonl
  - classifier:   sentence-transformer + LR

For each (condition, n_real, seed), train on (real subset + synthetic) and
evaluate on the held-out test, reporting macro F1, worst-class F1, ECE,
and paired statistics for full_attrforge vs full_classic.

Outputs:
    experiments/<base>_aggregated/banking77_scarce_real.json
    paper/figures/<base>_banking77_scarce_real.png
"""
from __future__ import annotations

import argparse
import json
import random
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
import synsmith  # noqa: E402
from synsmith.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def stratified_subsample(reals, n, seed):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in reals:
        by_label[r.label].append(r)
    labels = sorted(by_label.keys())
    per_class = max(1, n // len(labels))
    out = []
    for lbl in labels:
        items = list(by_label[lbl])
        rng.shuffle(items)
        out.extend(items[:per_class])
    rng.shuffle(out)
    return out[:n]


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def fit_eval(X_tr, y_tr, X_te, y_te, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1s = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_te == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_te != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_te == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s[lbl] = 0.0; continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        f1s[lbl] = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "macro_f1": float(np.mean(list(f1s.values()))),
        "worst_f1": float(min(f1s.values())),
        "per_class_f1": f1s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="banking77_run_001")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[10, 30, 50, 100, 200, 300])
    args = ap.parse_args()

    real_all = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/banking77_real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/banking77_real_test.jsonl")]
    y_test = np.array([r.label for r in real_test])
    labels = sorted(set(y_test.tolist()))
    test_texts = [r.text for r in real_test]

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_test_st = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    if not seed_dirs:
        print(f"No seed dirs found matching {args.base}_seed*. Run the generation first.")
        return

    aug = defaultdict(lambda: defaultdict(list))
    real_only = defaultdict(list)

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for n in args.sizes:
            sub = stratified_subsample(real_all, n, seed=seed)
            X_sub = enc.encode([r.text for r in sub], normalize_embeddings=True, show_progress_bar=False)
            r = fit_eval(X_sub, np.array([r.label for r in sub]), X_test_st, y_test, labels)
            real_only[n].append({"macro_f1": r["macro_f1"], "worst_f1": r["worst_f1"]})

        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            synth = load_synth(cond_dir)
            if not synth:
                continue
            X_synth = enc.encode([s.text for s in synth], normalize_embeddings=True, show_progress_bar=False)
            synth_labels = np.array([s.requested_attributes.get("intent", "?") for s in synth])
            for n in args.sizes:
                sub = stratified_subsample(real_all, n, seed=seed)
                X_sub = enc.encode([r.text for r in sub], normalize_embeddings=True, show_progress_bar=False)
                X_tr = np.concatenate([X_sub, X_synth], axis=0)
                y_tr = np.concatenate([np.array([r.label for r in sub]), synth_labels])
                r = fit_eval(X_tr, y_tr, X_test_st, y_test, labels)
                aug[cond_dir.name][n].append({"macro_f1": r["macro_f1"], "worst_f1": r["worst_f1"]})

    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]
    print(f"\n=== Banking77 augmentation: macro F1 (mean +- std over {len(seed_dirs)} seeds) ===")
    print(f"{'n':<5} {'real-only':<14}", end="")
    for c in conds:
        if c in aug:
            print(f" {c:<16}", end="")
    print()
    for n in args.sizes:
        r = [x["macro_f1"] for x in real_only[n]]
        rm = statistics.mean(r); rs = statistics.stdev(r) if len(r)>1 else 0
        print(f"{n:<5} {rm:.3f}+-{rs:.3f}  ", end="")
        for c in conds:
            if c not in aug:
                continue
            v = [x["macro_f1"] for x in aug[c][n]]
            m = statistics.mean(v); sd = statistics.stdev(v) if len(v)>1 else 0
            print(f" {m:.3f}+-{sd:.3f}    ", end="")
        print()

    print(f"\n=== Banking77 augmentation: worst-class F1 (mean +- std) ===")
    print(f"{'n':<5} {'real-only':<14}", end="")
    for c in conds:
        if c in aug:
            print(f" {c:<16}", end="")
    print()
    for n in args.sizes:
        r = [x["worst_f1"] for x in real_only[n]]
        rm = statistics.mean(r); rs = statistics.stdev(r) if len(r)>1 else 0
        print(f"{n:<5} {rm:.3f}+-{rs:.3f}  ", end="")
        for c in conds:
            if c not in aug:
                continue
            v = [x["worst_f1"] for x in aug[c][n]]
            m = statistics.mean(v); sd = statistics.stdev(v) if len(v)>1 else 0
            print(f" {m:.3f}+-{sd:.3f}    ", end="")
        print()

    print(f"\n=== Paired stats full_attrforge - full_classic ===")
    try:
        from scipy import stats as st
        for metric in ("macro_f1", "worst_f1"):
            for n in args.sizes:
                fc = [x[metric] for x in aug.get("full_classic", {}).get(n, [])]
                fa = [x[metric] for x in aug.get("full_attrforge", {}).get(n, [])]
                if not fc or not fa:
                    continue
                diffs = [a-b for a,b in zip(fa, fc)]
                t, p_t = st.ttest_rel(fa, fc)
                try:
                    _, p_w = st.wilcoxon(fa, fc, zero_method="zsplit")
                except ValueError:
                    p_w = float("nan")
                md = statistics.mean(diffs); sd = statistics.stdev(diffs) if len(diffs)>1 else 0
                print(f"  {metric:<10} n={n:<4} diff={md:+.3f}+-{sd:.3f}  paired-t p={p_t:.3f}  Wilcoxon p={p_w:.3f}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sizes": args.sizes,
        "labels": labels,
        "real_only": {n: real_only[n] for n in args.sizes},
        "augmented": {c: {n: aug[c][n] for n in args.sizes} for c in conds if c in aug},
    }
    (out_dir / "banking77_scarce_real.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/banking77_scarce_real.json")

    # Two-panel plot: macro and worst-class vs n
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sizes_arr = list(args.sizes)
    for ax, key, title in zip(axes, ["macro_f1", "worst_f1"],
                              ["Macro F1", "Worst-class F1"]):
        rm = [statistics.mean([x[key] for x in real_only[n]]) for n in sizes_arr]
        rs = [statistics.stdev([x[key] for x in real_only[n]]) if len(real_only[n])>1 else 0 for n in sizes_arr]
        ax.errorbar(sizes_arr, rm, yerr=rs, marker="s", linewidth=2, capsize=4,
                    color="#444444", linestyle="--", label="real-only")
        palette = {
            "naive": "#999999", "few_shot": "#bbbbbb",
            "self_critique": "#cccccc", "realism_only": "#888aff",
            "diversity_only": "#88aaaa",
            "full_classic": "#3a6ea5", "full_attrforge": "#c0392b",
        }
        for c in conds:
            if c not in aug:
                continue
            m = [statistics.mean([x[key] for x in aug[c][n]]) for n in sizes_arr]
            s = [statistics.stdev([x[key] for x in aug[c][n]]) if len(aug[c][n])>1 else 0 for n in sizes_arr]
            lw = 2.5 if c in {"full_classic", "full_attrforge"} else 1.2
            alpha = 1.0 if c in {"full_classic", "full_attrforge"} else 0.5
            ax.errorbar(sizes_arr, m, yerr=s, marker="o", linewidth=lw,
                        capsize=4, color=palette.get(c, "#999"), label=c, alpha=alpha)
        ax.set_xlabel("number of real training examples (stratified subsample)")
        ax.set_ylabel(title)
        ax.set_title(f"Banking77 10-class card/payment: {title} vs real-train size")
        ax.set_xticks(sizes_arr)
        ax.set_ylim(0, 1)
        ax.grid(linestyle=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{args.base}_scarce_real.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_scarce_real.png", dpi=160, bbox_inches="tight")
    print(f"Saved figure: {fig_dir}/{args.base}_scarce_real.png")


if __name__ == "__main__":
    main()
