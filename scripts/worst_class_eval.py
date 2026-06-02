"""Worst-class F1 sweep across conditions and real-train sizes.

The v1 paper used macro F1 as the headline. With 3 of 5 classes saturating at
F1=1.00 from real data alone, macro F1 averages near 0.893 from n=20 onward
and hides any synthetic-data signal. Worst-class F1 (the minimum per-class
F1 within each evaluation) does not saturate and is also what a practitioner
actually cares about for deployment.

This script reads the per-class F1 already computed in
`<base>_aggregated/per_class_aug.json` (sentence-transformer + LR) and:

1. Extracts the worst class per (condition, n_real, seed) tuple.
2. Reports mean +/- std across seeds per (condition, n_real) cell.
3. Computes paired-t and Wilcoxon for full_attrforge vs full_classic at every
   real-train size (the same comparison v1 reports as NS at macro F1).
4. Saves a JSON dump and a sweep figure with all 7 conditions across all 6
   real-train sizes.

Outputs:
    experiments/<base>_aggregated/worst_class.json
    paper/figures/<base>_worst_class.png
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
import synsmith  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    args = ap.parse_args()

    src = REPO / "experiments" / f"{args.base}_aggregated" / "per_class_aug.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    labels: list[str] = data["labels"]
    sizes: list[int] = data["sizes"]

    # Worst-class F1 per seed: min over classes of the per-class F1.
    # real_only and augmented are both keyed: [size][class] -> list[per-seed F1]
    real_only = data["real_only"]
    aug = data["augmented"]

    def worst_per_seed(per_class: dict) -> list[float]:
        # per_class is {class: [per-seed F1, ...]} for ONE size.
        # zip across classes to get the per-seed vector, then take min.
        rows = list(zip(*[per_class[lbl] for lbl in labels]))
        return [float(min(row)) for row in rows]

    real_worst = {str(n): worst_per_seed(real_only[str(n)]) for n in sizes}

    aug_worst: dict[str, dict[str, list[float]]] = {}
    for cond, per_size in aug.items():
        aug_worst[cond] = {str(n): worst_per_seed(per_size[str(n)]) for n in sizes}

    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]

    print(f"\nWorst-class F1 (min over classes, sentence-transformer + LR, "
          f"mean +/- std across 5 seeds):\n")
    print(f"{'n':<4} {'real-only':<14}", end="")
    for c in conds:
        print(f" {c:<16}", end="")
    print()
    for n in sizes:
        r = real_worst[str(n)]
        r_m = statistics.mean(r); r_s = statistics.stdev(r) if len(r) > 1 else 0
        print(f"{n:<4} {r_m:.3f}+-{r_s:.3f}", end="")
        for c in conds:
            v = aug_worst.get(c, {}).get(str(n), [])
            if not v:
                print(f"  {'n/a':<14}", end=""); continue
            m = statistics.mean(v); sd = statistics.stdev(v) if len(v) > 1 else 0
            print(f"  {m:.3f}+-{sd:.3f}", end="")
        print()

    print(f"\nPaired stats full_attrforge - full_classic on worst-class F1:")
    try:
        from scipy import stats as st
        for n in sizes:
            fc = aug_worst["full_classic"][str(n)]
            fa = aug_worst["full_attrforge"][str(n)]
            if not fc or not fa:
                continue
            diffs = [a - b for a, b in zip(fa, fc)]
            t, p_t = st.ttest_rel(fa, fc)
            try:
                w, p_w = st.wilcoxon(fa, fc, zero_method="zsplit")
            except ValueError:
                p_w = float("nan")
            print(f"  n={n}: mean diff = {statistics.mean(diffs):+.3f} "
                  f"+- {(statistics.stdev(diffs) if len(diffs)>1 else 0):.3f}; "
                  f"paired-t p={p_t:.3f}; Wilcoxon p={p_w:.3f}; "
                  f"per-seed: {[round(d,3) for d in diffs]}")
    except Exception as e:
        print(f"  scipy error: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sizes": sizes,
        "labels": labels,
        "real_only_worst": real_worst,
        "augmented_worst": aug_worst,
    }
    (out_dir / "worst_class.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/worst_class.json")

    # Plot
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sizes_arr = list(sizes)
    rm = [statistics.mean(real_worst[str(n)]) for n in sizes_arr]
    rs = [statistics.stdev(real_worst[str(n)]) if len(real_worst[str(n)]) > 1 else 0
          for n in sizes_arr]
    ax.errorbar(sizes_arr, rm, yerr=rs, marker="s", linewidth=2, capsize=4,
                color="#444444", linestyle="--", label="real-only")
    palette = {
        "naive": "#999999", "few_shot": "#bbbbbb",
        "self_critique": "#cccccc", "realism_only": "#888aff",
        "diversity_only": "#88aaaa",
        "full_classic": "#3a6ea5", "full_attrforge": "#c0392b",
    }
    for c in conds:
        if c not in aug_worst:
            continue
        m = [statistics.mean(aug_worst[c][str(n)]) for n in sizes_arr]
        s = [statistics.stdev(aug_worst[c][str(n)]) if len(aug_worst[c][str(n)]) > 1
             else 0 for n in sizes_arr]
        lw = 2.5 if c in {"full_classic", "full_attrforge"} else 1.2
        alpha = 1.0 if c in {"full_classic", "full_attrforge"} else 0.6
        ax.errorbar(sizes_arr, m, yerr=s, marker="o", linewidth=lw,
                    capsize=4, color=palette.get(c, "#999"), label=c, alpha=alpha)
    ax.set_xlabel("number of real training examples (stratified subsample)")
    ax.set_ylabel("worst-class F1 (min over the 5 classes)")
    ax.set_title("Augmentation worst-class F1 vs real-train size "
                 "(mean +/- std, 5 seeds)")
    ax.set_xticks(sizes_arr)
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.base}_worst_class.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_worst_class.png",
                dpi=160, bbox_inches="tight")
    print(f"Saved figure: {fig_dir}/{args.base}_worst_class.png")


if __name__ == "__main__":
    main()
