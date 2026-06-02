"""Quality-weighted augmentation: turn the verifier's per-sample verdicts into
classifier sample weights.

Hypothesis (the v2 method modification): AttrForge generates per-sample
attribute-verifier verdicts as a side product of the loop. Treating those
verdicts as soft sample weights for the downstream classifier (high-quality
samples count more, samples flagged by the verifier count less) extracts
information that v1's binary "include all 48 samples equally" protocol throws
away. AttrForge should benefit MORE than full_classic because its verifier
runs over a more diverse pool (more informative discriminations).

Protocol:
1. For each (condition, seed) tuple, collect every synthetic sample with its
   attribute_match boolean from the matching attribute_verdicts.jsonl file.
2. Build the augmented training set (30 real + 48 synthetic) with:
     - real samples: weight 1.0
     - synthetic verified True: weight 1.0
     - synthetic verified False: weight `--low-weight` (default 0.3)
   compare to an uniform-weight baseline (weight 1.0 for every sample).
3. Report uniform vs weighted F1 per condition; paired-t for AttrForge vs Classic.

This script can be run on the v1 N=5 data right now (existing verdicts) and
re-run after the N=10 extension completes for tighter statistics.
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


def load_synth_with_verdicts(cond_dir):
    """Walk every iteration dir under cond_dir, load samples and pair with verdict bool."""
    out = []  # list of (SyntheticSample, attribute_match_bool)
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        samples_p = iter_dir / "samples.jsonl"
        verdicts_p = iter_dir / "attribute_verdicts.jsonl"
        if not samples_p.exists():
            continue
        samples = {SyntheticSample.model_validate(r).sample_id:
                   SyntheticSample.model_validate(r) for r in load_jsonl(samples_p)}
        verdict_by_id: dict[str, bool] = {}
        if verdicts_p.exists():
            for v in load_jsonl(verdicts_p):
                # match could be True/False or missing (None) if the verifier didn't run
                am = v.get("attribute_match", None)
                if am is True or am is False:
                    verdict_by_id[v["sample_id"]] = am
        for sid, s in samples.items():
            # If the verifier wasn't enabled in this condition, treat every sample as 'verified True'
            # (i.e. uniform weight). The hypothesis only differentiates conditions that ACTUALLY ran
            # the verifier (full_classic and full_attrforge do).
            am = verdict_by_id.get(sid, True)
            out.append((s, am))
    return out


def fit_eval_weighted(X_train, y_train, sample_weight, X_test, y_test, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = clf.predict(X_test)
    f1s = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
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
    ap.add_argument("--base", required=True)
    ap.add_argument("--low-weight", type=float, default=0.3,
                    help="Weight assigned to verifier-flagged synthetic samples (default 0.3).")
    args = ap.parse_args()

    real_train = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_test.jsonl")]
    real_train_texts = [r.text for r in real_train]
    real_train_labels = np.array([r.label for r in real_train])
    test_texts = [r.text for r in real_test]
    test_labels = np.array([r.label for r in real_test])
    labels = sorted(set(test_labels.tolist()))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    bag = defaultdict(list)

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples_verdicts = load_synth_with_verdicts(cond_dir)
            if not samples_verdicts:
                continue
            texts = [s.text for s, _ in samples_verdicts]
            slabels = np.array([s.requested_attributes.get("intent", "?") for s, _ in samples_verdicts])
            verdict_bools = [v for _, v in samples_verdicts]
            X_synth = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            X_tr = np.concatenate([X_real_train, X_synth], axis=0)
            y_tr = np.concatenate([real_train_labels, slabels])

            # Uniform weights
            w_uniform = np.ones(len(X_tr))
            r_uni = fit_eval_weighted(X_tr, y_tr, w_uniform, X_test, test_labels, labels)

            # Verifier-weighted: real=1.0, synth match=1.0, synth fail=low_weight
            w_synth = np.array([1.0 if vb else args.low_weight for vb in verdict_bools])
            w_weighted = np.concatenate([np.ones(len(X_real_train)), w_synth])
            r_w = fit_eval_weighted(X_tr, y_tr, w_weighted, X_test, test_labels, labels)

            n_flagged = int(sum(1 for vb in verdict_bools if not vb))
            n_total = len(verdict_bools)

            bag[cond_dir.name].append({
                "seed": seed,
                "uniform_macro": r_uni["macro_f1"],
                "uniform_worst": r_uni["worst_f1"],
                "weighted_macro": r_w["macro_f1"],
                "weighted_worst": r_w["worst_f1"],
                "n_flagged": n_flagged, "n_total": n_total,
            })

    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]

    print(f"\n=== Quality-weighted augmentation (low_weight={args.low_weight} for verifier-flagged synth) ===")
    print(f"{'condition':<18} {'flagged/total':<14} {'uniform macro':<16} {'weighted macro':<16} {'macro gain':<14} {'uniform worst':<16} {'weighted worst':<16} {'worst gain':<14}")
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        nf = sum(r["n_flagged"] for r in rows); nt = sum(r["n_total"] for r in rows)
        flag_rate = f"{nf}/{nt}={nf/max(nt,1):.0%}"
        um = [r["uniform_macro"] for r in rows]; wm = [r["weighted_macro"] for r in rows]
        uw = [r["uniform_worst"] for r in rows]; ww = [r["weighted_worst"] for r in rows]
        mg = [a-b for a,b in zip(wm, um)]; wg = [a-b for a,b in zip(ww, uw)]
        def fmt(v): return f"{statistics.mean(v):.3f}+-{statistics.stdev(v) if len(v)>1 else 0:.3f}"
        print(f"{c:<18} {flag_rate:<14} {fmt(um):<16} {fmt(wm):<16} {fmt(mg):<14} {fmt(uw):<16} {fmt(ww):<16} {fmt(wg):<14}")

    print(f"\n=== Paired stats full_attrforge - full_classic ===")
    try:
        from scipy import stats as st
        for metric in ("weighted_macro", "weighted_worst"):
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
            md = statistics.mean(diffs); sd = statistics.stdev(diffs) if len(diffs)>1 else 0
            print(f"  {metric:<18} FA-FC diff={md:+.3f}+-{sd:.3f}  paired-t p={p_t:.3f}  Wilcoxon p={p_w:.3f}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_weighted.json").write_text(json.dumps(
        {"low_weight": args.low_weight, "augmented": {c: bag[c] for c in conds if c in bag}}, indent=2
    ), encoding="utf-8")
    print(f"\nSaved: {out_dir}/quality_weighted.json")


if __name__ == "__main__":
    main()
