"""Adversarial robustness via back-translation paraphrased test set.

Hypothesis: SynSmith's higher lexical diversity (Table 8: highest distinct-n,
lowest self-BLEU-4 of any iterated condition) should produce classifiers that
degrade LESS when the test items are paraphrased. If the diversity claim is
real, a classifier trained on a wider lexical distribution should handle
out-of-distribution surface forms more gracefully.

We paraphrase by back-translation using Helsinki-NLP MarianMT models, which
is more rigorous than LLM paraphrasing for this purpose:
  - deterministic and reproducible (no API, no closed-model drift),
  - introduces real lexical and syntactic variation,
  - free and offline,
  - can be applied at any scale.

Two paraphrases per test item: English -> German -> English (opus-mt-en-de
then opus-mt-de-en) and English -> French -> English. Combined N = 2 x 10 = 20.

Protocol:
1. Generate paraphrases via back-translation (if not cached).
2. For each (condition, seed) pair, train the augmentation classifier
   (30 real + 48 synthetic, sentence-transformer + LR) and evaluate on:
     - the clean test (the v1 number)
     - the paraphrased test
3. Report F1 drop and worst-class F1 drop per condition, plus paired stats
   for SynSmith vs full_classic.

Outputs:
    experiments/_splits/real_test_paraphrased.jsonl
    experiments/<base>_aggregated/robustness.json
    paper/figures/<base>_robustness.png
"""
from __future__ import annotations

import argparse
import json
import os
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
import synsmith  # noqa: E402  loads .env
from synsmith.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


BACKTRANSLATION_PIVOTS = [
    ("de", "Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en"),
    ("fr", "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"),
]


def _translate_batch(texts, model_name):
    """Translate a list of strings through a MarianMT model and return the outputs."""
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=200)
    out_ids = mdl.generate(**enc, max_length=200, num_beams=4)
    return [tok.decode(t, skip_special_tokens=True) for t in out_ids]


def paraphrase_test_items_backtranslation(real_test):
    """Paraphrase each test item once per pivot language via back-translation."""
    out = []
    texts = [r.text for r in real_test]
    for pivot, fwd_model, back_model in BACKTRANSLATION_PIVOTS:
        print(f"  Back-translating EN -> {pivot.upper()} -> EN via {fwd_model} / {back_model} ...")
        pivot_texts = _translate_batch(texts, fwd_model)
        back_texts = _translate_batch(pivot_texts, back_model)
        for idx, (item, back) in enumerate(zip(real_test, back_texts)):
            cleaned = back.strip().strip('"').strip("'").strip()
            if not cleaned:
                continue
            out.append({"text": cleaned, "label": item.label,
                        "_source": f"backtranslation_{pivot}",
                        "_origin_idx": idx,
                        "_pivot": pivot})
    return out


def load_synth(cond_dir):
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def fit_eval(X_train, y_train, X_test, y_test, labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    per_class_f1 = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            per_class_f1[lbl] = 0.0; continue
        p = tp / (tp + fp); r = tp / (tp + fn)
        per_class_f1[lbl] = 2 * p * r / (p + r) if (p + r) else 0.0
    macro = float(np.mean(list(per_class_f1.values())))
    worst = float(min(per_class_f1.values()))
    return {"macro_f1": macro, "worst_f1": worst, "per_class_f1": per_class_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n-paraphrases", type=int, default=2,
                    help="Number of paraphrases per test item")
    ap.add_argument("--paraphrase-only", action="store_true",
                    help="Just write the paraphrased test set, don't evaluate")
    ap.add_argument("--skip-paraphrase", action="store_true",
                    help="Skip paraphrase generation (use existing file)")
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    real_test_path = REPO / "experiments" / "_splits" / "real_test.jsonl"
    paraphrase_path = REPO / "experiments" / "_splits" / "real_test_paraphrased.jsonl"

    real_test = [RealExample.model_validate(r) for r in load_jsonl(real_test_path)]
    print(f"Loaded {len(real_test)} clean test items.")

    if not args.skip_paraphrase:
        paraphrased = paraphrase_test_items_backtranslation(real_test)
        with paraphrase_path.open("w", encoding="utf-8") as f:
            for r in paraphrased:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(paraphrased)} paraphrased items -> {paraphrase_path}")
    else:
        paraphrased = [json.loads(line) for line in paraphrase_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Loaded {len(paraphrased)} paraphrased items (cached).")

    if args.paraphrase_only:
        return

    real_train = [RealExample.model_validate(r) for r in load_jsonl(REPO / "experiments/_splits/real_train.jsonl")]
    real_train_texts = [r.text for r in real_train]
    real_train_labels = np.array([r.label for r in real_train])

    clean_test_texts = [r.text for r in real_test]
    clean_test_labels = np.array([r.label for r in real_test])
    para_test_texts = [r["text"] for r in paraphrased]
    para_test_labels = np.array([r["label"] for r in paraphrased])
    labels = sorted(set(clean_test_labels.tolist()))

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_clean_test = enc.encode(clean_test_texts, normalize_embeddings=True, show_progress_bar=False)
    X_para_test  = enc.encode(para_test_texts, normalize_embeddings=True, show_progress_bar=False)
    X_real_train = enc.encode(real_train_texts, normalize_embeddings=True, show_progress_bar=False)

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))

    # bag[condition] -> list of dicts {seed, macro_clean, macro_para, worst_clean, worst_para}
    bag = defaultdict(list)

    # Real-only first (no synthetic) - one per seed for traceability, even though
    # the augmented set with 30 real samples is deterministic across seeds.
    print("\n=== Real-only baselines ===")
    r_clean = fit_eval(X_real_train, real_train_labels, X_clean_test, clean_test_labels, labels)
    r_para  = fit_eval(X_real_train, real_train_labels, X_para_test, para_test_labels, labels)
    print(f"  real-only: clean macro={r_clean['macro_f1']:.3f} worst={r_clean['worst_f1']:.3f}; "
          f"paraphrased macro={r_para['macro_f1']:.3f} worst={r_para['worst_f1']:.3f}; "
          f"drop={r_clean['macro_f1']-r_para['macro_f1']:+.3f}")

    real_only_result = {
        "macro_clean": r_clean["macro_f1"], "macro_para": r_para["macro_f1"],
        "worst_clean": r_clean["worst_f1"], "worst_para": r_para["worst_f1"],
    }

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

            r_c = fit_eval(X_tr, y_tr, X_clean_test, clean_test_labels, labels)
            r_p = fit_eval(X_tr, y_tr, X_para_test, para_test_labels, labels)
            bag[cond_dir.name].append({
                "seed": seed,
                "macro_clean": r_c["macro_f1"], "macro_para": r_p["macro_f1"],
                "worst_clean": r_c["worst_f1"], "worst_para": r_p["worst_f1"],
            })

    # Summary
    conds = ["naive", "few_shot", "self_critique", "realism_only",
             "diversity_only", "full_classic", "full_attrforge"]

    print(f"\n=== Summary: F1 drop under paraphrase (mean +/- std across 5 seeds) ===")
    print(f"{'condition':<18} {'macro clean':<14} {'macro para':<14} {'macro drop':<14} "
          f"{'worst clean':<14} {'worst para':<14} {'worst drop':<14}")
    print(f"{'real-only':<18} {real_only_result['macro_clean']:.3f}         "
          f"{real_only_result['macro_para']:.3f}         "
          f"{real_only_result['macro_clean']-real_only_result['macro_para']:+.3f}         "
          f"{real_only_result['worst_clean']:.3f}         "
          f"{real_only_result['worst_para']:.3f}         "
          f"{real_only_result['worst_clean']-real_only_result['worst_para']:+.3f}")
    for c in conds:
        if c not in bag:
            continue
        rows = bag[c]
        mc = [r["macro_clean"] for r in rows]
        mp = [r["macro_para"]  for r in rows]
        wc = [r["worst_clean"] for r in rows]
        wp = [r["worst_para"]  for r in rows]
        md = [a-b for a,b in zip(mc, mp)]
        wd = [a-b for a,b in zip(wc, wp)]
        def msd(v):
            return f"{statistics.mean(v):.3f}+-{statistics.stdev(v) if len(v)>1 else 0:.3f}"
        print(f"{c:<18} {msd(mc):<14} {msd(mp):<14} {msd(md):<14} "
              f"{msd(wc):<14} {msd(wp):<14} {msd(wd):<14}")

    # Paired stats SynSmith vs Classic
    print("\n=== Paired stats: full_attrforge - full_classic ===")
    try:
        from scipy import stats as st
        for metric in ("macro_drop", "worst_drop"):
            fc_name, fa_name = "full_classic", "full_attrforge"
            fc_rows = bag.get(fc_name, [])
            fa_rows = bag.get(fa_name, [])
            if not fc_rows or not fa_rows:
                continue
            if metric == "macro_drop":
                fc = [r["macro_clean"] - r["macro_para"] for r in fc_rows]
                fa = [r["macro_clean"] - r["macro_para"] for r in fa_rows]
            else:
                fc = [r["worst_clean"] - r["worst_para"] for r in fc_rows]
                fa = [r["worst_clean"] - r["worst_para"] for r in fa_rows]
            diffs = [a-b for a,b in zip(fa, fc)]
            t, p = st.ttest_rel(fa, fc)
            try:
                _, p_w = st.wilcoxon(fa, fc, zero_method="zsplit")
            except ValueError:
                p_w = float("nan")
            md, sd = statistics.mean(diffs), statistics.stdev(diffs) if len(diffs) > 1 else 0
            print(f"  {metric:<12} (negative = SynSmith degrades less): "
                  f"diff={md:+.3f}+-{sd:.3f}; paired-t p={p:.3f}; Wilcoxon p={p_w:.3f}; "
                  f"per-seed: {[round(d,3) for d in diffs]}")
    except Exception as e:
        print(f"scipy: {e}")

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_paraphrases_per_item": args.n_paraphrases,
        "real_only": real_only_result,
        "augmented": {c: bag[c] for c in conds if c in bag},
    }
    (out_dir / "robustness.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_dir}/robustness.json")

    # Plot: clean vs paraphrased per condition (worst-class)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    iter_conds = [c for c in conds if c in bag]
    xs = list(range(len(iter_conds)))

    for ax, metric in zip(axes, [("macro", "Macro F1"), ("worst", "Worst-class F1")]):
        key, title = metric
        clean = [statistics.mean([r[f"{key}_clean"] for r in bag[c]]) for c in iter_conds]
        para  = [statistics.mean([r[f"{key}_para"]  for r in bag[c]]) for c in iter_conds]
        clean_sd = [statistics.stdev([r[f"{key}_clean"] for r in bag[c]]) if len(bag[c]) > 1 else 0 for c in iter_conds]
        para_sd  = [statistics.stdev([r[f"{key}_para"]  for r in bag[c]]) if len(bag[c]) > 1 else 0 for c in iter_conds]
        width = 0.38
        ax.bar([x - width/2 for x in xs], clean, width, yerr=clean_sd, capsize=3,
               color="#3a6ea5", label="clean test")
        ax.bar([x + width/2 for x in xs], para, width, yerr=para_sd, capsize=3,
               color="#c0392b", label="paraphrased test")
        ax.set_xticks(xs)
        ax.set_xticklabels(iter_conds, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title} clean vs paraphrased")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Adversarial robustness: F1 drop when the held-out real test "
                 "is LLM-paraphrased (mean +/- std, 5 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{args.base}_robustness.png", dpi=160, bbox_inches="tight")
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_robustness.png", dpi=160, bbox_inches="tight")
    print(f"Saved figure: {fig_dir}/{args.base}_robustness.png")


if __name__ == "__main__":
    main()
