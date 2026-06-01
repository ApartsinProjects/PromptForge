"""Multi-classifier downstream evaluation.

Reviewer BL5: the central "diversity-discriminability tradeoff" claim depends
on the downstream classifier being sensitive to surface variation. With only
TF-IDF + logistic regression, we cannot tell whether the gap is a real
property of the data or a brittleness of one weak classifier. This script
re-evaluates every condition's final batch with three downstream classifiers:

    1. TF-IDF + LR  (baseline, our default)
    2. Character n-gram + LR (less keyword-sensitive)
    3. Sentence-transformer embeddings + LR (deepest representation)

If the gap persists across all three, the diversity-discriminability claim is
robust to classifier choice. If it inverts on a stronger classifier, we report
that honestly.

Output:
    experiments/<base>_aggregated/multi_classifier.csv
    experiments/<base>_aggregated/multi_classifier.json
    paper/figures/<base>_multi_classifier.png
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO))
import attrforge  # noqa: E402  (loads .env)
from attrforge.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def load_synth_for_condition(condition_dir: Path) -> list[SyntheticSample]:
    out: list[SyntheticSample] = []
    for iter_dir in sorted(condition_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def fit_eval(
    name: str, X_train, y_train, X_test, y_test, labels
) -> dict:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = float(np.mean(y_pred == y_test))
    per_class_f1 = {}
    for lbl in labels:
        tp = int(((y_pred == lbl) & (y_test == lbl)).sum())
        fp = int(((y_pred == lbl) & (y_test != lbl)).sum())
        fn = int(((y_pred != lbl) & (y_test == lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            p = tp / (tp + fp); r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class_f1[lbl] = f1
    macro_f1 = float(np.mean(list(per_class_f1.values())))
    return {"classifier": name, "accuracy": acc, "macro_f1": macro_f1, "per_class_f1": per_class_f1}


def featurize_tfidf(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def featurize_char(train_texts, test_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    return v.fit_transform(train_texts), v.transform(test_texts)


def featurize_st(train_texts, test_texts):
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    return (
        enc.encode(train_texts, normalize_embeddings=True),
        enc.encode(test_texts, normalize_embeddings=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument(
        "--label-attribute", default="intent"
    )
    args = ap.parse_args()

    test_path = REPO / "experiments" / "_splits" / "real_test.jsonl"
    test = [RealExample.model_validate(r) for r in load_jsonl(test_path)]
    test_texts = [t.text for t in test]
    test_labels = np.array([t.label for t in test])
    labels = sorted(set(test_labels.tolist()))

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    if not seed_dirs:
        sys.exit(f"no seed dirs matching {args.base}_seed*")

    print(f"Evaluating {len(seed_dirs)} seeds × multiple classifiers")
    print(f"Test set: {len(test)} held-out real examples; {len(labels)} classes")

    bag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for sd in seed_dirs:
        for cond_dir in sorted(p for p in sd.iterdir() if p.is_dir() and p.name not in {"audit", "aggregated"}):
            samples = load_synth_for_condition(cond_dir)
            if not samples:
                continue
            train_texts = [s.text for s in samples]
            train_labels = np.array([s.requested_attributes.get(args.label_attribute, "?") for s in samples])

            for fname, featfn in [
                ("tfidf_word", featurize_tfidf),
                ("tfidf_char_3_5", featurize_char),
                ("st_minilm", featurize_st),
            ]:
                try:
                    X_train, X_test = featfn(train_texts, test_texts)
                    res = fit_eval(fname, X_train, train_labels, X_test, test_labels, labels)
                    bag[cond_dir.name][f"{fname}__macro_f1"].append(res["macro_f1"])
                    bag[cond_dir.name][f"{fname}__acc"].append(res["accuracy"])
                except Exception as e:
                    print(f"  {sd.name}/{cond_dir.name}/{fname} FAILED: {e}")

    print()
    print(f'{"condition":<18} {"tfidf F1":<14} {"char F1":<14} {"st_minilm F1":<14}')
    out_rows = []
    conds_order = ["naive", "few_shot", "self_critique", "realism_only", "diversity_only", "full_classic", "full_attrforge"]
    for cond in conds_order:
        if cond not in bag: continue
        d = bag[cond]
        def m(k):
            if not d.get(k): return ("n/a", "n/a")
            return (statistics.mean(d[k]), statistics.stdev(d[k]) if len(d[k]) > 1 else 0)
        tf = m("tfidf_word__macro_f1"); ch = m("tfidf_char_3_5__macro_f1"); st = m("st_minilm__macro_f1")
        def fmt(t):
            if t[0] == "n/a": return "n/a"
            return f"{t[0]:.2f}±{t[1]:.2f}"
        print(f'{cond:<18} {fmt(tf):<14} {fmt(ch):<14} {fmt(st):<14}')
        out_rows.append({
            "condition": cond,
            "tfidf_word_f1_mean": tf[0] if tf[0] != "n/a" else None,
            "tfidf_word_f1_sd": tf[1] if tf[0] != "n/a" else None,
            "tfidf_char_f1_mean": ch[0] if ch[0] != "n/a" else None,
            "tfidf_char_f1_sd": ch[1] if ch[0] != "n/a" else None,
            "st_minilm_f1_mean": st[0] if st[0] != "n/a" else None,
            "st_minilm_f1_sd": st[1] if st[0] != "n/a" else None,
        })

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "multi_classifier.json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    import csv
    with (out_dir / "multi_classifier.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = list(range(len(out_rows)))
    width = 0.25
    for i, (key, color, label) in enumerate([
        ("tfidf_word", "#3a6ea5", "TF-IDF word"),
        ("tfidf_char", "#c0392b", "TF-IDF char 3-5"),
        ("st_minilm", "#2e715a", "Sentence-transformer MiniLM"),
    ]):
        means = [r[f"{key}_f1_mean"] or 0 for r in out_rows]
        sds = [r[f"{key}_f1_sd"] or 0 for r in out_rows]
        pos = [x - width + i * width for x in xs]
        ax.bar(pos, means, width, yerr=sds, capsize=3, color=color, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["condition"] for r in out_rows], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("downstream macro F1")
    ax.set_title("Downstream classifier comparison (3 classifiers; mean ± std across 3 seeds, live gpt-4o-mini)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.base}_multi_classifier.png", dpi=160)
    fig.savefig(REPO / "docs" / "figures" / f"{args.base}_multi_classifier.png", dpi=160)
    print(f"\nSaved {fig_dir}/{args.base}_multi_classifier.png")


if __name__ == "__main__":
    main()
