"""Aggregate the OLD-framework SST-2 5-seed sweep on the full 872 test set
using the sentence-transformer + LR headline evaluator.

Computes per-seed accuracy and macro-F1 for each condition, then reports
mean +/- std across seeds. The baseline numbers from this script are the
OLD-framework reference that the v2.9.x re-eval will be compared against.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def _load_synth(cond_dir: Path) -> list[SyntheticSample]:
    samples = []
    for it in sorted(cond_dir.rglob("samples.jsonl")):
        for r in load_jsonl(it):
            samples.append(SyntheticSample.model_validate(r))
    return samples


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    real_test = [
        RealExample.model_validate(r)
        for r in load_jsonl(REPO / "experiments/_splits/sst2_real_test.jsonl")
    ]
    real_train = [
        RealExample.model_validate(r)
        for r in load_jsonl(REPO / "experiments/_splits/sst2_real_train.jsonl")
    ]
    y_test = np.array([r.label for r in real_test])
    print(f"Test set: {len(real_test)} items")
    print(f"Train seed: {len(real_train)} items")

    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    X_test = enc.encode(
        [r.text for r in real_test], normalize_embeddings=True, show_progress_bar=False
    )
    X_real_train = enc.encode(
        [r.text for r in real_train], normalize_embeddings=True, show_progress_bar=False
    )

    seeds = [17, 23, 41, 53, 89]
    conditions = [
        "real_only", "naive", "few_shot", "full_classic",
        "diversity_only", "realism_only", "full_attrforge",
    ]
    results: dict[str, dict[int, dict[str, float]]] = {c: {} for c in conditions}

    for seed in seeds:
        run_dir = REPO / f"experiments/sst2_run_001_seed{seed}"
        if not run_dir.exists():
            continue
        for cond in conditions:
            if cond == "real_only":
                X_train = X_real_train
                y_train = np.array([r.label for r in real_train])
            else:
                cond_dir = run_dir / cond
                if not cond_dir.exists():
                    continue
                samples = _load_synth(cond_dir)
                if not samples:
                    continue
                texts = [s.text for s in samples]
                labels = np.array([s.requested_attributes.get("intent", "?") for s in samples])
                if len(set(labels)) < 2:
                    continue
                X_train = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                y_train = labels
            clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            results[cond][seed] = {"acc": acc, "f1": f1, "n_train": len(X_train)}

    print()
    print(f"=== OLD framework SST-2 5-seed aggregation (sentence-transformer + LR, n_test={len(real_test)}) ===")
    print(f"{'Condition':<20} {'Seeds':<8} {'Mean Acc +- Std':<22} {'Mean F1 +- Std':<22} {'Mean N_train':<14}")
    print("-" * 90)
    for cond in conditions:
        cseeds = results[cond]
        if not cseeds:
            print(f"{cond:<20} (no data)")
            continue
        accs = [v["acc"] for v in cseeds.values()]
        f1s = [v["f1"] for v in cseeds.values()]
        ns = [v["n_train"] for v in cseeds.values()]
        n_seeds = len(cseeds)
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs, ddof=1) if n_seeds > 1 else 0)
        mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s, ddof=1) if n_seeds > 1 else 0)
        mean_n = float(np.mean(ns))
        print(
            f"{cond:<20} N={n_seeds:<6} {mean_acc:.4f} +- {std_acc:.4f}        {mean_f1:.4f} +- {std_f1:.4f}        {mean_n:.1f}"
        )

    # Save the aggregated table for later headline comparison
    out = REPO / "experiments/_diagnostics/sst2_old_framework_5seed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved aggregated results to {out}")

    # Quick paired comparison: full_attrforge vs real_only
    if results["full_attrforge"] and results["real_only"]:
        common = set(results["full_attrforge"].keys()) & set(results["real_only"].keys())
        if common:
            diffs_acc = [
                results["full_attrforge"][s]["acc"] - results["real_only"][s]["acc"]
                for s in sorted(common)
            ]
            print(
                f"\nPaired full_attrforge - real_only over {len(common)} seeds: "
                f"mean acc diff = {np.mean(diffs_acc):+.4f}, per-seed = {[f'{d:+.3f}' for d in diffs_acc]}"
            )


if __name__ == "__main__":
    main()
