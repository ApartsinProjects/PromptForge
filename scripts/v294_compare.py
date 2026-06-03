"""Compare v2.9.4 (Mode Hunter veto + Coverage Hole promotion + Realism
distribution-anchoring + Verifier read-text-first + Verifier real-anchor
calibration + Batch API) against the OLD framework on the same eval
harness (sentence-transformer + LR on the same test split).

Looks for runs at experiments/{dataset}_v294_seed{seed} for each
(dataset, seed) pair; falls back to the OLD experiments/{dataset}_run_001_seed{seed}
when v294 is missing.

Reports per-dataset:
  real-only baseline
  OLD framework SynSmith mean +/- std across seeds
  NEW framework v294 SynSmith mean +/- std across seeds
  delta v294 vs OLD
  delta v294 vs real-only
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from synsmith.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def _load_synth(cond_dir: Path) -> list[SyntheticSample]:
    out = []
    for it in sorted(cond_dir.rglob("samples.jsonl")):
        for r in load_jsonl(it):
            out.append(SyntheticSample.model_validate(r))
    return out


def _seed_eval(samples, enc, X_test, y_test, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    if not samples:
        return None
    texts = [s.text for s in samples]
    labels = np.array([s.requested_attributes.get("intent", "?") for s in samples])
    if len(set(labels)) < 2:
        return None
    X = enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=seed
    )
    clf.fit(X, labels)
    y_pred = clf.predict(X_test)
    return float(accuracy_score(y_test, y_pred)), float(
        f1_score(y_test, y_pred, average="macro")
    )


def compare_dataset(name, train_path, test_path, old_glob, new_glob, seeds, enc):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    real_train = [RealExample.model_validate(r) for r in load_jsonl(train_path)]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(test_path)]
    y_test = np.array([r.label for r in real_test])
    X_test = enc.encode(
        [r.text for r in real_test], normalize_embeddings=True, show_progress_bar=False
    )
    X_real = enc.encode(
        [r.text for r in real_train], normalize_embeddings=True, show_progress_bar=False
    )
    y_real = np.array([r.label for r in real_train])
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=17)
    clf.fit(X_real, y_real)
    y_pred = clf.predict(X_test)
    acc_ro = float(accuracy_score(y_test, y_pred))
    f1_ro = float(f1_score(y_test, y_pred, average="macro"))

    def gather(pattern):
        accs, f1s = [], []
        for s in seeds:
            cond_dir = REPO / pattern.format(seed=s) / "full_attrforge"
            if not cond_dir.exists():
                continue
            samples = _load_synth(cond_dir)
            res = _seed_eval(samples, enc, X_test, y_test, s)
            if res is None:
                continue
            accs.append(res[0])
            f1s.append(res[1])
        return accs, f1s

    old_accs, old_f1s = gather(old_glob)
    new_accs, new_f1s = gather(new_glob)

    def fmt(arr):
        if not arr:
            return "no data"
        if len(arr) == 1:
            return f"{arr[0]:.3f} (N=1)"
        return f"{np.mean(arr):.3f} ± {np.std(arr, ddof=1):.3f} (N={len(arr)})"

    print(f"\n=== {name} ===")
    print(f"  Real-only baseline:                acc {acc_ro:.3f}   f1 {f1_ro:.3f}")
    print(f"  OLD framework full_attrforge acc:  {fmt(old_accs)}")
    print(f"  NEW v2.9.4 full_attrforge acc:     {fmt(new_accs)}")
    if old_accs and new_accs:
        delta_new_old = float(np.mean(new_accs) - np.mean(old_accs))
        delta_new_ro = float(np.mean(new_accs) - acc_ro)
        print(f"  Delta v2.9.4 - OLD:  {delta_new_old:+.3f}pp")
        print(f"  Delta v2.9.4 - real: {delta_new_ro:+.3f}pp")


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    seeds = [17, 23, 41, 53, 89]
    print("=== v2.9.4 vs OLD framework comparison (sentence-transformer + LR) ===")
    compare_dataset(
        "SST-2",
        "experiments/_splits/sst2_real_train.jsonl",
        "experiments/_splits/sst2_real_test.jsonl",
        "experiments/sst2_run_001_seed{seed}",
        "experiments/sst2_v294_seed{seed}",
        seeds, enc,
    )
    compare_dataset(
        "Banking77",
        "experiments/_splits/banking77_real_train.jsonl",
        "experiments/_splits/banking77_real_test.jsonl",
        "experiments/banking77_run_001_seed{seed}",
        "experiments/banking77_v294_seed{seed}",
        seeds, enc,
    )
    compare_dataset(
        "TREC",
        "experiments/_splits/trec_real_train.jsonl",
        "experiments/_splits/trec_real_test.jsonl",
        "experiments/trec_run_001_seed{seed}",
        "experiments/trec_v294_seed{seed}",
        seeds, enc,
    )


if __name__ == "__main__":
    main()
