"""Multi-dataset sample-based root-cause inspection.

For each of (customer-support, Banking77, TREC, SST-2), under the OLD
framework runs, finds the test items where AttrForge fails but real-only
succeeds, and prints both the synth and real exemplars to surface
register / vocabulary / style mismatch patterns common across domains.

The goal is to validate whether the 3 fixes already identified on SST-2
(Mode Hunter veto, Coverage Hole promotion, Realism distribution-anchoring)
are GENERIC framework improvements -- i.e., do the same root causes
manifest across all datasets -- or whether new fault classes appear.
"""
from __future__ import annotations

import json
import os
import sys
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_synth(cd: Path) -> list:
    from attrforge.schema import SyntheticSample, load_jsonl
    samples = []
    for it in sorted(cd.glob("*/iter_*")):
        sj = it / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                samples.append(SyntheticSample.model_validate(r))
    return samples


def inspect_dataset(
    name: str,
    split_train: Path,
    split_test: Path,
    af_run: Path,
    label_attr: str = "intent",
    seed: int = 17,
    sample_failures: int = 12,
) -> None:
    from attrforge.schema import RealExample, load_jsonl
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 80}")
    print(f"=== {name} ({split_train.stem.split('_')[0]}) ===")
    print(f"{'=' * 80}")

    real_train = [RealExample.model_validate(r) for r in load_jsonl(split_train)]
    real_test = [RealExample.model_validate(r) for r in load_jsonl(split_test)]
    print(f"  {len(real_train)} train  /  {len(real_test)} test")

    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    X_real = enc.encode([r.text for r in real_train], normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode([r.text for r in real_test], normalize_embeddings=True, show_progress_bar=False)
    y_real = np.array([r.label for r in real_train])
    y_test = np.array([r.label for r in real_test])

    clf_r = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf_r.fit(X_real, y_real)
    y_pred_r = clf_r.predict(X_test)
    acc_r = (y_pred_r == y_test).mean()

    samples = load_synth(af_run)
    if not samples:
        print(f"  AttrForge run EMPTY at {af_run}")
        return
    af_texts = [s.text for s in samples]
    af_labels = np.array([s.requested_attributes.get(label_attr, "?") for s in samples])
    n_af_classes = len(set(af_labels))
    n_real_classes = len(set(y_real))
    if n_af_classes < 2:
        print(f"  AttrForge run has only {n_af_classes} classes, skipping")
        return
    X_af = enc.encode(af_texts, normalize_embeddings=True, show_progress_bar=False)
    clf_af = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    clf_af.fit(X_af, af_labels)
    y_pred_af = clf_af.predict(X_test)
    acc_af = (y_pred_af == y_test).mean()

    print(
        f"  Real-only acc {acc_r:.3f} | AttrForge acc {acc_af:.3f} "
        f"| {len(samples)} synth ({n_af_classes} classes vs {n_real_classes} real)"
    )

    real_correct = (y_pred_r == y_test)
    af_correct = (y_pred_af == y_test)
    af_loses = [
        i for i in range(len(real_test))
        if real_correct[i] and not af_correct[i]
    ]
    af_wins = [
        i for i in range(len(real_test))
        if af_correct[i] and not real_correct[i]
    ]
    print(f"  AttrForge LOSES (real right, AF wrong): {len(af_loses)} items")
    print(f"  AttrForge WINS (AF right, real wrong):   {len(af_wins)} items")
    print()

    rng = random.Random(seed)
    print(f"--- {min(sample_failures, len(af_loses))} sampled FAILURE items (AttrForge wrong, real right) ---")
    for i in rng.sample(af_loses, min(sample_failures, len(af_loses))):
        text = real_test[i].text[:110].replace("\n", " ")
        print(f"  [TRUE={y_test[i]:<20} AF={y_pred_af[i]:<20}] {text}")

    # For 3 failed items, show 3 nearest AttrForge synth samples to surface
    # what the generator IS producing vs what the test items look like.
    if af_loses and samples:
        print()
        print("--- 3 failed items + 3 nearest AttrForge synth samples per item ---")
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(X_test[af_loses[:3]], X_af)
        for k, i in enumerate(af_loses[:3]):
            test_text = real_test[i].text[:120].replace("\n", " ")
            print(f"  Test item {i} (TRUE={y_test[i]}): {test_text}")
            nearest = np.argsort(-sims[k])[:3]
            for j in nearest:
                synth_text = samples[j].text[:120].replace("\n", " ")
                req_label = samples[j].requested_attributes.get(label_attr, "?")
                print(f"    -> synth (req={req_label}, sim={sims[k][j]:.2f}): {synth_text}")
            print()


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    splits = REPO / "experiments" / "_splits"
    inspect_dataset(
        "Customer-support (5 classes, N=10 test, original)",
        splits / "real_train.jsonl",
        splits / "real_test.jsonl",
        REPO / "experiments" / "v2_n10" / "full_attrforge",
        seed=17,
        sample_failures=10,
    )
    inspect_dataset(
        "Banking77 (10 selected classes, N=400 test)",
        splits / "banking77_real_train.jsonl",
        splits / "banking77_real_test.jsonl",
        REPO / "experiments" / "banking77_run_001_seed17" / "full_attrforge",
        seed=17,
        sample_failures=12,
    )
    inspect_dataset(
        "TREC (6 classes, N=89 test)",
        splits / "trec_real_train.jsonl",
        splits / "trec_real_test.jsonl",
        REPO / "experiments" / "trec_run_001_seed17" / "full_attrforge",
        seed=17,
        sample_failures=10,
    )


if __name__ == "__main__":
    main()
