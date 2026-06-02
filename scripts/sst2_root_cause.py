"""SST-2 root-cause via specific failing datapoints.

Applies the new global rule: when a method has a gap to a published baseline,
inspect the SPECIFIC test items the method gets wrong vs a reference, to find
the systematic failure pattern. The fix then targets that pattern.

Loads SST-2 seed 17 full_attrforge synthetic samples + 60 real seed samples,
trains synth-only and real-only classifiers, finds the test items where
each succeeds and the other fails, and prints them for inspection.
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

from synsmith.schema import RealExample, SyntheticSample, load_jsonl  # noqa: E402


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    real_train = [
        RealExample.model_validate(r)
        for r in load_jsonl(REPO / "experiments/_splits/sst2_real_train.jsonl")
    ]
    real_test = [
        RealExample.model_validate(r)
        for r in load_jsonl(REPO / "experiments/_splits/sst2_real_test.jsonl")
    ]
    real_texts = [r.text for r in real_train]
    real_labels = np.array([r.label for r in real_train])
    test_texts = [r.text for r in real_test]
    test_labels = np.array([r.label for r in real_test])
    enc = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    X_real = enc.encode(real_texts, normalize_embeddings=True, show_progress_bar=False)
    X_test = enc.encode(test_texts, normalize_embeddings=True, show_progress_bar=False)

    # Load SynSmith seed 17 synth
    cd = REPO / "experiments/sst2_run_001_seed17/full_attrforge"
    samples = []
    for it in sorted(cd.glob("*/iter_*")):
        sj = it / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                samples.append(SyntheticSample.model_validate(r))
    af_texts = [s.text for s in samples]
    af_labels = np.array([s.requested_attributes.get("intent", "?") for s in samples])
    print(
        f"SynSmith seed 17: {len(samples)} synth "
        f"({sum(1 for l in af_labels if l == 'positive')} pos, "
        f"{sum(1 for l in af_labels if l == 'negative')} neg)"
    )

    clf_r = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=17
    )
    clf_r.fit(X_real, real_labels)
    y_r = clf_r.predict(X_test)
    X_af = enc.encode(af_texts, normalize_embeddings=True, show_progress_bar=False)
    clf_af = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced", random_state=17
    )
    clf_af.fit(X_af, af_labels)
    y_af = clf_af.predict(X_test)

    real_correct = (y_r == test_labels)
    af_correct = (y_af == test_labels)

    af_wrong_real_right = [
        i for i in range(len(test_texts)) if real_correct[i] and not af_correct[i]
    ]
    af_right_real_wrong = [
        i for i in range(len(test_texts)) if af_correct[i] and not real_correct[i]
    ]
    print()
    print(f"Real-only acc: {real_correct.mean():.3f}")
    print(f"SynSmith acc: {af_correct.mean():.3f}")
    print(
        f"Real RIGHT + SynSmith WRONG: {len(af_wrong_real_right)} items "
        f"(the cost gap)"
    )
    print(
        f"SynSmith RIGHT + Real WRONG: {len(af_right_real_wrong)} items "
        f"(SynSmith wins)"
    )

    rng = random.Random(17)
    print()
    print("=== SynSmith FAILS, real-only SUCCEEDS (the gap; sample 15) ===")
    for i in rng.sample(af_wrong_real_right, min(15, len(af_wrong_real_right))):
        print(
            f"  [TRUE={test_labels[i]:<9} AF={y_af[i]:<9}] {test_texts[i][:100]}"
        )
    print()
    print("=== SynSmith WINS over real-only (sample 10) ===")
    for i in rng.sample(af_right_real_wrong, min(10, len(af_right_real_wrong))):
        print(f"  [TRUE={test_labels[i]:<9} R={y_r[i]:<9}] {test_texts[i][:100]}")

    # Save full lists
    out = REPO / "experiments/_diagnostics/sst2_failing_items.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "af_wrong_real_right": [
                    (int(i), test_texts[i], str(test_labels[i]), str(y_af[i]), str(y_r[i]))
                    for i in af_wrong_real_right
                ],
                "af_right_real_wrong": [
                    (int(i), test_texts[i], str(test_labels[i]), str(y_af[i]), str(y_r[i]))
                    for i in af_right_real_wrong
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
