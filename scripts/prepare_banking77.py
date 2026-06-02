"""Prepare a Banking77 10-class semantically-overlapping subset.

Banking77 (Casanueva et al. 2020, mteb/banking77 on HuggingFace) has 77
fine-grained banking intents. Several CLUSTERS of intents semantically
overlap in everyday wording, e.g. all card-related issues use the same
vocabulary ("card", "blocked", "payment"). These clusters are exactly where
the v1 customer-support task's keyword anchor advantage breaks down and
where AttrForge's diversity should matter most.

We pick 10 intents from one dense cluster: "card and payment problems":
  - card_arrival
  - card_not_working
  - card_payment_fee_charged
  - card_payment_not_recognised
  - card_payment_wrong_exchange_rate
  - card_swallowed
  - declined_card_payment
  - lost_or_stolen_card
  - pending_card_payment
  - top_up_failed

These 10 share heavy vocabulary overlap ("card", "payment", "transaction").
A TF-IDF or sentence-transformer classifier with limited real data has to
make fine semantic distinctions. This is the regime the multi-critic loop
is built for.

Splits:
  - real seed for the generator (analogous to v1's 30-sample real_train):
      30 examples per class * 10 classes = 300 (we keep 5, 10, 15, 20, 25, 30
      stratified for the scarce-real sweep)
  - held-out real test: 50 examples per class * 10 classes = 500 items (vs v1's
      10-item test). This finally gives the worst-class F1 metric meaningful
      resolution.

Output:
  experiments/_splits/banking77_real_train.jsonl
  experiments/_splits/banking77_real_test.jsonl
  examples/banking77/config.yaml
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset


REPO = Path(__file__).resolve().parents[1]
OUT_SPLITS = REPO / "experiments" / "_splits"
OUT_CONFIG_DIR = REPO / "examples" / "banking77"

# 10 semantically overlapping card / payment intents
SUBSET_LABELS = [
    "card_arrival",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "declined_card_payment",
    "lost_or_stolen_card",
    "pending_card_payment",
    "top_up_failed",
]

N_TRAIN_PER_CLASS = 30
N_TEST_PER_CLASS = 50
SEED = 17


def main():
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    OUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading mteb/banking77...")
    ds_train = load_dataset("mteb/banking77", split="train")
    ds_test  = load_dataset("mteb/banking77", split="test")

    # Group by label_text
    by_label_train: dict[str, list[str]] = defaultdict(list)
    for r in ds_train:
        if r["label_text"] in SUBSET_LABELS:
            by_label_train[r["label_text"]].append(r["text"])
    by_label_test: dict[str, list[str]] = defaultdict(list)
    for r in ds_test:
        if r["label_text"] in SUBSET_LABELS:
            by_label_test[r["label_text"]].append(r["text"])

    rng = random.Random(SEED)

    train_split = []
    test_split  = []
    for label in SUBSET_LABELS:
        tr = by_label_train[label]
        te = by_label_test[label]
        rng.shuffle(tr)
        rng.shuffle(te)
        if len(tr) < N_TRAIN_PER_CLASS:
            print(f"  WARN: only {len(tr)} train items for {label}; using all")
        if len(te) < N_TEST_PER_CLASS:
            print(f"  WARN: only {len(te)} test items for {label}; using all")
        for t in tr[:N_TRAIN_PER_CLASS]:
            train_split.append({"text": t, "label": label})
        for t in te[:N_TEST_PER_CLASS]:
            test_split.append({"text": t, "label": label})

    print(f"Train: {len(train_split)} items ({N_TRAIN_PER_CLASS} per class).")
    print(f"Test:  {len(test_split)} items ({N_TEST_PER_CLASS} per class).")

    train_path = OUT_SPLITS / "banking77_real_train.jsonl"
    test_path  = OUT_SPLITS / "banking77_real_test.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for r in train_split:
            f.write(json.dumps(r) + "\n")
    with test_path.open("w", encoding="utf-8") as f:
        for r in test_split:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {train_path}")
    print(f"Saved {test_path}")

    # Write a config.yaml analogous to examples/customer_support/config.yaml.
    # We keep the same six attribute schema (label, difficulty, ambiguity, style,
    # noise, scenario type) so the generator and critics need no code change.
    config = {
        "task_name": "banking77_card_payment_10cls",
        "real_examples_path": str(train_path.relative_to(REPO)),
        "test_examples_path": str(test_path.relative_to(REPO)),
        "attribute_schema": {
            "intent": SUBSET_LABELS,
            "difficulty": ["easy", "medium", "hard"],
            "ambiguity": ["clear", "borderline", "ambiguous"],
            "style": ["formal", "colloquial", "abrupt"],
            "noise": ["clean", "minor_typos", "code_switching"],
            "scenario_type": ["transaction_issue", "card_handling", "general_inquiry"],
        },
        # Same per-iteration budget as v1 main_run_002 for direct comparability.
        "iterations": 3,
        "samples_per_iteration": 16,
        "seeds": [17, 23, 41, 53, 89],
    }
    config_path = OUT_CONFIG_DIR / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Saved {config_path}")


if __name__ == "__main__":
    main()
