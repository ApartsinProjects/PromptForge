"""v2 unified dashboard: run all v2 evaluations and print a single summary
table per metric, suitable for paper writing.

Runs (and re-runs) on whatever seed dirs exist under experiments/<base>_seed*:
  - worst_class_eval.py     -> worst_class.json
  - adversarial_robustness_eval.py (skip-paraphrase) -> robustness.json
  - tta_eval.py             -> tta.json
  - quality_weighted_eval.py -> quality_weighted.json

Then prints a final composite summary:

  Metric (higher is better unless noted):
      macro F1 single (v1 headline)
      macro F1 quality-weighted (v2 method modification)
      worst-class F1 single
      worst-class F1 quality-weighted
      |macro change| under back-translation (lower is better)
      |worst change| under back-translation (lower is better)
      TTA gain (TTA-mean macro - single macro)

For each metric: per-condition mean +/- std, and paired-t / Wilcoxon stats
for full_attrforge vs full_classic.

Run after every seed extension to see the headline numbers move in real time.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd, label):
    print(f"\n>>> {label}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  WARN: exit {r.returncode}; stderr tail:")
        print("  " + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
    # print the structured summary tail
    out = r.stdout
    tail = "\n".join(out.splitlines()[-25:])
    print(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="main_run_002")
    ap.add_argument("--skip-paraphrase", action="store_true", default=True,
                    help="Reuse the cached real_test_paraphrased.jsonl rather than regenerating.")
    args = ap.parse_args()

    print(f"v2 dashboard for base={args.base!r}\n")

    # 1) Per-class aggregation (must exist before worst-class)
    run([PY, "scripts/aggregate_multiseed.py", "--base", args.base],
        "1. multiseed aggregation (table.csv, summary.json, ...)")

    # 2) Worst-class F1 sweep
    if (REPO / f"experiments/{args.base}_aggregated/per_class_aug.json").exists():
        run([PY, "scripts/per_class_aug_eval.py", "--base", args.base],
            "2a. per-class aug (regenerate)")
    run([PY, "scripts/worst_class_eval.py", "--base", args.base],
        "2b. worst-class F1 sweep")

    # 3) Adversarial robustness (back-translation cached)
    if args.skip_paraphrase and (REPO / "experiments/_splits/real_test_paraphrased.jsonl").exists():
        run([PY, "scripts/adversarial_robustness_eval.py", "--base", args.base, "--skip-paraphrase"],
            "3. adversarial robustness via back-translation (cached paraphrases)")
    else:
        run([PY, "scripts/adversarial_robustness_eval.py", "--base", args.base],
            "3. adversarial robustness via back-translation")

    # 4) TTA
    run([PY, "scripts/tta_eval.py", "--base", args.base],
        "4. test-time augmentation via back-translation")

    # 5) Quality-weighted augmentation (the v2 method modification)
    run([PY, "scripts/quality_weighted_eval.py", "--base", args.base, "--low-weight", "0.3"],
        "5. quality-weighted augmentation (v2 method modification)")


if __name__ == "__main__":
    main()
