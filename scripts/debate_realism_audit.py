"""Post-hoc 3-judge debate realism audit (scout D3.1).

Runs the 3-judge debate realism critic on EXISTING full_attrforge final
batches (already in experiments/<base>_seed*/full_attrforge/...) and
compares the trio's verdicts against the single-judge baseline that the
loop already produced. No new generator runs are needed; the only API
cost is the 3-judge debate calls (~$0.03 per seed at 3x gpt-4o-mini-class
model pricing through OpenRouter).

Requires the OPENROUTER_API_KEY environment variable. The three default
judges are:

    openai/gpt-4o-mini       (OpenAI family, same as the v2 single judge)
    anthropic/claude-3-haiku (Anthropic family, anti-bias control)
    google/gemini-flash-1.5  (Google family, anti-bias control)

Outputs:
    experiments/<base>_aggregated/debate_realism_audit.json

The JSON carries per-seed per-condition: per-judge accuracy, majority-vote
accuracy, judge_agreement, ks_statistic, stopped_early, n_judges_called.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attrforge.critics.debate_discriminator import (  # noqa: E402
    DebateConfig,
    DebateJudge,
    RealismDebate,
)
from attrforge.schema import (  # noqa: E402
    RealExample,
    SyntheticSample,
    load_jsonl,
)


def load_synth(cond_dir: Path) -> list[SyntheticSample]:
    out = []
    for iter_dir in sorted(cond_dir.glob("*/iter_*")):
        sj = iter_dir / "samples.jsonl"
        if sj.exists():
            for r in load_jsonl(sj):
                out.append(SyntheticSample.model_validate(r))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument(
        "--condition",
        default="full_attrforge",
        help="Condition whose final batches we audit.",
    )
    ap.add_argument(
        "--n-real",
        type=int,
        default=8,
        help="How many real examples to pair against the synthetic batch.",
    )
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "ERROR: OPENROUTER_API_KEY is not set. Add it to .env or "
            "export it directly. Get a key at https://openrouter.ai/keys."
        )
        sys.exit(2)

    from scripts._splits_resolver import resolve_splits

    real_train_path, _ = resolve_splits(args.base)
    real_train = [
        RealExample.model_validate(r) for r in load_jsonl(real_train_path)
    ]
    # Use the first N real exemplars to keep the prompt size bounded.
    real_subset = real_train[: args.n_real]

    seed_dirs = sorted(
        (REPO / "experiments").glob(f"{args.base}_seed*")
    )
    if not seed_dirs:
        print(f"No seed dirs match {args.base}_seed*")
        sys.exit(2)

    print(f"Auditing {len(seed_dirs)} seeds with 3-judge debate on '{args.condition}'.")

    debate = RealismDebate(
        config=DebateConfig(
            judges=[
                DebateJudge(name="gpt-4o-mini", model="openai/gpt-4o-mini"),
                DebateJudge(
                    name="claude-3-haiku", model="anthropic/claude-3-haiku"
                ),
                DebateJudge(
                    name="gemini-flash-1.5", model="google/gemini-flash-1.5"
                ),
            ],
            ks_threshold=0.10,
            unanimity_threshold=0.80,
            seed=17,
        )
    )

    per_seed: list[dict] = []
    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        cond_dir = sd / args.condition
        if not cond_dir.exists():
            print(f"  seed {seed}: missing {args.condition} dir; skipping")
            continue
        synth = load_synth(cond_dir)
        if not synth:
            print(f"  seed {seed}: no synthetic samples; skipping")
            continue
        # Cap synthetic batch to match real subset size (avoid prompt bloat).
        synth = synth[: args.n_real]
        print(
            f"  seed {seed}: {len(real_subset)} real + {len(synth)} synthetic ..."
        )
        result = debate.judge(real_subset, synth)
        per_seed.append(
            {
                "seed": seed,
                "per_judge_accuracy": result.per_judge_accuracy,
                "majority_accuracy": result.majority_accuracy,
                "judge_agreement": result.judge_agreement,
                "ks_statistic": result.ks_statistic,
                "stopped_early": result.stopped_early,
                "n_judges_called": result.n_judges_called,
            }
        )

    if not per_seed:
        print("No results collected.")
        sys.exit(2)

    # Aggregate.
    print()
    print(f"=== 3-judge debate realism audit (N={len(per_seed)} seeds) ===")
    judges = list(per_seed[0]["per_judge_accuracy"].keys())
    for j in judges:
        accs = [
            r["per_judge_accuracy"][j]
            for r in per_seed
            if j in r["per_judge_accuracy"]
        ]
        if accs:
            print(
                f"  {j:<22} acc = {statistics.mean(accs):.3f} +- "
                f"{statistics.stdev(accs) if len(accs) > 1 else 0:.3f}"
            )
    mj = [r["majority_accuracy"] for r in per_seed]
    ag = [r["judge_agreement"] for r in per_seed]
    ks = [r["ks_statistic"] for r in per_seed]
    early = sum(1 for r in per_seed if r["stopped_early"])
    print(
        f"  majority-vote acc      = {statistics.mean(mj):.3f} +- "
        f"{statistics.stdev(mj) if len(mj) > 1 else 0:.3f}"
    )
    print(
        f"  judge agreement (unanimity) = {statistics.mean(ag):.3f} +- "
        f"{statistics.stdev(ag) if len(ag) > 1 else 0:.3f}"
    )
    print(
        f"  KS statistic (max judge spread) = {statistics.mean(ks):.3f} +- "
        f"{statistics.stdev(ks) if len(ks) > 1 else 0:.3f}"
    )
    print(
        f"  stopped early (KS halted before round 3): "
        f"{early}/{len(per_seed)} seeds"
    )

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "debate_realism_audit.json").write_text(
        json.dumps(per_seed, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_dir}/debate_realism_audit.json")


if __name__ == "__main__":
    main()
