"""Post-hoc Manifold-Entropy audit.

Runs the Manifold-Entropy critic (a deterministic sentence-transformer-
based diversity metric, in the same family as Vendi and Scendi but
measuring the effective rank of the sample-similarity kernel) on every
condition's final pooled batch across all seeds. Output: a per-condition
mean +/- std of three quantities:

    manifold_entropy   : Shannon entropy of normalized eigenvalues
    effective_rank     : exp(entropy), the "how many distinct directions
                         the batch spans" number
    eigenvalue_decay   : ratio of the 2nd-largest to the 1st-largest
                         eigenvalue (collapse alarm when low)

Saves: experiments/<base>_aggregated/manifold_entropy.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import synsmith  # noqa: E402
from synsmith.critics.manifold_entropy import (  # noqa: E402
    ManifoldEntropy,
    ManifoldEntropyConfig,
)
from synsmith.schema import SyntheticSample, load_jsonl  # noqa: E402


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
    args = ap.parse_args()

    critic = ManifoldEntropy(ManifoldEntropyConfig(use_embeddings=True))

    seed_dirs = sorted((REPO / "experiments").glob(f"{args.base}_seed*"))
    bag: dict[str, list[dict[str, float]]] = defaultdict(list)

    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("seed")[-1])
        except ValueError:
            continue
        for cond_dir in sorted(
            p for p in sd.iterdir()
            if p.is_dir() and p.name not in {"audit", "aggregated"}
        ):
            samples = load_synth(cond_dir)
            if not samples:
                continue
            r = critic.score(samples)
            bag[cond_dir.name].append(
                {
                    "seed": seed,
                    "manifold_entropy": r.manifold_entropy,
                    "effective_rank": r.effective_rank,
                    "eigenvalue_decay": r.eigenvalue_decay,
                    "n_samples": r.n_samples,
                }
            )

    conds = [
        "naive", "few_shot", "self_critique", "realism_only",
        "diversity_only", "full_classic", "full_attrforge",
        "no_pack", "no_mode_seeking", "no_mode_hunter", "no_coverage_hole",
    ]

    print()
    print(
        f"{'condition':<22} {'manifold_entropy':<24} {'effective_rank':<24} {'eigenvalue_decay':<20}"
    )
    for c in conds:
        rows = bag.get(c, [])
        if not rows:
            continue
        def fmt(key: str) -> str:
            v = [r[key] for r in rows]
            return f"{statistics.mean(v):.3f} +- {statistics.stdev(v) if len(v) > 1 else 0:.3f}"
        print(
            f"{c:<22} {fmt('manifold_entropy'):<24} {fmt('effective_rank'):<24} {fmt('eigenvalue_decay'):<20}"
        )

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {c: bag[c] for c in conds if c in bag}
    (out_dir / "manifold_entropy.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_dir}/manifold_entropy.json")

    # Iterated-vs-non-iterated ratio (paper-friendly summary).
    non_iter = [
        statistics.mean(r["effective_rank"] for r in bag[c])
        for c in ["naive", "few_shot"]
        if c in bag
    ]
    iter_set = [
        statistics.mean(r["effective_rank"] for r in bag[c])
        for c in [
            "self_critique", "realism_only", "diversity_only",
            "full_classic", "full_attrforge",
        ]
        if c in bag
    ]
    if non_iter and iter_set:
        print()
        print(f"Non-iterated effective_rank mean: {statistics.mean(non_iter):.3f}")
        print(f"Iterated     effective_rank mean: {statistics.mean(iter_set):.3f}")
        ratio = statistics.mean(iter_set) / max(statistics.mean(non_iter), 1e-9)
        print(f"Iterated/non-iterated effective_rank ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
