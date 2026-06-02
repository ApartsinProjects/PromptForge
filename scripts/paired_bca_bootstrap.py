"""Paired BCa bootstrap + sign-flip permutation re-analysis.

Implements the protocol from "When +1% Is Not Enough" (arXiv:2511.19794,
Nov 2025), specifically designed for small-seed evaluation budgets where
the conventional paired-t test under-powers gains in the 0.6-2.0 percentage-
point range.

Method: for a paired (control, treatment) pair of per-seed scores,
    1. Compute the per-seed delta.
    2. Resample seeds with replacement B=10000 times; compute B bootstrap
       means of the deltas. Apply BCa (bias-corrected and accelerated)
       intervals: shift the percentile interval by the bias factor z0
       and the acceleration factor a estimated by jackknife.
    3. Sign-flip permutation: for each of P=10000 permutations, randomly
       flip each delta's sign (under the null hypothesis: control and
       treatment are exchangeable). Two-sided p = fraction of permutations
       whose absolute mean delta >= observed |mean delta|.

The paired BCa CI corrects for skew and bias the percentile interval misses;
the sign-flip permutation is conditioned on the actual data, not a Normal
assumption.

Re-evaluates the following claims from the paper using existing data:
    H1: ensemble (sc+full_attrforge) macro F1 vs realism_only solo macro F1 at N=10
    H2: ensemble worst-class F1 vs full_attrforge solo worst-class F1 at N=10
    H3: ensemble worst-class F1 vs realism_only solo worst-class F1 at N=10
    H4: full_attrforge vs full_classic on full N=10 macro F1

Output:
    experiments/main_run_002_aggregated/paired_bca_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def paired_bca_ci(deltas: np.ndarray, alpha: float = 0.05, n_boot: int = 10_000, rng_seed: int = 17) -> tuple[float, float, float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI for the mean of paired deltas.

    Returns (mean, ci_lo, ci_hi, p_two_sided_via_ci_inversion).
    """
    rng = np.random.default_rng(rng_seed)
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    obs_mean = float(np.mean(deltas))
    # Bootstrap distribution
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[b] = deltas[idx].mean()
    # Bias correction z0
    from scipy.stats import norm
    p_below = float((boot_means < obs_mean).mean())
    if p_below in (0.0, 1.0):
        p_below = max(min(p_below, 1 - 1e-6), 1e-6)
    z0 = norm.ppf(p_below)
    # Acceleration via jackknife
    jk = np.empty(n)
    for i in range(n):
        jk[i] = np.delete(deltas, i).mean()
    jk_mean = jk.mean()
    numerator = float(np.sum((jk_mean - jk) ** 3))
    denominator = 6.0 * (float(np.sum((jk_mean - jk) ** 2)) ** 1.5)
    a = numerator / denominator if denominator > 0 else 0.0
    # Adjusted alpha
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)
    def bca_q(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    q_lo = bca_q(z_lo)
    q_hi = bca_q(z_hi)
    ci_lo = float(np.percentile(boot_means, 100 * q_lo))
    ci_hi = float(np.percentile(boot_means, 100 * q_hi))
    # p value via CI inversion: smallest alpha at which the CI excludes 0
    # (rough approximation; we'll report a CI-based "excludes zero?" flag)
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    return obs_mean, ci_lo, ci_hi, float(excludes_zero)


def sign_flip_p(deltas: np.ndarray, n_perm: int = 10_000, rng_seed: int = 17) -> float:
    """Two-sided sign-flip permutation test.

    Under the null hypothesis that control and treatment are exchangeable
    on a per-seed basis, each delta's sign is exchangeable. p = fraction of
    permutations whose |mean delta| >= |observed mean delta|.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(deltas)
    obs = abs(np.mean(deltas))
    if obs == 0:
        return 1.0
    n_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs(np.mean(deltas * signs)) >= obs:
            n_extreme += 1
    return (n_extreme + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="main_run_002",
        help="Aggregated base name (must have ensemble_deep.json).",
    )
    args = ap.parse_args()

    p_in = REPO / "experiments" / f"{args.base}_aggregated" / "ensemble_deep.json"
    d = json.load(open(p_in, encoding="utf-8"))
    solo = d["solo"]
    pairs = d["pairs"]

    target_pair = ("self_critique", "full_attrforge")
    rec = [
        p for p in pairs if (p["a"], p["b"]) == target_pair or (p["b"], p["a"]) == target_pair
    ][0]
    ens_macros = np.asarray(rec["macros"])
    ens_worsts = np.asarray(rec["worsts"])

    # H1: ensemble macro F1 vs realism_only solo macro F1
    r_macros = np.asarray(solo["realism_only"]["macros"])
    r_worsts = np.asarray(solo["realism_only"]["worsts"])
    # H4: full_attrforge vs full_classic on macro F1 (solo vs solo)
    af_macros = np.asarray(solo["full_attrforge"]["macros"])
    af_worsts = np.asarray(solo["full_attrforge"]["worsts"])
    fc_macros = np.asarray(solo["full_classic"]["macros"])
    fc_worsts = np.asarray(solo["full_classic"]["worsts"])

    tests = [
        ("H1: ENS macro vs realism_only solo macro", ens_macros - r_macros),
        ("H2: ENS worst vs full_attrforge solo worst", ens_worsts - af_worsts),
        ("H3: ENS worst vs realism_only solo worst", ens_worsts - r_worsts),
        ("H4: full_attrforge solo macro vs full_classic solo macro", af_macros - fc_macros),
        ("H5: full_attrforge solo worst vs full_classic solo worst", af_worsts - fc_worsts),
        ("H6: ENS macro vs full_attrforge solo macro", ens_macros - af_macros),
        ("H7: ENS macro vs full_classic solo macro", ens_macros - fc_macros),
    ]

    print()
    print(f"{'Hypothesis':<58} {'mean':<10} {'BCa 95% CI':<26} {'sign-flip p':<14}")
    out = {}
    for label, deltas in tests:
        m, lo, hi, _ = paired_bca_ci(deltas)
        p_sf = sign_flip_p(deltas)
        sig = " *" if (lo > 0 or hi < 0) else ("." if p_sf < 0.10 else "")
        print(
            f"  {label:<56} {m:+.3f}    [{lo:+.3f}, {hi:+.3f}]    p = {p_sf:.3f}{sig}"
        )
        out[label] = {
            "deltas": [float(x) for x in deltas],
            "mean_delta": float(m),
            "bca_ci_lo": float(lo),
            "bca_ci_hi": float(hi),
            "sign_flip_p": float(p_sf),
            "n_seeds": len(deltas),
        }

    out_dir = REPO / "experiments" / f"{args.base}_aggregated"
    (out_dir / "paired_bca_bootstrap.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {out_dir}/paired_bca_bootstrap.json")
    print(
        "\nLegend: * = BCa 95% CI excludes zero (effectively significant at 0.05);"
        " . = sign-flip p < 0.10."
    )


if __name__ == "__main__":
    main()
