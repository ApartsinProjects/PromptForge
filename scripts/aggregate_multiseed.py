"""Aggregate multi-seed experiment + post-hoc audit into the paper's final tables and figures.

Reads:
  experiments/<base>_seed<N>/all_summaries.json    (downstream + final_metrics)
  experiments/<base>_seed<N>/audit/audit_summary.json   (post-hoc audit)

Writes:
  experiments/<base>_aggregated/table.csv         (paper Table 2 contents)
  experiments/<base>_aggregated/per_iter.csv      (per-iter F1 for paper Figure 6)
  experiments/<base>_aggregated/summary.json      (everything together)
  paper/figures/<base>_main_metrics.png           (with error bars, replaces old plot)
  paper/figures/<base>_audit_differential.png     (with error bars across seeds)
  paper/figures/<base>_realism_curve.png          (discriminator accuracy over iterations)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1:
        return (vals[0], 0.0)
    return (statistics.mean(vals), statistics.stdev(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Run-id base; we look for <base>_seed*/.")
    args = ap.parse_args()

    base = args.base
    seed_dirs = sorted((REPO / "experiments").glob(f"{base}_seed*"))
    if not seed_dirs:
        # Maybe single-seed: try the bare name.
        single = REPO / "experiments" / base
        if single.exists():
            seed_dirs = [single]
    if not seed_dirs:
        raise SystemExit(f"no seed directories matching {base}_seed* or {base}")

    print(f"Aggregating {len(seed_dirs)} seed run(s): {[p.name for p in seed_dirs]}")

    # condition -> metric -> list[values across seeds]
    bag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # condition -> iter -> metric -> list[values across seeds]
    bag_per_iter: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for sd in seed_dirs:
        all_path = sd / "all_summaries.json"
        if not all_path.exists():
            continue
        for s in json.loads(all_path.read_text(encoding="utf-8")):
            if "error" in s:
                continue
            cond = s["condition"]
            m = s.get("final_metrics", {}) or {}
            ds = s.get("final_downstream", {}) or {}
            bag[cond]["acc"].append(ds.get("accuracy", 0.0))
            bag[cond]["macro_f1"].append(ds.get("macro_f1", 0.0))
            bag[cond]["disc_acc"].append(m.get("discriminator_accuracy"))
            bag[cond]["attr_match"].append(m.get("attribute_match_rate"))
            bag[cond]["near_dup"].append(m.get("near_duplicate_rate"))
            bag[cond]["combo_cov"].append(m.get("combination_coverage"))
            bag[cond]["pack_acc_loop"].append(m.get("pack_accuracy"))
            bag[cond]["ms_loop"].append(m.get("mode_seeking_ratio"))
            # Per-iteration downstream F1
            for di in s.get("per_iter_downstream", []) or []:
                it = di.get("iteration")
                bag_per_iter[cond][it]["macro_f1"].append(di.get("macro_f1", 0.0))
                bag_per_iter[cond][it]["acc"].append(di.get("accuracy", 0.0))

        # Post-hoc audit (run on every condition's final batch regardless of which critics were enabled).
        audit_path = sd / "audit" / "audit_summary.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            for row in audit.get("conditions", []):
                cond = row["condition"]
                bag[cond]["pack_audit"].append(row["pack_accuracy"])
                bag[cond]["pack_above_null"].append(row.get("pack_accuracy_above_null"))
                bag[cond]["ms_audit"].append(row["mode_seeking_ratio"])
                bag[cond]["ms_rel_real"].append(row["mode_seeking_relative_to_real"])
                bag[cond]["auroc"].append(row["coverage_auroc"])
                bag[cond]["hunter_new"].append(row["n_banned_phrasings_found"])
            bag["_meta"]["null_pack"].append(
                audit.get("null_pack_accuracy_real_vs_real", 0.5)
            )
            bag["_meta"]["real_ms"].append(audit.get("real_mode_seeking_ratio", 0.0))

    null_pack = (
        statistics.mean(bag["_meta"]["null_pack"])
        if bag["_meta"].get("null_pack")
        else 0.5
    )
    real_ms = (
        statistics.mean(bag["_meta"]["real_ms"])
        if bag["_meta"].get("real_ms")
        else 0.0
    )

    conditions = [c for c in bag if not c.startswith("_")]
    # Stable order matching paper Table 1.
    canonical = [
        "naive", "few_shot", "self_critique",
        "realism_only", "diversity_only", "full_classic", "full_attrforge",
    ]
    ordered = [c for c in canonical if c in conditions] + [c for c in conditions if c not in canonical]

    # ---- write the unified CSV ----
    out_dir = REPO / "experiments" / f"{base}_aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cond in ordered:
        d = bag[cond]
        row = {"condition": cond, "n_seeds": len(d["acc"])}
        for key in [
            "acc", "macro_f1", "attr_match", "disc_acc",
            "near_dup", "combo_cov",
            "pack_audit", "pack_above_null", "ms_audit", "ms_rel_real",
            "auroc", "hunter_new",
        ]:
            mean, sd = mean_std(d.get(key, []))
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
        rows.append(row)

    with (out_dir / "table.csv").open("w", encoding="utf-8", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---- per-iter CSV ----
    with (out_dir / "per_iter.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "iteration", "macro_f1_mean", "macro_f1_sd", "acc_mean", "acc_sd", "n_seeds"])
        for cond in ordered:
            for it in sorted(bag_per_iter[cond].keys()):
                d = bag_per_iter[cond][it]
                f1m, f1s = mean_std(d.get("macro_f1", []))
                am, asd = mean_std(d.get("acc", []))
                w.writerow([cond, it, f1m, f1s, am, asd, len(d.get("macro_f1", []))])

    # ---- summary JSON ----
    summary = {
        "base": base,
        "n_seeds": len(seed_dirs),
        "null_pack_accuracy_real_vs_real": null_pack,
        "real_mode_seeking_ratio": real_ms,
        "rows": rows,
        "per_iter": {
            cond: {
                str(it): {
                    "macro_f1_mean": mean_std(d.get("macro_f1", []))[0],
                    "macro_f1_sd": mean_std(d.get("macro_f1", []))[1],
                    "n_seeds": len(d.get("macro_f1", [])),
                }
                for it, d in bag_per_iter[cond].items()
            }
            for cond in ordered
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}/table.csv, per_iter.csv, summary.json")

    # ---- plots with error bars ----
    fig_dir = REPO / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_main(rows, fig_dir / f"{base}_main_metrics.png")
    _plot_audit(rows, null_pack, real_ms, fig_dir / f"{base}_audit_differential.png")
    _plot_realism(bag, ordered, fig_dir / f"{base}_realism_curve.png")
    _plot_per_iter(bag_per_iter, ordered, fig_dir / f"{base}_iteration_curves.png")
    _plot_adversary_metrics(rows, fig_dir / f"{base}_adversary_metrics.png")
    print(f"Wrote figures under {fig_dir}")


def _bar_with_error(ax, xs, means, stds, color, label=None):
    ax.bar(xs, means, color=color, yerr=stds, capsize=4, label=label)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(xs[i], m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)


def _plot_main(rows, out_path):
    conds = [r["condition"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = list(range(len(conds)))
    width = 0.38
    accs = [r["acc_mean"] for r in rows]
    acc_sd = [r["acc_sd"] for r in rows]
    f1s = [r["macro_f1_mean"] for r in rows]
    f1_sd = [r["macro_f1_sd"] for r in rows]
    ax.bar(
        [x - width / 2 for x in xs], accs, width,
        yerr=acc_sd, capsize=3, color="#3a6ea5", label="accuracy",
    )
    ax.bar(
        [x + width / 2 for x in xs], f1s, width,
        yerr=f1_sd, capsize=3, color="#c0392b", label="macro F1",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("downstream score on held-out real test")
    ax.set_title(
        f"Downstream metric by condition (mean ± std across {rows[0]['n_seeds']} seeds)"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_audit(rows, null_pack, real_ms, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    conds = [r["condition"] for r in rows]
    xs = list(range(len(conds)))

    # Pack
    means = [r["pack_audit_mean"] for r in rows]
    sds = [r["pack_audit_sd"] for r in rows]
    ax = axes[0, 0]
    _bar_with_error(ax, xs, means, sds, "#c0392b")
    ax.axhline(0.5, color="#888", linestyle=":", label="chance")
    ax.axhline(null_pack, color="#3a6ea5", linestyle="--", label=f"null real-vs-real ({null_pack:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Pack accuracy on every condition's final batch (lower = more diverse)")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.4)

    # MS relative to real
    means = [r["ms_rel_real_mean"] for r in rows]
    sds = [r["ms_rel_real_sd"] for r in rows]
    ax = axes[0, 1]
    _bar_with_error(ax, xs, means, sds, "#3a6ea5")
    ax.axhline(1.0, color="#888", linestyle="--", label=f"real ms = {real_ms:.3f}")
    ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, max(1.2, max(means) * 1.2 if means else 1.2))
    ax.set_title("Mode-seeking ratio relative to real (higher = more attribute-responsive)")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.4)

    # AUROC
    means = [r["auroc_mean"] for r in rows]
    sds = [r["auroc_sd"] for r in rows]
    ax = axes[1, 0]
    _bar_with_error(ax, xs, means, sds, "#7c4dff")
    ax.axhline(0.5, color="#888", linestyle=":", label="indistinguishable (0.5)")
    ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Coverage-hole AUROC (lower = more real-like coverage)")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Hunter
    means = [r["hunter_new_mean"] for r in rows]
    sds = [r["hunter_new_sd"] for r in rows]
    ax = axes[1, 1]
    _bar_with_error(ax, xs, means, sds, "#2e715a")
    ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    ax.set_title("LLM tics detected on final batch by Mode Hunter audit (lower = fewer)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Post-hoc adversary audit on every condition's final batch (mean ± std across seeds)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_realism(bag, ordered, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = list(range(len(ordered)))
    means = []
    sds = []
    for c in ordered:
        m, s = mean_std(bag[c].get("disc_acc", []))
        means.append(m)
        sds.append(s)
    ax.bar(xs, means, yerr=sds, capsize=4, color="#3a6ea5")
    ax.axhline(0.5, color="#888", linestyle=":", label="chance (target)")
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(ordered, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Realism: discriminator accuracy (closer to 0.5 = more realistic)")
    ax.set_ylabel("discriminator accuracy on mixed batch")
    ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_per_iter(bag_per_iter, ordered, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.get_cmap("tab10", max(len(ordered), 1))
    for i, c in enumerate(ordered):
        iters = sorted(bag_per_iter[c].keys())
        if not iters:
            continue
        means = [mean_std(bag_per_iter[c][it].get("macro_f1", []))[0] for it in iters]
        sds = [mean_std(bag_per_iter[c][it].get("macro_f1", []))[1] for it in iters]
        ax.errorbar(iters, means, yerr=sds, marker="o", label=c, color=cmap(i), capsize=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("downstream macro F1 (per-iter training set)")
    ax.set_title("Per-iteration downstream macro F1 (mean ± std)")
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_adversary_metrics(rows, out_path):
    conds = [r["condition"] for r in rows]
    xs = list(range(len(conds)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))

    def panel(ax, key, color, title, ylim=None):
        means = [r[f"{key}_mean"] for r in rows]
        sds = [r[f"{key}_sd"] for r in rows]
        _bar_with_error(ax, xs, means, sds, color)
        ax.set_xticks(xs); ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    panel(axes[0, 0], "pack_audit", "#c0392b",
          "Pack accuracy (POST-HOC audit, lower = more diverse)", (0, 1))
    panel(axes[0, 1], "ms_rel_real", "#3a6ea5",
          "Mode-seeking ratio relative to real (higher = better)")
    panel(axes[1, 0], "disc_acc", "#7c4dff",
          "Realism: discriminator accuracy (closer to 0.5 = more realistic)", (0, 1))
    panel(axes[1, 1], "attr_match", "#2e715a",
          "Attribute match rate (higher = better fidelity)", (0, 1.1))

    fig.suptitle("Headline metrics with error bars across seeds", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
