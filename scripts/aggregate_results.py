"""Aggregate experiments/<run_id>/ outputs into CSV tables and PNG plots.

Produces:
  experiments/<run_id>/aggregated/conditions.csv
  experiments/<run_id>/aggregated/per_iter.csv
  experiments/<run_id>/aggregated/per_class_f1.csv
  paper/figures/<run_id>_main_metrics.png
  paper/figures/<run_id>_pareto.png
  paper/figures/<run_id>_per_class_f1.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_summaries(run_dir: Path) -> list[dict]:
    all_path = run_dir / "all_summaries.json"
    if all_path.exists():
        return json.loads(all_path.read_text(encoding="utf-8"))
    # Fall back to scanning per-condition summary.json.
    out = []
    for cond_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        summary = cond_dir / "summary.json"
        if summary.exists():
            out.append(json.loads(summary.read_text(encoding="utf-8")))
    return out


def write_conditions_csv(summaries: list[dict], out_path: Path) -> None:
    fieldnames = [
        "condition",
        "iterations",
        "total_samples",
        "attribute_match_rate",
        "discriminator_accuracy",
        "synthetic_detection_rate",
        "near_duplicate_rate",
        "combination_coverage",
        "pack_accuracy",
        "mode_seeking_ratio",
        "banned_phrasings_total",
        "coverage_classifier_auroc",
        "downstream_accuracy",
        "downstream_macro_f1",
        "wall_time_seconds",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            if "error" in s:
                continue
            m = s.get("final_metrics", {}) or {}
            ds = s.get("final_downstream", {}) or {}
            w.writerow(
                {
                    "condition": s.get("condition"),
                    "iterations": s.get("iterations"),
                    "total_samples": s.get("total_samples"),
                    "attribute_match_rate": m.get("attribute_match_rate"),
                    "discriminator_accuracy": m.get("discriminator_accuracy"),
                    "synthetic_detection_rate": m.get("synthetic_detection_rate"),
                    "near_duplicate_rate": m.get("near_duplicate_rate"),
                    "combination_coverage": m.get("combination_coverage"),
                    "pack_accuracy": m.get("pack_accuracy"),
                    "mode_seeking_ratio": m.get("mode_seeking_ratio"),
                    "banned_phrasings_total": m.get("banned_phrasings_total"),
                    "coverage_classifier_auroc": m.get("coverage_classifier_auroc"),
                    "downstream_accuracy": ds.get("accuracy"),
                    "downstream_macro_f1": ds.get("macro_f1"),
                    "wall_time_seconds": s.get("wall_time_seconds"),
                }
            )


def write_per_iter_csv(summaries: list[dict], out_path: Path) -> None:
    rows: list[dict] = []
    for s in summaries:
        if "error" in s:
            continue
        for di in s.get("per_iter_downstream", []):
            rows.append(
                {
                    "condition": s["condition"],
                    "iteration": di.get("iteration"),
                    "accuracy": di.get("accuracy"),
                    "macro_f1": di.get("macro_f1"),
                    "n_train": di.get("n_train"),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["condition", "iteration", "accuracy", "macro_f1", "n_train"]
        )
        w.writeheader()
        w.writerows(rows)


def write_per_class_csv(summaries: list[dict], out_path: Path) -> None:
    rows: list[dict] = []
    for s in summaries:
        if "error" in s:
            continue
        ds = s.get("final_downstream", {}) or {}
        per_class = ds.get("per_class_f1", {}) or {}
        support = ds.get("per_class_support", {}) or {}
        for lbl, f1 in per_class.items():
            rows.append(
                {
                    "condition": s["condition"],
                    "label": lbl,
                    "f1": f1,
                    "support": support.get(lbl, 0),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "label", "f1", "support"])
        w.writeheader()
        w.writerows(rows)


def plot_main_metrics(summaries: list[dict], out_path: Path) -> None:
    conds = [s["condition"] for s in summaries if "error" not in s]
    f1 = [
        (s.get("final_downstream", {}) or {}).get("macro_f1", 0.0)
        for s in summaries
        if "error" not in s
    ]
    acc = [
        (s.get("final_downstream", {}) or {}).get("accuracy", 0.0)
        for s in summaries
        if "error" not in s
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(conds))
    width = 0.38
    ax.bar([i - width / 2 for i in x], acc, width, label="accuracy", color="#3a6ea5")
    ax.bar([i + width / 2 for i in x], f1, width, label="macro F1", color="#c0392b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("downstream score on held-out real test")
    ax.set_title("Downstream RQ4 metric by condition (train on synth, test on real)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_pareto(summaries: list[dict], out_path: Path) -> None:
    """Realism (low discriminator accuracy is good) vs diversity (high coverage is good)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for s in summaries:
        if "error" in s:
            continue
        m = s.get("final_metrics", {}) or {}
        x = 1.0 - float(m.get("discriminator_accuracy", 0.5) or 0.5)  # realism
        y = float(m.get("combination_coverage", 0.0) or 0.0)
        ds = (s.get("final_downstream", {}) or {}).get("macro_f1", 0.0)
        ax.scatter(x, y, s=80 + 200 * ds, alpha=0.7)
        ax.annotate(s["condition"], (x, y), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("realism (1 - discriminator accuracy; chance = 0.5)")
    ax.set_ylabel("diversity (combination coverage)")
    ax.set_title("Realism vs diversity (point size = downstream macro F1)")
    ax.grid(linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_per_class_f1(summaries: list[dict], out_path: Path) -> None:
    conds = [s["condition"] for s in summaries if "error" not in s]
    label_order: list[str] = []
    matrix: dict[str, list[float]] = {}
    for s in summaries:
        if "error" in s:
            continue
        ds = s.get("final_downstream", {}) or {}
        per_class = ds.get("per_class_f1", {}) or {}
        for lbl in per_class:
            if lbl not in label_order:
                label_order.append(lbl)
        for lbl in label_order:
            matrix.setdefault(lbl, []).append(per_class.get(lbl, 0.0))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(conds))
    width = 0.8 / max(1, len(label_order))
    for i, lbl in enumerate(label_order):
        offsets = [j - 0.4 + i * width + width / 2 for j in x]
        ax.bar(offsets, matrix[lbl], width, label=lbl)
    ax.set_xticks(list(x))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("per-class F1")
    ax.set_title("Per-class F1 on held-out real test")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_adversary_metrics(summaries: list[dict], out_path: Path) -> None:
    """The headline contribution: metrics that exist ONLY in SynSmith."""
    conds, pack, ms, banned, attr = [], [], [], [], []
    for s in summaries:
        if "error" in s:
            continue
        m = s.get("final_metrics", {}) or {}
        conds.append(s["condition"])
        pack.append(m.get("pack_accuracy"))
        ms.append(m.get("mode_seeking_ratio"))
        banned.append(m.get("banned_phrasings_total"))
        attr.append(m.get("attribute_match_rate"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    def bar_with_nans(ax, values, color, title, ylim=(0, 1), nan_label="not measured"):
        xs = list(range(len(values)))
        # Render present values as bars, mark missing with hatched cells.
        for i, v in enumerate(values):
            if v is None:
                ax.bar(i, 1.0, color="none", edgecolor="#888", hatch="//")
                ax.text(
                    i, 0.5, nan_label,
                    ha="center", va="center", rotation=90, color="#888", fontsize=8,
                )
            else:
                ax.bar(i, v, color=color)
                ax.text(
                    i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                )
        ax.set_xticks(xs)
        ax.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    bar_with_nans(
        axes[0, 0], pack, "#c0392b",
        "Pack accuracy (lower = more diverse; chance = 0.5)",
    )
    bar_with_nans(
        axes[0, 1], ms, "#3a6ea5",
        "Mode-seeking ratio (higher = more attribute responsive)",
        ylim=(0, max(0.3, max((v or 0) for v in ms) * 1.2)),
    )
    # Banned phrasings: y axis is integer; convert None -> 0 for display.
    banned_disp = [0 if v is None else v for v in banned]
    axes[1, 0].bar(range(len(banned_disp)), banned_disp, color="#7c4dff")
    for i, v in enumerate(banned):
        label = "off" if v is None else f"{int(v)}"
        axes[1, 0].text(i, banned_disp[i] + 0.1, label, ha="center", va="bottom", fontsize=8)
    axes[1, 0].set_xticks(range(len(conds)))
    axes[1, 0].set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    axes[1, 0].set_title("Banned phrasings accumulated by Mode Hunter", fontsize=10)
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.4)

    bar_with_nans(
        axes[1, 1], attr, "#2e715a",
        "Attribute match rate (higher = better fidelity)",
    )

    fig.suptitle("GAN-style adversary metrics by condition", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_curves(summaries: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in summaries:
        if "error" in s:
            continue
        di = s.get("per_iter_downstream", [])
        if not di:
            continue
        xs = [d.get("iteration", i) for i, d in enumerate(di)]
        ys = [d.get("macro_f1", 0.0) for d in di]
        ax.plot(xs, ys, marker="o", label=s["condition"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("downstream macro F1 (per-iter training set)")
    ax.set_title("Downstream macro F1 over iterations")
    ax.set_ylim(0, 1)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    run_dir = REPO / "experiments" / args.run_id
    if not run_dir.exists():
        sys.exit(f"no such run dir: {run_dir}")

    summaries = load_summaries(run_dir)
    if not summaries:
        sys.exit("no summaries found")

    agg_dir = run_dir / "aggregated"
    fig_dir = REPO / "paper" / "figures"

    write_conditions_csv(summaries, agg_dir / "conditions.csv")
    write_per_iter_csv(summaries, agg_dir / "per_iter.csv")
    write_per_class_csv(summaries, agg_dir / "per_class_f1.csv")

    plot_main_metrics(summaries, fig_dir / f"{args.run_id}_main_metrics.png")
    plot_pareto(summaries, fig_dir / f"{args.run_id}_pareto.png")
    plot_per_class_f1(summaries, fig_dir / f"{args.run_id}_per_class_f1.png")
    plot_iteration_curves(summaries, fig_dir / f"{args.run_id}_iteration_curves.png")
    plot_adversary_metrics(summaries, fig_dir / f"{args.run_id}_adversary_metrics.png")

    print(f"Wrote aggregated CSVs under {agg_dir}")
    print(f"Wrote figures under {fig_dir}")


if __name__ == "__main__":
    main()
