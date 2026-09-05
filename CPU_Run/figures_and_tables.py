"""Figures for the main comparison, the two failure directions, the placement ablation,
the rank sweep, the structure slices, and the leave-one-axis-out transfer.

Reads the aggregated tables and the main results CSV and writes PNGs to results/figures/.
Nothing is projected; a figure is produced only when the run that feeds it exists.

Run:  python CPU_Run/figures_and_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import FIGURES_DIR, MAIN_EVALUATION, RESULTS_DIR

logger = get_logger("figures_and_tables")

PRIMARY = "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy"


def _load_overall():
    df = pd.read_csv(MAIN_EVALUATION)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    for c in [PRIMARY, "accuracy_on_diff_condition", "accuracy_on_equal_condition",
              "gap_erasure_rate_on_diff_items", "gap_fabrication_rate_on_equal_items"]:
        if c in overall.columns:
            overall[c] = pd.to_numeric(overall[c], errors="coerce")
    return overall


def _bar(series, title, ylabel, path, colour="#3a6ea5", reference=None, reference_label=""):
    fig, ax = plt.subplots(figsize=(max(6, 0.62 * len(series)), 4.2))
    series.plot(kind="bar", ax=ax, color=colour)
    if reference is not None:
        ax.axhline(reference, color="#a5533a", linestyle="--", linewidth=1)
        ax.text(0.01, reference, reference_label, fontsize=7, color="#a5533a",
                va="bottom", transform=ax.get_yaxis_transform())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _surface_cue_reference(slice_name="test_all"):
    path = RESULTS_DIR / "surface_cue_ceiling_audit.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        row = df[df["split_name"] == slice_name]
        if row.empty:
            return None
        return float(row.iloc[0]["surface_cue_majority_ceiling_for_condition_and_answer"])
    except Exception:
        return None


def main():
    if not MAIN_EVALUATION.exists():
        logger.warning("No main_evaluation_results.csv; nothing to plot.")
        return
    overall = _load_overall()
    if overall.empty:
        logger.warning("Main results empty; nothing to plot.")
        return
    in_dist = overall[overall["scope"].astype(str).str.startswith("agrifacts_frozen_test")]
    ceiling = _surface_cue_reference()

    for tier in sorted(in_dist["tier"].unique()):
        sub = in_dist[in_dist["tier"] == tier]
        means = sub.groupby("method")[PRIMARY].mean().sort_values(ascending=False)
        _bar(means, f"Balanced awareness score ({tier})", "harmonic mean of diff and equal accuracy",
             FIGURES_DIR / f"main_comparison_{tier}.png",
             reference=ceiling, reference_label="state-blind majority ceiling")

        da = sub.groupby("method")[["accuracy_on_diff_condition", "accuracy_on_equal_condition"]].mean()
        fig, ax = plt.subplots(figsize=(max(6, 0.72 * len(da)), 4.2))
        da.plot(kind="bar", ax=ax, color=["#3a6ea5", "#6ea53a"])
        ax.set_title(f"Difference awareness against contextual awareness ({tier})")
        ax.set_ylabel("accuracy")
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"difference_awareness_{tier}.png", dpi=150)
        plt.close(fig)

        fails = sub.groupby("method")[["gap_erasure_rate_on_diff_items", "gap_fabrication_rate_on_equal_items"]].mean()
        if not fails.dropna(how="all").empty:
            fig, ax = plt.subplots(figsize=(max(6, 0.72 * len(fails)), 4.2))
            fails.plot(kind="bar", ax=ax, color=["#a5533a", "#8a6ea5"])
            ax.set_title(f"The two failure directions ({tier})")
            ax.set_ylabel("rate (lower is better)")
            ax.set_xlabel("")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f"failure_directions_{tier}.png", dpi=150)
            plt.close(fig)

        place = means[means.index.str.contains("placement|graft_proposed$", case=False, regex=True)]
        if len(place) >= 2:
            _bar(place, f"Placement ablation at a matched budget ({tier})", "balanced awareness score",
                 FIGURES_DIR / f"placement_ablation_{tier}.png")
        rank = means[means.index.str.contains("rank_sweep", case=False)]
        if len(rank) >= 2:
            _bar(rank, f"Rank sweep ({tier})", "balanced awareness score",
                 FIGURES_DIR / f"rank_sweep_{tier}.png")

    slice_path = RESULTS_DIR / "structure_slice_comparison_table.csv"
    if slice_path.exists():
        sl = pd.read_csv(slice_path)
        if not sl.empty:
            piv = sl.pivot_table(index="method", columns="test_slice_name", values=PRIMARY, aggfunc="mean")
            fig, ax = plt.subplots(figsize=(max(6, 0.72 * len(piv)), 4.2))
            piv.plot(kind="bar", ax=ax, color=["#3a6ea5", "#a5533a"])
            ax.set_title("Familiar against novel state-blind structure")
            ax.set_ylabel("balanced awareness score")
            ax.set_xlabel("")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / "structure_slice_comparison.png", dpi=150)
            plt.close(fig)

    transfer = overall[overall["scope"].astype(str).str.startswith("held_out_axis:")]
    if not transfer.empty:
        piv = transfer.pivot_table(index="held_out_axis", columns="base_method",
                                   values="gap_erasure_rate_on_diff_items", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(piv)), 4.2))
        piv.plot(kind="bar", ax=ax)
        ax.set_title("Residual gap erasure on the held-out axis (lower is better)")
        ax.set_ylabel("gap erasure rate")
        ax.set_xlabel("held-out axis")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "leave_one_axis_out_transfer.png", dpi=150)
        plt.close(fig)

    log_run_metadata("figures_and_tables", {"figures_dir": str(FIGURES_DIR)})
    logger.info("Figures written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
