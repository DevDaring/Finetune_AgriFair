"""Main comparison, difference-awareness, placement ablation, and rank-sweep figures.

Reads the aggregated tables and the main results CSV; writes PNGs to results/figures/.
No projected numbers; figures render only what real runs produced. CPU only.

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

PRIMARY = "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy"


def _load_overall():
    df = pd.read_csv(MAIN_EVALUATION)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    for c in [PRIMARY, "accuracy_on_neq_condition", "accuracy_on_eq_condition"]:
        overall[c] = pd.to_numeric(overall[c], errors="coerce")
    return overall


def _bar(series, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(series)), 4))
    series.plot(kind="bar", ax=ax, color="#3a6ea5")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not MAIN_EVALUATION.exists():
        logger.warning("No main_evaluation_results.csv; nothing to plot.")
        return
    overall = _load_overall()
    if overall.empty:
        logger.warning("Main results empty; nothing to plot.")
        return

    for tier in sorted(overall["tier"].unique()):
        sub = overall[overall["tier"] == tier]
        means = sub.groupby("method")[PRIMARY].mean().sort_values(ascending=False)
        _bar(means, f"Contextual fairness score ({tier})", "harmonic mean (neq, eq)", FIGURES_DIR / f"main_comparison_{tier}.png")

        # difference-awareness: neq vs eq accuracy
        da = sub.groupby("method")[["accuracy_on_neq_condition", "accuracy_on_eq_condition"]].mean()
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(da)), 4))
        da.plot(kind="bar", ax=ax)
        ax.set_title(f"Difference awareness vs contextual awareness ({tier})")
        ax.set_ylabel("accuracy")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"difference_awareness_{tier}.png", dpi=150)
        plt.close(fig)

        # placement ablation
        place = means[means.index.str.contains("placement|proposed", case=False, regex=True)]
        if len(place) >= 2:
            _bar(place, f"Placement ablation ({tier})", "contextual fairness", FIGURES_DIR / f"placement_ablation_{tier}.png")

        # rank sweep
        rank = means[means.index.str.contains("rank_sweep", case=False)]
        if len(rank) >= 2:
            _bar(rank, f"Rank sweep ({tier})", "contextual fairness", FIGURES_DIR / f"rank_sweep_{tier}.png")

    log_run_metadata("figures_and_tables", {"figures_dir": str(FIGURES_DIR)})
    logger.info("Figures written to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
