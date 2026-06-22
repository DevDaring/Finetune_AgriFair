"""Aggregate the main comparison, difference-awareness, external, and quantization tables.

Reads results/main_evaluation_results.csv and produces seed-averaged tables with mean and
standard deviation on the primary endpoint and the difference-awareness metrics
(Instruction.md Section 17). Runs on CPU; no GPU required.

Run:  python CPU_Run/aggregate_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import MAIN_EVALUATION, RESULTS_DIR

logger = get_logger("aggregate_results")

PRIMARY = "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy"


def _agg(df, value_cols):
    g = df.groupby(["tier", "method"], as_index=False)
    out = g[value_cols].agg(["mean", "std"])
    out.columns = ["_".join([c for c in col if c]).strip("_") for col in out.columns.to_flat_index()]
    return out


def main():
    if not MAIN_EVALUATION.exists():
        raise SystemExit("Run GPU_Run/evaluate_all.py first (main_evaluation_results.csv missing).")
    df = pd.read_csv(MAIN_EVALUATION)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    for c in [PRIMARY, "accuracy_on_neq_condition", "accuracy_on_eq_condition",
              "difference_aware_metric_wang_2025_recall_style",
              "contextual_awareness_metric_wang_2025_precision_style",
              "utility_retention_accuracy_on_held_out_general_set"]:
        overall[c] = pd.to_numeric(overall[c], errors="coerce")

    main_table = _agg(overall, [PRIMARY, "accuracy_on_neq_condition", "accuracy_on_eq_condition"])
    main_table.to_csv(RESULTS_DIR / "aggregated_main_comparison_table.csv", index=False)
    logger.info("Main comparison table: %d (tier,method) rows.", len(main_table))

    diff_table = _agg(overall, ["accuracy_on_neq_condition", "accuracy_on_eq_condition",
                                "difference_aware_metric_wang_2025_recall_style",
                                "contextual_awareness_metric_wang_2025_precision_style"])
    diff_table.to_csv(RESULTS_DIR / "difference_awareness_preservation_table.csv", index=False)

    # external generalization = the held-out axis slice (cross-axis transfer)
    ext = df[df["scope"].astype(str).str.contains("axis:gender")].copy()
    ext[PRIMARY] = pd.to_numeric(ext[PRIMARY], errors="coerce")
    if not ext.empty:
        _agg(ext, [PRIMARY]).to_csv(RESULTS_DIR / "aggregated_external_generalization_cross_axis_gender.csv", index=False)

    # quantization study
    quant = overall[overall["method"].astype(str).str.contains("qlora", case=False)].copy()
    if not quant.empty:
        _agg(quant, [PRIMARY]).to_csv(RESULTS_DIR / "quantization_study_table.csv", index=False)

    log_run_metadata("aggregate_results", {"methods": sorted(overall["method"].unique().tolist())})


if __name__ == "__main__":
    main()
