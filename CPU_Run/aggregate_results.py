"""Aggregate the main comparison, difference-awareness, transfer, and quantization tables.

Reads results/main_evaluation_results.csv and produces seed-averaged tables with mean and
standard deviation on the primary endpoint and on the difference-awareness metrics. The
frozen test set carries two slices, so every table is produced per slice as well as pooled:
a method's in-distribution number and its number on the structure-novel slice answer
different questions and are never averaged together.

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

PRIMARY = "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy"
VALUE_COLUMNS = [
    PRIMARY,
    "accuracy_on_diff_condition",
    "accuracy_on_equal_condition",
    "gap_erasure_rate_on_diff_items",
    "gap_fabrication_rate_on_equal_items",
    "difference_aware_metric_wang_2025_recall_style",
    "contextual_awareness_metric_wang_2025_precision_style",
    "capability_retention_accuracy_on_external_probe",
    "identity_swap_unlicensed_flip_rate",
    "identity_swap_invariance_rate_on_equal_items",
    "identity_swap_equivariance_rate_on_diff_items",
    "option_rotation_canonical_consistency_rate",
    "option_rotation_position_following_rate",
]


def _numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _agg(df, value_cols):
    cols = [c for c in value_cols if c in df.columns]
    grouped = df.groupby(["tier", "method"], as_index=False)[cols].agg(["mean", "std", "count"])
    grouped.columns = ["_".join([c for c in col if c]).strip("_") for col in grouped.columns.to_flat_index()]
    return grouped


def main():
    if not MAIN_EVALUATION.exists():
        raise SystemExit("Run GPU_Run/evaluate_all.py first (main_evaluation_results.csv missing).")
    df = _numeric(pd.read_csv(MAIN_EVALUATION), VALUE_COLUMNS)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    if overall.empty:
        raise SystemExit("No overall-scope rows in the evaluation results.")

    in_dist = overall[overall["scope"].astype(str).str.startswith("agrifacts_frozen_test")]
    _agg(in_dist, VALUE_COLUMNS).to_csv(RESULTS_DIR / "aggregated_main_comparison_table.csv", index=False)
    logger.info("Main comparison table: %d (tier,method) rows.", in_dist["method"].nunique())

    # per-slice tables, where the slice is carried on the per-item predictions
    from GPU_Run.common.checkpointing import read_jsonl
    from GPU_Run.common.paths import TEST_INSTANCES_FROZEN, per_item_prediction_paths
    from GPU_Run.common import metrics as M

    slice_of = {r["id"]: r.get("test_slice", "") for r in read_jsonl(TEST_INSTANCES_FROZEN)}
    slice_rows = []
    for (tier, method, seed), path in sorted(per_item_prediction_paths().items()):
        preds = read_jsonl(path)
        for slice_name in ("structure_familiar", "structure_novel"):
            sub = [p for p in preds if slice_of.get(p["id"], "") == slice_name]
            if not sub:
                continue
            slice_rows.append({
                "tier": tier, "method": method, "random_seed": seed, "test_slice_name": slice_name,
                "number_of_items": len(sub),
                "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy": round(M.balanced_awareness_score(sub), 4),
                "accuracy_on_diff_condition": round(M.condition_accuracy(sub, "diff"), 4),
                "accuracy_on_equal_condition": round(M.condition_accuracy(sub, "equal"), 4),
                "gap_erasure_rate_on_diff_items": round(M.gap_erasure_rate(sub), 4),
                "gap_fabrication_rate_on_equal_items": round(M.gap_fabrication_rate(sub), 4),
            })
    if slice_rows:
        pd.DataFrame(slice_rows).to_csv(RESULTS_DIR / "structure_slice_comparison_table.csv", index=False)
        logger.info("Structure-slice table: %d rows.", len(slice_rows))

    diff_cols = ["accuracy_on_diff_condition", "accuracy_on_equal_condition",
                 "difference_aware_metric_wang_2025_recall_style",
                 "contextual_awareness_metric_wang_2025_precision_style"]
    _agg(in_dist, diff_cols).to_csv(RESULTS_DIR / "difference_awareness_preservation_table.csv", index=False)

    transfer = overall[overall["scope"].astype(str).str.startswith("held_out_axis:")].copy()
    if not transfer.empty:
        transfer.to_csv(RESULTS_DIR / "leave_one_axis_out_transfer_table.csv", index=False)
        logger.info("Leave-one-axis-out transfer rows: %d.", len(transfer))

    per_axis = df[df["scope"].astype(str).str.contains(r"\|axis:", regex=True)].copy()
    if not per_axis.empty:
        per_axis.to_csv(RESULTS_DIR / "per_axis_breakdown_table.csv", index=False)

    quant = in_dist[in_dist["method"].astype(str).str.contains("qlora", case=False)]
    if not quant.empty:
        _agg(quant, [PRIMARY, "capability_retention_accuracy_on_external_probe"]).to_csv(
            RESULTS_DIR / "quantization_study_table.csv", index=False)

    log_run_metadata("aggregate_results", {"methods": sorted(in_dist["method"].unique().tolist())})


if __name__ == "__main__":
    main()
