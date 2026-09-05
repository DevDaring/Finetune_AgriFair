"""Dry run for the CPU analysis stage on synthetic fixtures.

Exercises the metric definitions, the two failure directions, identity-swap consistency,
option-rotation robustness, the non-regression-adjusted score, the DART severity map, the
prediction-filename parser, the paired bootstrap and Holm correction, the parameter-space
geometry diagnostics, and figure rendering. Synthetic numbers live only under
results/dry_run/ and never leak into real outputs.

The point of this file is that a definition which quietly drifts is caught before an
expensive run, so each check asserts a property the definition must have rather than a
value it happened to produce.

Run:  python Dry_Run/dry_run_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random

import numpy as np

from GPU_Run.common import metrics as M
from GPU_Run.common.logging_utils import get_logger, write_csv
from GPU_Run.common.paths import DRY_RUN_DIR

logger = get_logger("dry_run_analysis")


def _metric_checks(checks):
    preds = (
        [{"id": f"n{i}", "condition": "diff", "gold_canonical": "a", "pred_canonical": "a"} for i in range(6)]
        + [{"id": f"n{i}", "condition": "diff", "gold_canonical": "a", "pred_canonical": "c"} for i in range(6, 10)]
        + [{"id": f"e{i}", "condition": "equal", "gold_canonical": "c", "pred_canonical": "c"} for i in range(7)]
        + [{"id": f"e{i}", "condition": "equal", "gold_canonical": "c", "pred_canonical": "a"} for i in range(7, 10)]
    )
    wang = M.wang_contingency(preds)
    checks["wang_difference_aware_metric_correct"] = abs(
        wang["difference_aware_metric_wang_2025_recall_style"] - 0.6) < 1e-9
    checks["wang_gap_erasure_cell_counts_the_c_answers_on_diff_items"] = wang["wang_cell_C_diff_gap_erasure"] == 4
    checks["balanced_awareness_score_is_the_harmonic_mean"] = abs(
        M.balanced_awareness_score(preds) - M.harmonic_mean(0.6, 0.7)) < 1e-9
    checks["balanced_awareness_score_punishes_a_collapsed_condition"] = \
        M.balanced_awareness_score(preds) < 0.5 * (0.6 + 0.7)
    checks["gap_erasure_rate_correct"] = abs(M.gap_erasure_rate(preds) - 0.4) < 1e-9
    checks["gap_fabrication_rate_correct"] = abs(M.gap_fabrication_rate(preds) - 0.3) < 1e-9
    checks["census_citation_preservation_runs"] = M.citation_preservation_rate(
        "AgCensus2015-16 T2-4 X/All/area/SCvsST", "per T2-4 SCvsST") == 1.0

    erasure = [{"id": f"n{i}", "condition": "diff", "gold_canonical": "a", "pred_canonical": "c"} for i in range(8)] \
        + [{"id": f"e{i}", "condition": "equal", "gold_canonical": "c", "pred_canonical": "c"} for i in range(10)]
    fabrication = [{"id": f"n{i}", "condition": "diff", "gold_canonical": "a", "pred_canonical": "a"} for i in range(10)] \
        + [{"id": f"e{i}", "condition": "equal", "gold_canonical": "c", "pred_canonical": "a"} for i in range(6)]
    checks["failure_polarity_detects_gap_erasure"] = M.failure_polarity_index(erasure) > 0.9
    checks["failure_polarity_detects_gap_fabrication"] = M.failure_polarity_index(fabrication) < -0.9
    checks["polarity_label_matches_the_index"] = (
        M.polarity_label(M.failure_polarity_index(erasure)) == "gap_erasure_dominant"
        and M.polarity_label(M.failure_polarity_index(fabrication)) == "gap_fabrication_dominant"
        and M.polarity_label(0.0) == "bipolar_no_dominant_polarity")


def _consistency_checks(checks):
    original = [{"id": "x1", "condition": "diff", "pred_canonical": "a"},
                {"id": "x2", "condition": "equal", "pred_canonical": "c"}]
    licensed = [{"id": "x1::swap", "condition": "diff", "pred_canonical": "b"},
                {"id": "x2::swap", "condition": "equal", "pred_canonical": "c"}]
    unlicensed = [{"id": "x1::swap", "condition": "diff", "pred_canonical": "a"},
                  {"id": "x2::swap", "condition": "equal", "pred_canonical": "a"}]
    checks["identity_swap_flip_rate_zero_when_licensed"] = M.identity_swap_flip_rate(original, licensed) == 0.0
    checks["identity_swap_flip_rate_one_when_unlicensed"] = M.identity_swap_flip_rate(original, unlicensed) == 1.0
    checks["identity_swap_invariance_scores_equal_items_only"] = \
        M.identity_swap_invariance_rate(original, licensed) == 1.0 and \
        M.identity_swap_invariance_rate(original, unlicensed) == 0.0
    checks["identity_swap_equivariance_scores_diff_items_only"] = \
        M.identity_swap_equivariance_rate(original, licensed) == 1.0 and \
        M.identity_swap_equivariance_rate(original, unlicensed) == 0.0

    stable = {0: [{"id": "a", "pred_canonical": "a", "pred_display_letter": "a", "gold_canonical": "a", "condition": "diff"}],
              1: [{"id": "a", "pred_canonical": "a", "pred_display_letter": "c", "gold_canonical": "a", "condition": "diff"}]}
    positional = {0: [{"id": "a", "pred_canonical": "a", "pred_display_letter": "a", "gold_canonical": "a", "condition": "diff"}],
                  1: [{"id": "a", "pred_canonical": "b", "pred_display_letter": "a", "gold_canonical": "a", "condition": "diff"}]}
    checks["option_rotation_detects_a_content_stable_answer"] = \
        M.option_rotation_consistency(stable)["rotation_consistency_rate"] == 1.0
    checks["option_rotation_detects_position_following"] = \
        M.option_rotation_consistency(positional)["rotation_position_following_rate"] == 1.0

    honest = M.non_regression_adjusted_contextual_fairness(0.70, 0.70, 0.70, 0.60)
    traded = M.non_regression_adjusted_contextual_fairness(0.40, 0.90, 0.70, 0.60)
    checks["adjusted_score_penalises_an_awareness_trade"] = honest > traded
    checks["adjusted_score_equals_harmonic_mean_when_nothing_is_lost"] = abs(
        M.non_regression_adjusted_contextual_fairness(0.8, 0.8, 0.7, 0.7) - M.harmonic_mean(0.8, 0.8)) < 1e-9
    checks["fairness_gain_undefined_for_zero_invasiveness"] = \
        M.fairness_gain_per_unit(0.1, 0.0) != M.fairness_gain_per_unit(0.1, 0.0)

    dists = M.advice_drift_embedding_distances([("apply urea now", "apply urea now"),
                                                ("x", "completely different text here")])
    checks["advice_drift_orders_identical_below_different"] = len(dists) == 2 and dists[0] <= dists[1]
    checks["advice_jaccard_is_zero_for_identical_answers"] = \
        M.advice_content_word_jaccard_distance("apply urea now please", "apply urea now please") == 0.0
    checks["advice_structured_l1_notices_a_dropped_dosage"] = \
        M.advice_structured_l1("Use 2 kg of neem cake per plant.", "Use neem cake.") > 0


def _statistics_checks(checks):
    from CPU_Run.statistics_tests import holm, mcnemar, paired_bootstrap

    rng = random.Random(0)
    ref, cmp_ = [], []
    for i in range(120):
        cond = "diff" if i % 2 == 0 else "equal"
        gold = "a" if cond == "diff" else "c"
        ref.append({"id": str(i), "condition": cond, "gold_canonical": gold,
                    "pred_canonical": gold if rng.random() < 0.55 else "b"})
        cmp_.append({"id": str(i), "condition": cond, "gold_canonical": gold,
                     "pred_canonical": gold if rng.random() < 0.85 else "b"})
    boot = paired_bootstrap(ref, cmp_, resamples=200)
    checks["paired_bootstrap_reports_a_valid_p_value"] = 0.0 <= boot["p_value"] <= 1.0
    checks["paired_bootstrap_interval_is_ordered"] = boot["lower"] <= boot["upper"]
    checks["paired_bootstrap_detects_a_real_difference"] = boot["observed"] > 0 and boot["p_value"] < 0.05
    identical = paired_bootstrap(ref, ref, resamples=200)
    checks["paired_bootstrap_of_a_method_against_itself_is_null"] = abs(identical["observed"]) < 1e-12
    mc = mcnemar(ref, cmp_)
    checks["mcnemar_reports_discordant_counts"] = mc["only_comparison"] > 0 and 0.0 <= mc["p_value"] <= 1.0
    adjusted = holm([0.01, 0.04, 0.03])
    checks["holm_correction_is_monotone_and_inflates_p_values"] = \
        adjusted[0] <= adjusted[2] <= adjusted[1] and adjusted[0] >= 0.01


def _geometry_checks(checks):
    from CPU_Run import parameter_space_geometry as G

    rng = np.random.default_rng(0)
    W0 = rng.normal(size=(24, 16))
    dW = 0.01 * np.outer(rng.normal(size=24), rng.normal(size=16))  # rank-1 update
    diag = G.matrix_diagnostics(W0, dW)
    expected = [
        "bf16_aware_update_sparsity",
        "principal_angle_rotation_left_subspace_degrees",
        "principal_angle_rotation_right_subspace_degrees",
        "principal_angle_rotation_max_degrees",
        "normalized_spectral_shift",
        "update_mask_overlap_with_principal_mask",
        "update_mask_overlap_with_low_magnitude_mask",
        "stable_rank_of_update",
        "frobenius_norm_of_update",
        "relative_frobenius_drift",
        "hill_tail_exponent",
    ]
    checks["geometry_all_diagnostics_present"] = all(k in diag for k in expected)
    checks["geometry_all_diagnostics_finite"] = all(
        np.isfinite(diag[k]) for k in expected if k != "hill_tail_exponent")
    checks["geometry_rank_one_update_has_unit_stable_rank"] = abs(diag["stable_rank_of_update"] - 1.0) < 1e-6
    zero = G.matrix_diagnostics(W0, np.zeros_like(W0))
    checks["geometry_zero_update_preserves_the_spectrum"] = zero["normalized_spectral_shift"] < 1e-9
    checks["geometry_zero_update_has_unit_sparsity"] = abs(zero["bf16_aware_update_sparsity"] - 1.0) < 1e-9
    checks["geometry_zero_update_has_zero_drift"] = zero["relative_frobenius_drift"] < 1e-12
    big = G.matrix_diagnostics(W0, 0.5 * W0)
    checks["geometry_a_large_update_moves_the_spectrum_more_than_a_small_one"] = \
        big["normalized_spectral_shift"] > diag["normalized_spectral_shift"]
    inference_row = {
        "method_name": "baseline_fairsteer", "base_model_name": "dry_run",
        "weight_update_type": "inference_time_no_weight_update",
        "parameter_space_regime_label": "inference_time_no_weight_update",
    }
    try:
        write_csv(DRY_RUN_DIR / "dry_run_geometry_per_method.csv", [inference_row], G.PER_METHOD_COLUMNS)
        for cols in (G.PER_METHOD_COLUMNS, G.PER_LAYER_COLUMNS, G.CORR_COLUMNS, G.RUN_LOG_COLUMNS):
            write_csv(DRY_RUN_DIR / "dry_run_geometry_header_probe.csv", [], cols)
        checks["geometry_inference_time_row_and_headers_write"] = True
    except Exception as e:
        logger.warning("Geometry CSV check failed: %s", e)
        checks["geometry_inference_time_row_and_headers_write"] = False
    partial = G._partial_correlation([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.9, 0.85, 0.8, 0.7, 0.65, 0.6],
                                     [0.5, 0.6, 0.4, 0.7, 0.5, 0.6])
    lo, hi = G._bootstrap_pearson_ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.9, 0.85, 0.8, 0.7, 0.65, 0.6], resamples=100)
    checks["geometry_partial_correlation_in_range"] = -1.0 <= partial <= 1.0
    checks["geometry_bootstrap_interval_ordered"] = (lo != lo) or (hi != hi) or (lo <= hi)


def _judge_chain_checks(checks):
    """The answer-extraction judge is what rescues an output the parser could not read, on
    open-weight and hosted models alike. These assert its shape without spending a call."""
    from GPU_Run.common.clients import JudgeChain, build_judge_routes, primary_judge_chain, secondary_judge_chain

    routes = build_judge_routes()
    names = [r.name for r in routes]
    expected = ["deepseek-key-1", "deepseek-key-2", "mistral-key-1", "mistral-key-2"]
    checks["judge_chain_is_in_the_registered_order"] = names == [n for n in expected if n in names]
    checks["judge_chain_has_no_duplicate_routes"] = len(names) == len(set(names))
    checks["judge_agreement_uses_two_different_providers"] = (
        {r.provider for r in primary_judge_chain().routes} != {r.provider for r in secondary_judge_chain().routes})
    chain = JudgeChain(routes)
    checks["judge_is_disabled_by_the_offline_switch"] = _disabled_when_flag_set(chain)
    # the extraction prompt must never carry the question, the options or the gold answer
    from GPU_Run.common import prompts as P

    prompt = P.judge_answer_extraction_prompt("I would say option (b).")
    checks["judge_extraction_prompt_shows_only_the_model_output"] = (
        "b" in prompt and "Roughly equal" not in prompt and "Agriculture Census" not in prompt)


def _disabled_when_flag_set(chain) -> bool:
    import os

    previous = os.environ.get("DISABLE_JUDGE")
    os.environ["DISABLE_JUDGE"] = "1"
    try:
        disabled = not chain.available()
    finally:
        if previous is None:
            os.environ.pop("DISABLE_JUDGE", None)
        else:
            os.environ["DISABLE_JUDGE"] = previous
    return disabled


def _plumbing_checks(checks):
    from GPU_Run.common import dart_baseline as dart
    from GPU_Run.common.paths import parse_per_item_prediction_path, split_loao_method

    checks["dart_severity_map_ranks_gap_erasure_highest"] = (
        dart._severity("diff", "a", "c") == "extreme"
        and dart._severity("equal", "c", "a") == "severe"
        and dart._severity("diff", "a", "b") == "moderate"
        and dart._severity("diff", "a", "a") is None)
    checks["dart_oversampling_orders_by_severity"] = (
        dart.SEVERITY_OVERSAMPLING["extreme"] > dart.SEVERITY_OVERSAMPLING["severe"]
        > dart.SEVERITY_OVERSAMPLING["moderate"] > dart.SEVERITY_OVERSAMPLING["mild"])

    cases = {
        "per_item_predictions_smoke_baseline_dart_seed42.jsonl": ("smoke", "baseline_dart", 42),
        "per_item_predictions_smoke_baseline_igu_lora_seed42.jsonl": ("smoke", "baseline_igu_lora", 42),
        "per_item_predictions_broad-instruct_reference_vanilla_qlora_seed44.jsonl":
            ("broad-instruct", "reference_vanilla_qlora", 44),
        "per_item_predictions_small-instruct_frozen_base_seed42.jsonl": ("small-instruct", "frozen_base", 42),
        "per_item_predictions_small-instruct_graft_proposed_loao_gender_seed42.jsonl":
            ("small-instruct", "graft_proposed_loao_gender", 42),
    }
    checks["prediction_filename_parsing_correct"] = all(
        parse_per_item_prediction_path(name) == expected for name, expected in cases.items())
    checks["leave_one_axis_out_method_name_splits_correctly"] = (
        split_loao_method("graft_proposed_loao_gender") == ("graft_proposed", "gender")
        and split_loao_method("graft_proposed") == ("graft_proposed", None))


def main():
    checks = {}
    _metric_checks(checks)
    _consistency_checks(checks)
    _statistics_checks(checks)
    _geometry_checks(checks)
    _judge_chain_checks(checks)
    _plumbing_checks(checks)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.bar(["proposed", "baseline"], [0.7, 0.6])
        fig.savefig(DRY_RUN_DIR / "dry_run_figure.png", dpi=80)
        plt.close(fig)
        checks["figure_rendering_runs"] = (DRY_RUN_DIR / "dry_run_figure.png").exists()
    except Exception as e:
        logger.warning("Figure rendering failed: %s", e)
        checks["figure_rendering_runs"] = False

    report = {"checks": checks, "all_passed": all(checks.values())}
    (DRY_RUN_DIR / "dry_run_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in checks.items():
        logger.info("  %-62s %s", k, v)
    logger.info("dry_run_analysis all_passed=%s", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit("dry_run_analysis FAILED: " +
                         json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
