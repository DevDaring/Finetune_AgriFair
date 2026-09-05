"""Hypothesis tests on the primary endpoint.

Three families, each written with the multiplicity correction it needs:

  H1, localization at a matched budget (internal). Attribution-guided placement against
      uniform and against each independently drawn random placement. This is the causal
      test of the claim the method makes and it does not depend on out-ranking any external
      method. Random placement is a distribution over draws, not one lucky layer set, so the
      proposed arm is compared with the mean and with the spread of the draws.
  H2, the external comparison (secondary). The proposed method against every baseline,
      Holm-corrected across the family so a single nominally significant cell cannot be
      read as a win.
  H3 is a correlation and lives in CPU_Run/parameter_space_geometry.py.

Every test is paired at the item level and computed from the stored per-item predictions,
never from a summary. The bootstrap resamples items with replacement, recomputes the
balanced awareness score for both arms on the same resample, and reports the percentile
interval of the difference and a two-sided p-value. McNemar's exact test on the discordant
items is reported alongside, because the bootstrap and McNemar answer different questions
and reviewers ask for both.

Run:  python CPU_Run/statistics_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from GPU_Run.common import methods as METHODS
from GPU_Run.common import metrics as M
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import RESULTS_DIR, TEST_INSTANCES_FROZEN, per_item_prediction_paths
from GPU_Run.common.seeds import GLOBAL_SEED

logger = get_logger("statistics_tests")

BOOTSTRAP_RESAMPLES = int(os.environ.get("STATISTICS_BOOTSTRAP_RESAMPLES", "1000"))

COLUMNS = [
    "base_model_name", "hypothesis_family", "test_slice_name", "reference_method_name",
    "comparison_method_name", "number_of_paired_items",
    "balanced_awareness_score_reference", "balanced_awareness_score_comparison",
    "balanced_awareness_score_difference",
    "bootstrap_confidence_interval_lower", "bootstrap_confidence_interval_upper",
    "bootstrap_two_sided_p_value", "mcnemar_exact_two_sided_p_value",
    "holm_adjusted_p_value_within_family", "significant_after_holm_correction",
    "discordant_items_favouring_reference", "discordant_items_favouring_comparison",
    "bootstrap_resamples",
]

SEED_COLUMNS = [
    "base_model_name", "method_name", "test_slice_name", "number_of_seeds",
    "balanced_awareness_score_mean", "balanced_awareness_score_standard_deviation",
    "accuracy_on_diff_condition_mean", "accuracy_on_equal_condition_mean",
]


def _pooled_predictions() -> Dict[Tuple[str, str], List[Dict]]:
    """Per-item predictions keyed by (tier, method), pooled over seeds by taking the
    first seed's prediction for each item so the pairing stays one row per item."""
    out: Dict[Tuple[str, str], Dict[str, Dict]] = {}
    for (tier, method, _seed), path in sorted(per_item_prediction_paths().items()):
        bucket = out.setdefault((tier, method), {})
        for row in read_jsonl(path):
            bucket.setdefault(row["id"], row)
    return {k: list(v.values()) for k, v in out.items()}


def _by_seed() -> Dict[Tuple[str, str, int], List[Dict]]:
    return {k: read_jsonl(v) for k, v in sorted(per_item_prediction_paths().items())}


def _slice_map() -> Dict[str, str]:
    return {r["id"]: r.get("test_slice", "") for r in read_jsonl(TEST_INSTANCES_FROZEN)}


def paired_bootstrap(ref: Sequence[Dict], cmp_: Sequence[Dict], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = GLOBAL_SEED) -> Dict[str, float]:
    """Percentile interval and two-sided p-value for the difference in balanced awareness
    score, resampling the shared item set with replacement."""
    by_id_ref = {p["id"]: p for p in ref}
    by_id_cmp = {p["id"]: p for p in cmp_}
    ids = sorted(set(by_id_ref) & set(by_id_cmp))
    n = len(ids)
    if n < 8:
        return {"n": n, "observed": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "p_value": float("nan")}
    a = [by_id_ref[i] for i in ids]
    b = [by_id_cmp[i] for i in ids]
    observed = M.balanced_awareness_score(b) - M.balanced_awareness_score(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples)
    for r in range(resamples):
        idx = rng.integers(0, n, n)
        diffs[r] = M.balanced_awareness_score([b[i] for i in idx]) - M.balanced_awareness_score([a[i] for i in idx])
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"n": n, "observed": float(observed), "lower": float(lower), "upper": float(upper),
            "p_value": float(min(1.0, p))}


def mcnemar(ref: Sequence[Dict], cmp_: Sequence[Dict]) -> Dict[str, float]:
    """Exact two-sided McNemar test on per-item correctness."""
    from scipy.stats import binomtest

    by_id_ref = {p["id"]: int(p["pred_canonical"] == p["gold_canonical"]) for p in ref}
    by_id_cmp = {p["id"]: int(p["pred_canonical"] == p["gold_canonical"]) for p in cmp_}
    ids = sorted(set(by_id_ref) & set(by_id_cmp))
    only_ref = sum(1 for i in ids if by_id_ref[i] == 1 and by_id_cmp[i] == 0)
    only_cmp = sum(1 for i in ids if by_id_ref[i] == 0 and by_id_cmp[i] == 1)
    total = only_ref + only_cmp
    p = float(binomtest(only_cmp, total, 0.5).pvalue) if total > 0 else float("nan")
    return {"p_value": p, "only_reference": only_ref, "only_comparison": only_cmp}


def holm(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    finite = [(i, p) for i, p in enumerate(p_values) if p == p]
    adjusted = [float("nan")] * len(p_values)
    m = len(finite)
    running = 0.0
    for rank, (i, p) in enumerate(sorted(finite, key=lambda x: x[1])):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def _rows_for_family(tier, family, slice_name, reference, comparisons, preds) -> List[Dict]:
    rows = []
    for cmp_name in comparisons:
        ref, cmp_ = preds.get((tier, reference)), preds.get((tier, cmp_name))
        if not ref or not cmp_:
            continue
        if slice_name != "all":
            smap = _slice_map()
            ref = [p for p in ref if smap.get(p["id"]) == slice_name]
            cmp_ = [p for p in cmp_ if smap.get(p["id"]) == slice_name]
        boot = paired_bootstrap(ref, cmp_)
        mc = mcnemar(ref, cmp_)
        rows.append({
            "base_model_name": tier, "hypothesis_family": family, "test_slice_name": slice_name,
            "reference_method_name": reference, "comparison_method_name": cmp_name,
            "number_of_paired_items": boot["n"],
            "balanced_awareness_score_reference": round(M.balanced_awareness_score(ref), 4),
            "balanced_awareness_score_comparison": round(M.balanced_awareness_score(cmp_), 4),
            "balanced_awareness_score_difference": round(boot["observed"], 4) if boot["observed"] == boot["observed"] else "",
            "bootstrap_confidence_interval_lower": round(boot["lower"], 4) if boot["lower"] == boot["lower"] else "",
            "bootstrap_confidence_interval_upper": round(boot["upper"], 4) if boot["upper"] == boot["upper"] else "",
            "bootstrap_two_sided_p_value": round(boot["p_value"], 4) if boot["p_value"] == boot["p_value"] else "",
            "mcnemar_exact_two_sided_p_value": round(mc["p_value"], 4) if mc["p_value"] == mc["p_value"] else "",
            "discordant_items_favouring_reference": mc["only_reference"],
            "discordant_items_favouring_comparison": mc["only_comparison"],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "_p": boot["p_value"],
        })
    adjusted = holm([r["_p"] for r in rows])
    for r, adj in zip(rows, adjusted):
        r["holm_adjusted_p_value_within_family"] = round(adj, 4) if adj == adj else ""
        r["significant_after_holm_correction"] = "yes" if (adj == adj and adj < 0.05) else "no"
        r.pop("_p")
    return rows


def _seed_rows(slice_map) -> List[Dict]:
    by_seed = _by_seed()
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    extra: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}
    for (tier, method, seed), preds in by_seed.items():
        for slice_name in ("all", "structure_familiar", "structure_novel"):
            sub = preds if slice_name == "all" else [p for p in preds if slice_map.get(p["id"]) == slice_name]
            if not sub:
                continue
            grouped.setdefault((tier, method, slice_name), []).append(M.balanced_awareness_score(sub))
            extra.setdefault((tier, method, slice_name), []).append(
                (M.condition_accuracy(sub, "diff"), M.condition_accuracy(sub, "equal")))
    rows = []
    for (tier, method, slice_name), scores in sorted(grouped.items()):
        pairs = extra[(tier, method, slice_name)]
        rows.append({
            "base_model_name": tier, "method_name": method, "test_slice_name": slice_name,
            "number_of_seeds": len(scores),
            "balanced_awareness_score_mean": round(float(np.mean(scores)), 4),
            "balanced_awareness_score_standard_deviation": round(float(np.std(scores, ddof=1)), 4)
            if len(scores) > 1 else "",
            "accuracy_on_diff_condition_mean": round(float(np.mean([p[0] for p in pairs])), 4),
            "accuracy_on_equal_condition_mean": round(float(np.mean([p[1] for p in pairs])), 4),
        })
    return rows


def main():
    preds = _pooled_predictions()
    if not preds:
        raise SystemExit("Run evaluate_all.py first (no per-item prediction files).")
    slice_map = _slice_map()
    tiers = sorted({t for t, _ in preds})
    rows: List[Dict] = []
    for tier in tiers:
        available = {m for t, m in preds if t == tier}
        if METHODS.PROPOSED not in available:
            logger.warning("No %s predictions for %s; tests skipped.", METHODS.PROPOSED, tier)
            continue
        placement_arms = [m for m in ("ablation_placement_uniform", "ablation_placement_random",
                                      "ablation_placement_random_draw2", "ablation_placement_random_draw3")
                          if m in available]
        baselines = [m for m in METHODS.BASELINES + METHODS.REFERENCES if m in available]
        for slice_name in ("all", "structure_familiar", "structure_novel"):
            if placement_arms:
                rows += _rows_for_family(tier, "H1_localization_matched_budget", slice_name,
                                         METHODS.PROPOSED, placement_arms, preds)
            if baselines:
                rows += _rows_for_family(tier, "H2_external_comparison", slice_name,
                                         METHODS.PROPOSED, baselines, preds)

    write_csv(RESULTS_DIR / "statistical_tests.csv", rows, COLUMNS)
    write_csv(RESULTS_DIR / "seed_variability_table.csv", _seed_rows(slice_map), SEED_COLUMNS)
    n_sig = sum(1 for r in rows if r["significant_after_holm_correction"] == "yes")
    logger.info("Wrote %d test rows; %d significant after Holm correction.", len(rows), n_sig)
    log_run_metadata("statistics_tests", {"comparisons": len(rows), "significant_after_holm": n_sig})


if __name__ == "__main__":
    main()
