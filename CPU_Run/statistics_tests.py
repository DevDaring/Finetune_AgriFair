"""Paired bootstrap and paired Wilcoxon on the primary endpoint.

A paired bootstrap with 1000 resamples at p<0.05 on the contextual fairness score between
the proposed method and the automatically-selected strongest baseline, plus a paired Wilcoxon
signed-rank test on per-item correctness deltas (Instruction.md Section 7.4). No projected
values; everything comes from the per-item prediction files. CPU only.

Run:  python CPU_Run/statistics_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob

import numpy as np
import pandas as pd

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import MAIN_EVALUATION, RESULTS_DIR

logger = get_logger("statistics_tests")

PRIMARY = "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy"
COLUMNS = [
    "tier", "comparison", "proposed_method", "strongest_baseline",
    "contextual_fairness_delta", "bootstrap_p_value", "wilcoxon_p_value",
    "bootstrap_resamples", "n_items",
]


def _correctness_vector(path):
    rows = read_jsonl(path)
    return {r["id"]: int(r["pred_canonical"] == r["gold_canonical"]) for r in rows}


def _find_pred_file(tier, method):
    pats = sorted(glob.glob(str(RESULTS_DIR / f"per_item_predictions_{tier}_{method}_seed*.jsonl")))
    return Path(pats[0]) if pats else None


def _paired_bootstrap(delta_per_item, resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(delta_per_item, dtype=float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = d.mean()
    count = 0
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if (d[idx].mean() <= 0) == (obs > 0):
            count += 1
    # two-sided-ish: fraction of resamples crossing zero against the observed sign
    return count / resamples


def main():
    if not MAIN_EVALUATION.exists():
        raise SystemExit("Run evaluate_all.py first.")
    from scipy.stats import wilcoxon

    df = pd.read_csv(MAIN_EVALUATION)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    overall[PRIMARY] = pd.to_numeric(overall[PRIMARY], errors="coerce")

    rows = []
    for tier in sorted(overall["tier"].unique()):
        sub = overall[overall["tier"] == tier]
        means = sub.groupby("method")[PRIMARY].mean()
        if "xlora_bias_proposed" not in means.index:
            continue
        baselines = means.drop(index=[m for m in means.index if m.startswith(("xlora_bias_proposed", "frozen_base", "ablation"))], errors="ignore")
        if baselines.empty:
            continue
        strongest = baselines.idxmax()
        delta = float(means["xlora_bias_proposed"] - means[strongest])

        pf_prop = _find_pred_file(tier, "xlora_bias_proposed")
        pf_base = _find_pred_file(tier, strongest)
        boot_p = wil_p = float("nan")
        n_items = 0
        if pf_prop and pf_base:
            cp, cb = _correctness_vector(pf_prop), _correctness_vector(pf_base)
            common = sorted(set(cp) & set(cb))
            n_items = len(common)
            d = [cp[i] - cb[i] for i in common]
            boot_p = _paired_bootstrap(d)
            if any(x != 0 for x in d):
                try:
                    wil_p = float(wilcoxon(d).pvalue)
                except Exception:
                    wil_p = float("nan")
        rows.append({
            "tier": tier, "comparison": "proposed_vs_strongest_baseline",
            "proposed_method": "xlora_bias_proposed", "strongest_baseline": strongest,
            "contextual_fairness_delta": round(delta, 4),
            "bootstrap_p_value": round(boot_p, 4) if boot_p == boot_p else "",
            "wilcoxon_p_value": round(wil_p, 4) if wil_p == wil_p else "",
            "bootstrap_resamples": 1000, "n_items": n_items,
        })
        logger.info("tier=%s proposed vs %s delta=%.4f boot_p=%.4f", tier, strongest, delta, boot_p)

    write_csv(RESULTS_DIR / "statistical_tests.csv", rows, COLUMNS)
    log_run_metadata("statistics_tests", {"comparisons": len(rows)})


if __name__ == "__main__":
    main()
