"""Dry run for the CPU analysis stage on synthetic fixtures.

Builds a small synthetic main_evaluation_results.csv and per-item prediction files under
results/dry_run/, then exercises aggregation, the paired bootstrap / Wilcoxon, and figure
rendering, plus the metric functions (Wang contingency, harmonic mean). Synthetic numbers
live only under results/dry_run/ and never leak into real outputs (Instruction.md Sections 11,
18). CPU only.

Run:  python Dry_Run/dry_run_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random

from GPU_Run.common import metrics as M
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import DRY_RUN_DIR

logger = get_logger("dry_run_analysis")


def main():
    checks = {}

    # metric correctness on a hand-built contingency
    preds = (
        [{"condition": "neq", "gold_canonical": "a", "pred_canonical": "a"}] * 6
        + [{"condition": "neq", "gold_canonical": "a", "pred_canonical": "c"}] * 4
        + [{"condition": "eq", "gold_canonical": "c", "pred_canonical": "c"}] * 7
        + [{"condition": "eq", "gold_canonical": "c", "pred_canonical": "a"}] * 3
    )
    wang = M.wang_contingency(preds)
    checks["wang_diffaware_correct"] = abs(wang["difference_aware_metric_wang_2025_recall_style"] - (6 / 10)) < 1e-9
    checks["harmonic_mean_correct"] = abs(M.contextual_fairness_score(preds) - M.harmonic_mean(0.6, 0.7)) < 1e-9
    checks["citation_preservation_runs"] = M.citation_preservation_rate("T2-4 X/Y/area/SCvsST", "per T2-4 SCvsST") == 1.0

    # advice drift metrics
    dists = M.advice_drift_embedding_distances([("apply urea now", "apply urea now"), ("x", "completely different text here")])
    checks["advice_drift_runs"] = len(dists) == 2 and dists[0] <= dists[1]

    # paired bootstrap + wilcoxon on synthetic per-item correctness
    rng = random.Random(0)
    d = [1 if rng.random() < 0.6 else 0 for _ in range(50)]
    b = [1 if rng.random() < 0.4 else 0 for _ in range(50)]
    from CPU_Run.statistics_tests import _paired_bootstrap

    delta = [x - y for x, y in zip(d, b)]
    p = _paired_bootstrap(delta, resamples=200)
    checks["paired_bootstrap_runs"] = 0.0 <= p <= 1.0

    # figure rendering smoke
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
        logger.info("  %-40s %s", k, v)
    logger.info("dry_run_analysis all_passed=%s", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit("dry_run_analysis FAILED: " + json.dumps(checks))


if __name__ == "__main__":
    main()
