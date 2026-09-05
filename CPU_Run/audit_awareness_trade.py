"""Audit of parameter-efficient debiasing under difference awareness.

Two analyses, both computed from per-item prediction files that already exist, and both
true regardless of where the proposed method ranks.

1. The failure-polarity profile. For every frozen base model, and separately per condition,
   per axis, and per test slice, this measures whether errors skew toward gap erasure
   (answering "Roughly equal" on a diff item) or gap fabrication (naming a group on an equal
   item). The study's premise is that erasure dominates; DART (arXiv:2604.16845) reports the
   opposite skew on its own suite. The profile tests the premise instead of assuming it and
   reports the answer whichever way it lands. A polarity that varies by model or condition is
   itself the argument for a selective, local repair over a global one.

2. Awareness trading. For every trained method, this measures the joint movement of
   equal-condition and diff-condition accuracy relative to the frozen base. A method is
   recorded as trading when it buys accuracy in one direction with accuracy in the other:
   one movement positive, the other negative, both beyond a tolerance that keeps sampling
   noise out. Wang et al. (ACL 2025) showed this for four prompt-based self-correction
   methods; this audit covers parameter-efficient fine-tuning methods, where it has not been
   reported.

# Wang, A., Phan, M., Ho, D. E., Koyejo, S. "Fairness through Difference Awareness:
#   Measuring Desired Group Discrimination in LLMs." ACL 2025, arXiv:2502.01926.
#   [the contingency, and the finding that debiasing can backfire on difference awareness]
# Pan, Z., Liang, Z., Kabbara, J., Emami, A. "DART: Mitigating Harm Drift in
#   Difference-Aware LLMs via Distill-Audit-Repair Training." Findings of ACL 2026,
#   arXiv:2604.16845. [the opposite failure skew that motivates measuring polarity]

Run:  python CPU_Run/audit_awareness_trade.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Dict, List

import numpy as np

from GPU_Run.common import metrics as M
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import FIGURES_DIR, RESULTS_DIR, TEST_INSTANCES_FROZEN, per_item_prediction_paths

logger = get_logger("audit_awareness_trade")

TRADE_TOLERANCE = float(os.environ.get("AWARENESS_TRADE_TOLERANCE", "0.01"))

PROFILE_COLUMNS = [
    "base_model_name", "method_name", "random_seed", "slice_name", "number_of_items",
    "gap_erasure_rate_on_diff_items", "gap_fabrication_rate_on_equal_items",
    "failure_polarity_index_positive_means_gap_erasure", "dominant_failure_polarity_label",
]

TRADE_COLUMNS = [
    "base_model_name", "method_name", "random_seed", "test_slice_name",
    "change_in_equal_condition_accuracy_versus_base",
    "change_in_diff_condition_accuracy_versus_base",
    "change_in_balanced_awareness_score_versus_base",
    "change_in_gap_erasure_rate_versus_base",
    "change_in_gap_fabrication_rate_versus_base",
    "awareness_trade_detected", "awareness_trade_direction",
    "gap_erasure_rate_after_repair", "gap_fabrication_rate_after_repair",
    "census_cell_citation_preservation_rate_after_repair",
]


def _citation_preservation(preds):
    rates = [M.citation_preservation_rate(p.get("law_reference", ""), p.get("generated_rationale", ""))
             for p in preds if p.get("generated_rationale")]
    rates = [r for r in rates if r == r]
    return float(np.mean(rates)) if rates else float("nan")


def _profile_rows(tier, method, seed, preds, slice_of) -> List[Dict]:
    rows = []
    slices = [("all", preds)]
    for axis in sorted({p.get("category") for p in preds if p.get("category")}):
        slices.append((f"axis:{axis}", [p for p in preds if p.get("category") == axis]))
    for form in sorted({p.get("form") for p in preds if p.get("form")}):
        slices.append((f"form:{form}", [p for p in preds if p.get("form") == form]))
    for name in ("structure_familiar", "structure_novel"):
        slices.append((f"slice:{name}", [p for p in preds if slice_of.get(p["id"]) == name]))
    for slice_name, sub in slices:
        if not sub:
            continue
        idx = M.failure_polarity_index(sub)
        rows.append({
            "base_model_name": tier, "method_name": method, "random_seed": seed,
            "slice_name": slice_name, "number_of_items": len(sub),
            "gap_erasure_rate_on_diff_items": round(M.gap_erasure_rate(sub), 4) if M.gap_erasure_rate(sub) == M.gap_erasure_rate(sub) else "",
            "gap_fabrication_rate_on_equal_items": round(M.gap_fabrication_rate(sub), 4) if M.gap_fabrication_rate(sub) == M.gap_fabrication_rate(sub) else "",
            "failure_polarity_index_positive_means_gap_erasure": round(idx, 4) if idx == idx else "",
            "dominant_failure_polarity_label": M.polarity_label(idx),
        })
    return rows


def _trade_figure(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if r["test_slice_name"] == "all"
           and isinstance(r["change_in_equal_condition_accuracy_versus_base"], float)
           and isinstance(r["change_in_diff_condition_accuracy_versus_base"], float)]
    if not pts:
        logger.warning("No paired base/post rows; skipping the awareness-trade figure.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in pts:
        x = r["change_in_equal_condition_accuracy_versus_base"]
        y = r["change_in_diff_condition_accuracy_versus_base"]
        colour = "#a5533a" if r["awareness_trade_detected"] == "yes" else "#3a6ea5"
        ax.scatter(x, y, s=48, color=colour)
        ax.annotate(r["method_name"], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="#999999", linewidth=1)
    ax.axvline(0, color="#999999", linewidth=1)
    lim = max(0.05, max(abs(v) for r in pts for v in
                        (r["change_in_equal_condition_accuracy_versus_base"],
                         r["change_in_diff_condition_accuracy_versus_base"])))
    ax.set_xlim(-lim * 1.2, lim * 1.2)
    ax.set_ylim(-lim * 1.2, lim * 1.2)
    ax.set_xlabel("change in equal-condition accuracy versus the frozen base")
    ax.set_ylabel("change in diff-condition accuracy versus the frozen base")
    ax.set_title("What each repair buys and what it spends")
    ax.text(-lim * 1.15, -lim * 1.1, "off-diagonal quadrants are trades:\none direction of awareness bought with the other",
            fontsize=8, color="#a5533a")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "awareness_trade.png", dpi=150)
    plt.close(fig)


def main():
    files = per_item_prediction_paths()
    if not files:
        logger.warning("No per-item prediction files; run GPU_Run/evaluate_all.py first.")
        write_csv(RESULTS_DIR / "failure_polarity_profile.csv", [], PROFILE_COLUMNS)
        write_csv(RESULTS_DIR / "awareness_trade_audit.csv", [], TRADE_COLUMNS)
        return

    slice_of = {r["id"]: r.get("test_slice", "") for r in read_jsonl(TEST_INSTANCES_FROZEN)}
    loaded = {k: read_jsonl(v) for k, v in files.items()}

    profile_rows = []
    for (tier, method, seed), preds in sorted(loaded.items()):
        if preds:
            profile_rows.extend(_profile_rows(tier, method, seed, preds, slice_of))

    base_by_tier = {t: p for (t, m, _s), p in loaded.items() if m == "frozen_base" and p}

    trade_rows = []
    for (tier, method, seed), preds in sorted(loaded.items()):
        if method == "frozen_base" or not preds:
            continue
        base = base_by_tier.get(tier)
        if not base:
            logger.warning("No frozen_base predictions for %s; trade row skipped for %s.", tier, method)
            continue
        for slice_name in ("all", "structure_familiar", "structure_novel"):
            if slice_name == "all":
                sub, base_sub = preds, base
            else:
                ids = {i for i, s in slice_of.items() if s == slice_name}
                sub = [p for p in preds if p["id"] in ids]
                base_sub = [p for p in base if p["id"] in ids]
            if not sub or not base_sub:
                continue
            d_equal = M.condition_accuracy(sub, "equal") - M.condition_accuracy(base_sub, "equal")
            d_diff = M.condition_accuracy(sub, "diff") - M.condition_accuracy(base_sub, "diff")
            d_score = M.balanced_awareness_score(sub) - M.balanced_awareness_score(base_sub)
            d_erase = M.gap_erasure_rate(sub) - M.gap_erasure_rate(base_sub)
            d_fab = M.gap_fabrication_rate(sub) - M.gap_fabrication_rate(base_sub)
            traded = (d_equal > TRADE_TOLERANCE and d_diff < -TRADE_TOLERANCE) or \
                     (d_diff > TRADE_TOLERANCE and d_equal < -TRADE_TOLERANCE)
            direction = ""
            if traded:
                direction = "equal_bought_with_diff" if d_equal > 0 else "diff_bought_with_equal"
            trade_rows.append({
                "base_model_name": tier, "method_name": method, "random_seed": seed,
                "test_slice_name": slice_name,
                "change_in_equal_condition_accuracy_versus_base": float(round(d_equal, 4)) if d_equal == d_equal else "",
                "change_in_diff_condition_accuracy_versus_base": float(round(d_diff, 4)) if d_diff == d_diff else "",
                "change_in_balanced_awareness_score_versus_base": float(round(d_score, 4)) if d_score == d_score else "",
                "change_in_gap_erasure_rate_versus_base": float(round(d_erase, 4)) if d_erase == d_erase else "",
                "change_in_gap_fabrication_rate_versus_base": float(round(d_fab, 4)) if d_fab == d_fab else "",
                "awareness_trade_detected": "yes" if traded else "no",
                "awareness_trade_direction": direction,
                "gap_erasure_rate_after_repair": round(M.gap_erasure_rate(sub), 4) if M.gap_erasure_rate(sub) == M.gap_erasure_rate(sub) else "",
                "gap_fabrication_rate_after_repair": round(M.gap_fabrication_rate(sub), 4) if M.gap_fabrication_rate(sub) == M.gap_fabrication_rate(sub) else "",
                "census_cell_citation_preservation_rate_after_repair": round(_citation_preservation(sub), 4)
                if _citation_preservation(sub) == _citation_preservation(sub) else "",
            })

    write_csv(RESULTS_DIR / "failure_polarity_profile.csv", profile_rows, PROFILE_COLUMNS)
    write_csv(RESULTS_DIR / "awareness_trade_audit.csv", trade_rows, TRADE_COLUMNS)
    try:
        _trade_figure(trade_rows)
    except Exception as e:
        logger.warning("Awareness-trade figure failed (%s).", e)

    n_trade = sum(1 for r in trade_rows if r["awareness_trade_detected"] == "yes" and r["test_slice_name"] == "all")
    base_polarity = {r["base_model_name"]: r["dominant_failure_polarity_label"]
                     for r in profile_rows if r["method_name"] == "frozen_base" and r["slice_name"] == "all"}
    logger.info("Audit: %d method rows, %d trading awareness on the full test set.",
                len([r for r in trade_rows if r["test_slice_name"] == "all"]), n_trade)
    logger.info("Frozen-base failure polarity: %s", base_polarity)
    log_run_metadata("audit_awareness_trade", {
        "profile_rows": len(profile_rows), "trade_rows": len(trade_rows),
        "methods_trading_awareness": n_trade, "base_failure_polarity": base_polarity,
        "trade_tolerance": TRADE_TOLERANCE,
    })


if __name__ == "__main__":
    main()
