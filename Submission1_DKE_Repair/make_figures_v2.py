"""Figures for the repaired DKE manuscript, drawn from the version-2 result tables.

    python -m Submission1_DKE_Repair.make_figures_v2

fig_wording_v2.pdf   E1: accuracy under the two wordings, and how often each system simply
                     answers "roughly equal" once the question is fully specified.
fig_evidence_v2.pdf  M3: the corrected three-category response distribution, counted on semantic
                     categories rather than raw answer strings, beside accuracy by relation.
fig_advice_v2.pdf    M4: both-reader and either-reader proportions with case-clustered intervals.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Submission1_Code_Phase2 import common as C

OUT = "results_submission1_dke_repair_v2"
# Kept short: three panels across a text-width figure leave no room for two-line labels.
SHORT = {"broad-instruct|frozen_base|seed42": "broad\nunmod.",
         "broad-instruct|graft_proposed|seed42": "broad\nadapt.",
         "small-instruct|frozen_base|seed42": "small\nunmod.",
         "small-instruct|graft_proposed|seed42": "small\nadapt."}
INK, ACC, GREY = "#1b2a3a", "#b4552d", "#8c94a0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.dpi": 200})


def _rows(p: Path):
    return list(csv.DictReader(p.open(encoding="utf-8")))


def wording(base: Path, dest: Path) -> None:
    rows = _rows(base / "r1_corrected_results.csv")
    labels = [SHORT[r["system"]] for r in rows]
    a = [float(r["wording_a_accuracy"]) for r in rows]
    b = [float(r["wording_b_accuracy"]) for r in rows]
    d = [float(r["a_minus_b"]) for r in rows]
    lo = [x - float(r["ci_lower"]) for x, r in zip(d, rows)]
    hi = [float(r["ci_upper"]) - x for x, r in zip(d, rows)]

    # how often each system just answers "roughly equal"
    preds = list(C.read_jsonl(base / "v2_predictions.jsonl"))
    share = []
    for r in rows:
        v = [p for p in preds if p["study"] == "r1_corrected" and p["system"] == r["system"]]
        eq = sum(1 for p in v if "roughly equal" in str(p.get("picked_choice", "")).lower())
        share.append(eq / len(v) if v else 0.0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.4, 3.0),
                                        gridspec_kw={"width_ratios": [1.15, 1, 1]})
    x = np.arange(len(rows))
    ax1.bar(x - 0.19, a, 0.38, label="wording A", color=GREY)
    ax1.bar(x + 0.19, b, 0.38, label="wording B", color=ACC)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=7.5); ax1.set_ylim(0, 1)
    ax1.set_ylabel("accuracy, 34 corrected\ncomparisons")
    ax1.axhline(8 / 34, ls=":", lw=1, color=INK)
    ax1.text(-0.45, 8 / 34 + 0.02, "all-equal answer", fontsize=7)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper right")

    ax2.axhline(0, color=INK, lw=1)
    ax2.errorbar(x, d, yerr=[lo, hi], fmt="o", color=ACC, capsize=3, lw=1.4, ms=5)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5)
    ax2.set_ylabel("accuracy lost, A minus B")

    ax3.bar(x, share, 0.55, color=[ACC if s > 0.9 else GREY for s in share])
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7.5); ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("share answered\n“roughly equal”")
    fig.tight_layout()
    fig.savefig(dest / "fig_wording_v2.pdf"); plt.close(fig)


def evidence(base: Path, dest: Path) -> None:
    sem = _rows(base / "r2_semantic_response_distribution.csv")
    old = _rows(C.CODES_ROOT / "results_submission1_phase2" / "tables" / "r2_evidence_results.csv")
    acc = {r["system"]: r for r in old}
    labels = [SHORT[r["system"]] for r in sem]
    x = np.arange(len(sem))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, (k, lab, col) in enumerate([("first_group", "first higher", GREY),
                                       ("second_group", "second higher", INK),
                                       ("roughly_equal", "about equal", ACC)]):
        vals = [int(r[k]) / int(r["n"]) for r in sem]
        a1.bar(x + (i - 1) * 0.27, vals, 0.26, label=lab, color=col)
    a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_ylim(0, 1)
    a1.set_ylabel("share of responses\nin each semantic category")
    a1.axhline(1 / 3, ls=":", lw=1, color=INK)
    a1.legend(frameon=False, fontsize=7.5, loc="upper left")

    rel = [("accuracy_first_higher", "first higher", GREY),
           ("accuracy_second_higher", "second higher", INK),
           ("accuracy_approximately_equal", "about equal", ACC)]
    for i, (k, lab, col) in enumerate(rel):
        vals = [float(acc[r["system"]][k]) for r in sem]
        a2.bar(x + (i - 1) * 0.27, vals, 0.26, color=col)
    a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylim(0, 1)
    a2.set_ylabel("accuracy, by the relation\nthe numbers encode")
    a2.axhline(1 / 3, ls=":", lw=1, color=INK)
    fig.tight_layout()
    fig.savefig(dest / "fig_evidence_v2.pdf"); plt.close(fig)


def advice(base: Path, dest: Path) -> None:
    res = json.load((base / "r3_reader_agreement_and_case_cluster_results.json").open())
    rows = _rows(base / "r3_per_system_reader_agreement.csv")
    labels = [SHORT[r["system"]] for r in rows]
    both = [float(r["both_reader_proportion"]) for r in rows]
    lo = [b - float(r["case_cluster_lower"]) for b, r in zip(both, rows)]
    hi = [float(r["case_cluster_upper"]) - b for b, r in zip(both, rows)]
    either = [int(r["either_reader_substantive"]) / int(r["n_pairs"]) for r in rows]

    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    x = np.arange(len(rows))
    for i, (b, e) in enumerate(zip(both, either)):
        ax.plot([i, i], [b, e], color=GREY, lw=6, solid_capstyle="butt", zorder=1)
    ax.errorbar(x, both, yerr=[lo, hi], fmt="o", color=ACC, capsize=3, lw=1.4, ms=5, zorder=2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("identity pairs labelled a\nsubstantive change")
    ax.set_ylim(-0.02, max(either) * 1.35)
    h = [plt.Line2D([], [], color=GREY, lw=6, label="both-reader to either-reader"),
         plt.Line2D([], [], color=ACC, marker="o", ms=5, lw=1.4,
                    label="both-reader, case-clustered 95% CI")]
    ax.legend(handles=h, frameon=False, fontsize=7, loc="upper right", handlelength=1.6)
    fig.tight_layout()
    fig.savefig(dest / "fig_advice_v2.pdf"); plt.close(fig)


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    base = C.CODES_ROOT / OUT
    dest = C.CODES_ROOT.parent / "Submission1" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    wording(base, dest); evidence(base, dest); advice(base, dest)
    print(f"[make_figures_v2] fig_wording_v2.pdf, fig_evidence_v2.pdf, fig_advice_v2.pdf -> {dest}")


if __name__ == "__main__":
    main()
