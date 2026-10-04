"""Figures for the DKE manuscript, drawn from the phase-2 result tables only.

    python -m Submission1_Code_Phase2.make_figures

Three figures, one per research question, as the revision plan asks:
  fig_template_transfer.pdf  R1  accuracy on the original wording against a new wording
  fig_evidence.pdf           R2  accuracy by the relation the supplied numbers actually encode
  fig_advice_floor.pdf       R3  conservative floor for identity-driven change, with exact CIs

Nothing here recomputes a result. Each panel reads a committed CSV so the figure and the text
cannot drift apart.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Submission1_Code_Phase2 import common as C

# Labels match the manuscript's wording exactly: the text says "unmodified", not "frozen".
SHORT = {"broad-instruct|frozen_base|seed42": "broad\nunmodified",
         "broad-instruct|graft_proposed|seed42": "broad\nadapted",
         "small-instruct|frozen_base|seed42": "small\nunmodified",
         "small-instruct|graft_proposed|seed42": "small\nadapted"}
INK, ACC, GREY = "#1b2a3a", "#b4552d", "#8c94a0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "figure.dpi": 200})


def _rows(p: Path):
    return list(csv.DictReader(p.open(encoding="utf-8")))


def template_transfer(tables: Path, dest: Path) -> None:
    rows = _rows(tables / "r1_template_transfer_results.csv")
    labels = [SHORT[r["system"]] for r in rows]
    orig = [float(r["original_family_accuracy"]) for r in rows]
    new = [float(r["new_family_accuracy"]) for r in rows]
    drop = [float(r["original_minus_new_wording"]) for r in rows]
    lo = [d - float(r["ci_lower"]) for d, r in zip(drop, rows)]
    hi = [float(r["ci_upper"]) - d for d, r in zip(drop, rows)]

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(rows))
    a.bar(x - 0.19, orig, 0.38, label="original wording", color=GREY)
    a.bar(x + 0.19, new, 0.38, label="new wording", color=ACC)
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylim(0, 1)
    a.set_ylabel("accuracy on 34 fresh comparisons")
    a.axhline(1 / 3, ls=":", lw=1, color=INK)
    a.text(len(rows) - 0.55, 1 / 3 + 0.02, "chance", fontsize=7.5, color=INK)
    a.legend(frameon=False, fontsize=7.5, loc="upper left")

    b.axhline(0, color=INK, lw=1)
    b.errorbar(x, drop, yerr=[lo, hi], fmt="o", color=ACC, capsize=3, lw=1.4, ms=5)
    b.set_xticks(x); b.set_xticklabels(labels)
    b.set_ylabel("accuracy lost to rewording\n(percentage points / 100)")
    fig.tight_layout()
    fig.savefig(dest / "fig_template_transfer.pdf"); plt.close(fig)


def evidence(tables: Path, dest: Path) -> None:
    rows = _rows(tables / "r2_evidence_results.csv")
    labels = [SHORT[r["system"]] for r in rows]
    rel = [("accuracy_first_higher", "first higher", GREY),
           ("accuracy_second_higher", "second higher", INK),
           ("accuracy_approximately_equal", "about equal", ACC)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    x = np.arange(len(rows))
    for i, (k, lab, col) in enumerate(rel):
        a.bar(x + (i - 1) * 0.27, [float(r[k]) for r in rows], 0.26, label=lab, color=col)
    a.set_xticks(x); a.set_xticklabels(labels); a.set_ylim(0, 1)
    a.set_ylabel("accuracy by the relation\nthe supplied numbers encode")
    a.axhline(1 / 3, ls=":", lw=1, color=INK)
    a.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")

    share = [float(r["most_frequent_answer_share"]) for r in rows]
    b.bar(x, share, 0.5, color=[ACC if s > 0.5 else GREY for s in share])
    b.axhline(1 / 3, ls=":", lw=1, color=INK)
    b.text(-0.42, 1 / 3 + 0.02, "balanced design", fontsize=7.5, color=INK)
    b.set_xticks(x); b.set_xticklabels(labels); b.set_ylim(0, 1)
    b.set_ylabel("share taken by the single\nmost frequent answer")
    fig.tight_layout()
    fig.savefig(dest / "fig_evidence.pdf"); plt.close(fig)


def advice_floor(advice: Path, dest: Path) -> None:
    rows = [r for r in _rows(advice / "r3_adjudicated_change.csv")
            if r["case_type"] == "identity_irrelevant"]
    labels = [SHORT[r["system"]] for r in rows]
    floor = [float(r["share_floor"]) for r in rows]
    ceil = [float(r["share_ceiling"]) for r in rows]
    lo = [f - float(r["floor_ci_low"]) for f, r in zip(floor, rows)]
    hi = [float(r["floor_ci_high"]) - f for f, r in zip(floor, rows)]

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(rows))
    for i, (f, c) in enumerate(zip(floor, ceil)):
        ax.plot([i, i], [f, c], color=GREY, lw=6, solid_capstyle="butt", zorder=1)
    ax.errorbar(x, floor, yerr=[lo, hi], fmt="o", color=ACC, capsize=3, lw=1.4, ms=5, zorder=2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pairs whose advice changed when\nonly the asker's identity changed")
    ax.set_ylim(-0.02, max(ceil) * 1.32)
    handles = [plt.Line2D([], [], color=GREY, lw=6, label="floor to ceiling"),
               plt.Line2D([], [], color=ACC, marker="o", ms=5, lw=1.4,
                          label="floor, exact 95% CI")]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right",
              handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(dest / "fig_advice_floor.pdf"); plt.close(fig)


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    cfg = C.load_config()
    out = C.CODES_ROOT / cfg["output_directory"]
    dest = C.CODES_ROOT.parent / "Submission1" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    template_transfer(out / "tables", dest)
    evidence(out / "tables", dest)
    advice_floor(out / "advice", dest)
    print(f"[make_figures] fig_template_transfer.pdf, fig_evidence.pdf, fig_advice_floor.pdf -> {dest}")


if __name__ == "__main__":
    main()
