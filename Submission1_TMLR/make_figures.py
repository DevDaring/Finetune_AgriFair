"""Figures for the TMLR manuscript, drawn from results_submission1_tmlr/analysis.

    python -m Submission1_TMLR.make_figures

fig_ranking.pdf    accuracy from memory (155 verified comparisons) against accuracy with supplied
                   numbers (288 numerical prompts), one point per system (34).
fig_followups.pdf  share of "roughly equal" answers under the standard prompt and the follow-up
                   conditions, for the unmodified models.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Submission1_Code_Phase2 import common as C

AN = C.CODES_ROOT / "results_submission1_tmlr" / "analysis"
DEST = C.CODES_ROOT.parent / "Submission1" / "figures_tmlr"
FAM_COLOR = {"Llama-3.2-3B": "#0072B2", "Qwen3-4B": "#D55E00", "Ministral-8B": "#009E73",
             "Gemma-3-12B": "#CC79A7", "Qwen3-Next-80B": "#E69F00", "DeepSeek-V3.2": "#000000"}
MARKER = {"unmodified": "o", "attribution-guided LoRA": "^", "plain LoRA": "s", "random-placement LoRA": "D"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def read(name):
    return list(csv.DictReader((AN / name).open(encoding="utf-8")))


def family_method(label: str):
    parts = [p.strip() for p in label.split(",")]
    return parts[0], (parts[1] if len(parts) > 1 else "unmodified")


def fig_ranking():
    rows = read("h5_ranking_per_system.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for r in rows:
        fam, meth = family_method(r["system"])
        ax.scatter(float(r["accuracy_from_memory_155"]), float(r["accuracy_supplied_numbers"]),
                   c=FAM_COLOR[fam], marker=MARKER[meth], s=28 if meth == "unmodified" else 18,
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xlabel("Accuracy from memory (155 verified comparisons)")
    ax.set_ylabel("Accuracy with supplied numbers\n(288 numerical prompts)")
    ax.set_xlim(0, 0.85); ax.set_ylim(0.3, 0.9)
    fh = [plt.Line2D([], [], marker="o", ls="", c=c, label=f) for f, c in FAM_COLOR.items()]
    mh = [plt.Line2D([], [], marker=m, ls="", c="grey", label=k) for k, m in MARKER.items()]
    leg1 = ax.legend(handles=fh, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="Family")
    ax.add_artist(leg1)
    ax.legend(handles=mh, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False, title="Version")
    fig.tight_layout()
    fig.savefig(DEST / "fig_ranking.pdf"); plt.close(fig)


COND_COLOR = {"standard": "#4D4D4D", "no rule": "#E69F00", "+ cannot tell": "#56B4E9",
              "real table": "#009E73", "reasoning": "#CC79A7"}


def fig_followups():
    """Two panels for the six unmodified models: share of 'roughly equal' answers and accuracy, under
    the standard prompt and each follow-up condition (155 verified comparisons, both wordings)."""
    order = list(FAM_COLOR)
    rows = [r for r in read("followups.csv") if family_method(r["system"])[1] == "unmodified"]
    rows.sort(key=lambda r: order.index(family_method(r["system"])[0]))
    panels = [("Share of 'roughly equal' answers", [("standard_equal_share", "standard"), ("norule_equal_share", "no rule"),
                                                      ("abstain_equal_share", "+ cannot tell"), ("cot_equal_share", "reasoning")]),
              ("Accuracy", [("standard_accuracy", "standard"), ("norule_accuracy", "no rule"),
                            ("realtable_accuracy", "real table"), ("cot_accuracy", "reasoning")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, (title, conds) in zip(axes, panels):
        w = 0.8 / len(conds)
        for i, (key, lab) in enumerate(conds):
            xs, ys = [], []
            for j, r in enumerate(rows):
                if r.get(key) not in (None, ""):
                    xs.append(j + (i - (len(conds) - 1) / 2) * w); ys.append(float(r[key]))
            ax.bar(xs, ys, w * 0.92, label=lab, color=COND_COLOR[lab])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([family_method(r["system"])[0] for r in rows], rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("Proportion of 310 prompts\n(155 comparisons x 2 wordings)")
    handles = {lab: plt.Rectangle((0, 0), 1, 1, color=c) for lab, c in COND_COLOR.items()}
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=5, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(DEST / "fig_followups.pdf"); plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fig_ranking()
    if (AN / "followups.csv").exists():
        fig_followups()
    print("figures ->", DEST)


if __name__ == "__main__":
    main()
