"""Figures for the TMLR manuscript, drawn from results_submission1_tmlr/analysis.

    python -m Submission1_TMLR.make_figures

fig_ranking.pdf    accuracy from memory (154 verified comparisons) against accuracy with supplied
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
FAM_COLOR = {"Llama-3.2-3B": "#0072B2", "Qwen3-4B": "#D55E00", "Ministral-8B": "#009E73", "Gemma-3-12B": "#CC79A7",
             "Llama-3.3-70B": "#0072B2", "Qwen3-32B": "#D55E00", "Mistral-Large-3": "#009E73", "Gemma-3-27B": "#CC79A7",
             "Qwen3-Next-80B": "#E69F00", "DeepSeek-V3.2": "#000000", "gpt-oss-120B": "#56B4E9", "Kimi-K2.5": "#F0E442",
             "GLM-5": "#999999"}
HOSTED = {"Llama-3.3-70B", "Qwen3-32B", "Mistral-Large-3", "Gemma-3-27B", "Qwen3-Next-80B", "DeepSeek-V3.2",
          "gpt-oss-120B", "Kimi-K2.5", "GLM-5"}
MARKER = {"unmodified": "o", "attribution-guided LoRA": "^", "plain LoRA": "s", "random-placement LoRA": "D"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})


def read(name):
    return list(csv.DictReader((AN / name).open(encoding="utf-8")))


def family_method(label: str):
    parts = [p.strip() for p in label.split(",")]
    return parts[0], (parts[1] if len(parts) > 1 else "unmodified")


# one colour per family line: a GPU family and its larger hosted sibling share it
LINE = {"Llama-3.2-3B": "Llama (3.2-3B, 3.3-70B)", "Llama-3.3-70B": "Llama (3.2-3B, 3.3-70B)",
        "Qwen3-4B": "Qwen3 (4B, 32B)", "Qwen3-32B": "Qwen3 (4B, 32B)",
        "Ministral-8B": "Mistral (Ministral-8B, Large-3)", "Mistral-Large-3": "Mistral (Ministral-8B, Large-3)",
        "Gemma-3-12B": "Gemma-3 (12B, 27B)", "Gemma-3-27B": "Gemma-3 (12B, 27B)",
        "Qwen3-Next-80B": "Qwen3-Next-80B", "DeepSeek-V3.2": "DeepSeek-V3.2", "gpt-oss-120B": "gpt-oss-120B",
        "Kimi-K2.5": "Kimi-K2.5", "GLM-5": "GLM-5"}
LINE_COLOR = {"Llama (3.2-3B, 3.3-70B)": "#0072B2", "Qwen3 (4B, 32B)": "#D55E00",
              "Mistral (Ministral-8B, Large-3)": "#009E73", "Gemma-3 (12B, 27B)": "#CC79A7",
              "Qwen3-Next-80B": "#E69F00", "DeepSeek-V3.2": "#000000", "gpt-oss-120B": "#56B4E9",
              "Kimi-K2.5": "#B8860B", "GLM-5": "#7F7F7F"}


def fig_ranking():
    rows = read("h5_ranking_per_system.csv")
    fig, ax = plt.subplots(figsize=(6.0, 5.3))
    for r in rows:
        fam, meth = family_method(r["system"])
        hosted = fam in HOSTED
        ax.scatter(float(r["accuracy_from_memory_154"]), float(r["accuracy_supplied_numbers"]),
                   c=LINE_COLOR[LINE[fam]], marker="P" if hosted else MARKER[meth],
                   s=60 if hosted else (44 if meth == "unmodified" else 28),
                   edgecolors="white", linewidths=0.4, zorder=3)
    pos = {family_method(r["system"]): (float(r["accuracy_from_memory_154"]), float(r["accuracy_supplied_numbers"])) for r in rows}
    for (fam, meth), (dx, dy), ha in ((("Qwen3-4B", "unmodified"), (0.03, 0.0), "left"),
                                      (("gpt-oss-120B", "unmodified"), (-0.03, 0.0), "right"),
                                      (("Llama-3.3-70B", "unmodified"), (0.0, -0.06), "center"),
                                      (("Llama-3.2-3B", "unmodified"), (0.03, 0.0), "left")):
        if (fam, meth) in pos:
            x, y = pos[(fam, meth)]
            ax.annotate(fam, (x, y), (x + dx, y + dy), fontsize=8, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="0.3"))
    ax.set_xlabel("Accuracy from memory (154 verified census comparisons)")
    ax.set_ylabel("Accuracy with supplied numbers\n(288 numerical prompts)")
    ax.set_xlim(0, 0.9); ax.set_ylim(0.3, 1.04)
    ax.grid(lw=0.3, alpha=0.4)
    fh = [plt.Line2D([], [], marker="o", ls="", c=c, label=f) for f, c in LINE_COLOR.items()]
    mh = [plt.Line2D([], [], marker=m, ls="", c="grey", label=k) for k, m in MARKER.items()]
    mh.append(plt.Line2D([], [], marker="P", ls="", c="grey", label="hosted, unmodified"))
    fig.legend(handles=fh, loc="lower center", bbox_to_anchor=(0.5, 0.14), ncol=3, frameon=False,
               fontsize=7.5, title="Model family", title_fontsize=8)
    fig.legend(handles=mh, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
               fontsize=7.5, title="Version", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.31, 1, 1))
    fig.savefig(DEST / "fig_ranking.pdf"); plt.close(fig)


COND_STYLE = {"standard": ("#4D4D4D", "o"), "no rule": ("#E69F00", "s"), "+ cannot tell": ("#56B4E9", "^"),
              "real table": ("#009E73", "D"), "reasoning": ("#CC79A7", "v")}


def fig_followups():
    """Dot plot for the 13 unmodified models: share of 'roughly equal' answers and accuracy, under the
    standard prompt and each follow-up condition (154 verified comparisons, both wordings). Conditions
    differ in marker shape as well as colour, so the figure reads in greyscale."""
    order = list(FAM_COLOR)
    rows = [r for r in read("followups.csv") if family_method(r["system"])[1] == "unmodified"]
    rows.sort(key=lambda r: order.index(family_method(r["system"])[0]), reverse=True)
    panels = [("Share of \u201croughly equal\u201d answers", [("standard_equal_share", "standard"), ("norule_equal_share", "no rule"),
                                                      ("abstain_equal_share", "+ cannot tell"), ("cot_equal_share", "reasoning")]),
              ("Accuracy", [("standard_accuracy", "standard"), ("norule_accuracy", "no rule"),
                            ("realtable_accuracy", "real table"), ("cot_accuracy", "reasoning")])]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.2), sharey=True)
    for ax, (title, conds) in zip(axes, panels):
        for j in range(len(rows)):
            if j % 2 == 0:
                ax.axhspan(j - 0.5, j + 0.5, color="#F2F2F2", zorder=0, lw=0)
        for i, (key, lab) in enumerate(conds):
            col, mk = COND_STYLE[lab]
            off = (i - (len(conds) - 1) / 2) * 0.17
            xs = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
            ys = [j + off for j, r in enumerate(rows) if r.get(key) not in (None, "")]
            ax.scatter(xs, ys, s=26, marker=mk, color=col, edgecolors="black", linewidths=0.3, zorder=3, label=lab)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([family_method(r["system"])[0] for r in rows], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("Proportion of 308 prompts", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", lw=0.3, alpha=0.6)
    handles = [plt.Line2D([], [], marker=m, ls="", color=c, markeredgecolor="black", markeredgewidth=0.3,
                          markersize=6, label=lab) for lab, (c, m) in COND_STYLE.items()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(DEST / "fig_followups.pdf"); plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    fig_ranking()
    if (AN / "followups.csv").exists():
        fig_followups()
    print("figures ->", DEST)


if __name__ == "__main__":
    main()
