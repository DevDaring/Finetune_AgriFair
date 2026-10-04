"""LaTeX tables for the TMLR manuscript, written from results_submission1_tmlr/analysis/*.csv.

    python -m Submission1_TMLR.make_tables

Every number in a table comes from the analysis files, so the manuscript and the released outputs
cannot drift apart. Tables are written to Submission1/tables_tmlr/ and \\input by TMLR_AgriFair.tex.
"""
from __future__ import annotations

import collections
import csv
import json

from Submission1_Code_Phase2 import common as C

AN = C.CODES_ROOT / "results_submission1_tmlr" / "analysis"
DEST = C.CODES_ROOT.parent / "Submission1" / "tables_tmlr"
FAMS = ["Llama-3.2-3B", "Qwen3-4B", "Ministral-8B", "Gemma-3-12B"]
LARGE = ["Qwen3-Next-80B", "DeepSeek-V3.2"]
METHODS = ["unmodified", "attribution-guided LoRA", "plain LoRA", "random-placement LoRA"]
SHORT = {"unmodified": "unmodified", "attribution-guided LoRA": "attribution-guided",
         "plain LoRA": "plain LoRA", "random-placement LoRA": "random placement"}


def read(name):
    p = AN / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def f3(x):
    return "--" if x in (None, "", "nan") else f"{float(x):.3f}"


def sgn(x):
    return "--" if x in (None, "", "nan") else f"${float(x):+.3f}$"


def ci(lo, hi):
    return f"$[{float(lo):+.2f},\\ {float(hi):+.2f}]$"


def split(label):
    parts = [p.strip() for p in label.split(",")]
    fam = parts[0]
    meth = parts[1] if len(parts) > 1 else "unmodified"
    seed = None
    if len(parts) > 2:   # "seed 43" -> 43; "draw 2" -> "draw 2"
        seed = int(parts[2].replace("seed", "")) if parts[2].startswith("seed") else parts[2]
    return fam, meth, seed


def rng(vals):
    vals = [float(v) for v in vals if v not in (None, "")]
    if not vals:
        return "--"
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return f"{m:.3f}"
    return f"{m:.3f} {{\\scriptsize[{min(vals):.2f}, {max(vals):.2f}]}}"


def write(name, body):
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / name).write_text(body, encoding="utf-8")


# ---------------------------------------------------------------- main-text tables
def tab_overview():
    """One row per family x method: memory accuracy, share 'roughly equal', supplied-number accuracy,
    fully correct scenarios. Mean over seeds with [min, max]."""
    mem = {r["system"]: r for r in read("h2_item_type.csv") if r["set"] == "all verified (155)"}
    num = {r["system"]: r for r in read("h4_numerical.csv")}
    groups = collections.defaultdict(list)
    for s in mem:
        fam, meth, _ = split(s)
        groups[(fam, meth)].append(s)
    lines = []
    for fam in LARGE + FAMS:
        meths = ["unmodified"] if fam in LARGE else METHODS
        for i, meth in enumerate(meths):
            ss = groups.get((fam, meth), [])
            if not ss:
                continue
            cell = [rng([mem[s]["accuracy_overall"] for s in ss]), rng([mem[s]["accuracy_difference_items"] for s in ss]),
                    rng([mem[s]["share_roughly_equal_answers"] for s in ss]),
                    rng([num[s]["accuracy_overall"] for s in ss if s in num]),
                    rng([num[s]["fully_correct_scenarios_pooled"] for s in ss if s in num])]
            name = fam if i == 0 else ""
            lines.append(f"{name} & {SHORT[meth]} & {len(ss)} & " + " & ".join(cell) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write("tab_overview.tex", "\n".join(lines) + "\n")


def tab_followups():
    rows = read("followups.csv")
    lines = []
    order = LARGE + FAMS
    rows = sorted(rows, key=lambda r: (order.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1])))
    for r in rows:
        fam, meth, seed = split(r["system"])
        if meth not in ("unmodified", "attribution-guided LoRA") or (seed not in (None, 42)):
            continue
        name = fam + ("" if meth == "unmodified" else ", attr.-guided")
        lines.append(" & ".join([name, f3(r["standard_equal_share"]), f3(r.get("norule_equal_share")),
                                 f3(r.get("abstain_equal_share")), f3(r.get("abstain_cant_tell_share")),
                                 f3(r["standard_accuracy"]), f3(r.get("realtable_accuracy")),
                                 f3(r.get("cot_accuracy")), f3(r.get("cot_no_final_answer_share"))]) + r" \\")
    write("tab_followups.tex", "\n".join(lines) + "\n")


def tab_finetune():
    """H4 per family x method: fine-tuned minus unmodified on both outcomes, one row per seed."""
    rows = read("h4_finetune_contrasts.csv")
    by = collections.defaultdict(dict)
    for r in rows:
        key = "num" if r["outcome"].startswith("fully") else "mem"
        by[(r["family"], r["method"], r["variant"])][key] = r
    lines = []
    for fam in FAMS:
        first = True
        for meth in METHODS[1:]:
            for seed in ("seed 42", "seed 43", "seed 44", "draw 1", "draw 2", "draw 3"):
                d = by.get((fam, meth, seed))
                if not d:
                    continue
                cells = []
                for key in ("mem", "num"):
                    r = d.get(key)
                    if r:
                        star = "$^{*}$" if float(r["p_holm_h4"]) < 0.05 else ""
                        cells += [sgn(r["difference"]) + star, ci(r["ci_lower"], r["ci_upper"])]
                    else:
                        cells += ["--", "--"]
                lines.append(" & ".join([fam if first else "", SHORT[meth], seed.replace("seed ", "")] + cells) + r" \\")
                first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write("tab_finetune.tex", "\n".join(lines) + "\n")


def tab_methods():
    rows = read("h6_method_contrasts.csv")
    lines = []
    for fam in FAMS:
        rr = [r for r in rows if r["family"] == fam]
        pairs = collections.OrderedDict()
        for r in rr:
            pairs.setdefault(r["contrast"], {})["num" if r["outcome"].startswith("fully") else "mem"] = r
        first = True
        for contrast, d in pairs.items():
            a, b = contrast.split(" minus ")
            fa, ma, sa = split(a); fb, mb, sb = split(b)
            lab = f"{SHORT[ma]} {sa} $-$ {SHORT[mb]} {sb}".replace("random placement draw", "random draw")
            cells = []
            for key in ("mem", "num"):
                r = d.get(key)
                star = "$^{*}$" if r and float(r["p_holm_h6_within_family"]) < 0.05 else ""
                cells += [sgn(r["difference"]) + star, ci(r["ci_lower"], r["ci_upper"])] if r else ["--", "--"]
            lines.append(" & ".join([fam if first else "", lab] + cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write("tab_methods.tex", "\n".join(lines) + "\n")


def tab_altered():
    rows = [r for r in read("altered_tables.csv") if r.get("scope", "all 48") == "all 48"]
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["system"]][r["alteration"]] = r
    order = LARGE + FAMS
    lines = []
    for s in sorted(by, key=lambda s: (order.index(split(s)[0]), METHODS.index(split(s)[1]), split(s)[2] or 0)):
        fam, meth, seed = split(s)
        if meth != "unmodified":
            continue
        d = by[s]
        cells = [f3(d["irrelevant_column"]["unchanged_accuracy"])]
        for v in ("irrelevant_column", "row_order_reversed"):
            r = d.get(v)
            star = "$^{*}$" if r and r.get("p_holm") not in (None, "") and float(r["p_holm"]) < 0.05 else ""
            cells += [sgn(r["altered_minus_unchanged"]) + star] if r else ["--"]
        cells += [f3(d["values_removed"]["altered_accuracy"])]   # share choosing 'values not in the table'

        lines.append(" & ".join([fam] + cells) + r" \\")
    write("tab_altered.tex", "\n".join(lines) + "\n")


def tab_repro():
    rows = read("reproducibility.csv")
    pub = [r for r in rows if r["n_compared_with_published"] not in ("", "0")]
    fl = [r for r in rows if r["n_compared_sdpa_vs_flash"] not in ("", "0")]
    lines = []
    for r in pub:
        lines.append(f"{r['system']} & {r['n_compared_with_published']} & {f3(r['identical_answer_vs_published'])} \\\\")
    write("tab_repro_published.tex", "\n".join(lines) + "\n")
    tot = sum(int(r["n_compared_sdpa_vs_flash"]) for r in fl)
    chg = sum(int(r["answers_changed_by_flash"]) for r in fl)
    mx = max(fl, key=lambda r: float(r["share_changed_by_flash"])) if fl else None
    acc = [abs(float(r["accuracy_flash"]) - float(r["accuracy_sdpa"])) for r in fl]
    C.write_json(DEST / "flash_summary.json", {"systems": len(fl), "answers": tot, "changed": chg,
                                               "share": round(chg / tot, 4) if tot else None,
                                               "max_system": mx["system"] if mx else None,
                                               "max_share": float(mx["share_changed_by_flash"]) if mx else None,
                                               "max_abs_accuracy_change": round(max(acc), 4) if acc else None})


# ---------------------------------------------------------------- appendix: every system
def tab_all_systems():
    mem = {r["system"]: r for r in read("h2_item_type.csv") if r["set"] == "all verified (155)"}
    wd = {r["system"]: r for r in read("h3_wording.csv") if r["set"] == "all verified (155)"}
    num = {r["system"]: r for r in read("h4_numerical.csv")}
    ro = {r["system"]: r for r in read("reordered_options.csv")}
    order = LARGE + FAMS
    lines = []
    for s in sorted(mem, key=lambda s: (order.index(split(s)[0]), METHODS.index(split(s)[1]), split(s)[2] or 0)):
        fam, meth, seed = split(s)
        name = f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")
        w = wd[s]
        star = "$^{*}$" if float(w["p_holm_h3"]) < 0.05 else ""
        lines.append(" & ".join([name, f3(mem[s]["accuracy_overall"]), f3(mem[s]["accuracy_difference_items"]),
                                 f3(mem[s]["accuracy_equal_items"]), f3(mem[s]["share_roughly_equal_answers"]),
                                 sgn(w["difference"]) + star, f3(ro[s]["same_meaning"]) if s in ro else "--",
                                 f3(num[s]["accuracy_overall"]), f3(num[s]["fully_correct_scenarios_pooled"])]) + r" \\")
    write("tab_all_systems.tex", "\n".join(lines) + "\n")


def tab_altered_full():
    rows = read("altered_tables.csv")
    order = LARGE + FAMS
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["system"]][(r["alteration"], r.get("scope", "all 48"))] = r
    lines = []
    for s in sorted(by, key=lambda s: (order.index(split(s)[0]), METHODS.index(split(s)[1]), split(s)[2] or 0)):
        fam, meth, seed = split(s)
        name = f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")
        d = by[s]
        g = lambda v, sc, k: d.get((v, sc), {}).get(k)
        cells = [f3(g("irrelevant_column", "published 12", "unchanged_accuracy")),
                 f3(g("irrelevant_column", "new 36", "unchanged_accuracy"))]
        for v in ("irrelevant_column", "row_order_reversed", "values_removed"):
            cells += [sgn(g(v, "published 12", "altered_minus_unchanged")), sgn(g(v, "new 36", "altered_minus_unchanged"))]
        lines.append(" & ".join([name] + cells) + r" \\")
    write("tab_altered_full.tex", "\n".join(lines) + "\n")


def tab_neutral():
    rows = read("neutral_wording.csv")
    order = LARGE + FAMS
    lines = []
    for r in sorted(rows, key=lambda r: (order.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1]), split(r["system"])[2] or 0)):
        fam, meth, seed = split(r["system"])
        name = f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")
        lines.append(f"{name} & {f3(r['agricultural_accuracy'])} & {f3(r['neutral_accuracy'])} & {sgn(r['neutral_minus_agricultural'])} \\\\")
    write("tab_neutral.tex", "\n".join(lines) + "\n")


def _effect(rows, pkey):
    """mean difference [min, max] over seeds, and the count of Holm-significant increases and decreases."""
    ds = [float(r["difference"]) for r in rows]
    up = sum(float(r["difference"]) > 0 and float(r[pkey]) < 0.05 for r in rows)
    dn = sum(float(r["difference"]) < 0 and float(r[pkey]) < 0.05 for r in rows)
    m = sum(ds) / len(ds)
    span = f" {{\\scriptsize[{min(ds):+.2f}, {max(ds):+.2f}]}}" if len(ds) > 1 else ""
    return f"${m:+.3f}${span}", f"{up}/{dn}"


def tab_finetune_compact():
    rows = read("h4_finetune_contrasts.csv")
    lines = []
    for fam in FAMS:
        first = True
        for meth in METHODS[1:]:
            rr = [r for r in rows if r["family"] == fam and r["method"] == meth]
            if not rr:
                continue
            cells = []
            for pre in ("accuracy from memory", "fully correct"):
                sel = [r for r in rr if r["outcome"].startswith(pre)]
                cells += list(_effect(sel, "p_holm_h4")) if sel else ["--", "--"]
            lines.append(" & ".join([fam if first else "", SHORT[meth], str(len(rr) // 2)] + cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write("tab_finetune_compact.tex", "\n".join(lines) + "\n")


def tab_methods_compact():
    rows = read("h6_method_contrasts.csv")
    kinds = [("attribution-guided LoRA", "plain LoRA"), ("attribution-guided LoRA", "random-placement LoRA"),
             ("plain LoRA", "random-placement LoRA")]
    lines = []
    for fam in FAMS:
        first = True
        for ka, kb in kinds:
            rr = [r for r in rows if r["family"] == fam and split(r["contrast"].split(" minus ")[0])[1] == ka
                  and split(r["contrast"].split(" minus ")[1])[1] == kb]
            if not rr:
                continue
            cells = []
            for pre in ("accuracy from memory", "fully correct"):
                sel = [r for r in rr if r["outcome"].startswith(pre)]
                cells += list(_effect(sel, "p_holm_h6_within_family")) if sel else ["--", "--"]
            lab = f"{SHORT[ka]} $-$ {SHORT[kb]}"
            lines.append(" & ".join([fam if first else "", lab, str(len(rr) // 2)] + cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write("tab_methods_compact.tex", "\n".join(lines) + "\n")


def tab_gap():
    rows = read("h2_gap_axis.csv")
    order = LARGE + FAMS
    lines = []
    for r in sorted(rows, key=lambda r: (order.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1]), str(split(r["system"])[2] or 0))):
        fam, meth, seed = split(r["system"])
        name = f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")
        lines.append(" & ".join([name] + [f3(r[k]) for k in ("gap 10-25_equal_share", "gap 25-50_equal_share", "gap 50+_equal_share",
                                                            "landholding_equal_share", "social_group_equal_share")]) + r" \\")
    write("tab_gap.tex", "\n".join(lines) + "\n")


def main():
    for fn in (tab_overview, tab_followups, tab_finetune, tab_methods, tab_altered, tab_repro,
               tab_all_systems, tab_altered_full, tab_neutral, tab_finetune_compact, tab_methods_compact, tab_gap):
        fn()
    print("tables ->", DEST, sorted(p.name for p in DEST.iterdir()))


if __name__ == "__main__":
    main()
