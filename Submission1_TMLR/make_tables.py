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
SIBLING = {"Llama-3.2-3B": "Llama-3.3-70B", "Qwen3-4B": "Qwen3-32B", "Ministral-8B": "Mistral-Large-3",
           "Gemma-3-12B": "Gemma-3-27B"}                       # larger hosted sibling of each GPU family (round 2)
LARGE = ["Qwen3-Next-80B", "DeepSeek-V3.2", "gpt-oss-120B", "Kimi-K2.5", "GLM-5"]   # frontier open-weight, hosted
ORDER = [n for f in FAMS for n in (f, SIBLING[f])] + LARGE
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
    mem = {r["system"]: r for r in read("h2_item_type.csv") if r["set"] == "all verified (154)"}
    num = {r["system"]: r for r in read("h4_numerical.csv")}
    groups = collections.defaultdict(list)
    for s in mem:
        fam, meth, _ = split(s)
        groups[(fam, meth)].append(s)
    def cells(ss):
        return [rng([mem[s]["accuracy_overall"] for s in ss]), rng([mem[s]["accuracy_difference_items"] for s in ss]),
                rng([mem[s]["share_roughly_equal_answers"] for s in ss]),
                rng([num[s]["accuracy_overall"] for s in ss if s in num]),
                rng([num[s]["fully_correct_scenarios_pooled"] for s in ss if s in num])]
    lines = []
    for fam in FAMS:
        for i, meth in enumerate(METHODS):
            ss = groups.get((fam, meth), [])
            if ss:
                lines.append(f"{fam if i == 0 else ''} & {SHORT[meth]} & {len(ss)} & " + " & ".join(cells(ss)) + r" \\")
        sib = groups.get((SIBLING[fam], "unmodified"), [])
        if sib:
            lines.append(f"{SIBLING[fam]} & unmodified, hosted & 1 & " + " & ".join(cells(sib)) + r" \\")
        lines.append(r"\midrule")
    for fam in LARGE:
        ss = groups.get((fam, "unmodified"), [])
        if ss:
            lines.append(f"{fam} & unmodified, hosted & 1 & " + " & ".join(cells(ss)) + r" \\")
    lines.append(r"\bottomrule")
    write("tab_overview.tex", "\n".join(lines) + "\n")


def tab_followups():
    rows = read("followups.csv")
    lines = []
    order = ORDER
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
    order = ORDER
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
    mem = {r["system"]: r for r in read("h2_item_type.csv") if r["set"] == "all verified (154)"}
    wd = {r["system"]: r for r in read("h3_wording.csv") if r["set"] == "all verified (154)"}
    num = {r["system"]: r for r in read("h4_numerical.csv")}
    ro = {r["system"]: r for r in read("reordered_options.csv")}
    order = ORDER
    lines = []
    for s in sorted(mem, key=lambda s: (order.index(split(s)[0]), METHODS.index(split(s)[1]), str(split(s)[2] or 0))):
        if s not in wd or s not in num:          # a system whose run is still incomplete
            continue
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
    order = ORDER
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
    order = ORDER
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


def tab_rules():
    """E3/E4: share of 'roughly equal' answers under the standard rule, the three rule variants and few-shot,
    plus few-shot accuracy, for every unmodified model and the attribution-guided seed-42 adapters."""
    rows = read("followups.csv")
    if not rows or "fewshot_equal_share" not in rows[0] and "rule_first_equal_share" not in rows[0]:
        return
    rows = sorted(rows, key=lambda r: (ORDER.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1])))
    lines = []
    for r in rows:
        fam, meth, seed = split(r["system"])
        if meth not in ("unmodified", "attribution-guided LoRA") or seed not in (None, "seed 42", 42):
            continue
        name = fam + ("" if meth == "unmodified" else ", attr.-guided")
        star = lambda k: "$^{*}$" if r.get(f"{k}_p_holm") not in (None, "") and float(r[f"{k}_p_holm"]) < 0.05 else ""
        lines.append(" & ".join([name, f3(r["standard_equal_share"]), f3(r.get("norule_equal_share")) + star("norule"),
                                 f3(r.get("rule_2_20_equal_share")) + star("rule_2_20"),
                                 f3(r.get("rule_reversed_equal_share")) + star("rule_reversed"),
                                 f3(r.get("rule_first_equal_share")) + star("rule_first"),
                                 f3(r.get("fewshot_equal_share")), f3(r["standard_accuracy"]),
                                 f3(r.get("fewshot_accuracy")) + star("fewshot")]) + r" \\")
    write("tab_rules.tex", "\n".join(lines) + "\n")


def tab_logprobs():
    rows = read("logprobs.csv")
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (ORDER.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1]), str(split(r["system"])[2] or 0)))
    lines = []
    for r in rows:
        fam, meth, seed = split(r["system"])
        name = f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")
        lines.append(" & ".join([name, f3(r.get("standard_mean_p_equal")), f3(r.get("standard_share_p_equal_over_0.8")),
                                 f3(r.get("standard_near_tie_share")), f3(r.get("standard_argmax_accuracy")),
                                 f3(r.get("standard_ece")), f3(r.get("norule_mean_p_equal")),
                                 f3(r.get("numerical_mean_p_equal")), f3(r.get("numerical_argmax_accuracy"))]) + r" \\")
    write("tab_logprobs.tex", "\n".join(lines) + "\n")
    main_rows = [l for l, r in zip(lines, rows) if split(r["system"])[1] == "unmodified"
                 or (split(r["system"])[1] == "attribution-guided LoRA" and split(r["system"])[2] in ("seed 42", 42))]
    write("tab_logprobs_main.tex", "\n".join(main_rows) + "\n")


def tab_baselines():
    rows = read("baselines.csv")
    names = {"cross_state_prior": "Cross-state prior (leave one state out)", "constant_equal": "Constant ``roughly equal''",
             "constant_first": "Constant first entity", "constant_second": "Constant second entity"}
    lines = [f"{names[r['policy']]} & {f3(r['accuracy_154'])} & {f3(r['accuracy_difference_items'])} & "
             f"{f3(r['accuracy_equal_items'])} & {f3(r['share_roughly_equal_answers'])} \\\\" for r in rows]
    write("tab_baselines.tex", "\n".join(lines) + "\n")


def tab_gap():
    rows = read("h2_gap_axis.csv")
    order = ORDER
    lines = []
    for r in sorted(rows, key=lambda r: (order.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1]), str(split(r["system"])[2] or 0))):
        fam, meth, seed = split(r["system"])
        name = f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")
        lines.append(" & ".join([name] + [f3(r[k]) for k in ("gap 10-25_equal_share", "gap 25-50_equal_share", "gap 50+_equal_share",
                                                            "landholding_equal_share", "social_group_equal_share")]) + r" \\")
    write("tab_gap.tex", "\n".join(lines) + "\n")


AN3 = C.CODES_ROOT / "results_submission1_tmlr" / "analysis_round3"


def read3(name):
    p = AN3 / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def _key(r):
    fam, meth, seed = split(r["system"])
    return (ORDER.index(fam) if fam in ORDER else 99, METHODS.index(meth), str(seed or 0))


def _name(r):
    fam, meth, seed = split(r["system"])
    return fam if meth == "unmodified" else f"{fam}, {SHORT[meth]}" + (f" {seed}" if seed else "")


def _rows3(name, unmodified_only):
    rows = sorted(read3(name), key=_key)
    return [r for r in rows if not unmodified_only or split(r["system"])[1] == "unmodified"]


def tab_round3():
    """Round-3 tables: main-text versions (unmodified models) and appendix versions (every system)."""
    specs = {
        "wdi": ("p2_wdi.csv", ["accuracy_memory", "equal_share", "norule_equal_share", "accuracy_supplied_numbers",
                               "fully_correct_scenarios", "share_prior_answer"]),
        "mnli": ("p1_mnli.csv", ["mnli_plain_accuracy", "mnli_strict_accuracy", "mnli_lenient_accuracy",
                                 "mnli_plain_neither_share", "mnli_strict_neither_share", "mnli_lenient_neither_share"]),
        "adv": ("p5_adversarial.csv", ["accuracy", "share_prior_answer", "share_roughly_equal"]),
        "recall": ("p6_recall.csv", ["parse_rate", "median_abs_error_pp", "implied_accuracy", "mc_accuracy_same_items",
                                     "implied_correct_when_mc_said_equal_on_diff"]),
        "drift": ("p3_drifted_leaderboard.csv", ["accuracy_drifted_v1", "accuracy_corrected", "accuracy_norule",
                                                 "rank_drifted", "rank_corrected"]),
    }
    for tag, (fname, cols) in specs.items():
        for suffix, only in (("", True), ("_full", False)):
            rows = _rows3(fname, only)
            if not rows:
                continue
            lines = []
            for r in rows:
                cells = []
                for c in cols:
                    v = r.get(c)
                    cells.append(str(v) if c.startswith("rank") or c == "median_abs_error_pp" and v not in (None, "") else f3(v))
                if tag == "adv":
                    cells.insert(1, r.get("ci", "").replace("[", "{\\scriptsize[").replace("]", "]}"))
                lines.append(" & ".join([_name(r)] + cells) + r" \\")
            write(f"tab_r3_{tag}{suffix}.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- revision tables (5 Oct 2026)
ANR = C.CODES_ROOT / "results_submission1_tmlr" / "analysis_review"


def readr(name):
    p = ANR / name
    return list(csv.DictReader(p.open(encoding="utf-8"))) if p.exists() else []


def ci2(text):
    a, b = (float(x) for x in text.strip("[]").split(","))
    return f"{{\\scriptsize[{a:+.2f}, {b:+.2f}]}}"


def tau_cell(t):
    return f"{t['kendall_tau']:.2f} {{\\scriptsize[{t['ci_lower']:.2f}, {t['ci_upper']:.2f}]}}"


def tab_review():
    import re
    # main-text overview without the adapter ranges (the appendix lists every adapter)
    ov = (DEST / "tab_overview.tex").read_text(encoding="utf-8")
    write("tab_overview_main.tex", re.sub(r" \{\\scriptsize\[[^\]]*\]\}", "", ov))
    summ = json.loads((ANR / "summary_review.json").read_text())
    d = summ["R1_drift"]
    names = [("drift_only (drifted vs corrected, both without rule)", "Drift only", "drifted vs corrected, both without the rule"),
             ("rule_only (corrected without vs with rule)", "Rule only", "corrected, without vs with the rule"),
             ("both (drifted without rule vs corrected with rule)", "Both", "drifted without the rule vs corrected with it")]
    write("tab_drift_iso.tex", "\n".join(f"{a} & {b} & {tau_cell(d['all'][k])} & {tau_cell(d['unmodified'][k])} \\\\"
                                           for k, a, b in names) + "\n")
    rt = summ["R3_real_table"]
    lines = [f"Closed book vs real table (same 154 comparisons) & {tau_cell(rt['all']['closed_book_vs_real_table'])} & {tau_cell(rt['unmodified']['closed_book_vs_real_table'])} \\\\",
             f"Closed book vs numerical set & {tau_cell(rt['all']['closed_book_vs_numerical_set'])} & {tau_cell(rt['unmodified']['closed_book_vs_numerical_set'])} \\\\",
             f"Real table vs numerical set & {tau_cell(rt['all']['real_table_vs_numerical_set'])} & {tau_cell(rt['unmodified']['real_table_vs_numerical_set'])} \\\\"]
    write("tab_rank_matched.tex", "\n".join(lines) + "\n")
    # matched real table, per system
    rows = sorted(readr("r3_real_table_matched.csv"), key=_key)
    lines = [" & ".join([_name(r), f3(r["closed_book"]), f3(r["real_table"]), sgn(r["real_table_minus_closed_book"]),
                         ci2(r["ci"]), f3(r["closed_book_diff_items"]),
                         f3(r["real_table_diff_items"]), f3(r["real_table_equal_share"])]) + r" \\" for r in rows]
    write("tab_realtable_full.tex", "\n".join(lines) + "\n")
    # placement, every seed and draw averaged
    pl = readr("r6_placement_all_seeds.csv")
    cname = {"mean attribution-guided minus mean random": "attribution-guided $-$ random placement",
             "mean plain minus mean random": "plain LoRA $-$ random placement",
             "mean attribution-guided minus mean plain": "attribution-guided $-$ plain LoRA"}
    lines = []
    for fam in FAMS:
        for k, (con, lab) in enumerate(cname.items()):
            m = next(r for r in pl if r["family"] == fam and r["contrast"] == con and r["outcome"] == "memory accuracy")
            fc = next(r for r in pl if r["family"] == fam and r["contrast"] == con and r["outcome"] == "fully correct")
            cell = lambda r: f"{sgn(r['difference'])} {{\\scriptsize[{float(r['ci_lower']):+.2f}, {float(r['ci_upper']):+.2f}]}}"
            lines.append(" & ".join([fam if k == 0 else "", lab, cell(m), cell(fc)]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    write("tab_placement.tex", "\n".join(lines) + "\n")
    # hosted repeatability
    rows = sorted(readr("r5_hosted_repeatability.csv"), key=_key)
    lines = [" & ".join([_name(r), f3(r["same_choice"]), f3(r["accuracy_run1"]), f3(r["accuracy_run2"]), sgn(r["run1_minus_run2"]),
                         ci2(r["ci"])]) + r" \\" for r in rows]
    write("tab_repeat.tex", "\n".join(lines) + "\n")
    # lenient reading: systems with any unparsed answer on the 154 comparisons
    rows = sorted([r for r in readr("r4_lenient_reading.csv") if float(r["unparsed_strict"]) > 0], key=_key)
    lines = [" & ".join([_name(r), f3(r["unparsed_strict"]), str(int(r["recovered_other_key"]) + int(r["recovered_bare_letter"]) + int(r["recovered_option_text"])),
                         f3(r["accuracy_strict"]), f3(r["accuracy_lenient"]), f3(r["equal_share_strict"]), f3(r["equal_share_lenient"])]) + r" \\" for r in rows]
    write("tab_lenient.tex", "\n".join(lines) + "\n")
    # numeric recall with the tolerance subset (unmodified models)
    rows = sorted([r for r in readr("r2_recall_tolerance.csv") if r["unmodified"] == "True"], key=_key)
    p6 = {r["system"]: r for r in read3("p6_recall.csv")}
    lines = []
    for r in rows:
        q = p6.get(r["system"], {})
        lines.append(" & ".join([_name(r), str(r["parsed"]), q.get("median_abs_error_pp") or "--", f3(r.get("implied_accuracy")),
                                 f3(r.get("mc_accuracy")), str(r.get("within_10_n") or 0), f3(r.get("within_10_implied_accuracy")),
                                 f3(r.get("within_10_mc_accuracy")), f3(r.get("within_10_prior_accuracy"))]) + r" \\")
    write("tab_recall_tol.tex", "\n".join(lines) + "\n")
    # MNLI answers by gold label, pooled over the 13 unmodified models
    rows = readr("r8_mnli_by_gold.csv")
    lines = [" & ".join([r["condition"], r["gold"], f3(r["answered_entailment"]), f3(r["answered_contradiction"]),
                         f3(r["answered_neither"])]) + r" \\" for r in rows]
    write("tab_mnli_gold.tex", "\n".join(lines) + "\n")
    # rule variants with the standard prompt on the 105 comparisons the 2/20 variant keeps
    std105 = {r["system"]: r for r in readr("r7_rule_2_20_matched.csv")}
    rows = read("followups.csv")
    rows = sorted(rows, key=lambda r: (ORDER.index(split(r["system"])[0]), METHODS.index(split(r["system"])[1])))
    lines = []
    for r in rows:
        fam, meth, seed = split(r["system"])
        if meth not in ("unmodified", "attribution-guided LoRA") or seed not in (None, "seed 42", 42):
            continue
        name = fam + ("" if meth == "unmodified" else ", attr.-guided")
        star = lambda k: "$^{*}$" if r.get(f"{k}_p_holm") not in (None, "") and float(r[f"{k}_p_holm"]) < 0.05 else ""
        lines.append(" & ".join([name, f3(r["standard_equal_share"]), f3(r.get("norule_equal_share")) + star("norule"),
                                 f3(std105.get(r["system"], {}).get("standard_equal_share_same_items")),
                                 f3(r.get("rule_2_20_equal_share")),
                                 f3(r.get("rule_reversed_equal_share")) + star("rule_reversed"),
                                 f3(r.get("rule_first_equal_share")) + star("rule_first"),
                                 f3(r.get("fewshot_equal_share"))]) + r" \\")
    write("tab_rules.tex", "\n".join(lines) + "\n")


def main():
    for fn in (tab_overview, tab_followups, tab_finetune, tab_methods, tab_altered, tab_repro,
               tab_all_systems, tab_altered_full, tab_neutral, tab_finetune_compact, tab_methods_compact, tab_gap, tab_baselines, tab_rules, tab_logprobs, tab_round3, tab_review):
        fn()
    print("tables ->", DEST, sorted(p.name for p in DEST.iterdir()))


if __name__ == "__main__":
    main()
