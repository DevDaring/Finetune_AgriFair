"""Round-3 analysis (Future_PLan.md "Round 3"; hypotheses H11, H12, H14, H15, H16).

    python -m Submission1_TMLR.analyse_round3

Inputs : gpu_round3_predictions.jsonl, bedrock_round3_predictions.jsonl (or their .gz copies), plus the
         main and follow-up outputs for the paired contrasts.
Outputs: results_submission1_tmlr/analysis_round3/*.csv and summary_round3.json.
Statistics as in the plan: cluster bootstrap (10,000 draws), McNemar where each cluster is one binary
pair, cluster-level sign-flip otherwise, Holm within each family, Kendall's tau with a cluster bootstrap.
"""
from __future__ import annotations

import collections
import itertools
import json
import re
from typing import Dict, List

import numpy as np

from Submission1_Code_Phase2 import common as C
from Submission1_TMLR import analyse as A

OUT = A.OUT
AN = OUT / "analysis_round3"
EQ = "roughly equal"


def load() -> List[Dict]:
    """One row per (prompt, system) across files. The renamed items come only from their own files: the
    Bedrock round-3 repair pass answered them a second time (see api_determinism)."""
    rows, seen = [], set()
    for name in ("gpu_round3_predictions.jsonl", "bedrock_renamed_predictions.jsonl", "bedrock_round3_predictions.jsonl"):
        found = A.rows_of(OUT / name)
        for r in found or []:
            if name == "bedrock_round3_predictions.jsonl" and r.get("experiment") == "renamed":
                continue
            k = (r["prompt_id"], r["system"])
            if k not in seen and not r.get("error"):
                seen.add(k); rows.append(r)
    return A.substitute_256(rows, ("gpu_retry256_round3_predictions.jsonl",))


def api_determinism() -> Dict:
    """Hosted models answered the renamed prompts twice at temperature 0 (their own stream, and the round-3
    repair pass). Share of identical chosen options per model, over the prompts answered both times."""
    first = {(r["prompt_id"], r["system"]): r for r in (A.rows_of(OUT / "bedrock_renamed_predictions.jsonl") or []) if not r.get("error")}
    second = {(r["prompt_id"], r["system"]): r for r in (A.rows_of(OUT / "bedrock_round3_predictions.jsonl") or [])
              if r.get("experiment") == "renamed" and not r.get("error")}
    both = sorted(set(first) & set(second))
    out = {}
    for s in sorted({k[1] for k in both}):
        ks = [k for k in both if k[1] == s]
        out[A.label(s)] = {"pairs": len(ks), "same_choice": round(float(np.mean([first[k].get("picked_choice") == second[k].get("picked_choice") for k in ks])), 4),
                           "same_text": round(float(np.mean([first[k].get("raw_output") == second[k].get("raw_output") for k in ks])), 4)}
    return out


def unmod(sid: str) -> bool:
    return A.method(sid) in ("frozen_base", "unmodified")


def tau_with_ci(acc_a: Dict[str, Dict], acc_b: Dict[str, Dict], draws: int = 2000):
    """acc_x: system -> {cluster: mean outcome}. Kendall tau of system means, CI by resampling clusters
    (the same draw of clusters for both measures when they share clusters)."""
    systems = sorted(set(acc_a) & set(acc_b))
    if not systems:
        return None
    ua = set().union(*(acc_a[s] for s in systems)); ub = set().union(*(acc_b[s] for s in systems))
    systems = [s for s in systems if set(acc_a[s]) >= ua and set(acc_b[s]) >= ub]   # complete cases only
    if len(systems) < 3:
        return None
    ca, cb = sorted(ua), sorted(ub)
    Ma = np.array([[acc_a[s][c] for c in ca] for s in systems]); Mb = np.array([[acc_b[s][c] for c in cb] for s in systems])

    def tau(x, y):
        conc = disc = 0
        for i, j in itertools.combinations(range(len(x)), 2):
            sg = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); conc += sg > 0; disc += sg < 0
        return (conc - disc) / (len(x) * (len(x) - 1) / 2), int(disc)
    t, d = tau(Ma.mean(1), Mb.mean(1))
    rng = np.random.default_rng(A.SEED); boots = []
    shared = ca == cb and Ma.shape == Mb.shape
    for _ in range(draws):
        ia = rng.integers(0, len(ca), len(ca)); ib = ia if shared else rng.integers(0, len(cb), len(cb))
        boots.append(tau(Ma[:, ia].mean(1), Mb[:, ib].mean(1))[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_systems": len(systems), "kendall_tau": round(t, 4), "ci_lower": round(float(lo), 4),
            "ci_upper": round(float(hi), 4), "discordant_pairs": d, "total_pairs": len(systems) * (len(systems) - 1) // 2}


def by_cluster(rows, key="comparison_id", val=lambda r: float(bool(r["correct"]))):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        acc[r["system"]][r[key]].append(val(r))
    return {s: {c: float(np.mean(v)) for c, v in d.items()} for s, d in acc.items()}


# ---------------------------------------------------------------- P3: drifted questions and the leaderboard
def p3(rows, main_rows) -> Dict:
    drift = [r for r in rows if r["experiment"] == "drifted"]
    if not drift:
        return {}
    ids = {r["comparison_id"] for r in drift}
    corr = [r for r in main_rows if r["experiment"] == "verified" and r["comparison_id"] in ids]
    norule = [r for r in main_rows if r["experiment"] == "norule" and r["comparison_id"] in ids]
    a_d, a_c, a_n = by_cluster(drift), by_cluster(corr), by_cluster(norule)
    table = []
    systems = sorted(set(a_d) & set(a_c))
    rank = lambda acc: {s: i + 1 for i, s in enumerate(sorted(systems, key=lambda s: -np.mean(list(acc[s].values()))))}
    rd, rc = rank(a_d), rank(a_c)
    for s in systems:
        table.append({"system": A.label(s), "system_id": s,
                      "accuracy_drifted_v1": round(float(np.mean(list(a_d[s].values()))), 4),
                      "accuracy_corrected": round(float(np.mean(list(a_c[s].values()))), 4),
                      "accuracy_norule": round(float(np.mean(list(a_n[s].values()))), 4) if s in a_n else None,
                      "rank_drifted": rd[s], "rank_corrected": rc[s],
                      "equal_share_drifted": round(float(np.mean([A.is_equal(r) for r in drift if r["system"] == s])), 4)})
    C.write_csv(AN / "p3_drifted_leaderboard.csv", table, columns=list(table[0]))
    top = sorted(systems, key=lambda s: rc[s])[:5]
    return {"h12_drifted_vs_corrected": tau_with_ci(a_d, a_c),
            "drifted_vs_norule": tau_with_ci(a_d, {s: v for s, v in a_n.items() if s in a_d}) if a_n else None,
            "top5_corrected_and_their_drifted_rank": {A.label(s): {"corrected": rc[s], "drifted": rd[s]} for s in top}}


# ---------------------------------------------------------------- P1: MNLI instruction effect
MNLI_LETTERS = {"a": "entailment", "b": "contradiction", "c": "neither"}


def p1(rows) -> Dict:
    m = [r for r in rows if r["experiment"].startswith("mnli_")]
    if not m:
        return {}
    neither = lambda r: str(r.get("picked_choice") or "") == "neither"

    def lenient(r):
        """Secondary reading (labelled as such): a reply that is a bare option letter counts as that option."""
        if r.get("parse_ok"):
            return r.get("picked_choice")
        raw = (r.get("raw_output") or "").strip().strip("().").lower()
        return MNLI_LETTERS.get(raw)
    out, contrasts = [], []
    for s in sorted({r["system"] for r in m}):
        rec = {"system": A.label(s), "system_id": s, "unmodified": unmod(s)}
        cond = {}
        for c in ("mnli_plain", "mnli_strict", "mnli_lenient"):
            v = {r["item_id"]: r for r in m if r["system"] == s and r["experiment"] == c}
            cond[c] = v
            if v:
                rec[f"{c}_accuracy"] = round(float(np.mean([bool(r["correct"]) for r in v.values()])), 4)
                rec[f"{c}_neither_share"] = round(float(np.mean([neither(r) for r in v.values()])), 4)
                rec[f"{c}_parse_rate"] = round(float(np.mean([bool(r["parse_ok"]) for r in v.values()])), 4)
                rec[f"{c}_neither_share_lenient"] = round(float(np.mean([lenient(r) == "neither" for r in v.values()])), 4)
                rec[f"{c}_accuracy_lenient"] = round(float(np.mean([str(lenient(r) or "").lower() == r["gold_choice_text"].lower() for r in v.values()])), 4)
        for c in ("mnli_strict", "mnli_lenient"):
            if cond.get(c) and cond.get("mnli_plain"):
                a = {k: float(neither(r)) for k, r in cond[c].items()}
                b = {k: float(neither(r)) for k, r in cond["mnli_plain"].items()}
                res = A.paired(a, b)
                rec[f"{c}_minus_plain_neither"] = res["difference"]; rec[f"{c}_ci"] = f"[{res['ci_lower']}, {res['ci_upper']}]"
                rec[f"{c}_p"] = res["p_mcnemar"]
        out.append(rec)
    for c in ("mnli_strict", "mnli_lenient"):
        fam = [r for r in out if r["unmodified"] and f"{c}_p" in r]
        for r, adj in zip(fam, A.holm([r[f"{c}_p"] for r in fam])):
            r[f"{c}_p_holm"] = round(adj, 6)
        fam = [r for r in out if not r["unmodified"] and f"{c}_p" in r]
        for r, adj in zip(fam, A.holm([r[f"{c}_p"] for r in fam])):
            r[f"{c}_p_holm"] = round(adj, 6)
    C.write_csv(AN / "p1_mnli.csv", out, columns=list(dict.fromkeys(k for r in out for k in r)))
    acc = {c: by_cluster([r for r in m if r["experiment"] == c], key="item_id") for c in ("mnli_plain", "mnli_strict", "mnli_lenient")}
    um = lambda d: {s: v for s, v in d.items() if unmod(s)}
    # fine-tuning effect on "neither" (plain prompt), adapter minus its unmodified base, Holm across adapters
    ft = []
    plain = {s: {r["item_id"]: float(neither(r)) for r in m if r["system"] == s and r["experiment"] == "mnli_plain"}
             for s in {r["system"] for r in m}}
    for s, v in sorted(plain.items()):                     # sorted: set order differs between runs
        if s.startswith("bedrock") or unmod(s):
            continue
        base = f"{A.family(s)}|frozen_base|seed42"
        if base in plain:
            ft.append({"system": A.label(s), **A.paired(v, plain[base])})
    for r, adj in zip(ft, A.holm([r["p_mcnemar"] for r in ft])):
        r["p_holm"] = round(adj, 6)
    if ft:
        C.write_csv(AN / "p1_mnli_finetune_neither.csv", ft, columns=list(ft[0]))
    n_up = sum(1 for r in out if r["unmodified"] and r.get("mnli_strict_p_holm", 1) < 0.05 and r["mnli_strict_minus_plain_neither"] > 0)
    return {"h11_unmodified_with_more_neither_under_strict": n_up,
            "h11_unmodified_systems": sum(r["unmodified"] for r in out),
            "tau_plain_vs_strict_all": tau_with_ci(acc["mnli_plain"], acc["mnli_strict"]),
            "tau_plain_vs_strict_unmodified": tau_with_ci(um(acc["mnli_plain"]), um(acc["mnli_strict"])),
            "tau_plain_vs_lenient_all": tau_with_ci(acc["mnli_plain"], acc["mnli_lenient"]),
            "adapters_with_more_neither_than_base_plain": sum(1 for r in ft if r["p_holm"] < 0.05 and r["difference"] > 0),
            "adapters_with_less_neither_than_base_plain": sum(1 for r in ft if r["p_holm"] < 0.05 and r["difference"] < 0),
            "adapters": len(ft)}


# ---------------------------------------------------------------- P5: prior-adversarial census set
def p5(rows) -> Dict:
    a = [r for r in rows if r["experiment"] == "adversarial"]
    if not a:
        return {}
    out = []
    for s in sorted({r["system"] for r in a}):
        v = [r for r in a if r["system"] == s]
        acc = {}; pri = collections.defaultdict(list)
        for r in v:
            acc.setdefault(r["comparison_id"], []).append(float(bool(r["correct"])))
            pri[r["comparison_id"]].append(float(r.get("picked_choice") == r["prior_answer"]))
        d = np.array([np.mean(x) for x in acc.values()]); lo, hi = A.boot_ci(d)
        out.append({"system": A.label(s), "system_id": s, "n_prompts": len(v),
                    "accuracy": round(float(d.mean()), 4), "ci": f"[{lo:.3f}, {hi:.3f}]",
                    "share_prior_answer": round(float(np.mean([x for l in pri.values() for x in l])), 4),
                    "share_roughly_equal": round(float(np.mean([A.is_equal(r) for r in v])), 4),
                    "prior_beats_correct": bool(np.mean([x for l in pri.values() for x in l]) > d.mean())})
    C.write_csv(AN / "p5_adversarial.csv", out, columns=list(out[0]))
    return {"h14_systems_choosing_prior_more_than_correct": sum(r["prior_beats_correct"] for r in out), "systems": len(out),
            "best": max(out, key=lambda r: r["accuracy"])["system"], "best_accuracy": max(r["accuracy"] for r in out)}


# ---------------------------------------------------------------- P2: WDI replication
def p2(rows) -> Dict:
    w = [r for r in rows if r["experiment"] in ("wdi_standard", "wdi_norule", "wdi_numerical")]
    if not w:
        return {}
    out = []
    for s in sorted({r["system"] for r in w}):
        st = [r for r in w if r["system"] == s and r["experiment"] == "wdi_standard"]
        nr = [r for r in w if r["system"] == s and r["experiment"] == "wdi_norule"]
        nu = [r for r in w if r["system"] == s and r["experiment"] == "wdi_numerical"]
        rec = {"system": A.label(s), "system_id": s}
        if st:
            eqs = float(np.mean([A.is_equal(r) for r in st]))
            rec.update({"accuracy_memory": round(float(np.mean([bool(r["correct"]) for r in st])), 4),
                        "accuracy_difference_items": round(float(np.mean([bool(r["correct"]) for r in st if r["condition"] == "diff"])), 4),
                        "accuracy_equal_items": round(float(np.mean([bool(r["correct"]) for r in st if r["condition"] == "equal"])), 4),
                        "equal_share": round(eqs, 4), "defaults_to_equal": eqs >= 0.8,
                        "share_prior_answer": round(float(np.mean([r.get("picked_choice") == r.get("prior_answer") for r in st])), 4),
                        "parse_rate": round(float(np.mean([bool(r["parse_ok"]) for r in st])), 4)})
            a = {(r["comparison_id"], r["wording"]): float(r["wording"] == "wording_a" and bool(r["correct"])) for r in st}
            wa = {r["comparison_id"]: float(bool(r["correct"])) for r in st if r["wording"] == "wording_a"}
            wb = {r["comparison_id"]: float(bool(r["correct"])) for r in st if r["wording"] == "wording_b"}
            res = A.paired(wa, wb); rec.update({"wording_a_minus_b": res["difference"], "wording_p": res["p_mcnemar"]})
        if st and nr:
            a = {(r["comparison_id"], r["wording"]): float(A.is_equal(r)) for r in nr}
            b = {(r["comparison_id"], r["wording"]): float(A.is_equal(r)) for r in st}
            res = A.paired_cluster(a, b, A.first)
            rec.update({"norule_equal_share": round(float(np.mean(list(a.values()))), 4),
                        "norule_accuracy": round(float(np.mean([bool(r["correct"]) for r in nr])), 4),
                        "norule_minus_standard_equal": res["difference"], "norule_ci": f"[{res['ci_lower']}, {res['ci_upper']}]",
                        "norule_p": res["p_signflip"]})
        if nu:
            g = collections.defaultdict(list)
            for r in nu:
                g[(r["bundle_id"], r["wording"])].append(bool(r["correct"]))
            rec.update({"accuracy_supplied_numbers": round(float(np.mean([bool(r["correct"]) for r in nu])), 4),
                        "fully_correct_scenarios": round(float(np.mean([len(v) == 3 and all(v) for v in g.values()])), 4)})
        out.append(rec)
    for key in ("norule_p", "wording_p"):
        fam = [r for r in out if key in r]
        for r, adj in zip(fam, A.holm([r[key] for r in fam])):
            r[key + "_holm"] = round(adj, 6)
    C.write_csv(AN / "p2_wdi.csv", out, columns=list(dict.fromkeys(k for r in out for k in r)))
    mem = by_cluster([r for r in w if r["experiment"] == "wdi_standard"])
    num = by_cluster([r for r in w if r["experiment"] == "wdi_numerical"], key="bundle_id")
    defaulting = [r for r in out if r.get("defaults_to_equal")]
    base = json.loads((OUT / "wdi" / "prior_answers.json").read_text())["accuracy"]
    return {"systems": len(out), "h15a_defaulting_systems": [r["system"] for r in defaulting],
            "h15b_defaulting_with_lower_equal_share_without_rule": sum(1 for r in defaulting if r.get("norule_p_holm", 1) < 0.05
                                                                      and r["norule_minus_standard_equal"] < 0),
            "h15c_tau_memory_vs_supplied": tau_with_ci(mem, num),
            "wording_effects_holm": sum(1 for r in out if r.get("wording_p_holm", 1) < 0.05),
            "baselines": base, "systems_above_regional_prior": sum(1 for r in out if r.get("accuracy_memory", 0) > base["regional_prior"])}


# ---------------------------------------------------------------- P6: numeric recall
NUM = re.compile(r'"share_([12])_percent"\s*:\s*"?\s*(-?\d+(?:\.\d+)?)')


def implied(s1: float, s2: float, e1: str, e2: str):
    gap = abs(s1 - s2)
    return "Roughly equal" if gap < 5 else ((e1 if s1 > s2 else e2) if gap >= 10 else None)


def p6(rows, main_rows) -> Dict:
    rc = [r for r in rows if r["experiment"] == "recall"]
    if not rc:
        return {}
    mc = collections.defaultdict(dict)
    for r in main_rows:
        if r["experiment"] in ("verified", "extended"):
            mc[r["system"]].setdefault(r["comparison_id"], []).append(r)
    out = []
    for s in sorted({r["system"] for r in rc}):
        v = [r for r in rc if r["system"] == s]
        errs, imp_ok, mc_ok, eq_imp = [], {}, {}, []
        parsed = 0
        for r in v:
            got = dict(NUM.findall(r.get("raw_output") or ""))
            if "1" not in got or "2" not in got:
                continue
            parsed += 1
            s1, s2 = float(got["1"]), float(got["2"])
            errs += [abs(s1 - r["share1_pct"]), abs(s2 - r["share2_pct"])]
            lab = implied(s1, s2, r["entity1"], r["entity2"])
            imp_ok[r["comparison_id"]] = float(lab is not None and lab.lower() == r["gold_choice_text"].lower())
            m = mc.get(s, {}).get(r["comparison_id"])
            if m:
                mc_ok[r["comparison_id"]] = float(np.mean([bool(x["correct"]) for x in m]))
                if r["condition"] == "diff":
                    eq_imp += [imp_ok[r["comparison_id"]] for x in m if A.is_equal(x)]
        res = A.paired(imp_ok, mc_ok) if mc_ok else {}
        out.append({"system": A.label(s), "system_id": s, "parse_rate": round(parsed / len(v), 4),
                    "median_abs_error_pp": round(float(np.median(errs)), 2) if errs else None,
                    "implied_accuracy": round(float(np.mean(list(imp_ok.values()))), 4) if imp_ok else None,
                    "mc_accuracy_same_items": round(float(np.mean([mc_ok[k] for k in imp_ok if k in mc_ok])), 4) if mc_ok else None,
                    "implied_minus_mc": res.get("difference"), "ci": f"[{res.get('ci_lower')}, {res.get('ci_upper')}]" if res else None,
                    "p": res.get("p_mcnemar"),
                    "implied_correct_when_mc_said_equal_on_diff": round(float(np.mean(eq_imp)), 4) if eq_imp else None})
    fam = [r for r in out if r["p"] is not None]
    for r, adj in zip(fam, A.holm([r["p"] for r in fam])):
        r["p_holm"] = round(adj, 6)
    C.write_csv(AN / "p6_recall.csv", out, columns=list(dict.fromkeys(k for r in out for k in r)))
    return {"systems": len(out)}


def p7(rows, main_rows) -> Dict:
    """H17: entity names. Renamed vs original prompt on the same comparison and wording; accuracy and the
    share of 'roughly equal'; cluster = comparison; Holm across systems; defaulting status on the 154
    with the renamed items swapped in."""
    rn = [r for r in rows if r["experiment"] == "renamed"]
    if not rn:
        return {}
    orig = {(r["system"], r["comparison_id"], r["wording"]): r for r in main_rows + rows
            if r["experiment"] in ("verified", "extended", "adversarial")}
    std154 = collections.defaultdict(dict)
    for r in main_rows:
        if r["experiment"] in ("verified", "extended"):
            std154[r["system"]][(r["comparison_id"], r["wording"])] = A.is_equal(r)
    out = []
    for s in sorted({r["system"] for r in rn}):
        v = [r for r in rn if r["system"] == s and (s, r["comparison_id"], r["wording"]) in orig]
        if not v:
            continue
        a_acc = {(r["comparison_id"], r["wording"]): float(bool(r["correct"])) for r in v}
        b_acc = {(r["comparison_id"], r["wording"]): float(bool(orig[(s, r["comparison_id"], r["wording"])]["correct"])) for r in v}
        a_eq = {(r["comparison_id"], r["wording"]): float(A.is_equal(r)) for r in v}
        b_eq = {(r["comparison_id"], r["wording"]): float(A.is_equal(orig[(s, r["comparison_id"], r["wording"])])) for r in v}
        ra, re_ = A.paired_cluster(a_acc, b_acc, A.first), A.paired_cluster(a_eq, b_eq, A.first)
        swapped = dict(std154.get(s, {}))
        for r in v:
            if r["origin"] in ("verified", "extended"):
                swapped[(r["comparison_id"], r["wording"])] = A.is_equal(r)
        old_share = float(np.mean(list(std154[s].values()))) if std154.get(s) else None
        new_share = float(np.mean(list(swapped.values()))) if swapped else None
        out.append({"system": A.label(s), "system_id": s, "n_prompts": len(v),
                    "renamed_minus_original_accuracy": ra["difference"], "acc_ci": f"[{ra['ci_lower']}, {ra['ci_upper']}]", "acc_p": ra["p_signflip"],
                    "renamed_minus_original_equal_share": re_["difference"], "eq_ci": f"[{re_['ci_lower']}, {re_['ci_upper']}]", "eq_p": re_["p_signflip"],
                    "equal_share_154_original": round(old_share, 4) if old_share is not None else None,
                    "equal_share_154_with_renamed": round(new_share, 4) if new_share is not None else None,
                    "default_status_changes": (old_share is not None and new_share is not None and (old_share >= 0.8) != (new_share >= 0.8))})
    for key in ("acc_p", "eq_p"):
        for r, adj in zip(out, A.holm([r[key] for r in out])):
            r[key + "_holm"] = round(adj, 6)
    C.write_csv(AN / "p7_renamed.csv", out, columns=list(out[0]))
    return {"systems": len(out),
            "holm_significant_accuracy_changes": sum(r["acc_p_holm"] < 0.05 for r in out),
            "holm_significant_equal_share_changes": sum(r["eq_p_holm"] < 0.05 for r in out),
            "default_status_changes": [r["system"] for r in out if r["default_status_changes"]]}


def budget_256(main_rows) -> Dict:
    """Budget sensitivity (logged in the plan): for the GPU systems re-run at 256 tokens, recompute P6 and
    P7 from the 256-token outputs and write them beside the primary 24/40-token results."""
    retry = A.rows_of(OUT / "gpu_retry256_round3_predictions.jsonl")
    if not retry:
        return {}
    retry = [r for r in retry if not r.get("error")]
    # compare like with like: where a system's main-set answers were also re-run at 256 tokens (round 1),
    # use those as its baseline instead of the 24-token answers
    r1 = [r for r in (A.rows_of(OUT / "gpu_retry256_predictions.jsonl") or []) if not r.get("error")]
    r1_sys = {r["system"] for r in r1}
    main_rows = [r for r in main_rows if r["system"] not in r1_sys] + r1
    out = {"baseline_256_for": sorted(A.label(x) for x in r1_sys)}
    global AN
    saved = AN
    try:
        AN = AN / "budget256"; AN.mkdir(parents=True, exist_ok=True)
        out["p6"] = p6(retry, main_rows)
        out["p7"] = p7(retry, main_rows)
        out["systems"] = sorted({A.label(r["system"]) for r in retry})
    finally:
        AN = saved
    return out


def main() -> None:
    AN.mkdir(parents=True, exist_ok=True)
    rows = load()
    if not rows:
        raise SystemExit("no round-3 predictions found")
    main_rows = A.load()
    summary = {"n_rows": len(rows), "systems": len({r["system"] for r in rows}),
               "by_experiment": dict(collections.Counter(r["experiment"] for r in rows)),
               "p3": p3(rows, main_rows), "p1": p1(rows), "p5": p5(rows), "p2": p2(rows), "p6": p6(rows, main_rows),
               "p7": p7(rows, main_rows), "api_determinism": api_determinism(),
               "budget256": budget_256(main_rows)}
    C.write_json(AN / "summary_round3.json", summary)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
