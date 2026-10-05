"""Analyses added in revision (5 Oct 2026), all computed from existing outputs; no new model runs.

    python -m Submission1_TMLR.analyse_review

R1  drift isolated from the rule: the 34 comparisons as drifted (no rule), corrected without the rule and
    corrected with the rule; Kendall's tau between each pair, with cluster-bootstrap intervals.
R2  numeric recall within a tolerance: the share of comparisons whose two recalled shares are both within
    5 or 10 points, and multiple-choice accuracy, implied accuracy and the prior on that subset.
R3  the matched evidence comparison: the same 154 comparisons without and with the real census table.
R4  strict against lenient answer reading on the 154 comparisons.
R5  hosted repeatability: accuracy of the two answers to the 296 renamed prompts, and their rankings.
R6  placement: mean of three attribution-guided (and plain) adapters minus mean of three random draws.
R7  the standard prompt on the 105 comparisons that the 2/20 rule variant keeps.
R8  MNLI answer distribution by gold label, unmodified models.
Outputs: results_submission1_tmlr/analysis_review/*.csv and summary_review.json.
"""
from __future__ import annotations

import collections
import json
import re

import numpy as np

from Submission1_Code_Phase2 import common as C
from Submission1_TMLR import analyse as A
from Submission1_TMLR import analyse_round3 as R3

OUT = A.OUT
AN = OUT / "analysis_review"
DRAWS = 10000


def unmod(s):
    return A.method(s) in ("frozen_base", "unmodified")


def mean_acc(d):
    return float(np.mean(list(d.values())))


def boot_diff(a: dict, b: dict, cluster=lambda k: k):
    """Mean of a minus mean of b over shared keys; interval by resampling clusters."""
    keys = sorted(set(a) & set(b))
    cl = collections.defaultdict(list)
    for k in keys:
        cl[cluster(k)].append(a[k] - b[k])
    d = np.array([np.mean(v) for v in cl.values()])
    rng = np.random.default_rng(A.SEED)
    bs = d[rng.integers(0, len(d), (DRAWS, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return round(float(d.mean()), 4), round(float(lo), 4), round(float(hi), 4)


# ---------------------------------------------------------------- R1
def r1(main_rows, r3_rows):
    drift = [r for r in r3_rows if r["experiment"] == "drifted"]
    ids = {r["comparison_id"] for r in drift}
    corr = [r for r in main_rows if r["experiment"] == "verified" and r["comparison_id"] in ids]
    norule = [r for r in main_rows if r["experiment"] == "norule" and r["comparison_id"] in ids]
    a = {"drifted_no_rule": R3.by_cluster(drift), "corrected_no_rule": R3.by_cluster(norule),
         "corrected_with_rule": R3.by_cluster(corr)}
    common = set(a["drifted_no_rule"]) & set(a["corrected_no_rule"]) & set(a["corrected_with_rule"])
    out = {"n_comparisons": len(ids), "n_systems_all_three": len(common)}
    for scope, keep in (("all", lambda s: s in common), ("unmodified", lambda s: s in common and unmod(s))):
        sub = {k: {s: v for s, v in d.items() if keep(s)} for k, d in a.items()}
        out[scope] = {
            "drift_only (drifted vs corrected, both without rule)": R3.tau_with_ci(sub["drifted_no_rule"], sub["corrected_no_rule"]),
            "rule_only (corrected without vs with rule)": R3.tau_with_ci(sub["corrected_no_rule"], sub["corrected_with_rule"]),
            "both (drifted without rule vs corrected with rule)": R3.tau_with_ci(sub["drifted_no_rule"], sub["corrected_with_rule"]),
            "n_systems": len(sub["drifted_no_rule"])}
    rows = []
    for s in sorted(common):
        rows.append({"system": A.label(s), "system_id": s,
                     **{k: round(mean_acc(d[s]), 4) for k, d in a.items()}})
    for key in ("drifted_no_rule", "corrected_no_rule", "corrected_with_rule"):
        order = sorted(rows, key=lambda r: -r[key])
        for i, r in enumerate(order):
            r[f"rank_{key}"] = i + 1
    C.write_csv(AN / "r1_drift_isolated.csv", rows, columns=list(rows[0]))
    diffs = [r["drifted_no_rule"] - r["corrected_no_rule"] for r in rows]
    out["mean_accuracy_change_drifted_minus_corrected_no_rule"] = round(float(np.mean(diffs)), 4)
    out["systems_lower_on_drifted"] = sum(d < 0 for d in diffs)
    out["largest_rank_moves_drift_only"] = sorted(
        [(r["system"], r["rank_corrected_no_rule"], r["rank_drifted_no_rule"]) for r in rows],
        key=lambda x: -abs(x[1] - x[2]))[:6]
    return out


# ---------------------------------------------------------------- R2
def r2(main_rows, r3_rows):
    prior = json.loads((OUT / "analysis" / "baseline_cross_state_prior_answers.json").read_text())
    rc = [r for r in r3_rows if r["experiment"] == "recall"]
    mc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in main_rows:
        if r["experiment"] in ("verified", "extended"):
            mc[r["system"]][r["comparison_id"]].append(float(bool(r["correct"])))
    out = []
    for s in sorted({r["system"] for r in rc}):
        rec = {"system": A.label(s), "system_id": s, "unmodified": unmod(s)}
        items = []
        for r in (x for x in rc if x["system"] == s):
            got = dict(R3.NUM.findall(r.get("raw_output") or ""))
            if "1" not in got or "2" not in got:
                continue
            s1, s2 = float(got["1"]), float(got["2"])
            e = max(abs(s1 - r["share1_pct"]), abs(s2 - r["share2_pct"]))
            lab = R3.implied(s1, s2, r["entity1"], r["entity2"])
            items.append({"cid": r["comparison_id"], "err": e,
                          "implied": float(lab is not None and lab.lower() == r["gold_choice_text"].lower()),
                          "mc": float(np.mean(mc[s][r["comparison_id"]])) if mc[s].get(r["comparison_id"]) else None,
                          "prior": float(prior.get(r["comparison_id"], "").lower() == r["gold_choice_text"].lower())})
        rec["parsed"] = len(items)
        for tol in (5, 10):
            sub = [i for i in items if i["err"] <= tol and i["mc"] is not None]
            rec[f"within_{tol}_share"] = round(len(sub) / 154, 4)
            rec[f"within_{tol}_n"] = len(sub)
            if sub:
                rec[f"within_{tol}_mc_accuracy"] = round(float(np.mean([i["mc"] for i in sub])), 4)
                rec[f"within_{tol}_implied_accuracy"] = round(float(np.mean([i["implied"] for i in sub])), 4)
                rec[f"within_{tol}_prior_accuracy"] = round(float(np.mean([i["prior"] for i in sub])), 4)
        allp = [i for i in items if i["mc"] is not None]
        if allp:
            rec["implied_accuracy"] = round(float(np.mean([i["implied"] for i in allp])), 4)
            rec["mc_accuracy"] = round(float(np.mean([i["mc"] for i in allp])), 4)
            rec["prior_accuracy_same_items"] = round(float(np.mean([i["prior"] for i in allp])), 4)
            d, lo, hi = boot_diff({i["cid"]: i["implied"] for i in allp}, {i["cid"]: i["mc"] for i in allp})
            rec.update({"implied_minus_mc": d, "implied_minus_mc_ci": f"[{lo}, {hi}]"})
            d, lo, hi = boot_diff({i["cid"]: i["implied"] for i in allp}, {i["cid"]: i["prior"] for i in allp})
            rec.update({"implied_minus_prior": d, "implied_minus_prior_ci": f"[{lo}, {hi}]"})
        out.append(rec)
    C.write_csv(AN / "r2_recall_tolerance.csv", out, columns=list(dict.fromkeys(k for r in out for k in r)))
    return {r["system"]: {k: r.get(k) for k in ("parsed", "within_10_n", "within_10_mc_accuracy",
                                                   "within_10_implied_accuracy", "implied_accuracy", "mc_accuracy",
                                                   "prior_accuracy_same_items", "implied_minus_mc_ci", "implied_minus_prior",
                                                   "implied_minus_prior_ci")}
            for r in out if r["unmodified"]}


# ---------------------------------------------------------------- R3
def r3(main_rows):
    std = [r for r in main_rows if r["experiment"] in ("verified", "extended")]
    rt = [r for r in main_rows if r["experiment"] == "realtable"]
    num = [r for r in main_rows if r["experiment"] == "numerical"]
    has_rt = {r["system"] for r in rt}
    out = {}

    def bal(rows):
        """class-balanced accuracy per comparison cluster: weight equal and difference items equally."""
        by = R3.by_cluster(rows)
        cond = {r["comparison_id"]: r["condition"] for r in rows}
        n_eq = sum(1 for c in set(cond) if cond[c] == "equal"); n_df = len(set(cond)) - n_eq
        w = {c: (0.5 / n_eq if cond[c] == "equal" else 0.5 / n_df) * len(set(cond)) for c in cond}
        return {s: {c: v * w[c] for c, v in d.items()} for s, d in by.items()}

    for scope, keep in (("all", lambda s: s in has_rt), ("unmodified", lambda s: s in has_rt and unmod(s))):
        f = lambda rows: [r for r in rows if keep(r["system"])]
        out[scope] = {
            "closed_book_vs_real_table": R3.tau_with_ci(R3.by_cluster(f(std)), R3.by_cluster(f(rt))),
            "closed_book_vs_real_table_balanced": R3.tau_with_ci(bal(f(std)), bal(f(rt))),
            "closed_book_vs_numerical_set": R3.tau_with_ci(R3.by_cluster(f(std)), R3.by_cluster(f(num), key="bundle_id")),
            "real_table_vs_numerical_set": R3.tau_with_ci(R3.by_cluster(f(rt)), R3.by_cluster(f(num), key="bundle_id"))}
    allsys = {s for s in {r["system"] for r in std}}
    um = lambda rows: [r for r in rows if unmod(r["system"])]
    out["unmodified_all13_closed_book_vs_numerical"] = R3.tau_with_ci(R3.by_cluster(um(std)), R3.by_cluster(um(num), key="bundle_id"))
    rows = []
    for s in sorted(has_rt):
        a = [r for r in std if r["system"] == s]; b = [r for r in rt if r["system"] == s]
        acc = lambda v, c=None: round(float(np.mean([bool(r["correct"]) for r in v if c is None or r["condition"] == c])), 4)
        ka = {(r["comparison_id"], r["wording"]): float(bool(r["correct"])) for r in b}
        kb = {(r["comparison_id"], r["wording"]): float(bool(r["correct"])) for r in a}
        d, lo, hi = boot_diff(ka, kb, A.first)
        rows.append({"system": A.label(s), "system_id": s, "unmodified": unmod(s),
                     "closed_book": acc(a), "closed_book_diff_items": acc(a, "diff"), "closed_book_equal_items": acc(a, "equal"),
                     "real_table": acc(b), "real_table_diff_items": acc(b, "diff"), "real_table_equal_items": acc(b, "equal"),
                     "real_table_equal_share": round(float(np.mean([A.is_equal(r) for r in b])), 4),
                     "real_table_minus_closed_book": d, "ci": f"[{lo}, {hi}]"})
    C.write_csv(AN / "r3_real_table_matched.csv", rows, columns=list(rows[0]))
    out["systems_real_table_ge_0.9"] = [r["system"] for r in rows if r["real_table"] >= 0.9]
    out["unmodified_still_default_with_table"] = [r["system"] for r in rows if r["unmodified"] and r["real_table_equal_share"] >= 0.8]
    out["systems_with_real_table"] = len(rows)
    return out


# ---------------------------------------------------------------- R4
KEYED = re.compile(r'"[^"]{1,40}"\s*:\s*"\(?([a-dA-D])\)?"')
BARE = re.compile(r'^\W*\(?([a-dA-D])\)?\W*$')


def lenient_pick(r):
    """Secondary reading: a parsed answer stands; otherwise one unambiguous letter under any JSON key, or a
    reply that is only a letter, or a reply that is exactly one option text."""
    if r.get("parse_ok"):
        return r.get("picked_choice"), "strict"
    raw = (r.get("raw_output") or "").strip()
    ls = {x.lower() for x in KEYED.findall(raw)}
    if len(ls) == 1:
        i = "abcd".index(ls.pop())
        return (r["choices"][i], "other_key") if i < len(r["choices"]) else (None, "none")
    m = BARE.match(raw)
    if m:
        i = "abcd".index(m.group(1).lower())
        return (r["choices"][i], "bare_letter") if i < len(r["choices"]) else (None, "none")
    hit = [c for c in r["choices"] if raw.strip(' ."\'').lower() == c.lower()]
    if len(hit) == 1:
        return hit[0], "option_text"
    return None, "none"


def r4(main_rows):
    std = [r for r in main_rows if r["experiment"] in ("verified", "extended")]
    num = [r for r in main_rows if r["experiment"] == "numerical"]
    out, strict_m, len_m, strict_n, len_n = [], {}, {}, {}, {}
    for s in sorted({r["system"] for r in std}):
        v = [r for r in std if r["system"] == s]
        picks = [lenient_pick(r) for r in v]
        ok = lambda p, r: p is not None and p.lower() == r["gold_choice_text"].lower()
        lacc = [float(ok(p, r)) for (p, _), r in zip(picks, v)]
        leq = [float(p is not None and A.EQUAL in p.lower()) for p, _ in picks]
        kinds = collections.Counter(k for _, k in picks)
        strict_m[s] = {(r["comparison_id"], r["wording"]): float(bool(r["correct"])) for r in v}
        len_m[s] = {(r["comparison_id"], r["wording"]): x for r, x in zip(v, lacc)}
        vn = [r for r in num if r["system"] == s]
        strict_n[s] = {r["prompt_id"]: float(bool(r["correct"])) for r in vn}
        len_n[s] = {r["prompt_id"]: float(ok(lenient_pick(r)[0], r)) for r in vn}
        out.append({"system": A.label(s), "system_id": s, "n": len(v),
                    "unparsed_strict": round(1 - float(np.mean([bool(r["parse_ok"]) for r in v])), 4),
                    "recovered_other_key": kinds["other_key"], "recovered_bare_letter": kinds["bare_letter"],
                    "recovered_option_text": kinds["option_text"], "still_unparsed": kinds["none"],
                    "accuracy_strict": round(float(np.mean([bool(r["correct"]) for r in v])), 4),
                    "accuracy_lenient": round(float(np.mean(lacc)), 4),
                    "equal_share_strict": round(float(np.mean([A.is_equal(r) for r in v])), 4),
                    "equal_share_lenient": round(float(np.mean(leq)), 4),
                    "accuracy_numerical_strict": round(float(np.mean(list(strict_n[s].values()))), 4),
                    "accuracy_numerical_lenient": round(float(np.mean(list(len_n[s].values()))), 4)})
    C.write_csv(AN / "r4_lenient_reading.csv", out, columns=list(out[0]))

    def tau(x, y):
        ss = sorted(x); a = np.array([np.mean(list(x[s].values())) for s in ss]); b = np.array([np.mean(list(y[s].values())) for s in ss])
        import itertools
        c = d = 0
        for i, j in itertools.combinations(range(len(ss)), 2):
            g = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]); c += g > 0; d += g < 0
        return round((c - d) / (len(ss) * (len(ss) - 1) / 2), 4)
    by_cl = lambda d: {s: {k[0]: v for k, v in x.items()} for s, x in d.items()}
    return {"systems_with_any_unparsed": sum(r["unparsed_strict"] > 0 for r in out),
            "default_status_changes": [r["system"] for r in out if (r["equal_share_strict"] >= 0.8) != (r["equal_share_lenient"] >= 0.8)],
            "defaulting_lenient": sum(r["equal_share_lenient"] >= 0.8 for r in out),
            "max_accuracy_gain": max((r["accuracy_lenient"] - r["accuracy_strict"], r["system"]) for r in out),
            "tau_memory_strict_vs_lenient": tau(strict_m, len_m),
            "tau_memory_vs_numbers_lenient": tau(len_m, len_n),
            "tau_memory_vs_numbers_strict_check": tau(strict_m, strict_n)}


# ---------------------------------------------------------------- R5
def r5():
    first = {(r["prompt_id"], r["system"]): r for r in (A.rows_of(OUT / "bedrock_renamed_predictions.jsonl") or []) if not r.get("error")}
    second = {(r["prompt_id"], r["system"]): r for r in (A.rows_of(OUT / "bedrock_round3_predictions.jsonl") or [])
              if r.get("experiment") == "renamed" and not r.get("error")}
    both = sorted(set(first) & set(second))
    out = []
    acc1, acc2 = {}, {}
    for s in sorted({k[1] for k in both}):
        ks = [k for k in both if k[1] == s]
        a = {(first[k]["comparison_id"], first[k]["wording"]): float(bool(first[k]["correct"])) for k in ks}
        b = {(second[k]["comparison_id"], second[k]["wording"]): float(bool(second[k]["correct"])) for k in ks}
        acc1[s], acc2[s] = R3.by_cluster([first[k] for k in ks]), R3.by_cluster([second[k] for k in ks])
        d, lo, hi = boot_diff(a, b, A.first)
        same_hash = np.mean([first[k].get("prompt_sha256") == second[k].get("prompt_sha256") for k in ks])
        same_model = np.mean([first[k].get("bedrock_model_id") == second[k].get("bedrock_model_id") for k in ks])
        same_budget = np.mean([first[k].get("max_new_tokens") == second[k].get("max_new_tokens") for k in ks])
        out.append({"system": A.label(s), "pairs": len(ks),
                    "same_choice": round(float(np.mean([first[k].get("picked_choice") == second[k].get("picked_choice") for k in ks])), 4),
                    "accuracy_run1": round(float(np.mean(list(a.values()))), 4), "accuracy_run2": round(float(np.mean(list(b.values()))), 4),
                    "run1_minus_run2": d, "ci": f"[{lo}, {hi}]",
                    "equal_share_run1": round(float(np.mean([A.is_equal(first[k]) for k in ks])), 4),
                    "equal_share_run2": round(float(np.mean([A.is_equal(second[k]) for k in ks])), 4),
                    "same_prompt_hash": round(float(same_hash), 4), "same_model_id": round(float(same_model), 4),
                    "same_token_budget": round(float(same_budget), 4)})
    C.write_csv(AN / "r5_hosted_repeatability.csv", out, columns=list(out[0]))
    return {"tau_run1_vs_run2_hosted": R3.tau_with_ci(acc1, acc2),
            "max_abs_accuracy_change": max(abs(r["run1_minus_run2"]) for r in out),
            "rank_run1": [r["system"] for r in sorted(out, key=lambda r: -r["accuracy_run1"])],
            "rank_run2": [r["system"] for r in sorted(out, key=lambda r: -r["accuracy_run2"])],
            "n_comparisons": len({k[0].rsplit("-wording", 1)[0] for k in both}) // max(1, len(out)) if out else 0}


# ---------------------------------------------------------------- R6
def r6(main_rows):
    comp = collections.defaultdict(dict); joint = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in main_rows:
        if r["experiment"] in ("verified", "extended"):
            comp[r["system"]][(r["comparison_id"], r["wording"])] = float(bool(r["correct"]))
        elif r["experiment"] == "numerical":
            joint[r["system"]][(r["bundle_id"], r["wording"])].append(bool(r["correct"]))
    full = {s: {k: float(len(v) == 3 and all(v)) for k, v in d.items()} for s, d in joint.items()}
    out = []
    for fam in sorted({A.family(s) for s in comp if not s.startswith("bedrock")}):
        group = {"attribution-guided": [f"{fam}|graft_proposed|seed{x}" for x in (42, 43, 44)],
                 "plain": [f"{fam}|reference_vanilla_lora|seed{x}" for x in (42, 43, 44)],
                 "random": [f"{fam}|{d}|seed42" for d in A.DRAW]}
        for outcome, src in (("memory accuracy", comp), ("fully correct", full)):
            avg = {g: {k: float(np.mean([src[s][k] for s in ss])) for k in src[ss[0]]} for g, ss in group.items()}
            for a, b in (("attribution-guided", "random"), ("plain", "random"), ("attribution-guided", "plain")):
                d, lo, hi = boot_diff(avg[a], avg[b], A.first)
                spread = lambda g: round(max(mean_acc(src[s]) for s in group[g]) - min(mean_acc(src[s]) for s in group[g]), 4)
                out.append({"family": A.FAMILY_NAME[fam], "outcome": outcome, "contrast": f"mean {a} minus mean {b}",
                            "difference": d, "ci_lower": lo, "ci_upper": hi, "excludes_zero": lo > 0 or hi < 0,
                            f"spread_{a}": spread(a), f"spread_{b}": spread(b)})
    C.write_csv(AN / "r6_placement_all_seeds.csv", out, columns=list(dict.fromkeys(k for r in out for k in r)))
    return {"guided_vs_random_excluding_zero": [(r["family"], r["outcome"], r["difference"]) for r in out
                                                if r["contrast"] == "mean attribution-guided minus mean random" and r["excludes_zero"]],
            "guided_vs_random_all": [(r["family"], r["outcome"], r["difference"], r["ci_lower"], r["ci_upper"]) for r in out
                                     if r["contrast"] == "mean attribution-guided minus mean random"]}


# ---------------------------------------------------------------- R7
def r7(main_rows):
    v220 = [r for r in main_rows if r["experiment"] == "rule_2_20"]
    keep = {r["comparison_id"] for r in v220}
    std = [r for r in main_rows if r["experiment"] in ("verified", "extended") and r["comparison_id"] in keep]
    out = []
    for s in sorted({r["system"] for r in v220}):
        a = [r for r in v220 if r["system"] == s]; b = [r for r in std if r["system"] == s]
        out.append({"system": A.label(s), "system_id": s, "n_comparisons": len(keep),
                    "standard_equal_share_same_items": round(float(np.mean([A.is_equal(r) for r in b])), 4),
                    "rule_2_20_equal_share": round(float(np.mean([A.is_equal(r) for r in a])), 4),
                    "standard_accuracy_same_items": round(float(np.mean([bool(r["correct"]) for r in b])), 4),
                    "rule_2_20_accuracy": round(float(np.mean([bool(r["correct"]) for r in a])), 4)})
    C.write_csv(AN / "r7_rule_2_20_matched.csv", out, columns=list(out[0]))
    return {"n_comparisons": len(keep), "n_equal": len({r["comparison_id"] for r in v220 if r["condition"] == "equal"})}


# ---------------------------------------------------------------- R8
def r8(r3_rows):
    m = [r for r in r3_rows if r["experiment"] in ("mnli_plain", "mnli_strict") and unmod(r["system"])]
    out = []
    for cond in ("mnli_plain", "mnli_strict"):
        for gold in ("entailment", "contradiction", "neither"):
            v = [r for r in m if r["experiment"] == cond and r["gold_choice_text"].lower() == gold]
            c = collections.Counter(str(r.get("picked_choice") or "unparsed").lower() for r in v)
            out.append({"condition": cond.replace("mnli_", ""), "gold": gold, "n": len(v),
                        **{f"answered_{k}": round(c[k] / len(v), 4) for k in ("entailment", "contradiction", "neither", "unparsed")}})
    C.write_csv(AN / "r8_mnli_by_gold.csv", out, columns=list(out[0]))
    return out


def main():
    AN.mkdir(parents=True, exist_ok=True)
    main_rows = A.load()
    r3_rows = R3.load()
    s = {"R1_drift": r1(main_rows, r3_rows), "R2_recall": r2(main_rows, r3_rows), "R3_real_table": r3(main_rows),
         "R4_lenient": r4(main_rows), "R5_hosted": r5(), "R6_placement": r6(main_rows), "R7_rule_2_20": r7(main_rows),
         "R8_mnli": r8(r3_rows)}
    C.write_json(AN / "summary_review.json", s)
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
