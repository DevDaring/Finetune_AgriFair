"""Pre-registered analysis for the TMLR extension (plan: Submission1/TMLR_Research_Plan.md, sections 2 and 4).

    python -m Submission1_TMLR.analyse

Inputs : results_submission1_tmlr/gpu_main_predictions.jsonl (sdpa), bedrock_main_predictions.jsonl,
         gpu_flash_predictions.jsonl (FlashAttention-2 sensitivity only).
Outputs: results_submission1_tmlr/analysis/*.csv and summary.json.

Definitions follow the published analysis (Submission1_DKE_Repair.analyse_v2) so that results
for the original four systems are directly comparable. Unparsed answers count as wrong.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from typing import Dict, List, Tuple

import numpy as np

from Submission1_Code_Phase2 import common as C

OUT = C.CODES_ROOT / "results_submission1_tmlr"
DRAWS, SEED = 10000, 20261004
EQUAL = "roughly equal"
FAMILY_NAME = {"small-instruct": "Llama-3.2-3B", "broad-instruct": "Qwen3-4B",
               "general-instruct-2": "Ministral-8B", "general-instruct": "Gemma-3-12B",
               "qwen3-next-80b-a3b": "Qwen3-Next-80B", "deepseek-v3.2": "DeepSeek-V3.2"}
METHOD_NAME = {"frozen_base": "unmodified", "graft_proposed": "attribution-guided LoRA",
               "reference_vanilla_lora": "plain LoRA", "ablation_placement_random": "random-placement LoRA",
               "unmodified": "unmodified"}


# ------------------------------------------------------------------ loading
_LAST = None


def rescore_cot(r: Dict) -> Dict:
    """Reasoning replies end with the answer, so the LAST answer_choice_letter is the answer (the published
    first-object parser would read an interim object). No final JSON counts as wrong and is recorded."""
    import re
    global _LAST
    _LAST = _LAST or re.compile(r'"answer_choice_letter"\s*:\s*"\(?([A-Da-d])\)?"')
    found = _LAST.findall(r.get("raw_output") or "")
    letter = found[-1].lower() if found else None
    ok = letter is not None and "abcd".index(letter) < len(r["choices"])
    picked = r["choices"]["abcd".index(letter)] if ok else None
    return {**r, "display_letter": letter if ok else None, "picked_choice": picked, "parse_ok": ok,
            "correct": bool(ok and picked.strip().lower() == str(r["gold_choice_text"]).strip().lower()),
            "cot_no_final_answer": not ok}


def rows_of(p):
    """A predictions file, or its gzipped copy (the public repository stores the large files gzipped)."""
    import gzip
    from Submission1_TMLR.exclusions import EXCLUDED_COMPARISONS
    rows = None
    if p.exists():
        rows = C.read_jsonl(p)
    else:
        gz = p.with_name(p.name + ".gz")
        if gz.exists():
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
    if rows is None:
        return None
    return [r for r in rows if r.get("comparison_id") not in EXCLUDED_COMPARISONS]


def load() -> List[Dict]:
    rows = []
    for name in ("gpu_main_predictions.jsonl", "bedrock_main_predictions.jsonl",
                 "gpu_followups_predictions.jsonl", "bedrock_followups_predictions.jsonl",
                 "gpu_cot_predictions.jsonl", "bedrock_cot_predictions.jsonl",
                 "gpu_altered48_predictions.jsonl", "bedrock_altered48_predictions.jsonl",
                 "gpu_draws_predictions.jsonl"):
        found = rows_of(OUT / name)
        if found is not None:
            seen = set()
            for r in found:
                k = (r["prompt_id"], r["system"])
                if k not in seen and not r.get("error"):
                    seen.add(k); rows.append(rescore_cot(r) if r.get("experiment") == "cot" else r)
    return rows


DRAW = {"ablation_placement_random": 1, "ablation_placement_random_draw2": 2, "ablation_placement_random_draw3": 3}


def canon(m: str) -> str:
    """the three random-placement draws are one method (added 4 Oct 2026: draws 2 and 3)."""
    return "ablation_placement_random" if m in DRAW else m


def variant(sid: str) -> str:
    p = sid.split("|")
    if p[0] == "bedrock" or p[1] == "frozen_base":
        return ""
    return f"draw {DRAW[p[1]]}" if p[1] in DRAW else f"seed {p[2].replace('seed', '')}"


def label(sid: str) -> str:
    parts = sid.split("|")
    if parts[0] == "bedrock":
        return FAMILY_NAME[parts[1]]
    fam, method, seed = parts
    return f"{FAMILY_NAME[fam]}, {METHOD_NAME[canon(method)]}" + ("" if method == "frozen_base" else f", {variant(sid)}")


def family(sid: str) -> str:
    p = sid.split("|")
    return p[1] if p[0] == "bedrock" else p[0]


def method(sid: str) -> str:
    p = sid.split("|")
    return "unmodified" if p[0] == "bedrock" else p[1]


def seed_of(sid: str):
    p = sid.split("|")
    return None if p[0] == "bedrock" else int(p[2].replace("seed", ""))


def is_equal(r: Dict) -> bool:
    return EQUAL in str(r.get("picked_choice") or "").lower()


def category(r: Dict) -> str:
    """first entity, second entity, roughly equal, or unparsed (numerical sets)."""
    pc = r.get("picked_choice")
    if pc is None:
        return "unparsed"
    if EQUAL in pc.lower():
        return "equal"
    if pc == r.get("group1"):
        return "first"
    if pc == r.get("group2"):
        return "second"
    return "other"


# ------------------------------------------------------------------ statistics
def boot_ci(diffs: np.ndarray, seed: int = SEED) -> Tuple[float, float]:
    if len(diffs) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(DRAWS, len(diffs)))
    m = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def signflip_p(d: np.ndarray, seed: int = SEED) -> float:
    """Two-sided paired sign-flip test on cluster-level differences (Monte Carlo, DRAWS flips)."""
    if len(d) == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(DRAWS, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((1 + np.sum(null >= obs - 1e-12)) / (DRAWS + 1))


def paired_cluster(a: Dict, b: Dict, cluster) -> Dict:
    """a, b: unit -> 0/1 outcome. Units sharing a cluster (e.g. the two wordings of one comparison or
    scenario) are averaged first; the interval resamples clusters and the p-value is a cluster-level
    sign-flip test, because McNemar assumes independent pairs (deviation logged 4 Oct 2026)."""
    keys = sorted(set(a) & set(b))
    by = collections.defaultdict(list)
    for k in keys:
        by[cluster(k)].append(a[k] - b[k])
    d = np.array([np.mean(v) for _, v in sorted(by.items())], float)
    lo, hi = boot_ci(d)
    return {"n_units": len(keys), "n_clusters": len(d),
            "difference": round(float(np.mean([a[k] - b[k] for k in keys])), 4) if keys else None,
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4),
            "discordant_a_only": sum(1 for k in keys if a[k] == 1 and b[k] == 0),
            "discordant_b_only": sum(1 for k in keys if a[k] == 0 and b[k] == 1),
            "p_signflip": round(signflip_p(d), 6)}


def first(k):
    return k[0]


def paired(a: Dict, b: Dict) -> Dict:
    """a, b: cluster -> mean outcome (0..1). Returns difference a-b with bootstrap CI and McNemar on
    binary clusters (clusters with mixed outcomes enter the bootstrap but not the McNemar counts)."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys], float)
    lo, hi = boot_ci(d)
    bb = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    cc = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    return {"n_clusters": len(keys), "difference": round(float(d.mean()), 4) if len(d) else None,
            "ci_lower": round(lo, 4), "ci_upper": round(hi, 4), "discordant_a_only": bb,
            "discordant_b_only": cc, "p_mcnemar": round(mcnemar_exact(bb, cc), 6)}


# ------------------------------------------------------------------ H2, H3: verified and extended comparisons
def comparisons(rows: List[Dict]) -> Dict[str, List[Dict]]:
    sel = [r for r in rows if r["experiment"] in ("verified", "extended")]
    systems = sorted({r["system"] for r in sel})
    wording, itemtype = [], []
    for sid in systems:
        s = [r for r in sel if r["system"] == sid]
        for scope, exps in (("verified (34)", {"verified"}), ("extended (120)", {"extended"}),
                            ("all verified (154)", {"verified", "extended"})):
            v = [r for r in s if r["experiment"] in exps]
            a = {r["comparison_id"]: float(bool(r["correct"])) for r in v if r["wording"] == "wording_a"}
            b = {r["comparison_id"]: float(bool(r["correct"])) for r in v if r["wording"] == "wording_b"}
            res = paired(a, b)
            wording.append({"system": label(sid), "system_id": sid, "set": scope,
                            "accuracy_wording_a": round(sum(a.values()) / len(a), 4) if a else None,
                            "accuracy_wording_b": round(sum(b.values()) / len(b), 4) if b else None,
                            **res, "parse_rate": round(sum(bool(r["parse_ok"]) for r in v) / len(v), 4) if v else None})
            diff = [r for r in v if r["condition"] == "diff"]
            eq = [r for r in v if r["condition"] == "equal"]
            itemtype.append({"system": label(sid), "system_id": sid, "set": scope, "n_responses": len(v),
                             "accuracy_overall": round(sum(bool(r["correct"]) for r in v) / len(v), 4) if v else None,
                             "accuracy_difference_items": round(sum(bool(r["correct"]) for r in diff) / len(diff), 4) if diff else None,
                             "n_difference_responses": len(diff),
                             "accuracy_equal_items": round(sum(bool(r["correct"]) for r in eq) / len(eq), 4) if eq else None,
                             "n_equal_responses": len(eq),
                             "share_roughly_equal_answers": round(sum(is_equal(r) for r in v) / len(v), 4) if v else None,
                             "defaults_to_equal": bool(v) and sum(is_equal(r) for r in v) / len(v) >= 0.8})
    # Holm within H3: one contrast per system, on the 154-comparison scope (primary)
    prim = [w for w in wording if w["set"] == "all verified (154)"]
    for w, adj in zip(prim, holm([w["p_mcnemar"] for w in prim])):
        w["p_holm_h3"] = round(adj, 6)
    return {"wording": wording, "itemtype": itemtype}


def reordered(rows: List[Dict]) -> List[Dict]:
    base = {(r["system"], r["comparison_id"], r["wording"]): r for r in rows if r["experiment"] == "verified"}
    perm = {(r["system"], r["comparison_id"], r["wording"]): r for r in rows if r["experiment"] == "reordered"}
    out = []
    for sid in sorted({k[0] for k in perm}):
        keys = [k for k in perm if k[0] == sid and k in base]
        if not keys:
            continue
        n = len(keys)
        out.append({"system": label(sid), "system_id": sid, "n_responses": n,
                    "equal_share_at_c": round(sum(is_equal(base[k]) for k in keys) / n, 4),
                    "equal_share_moved": round(sum(is_equal(perm[k]) for k in keys) / n, 4),
                    "same_meaning": round(sum(base[k].get("picked_choice") == perm[k].get("picked_choice") for k in keys) / n, 4),
                    "same_letter": round(sum(base[k].get("display_letter") == perm[k].get("display_letter") for k in keys) / n, 4),
                    "accuracy_at_c": round(sum(bool(base[k]["correct"]) for k in keys) / n, 4),
                    "accuracy_moved": round(sum(bool(perm[k]["correct"]) for k in keys) / n, 4)})
    return out


# ------------------------------------------------------------------ H4, H5, H6: numerical set
def numerical(rows: List[Dict]) -> Tuple[List[Dict], Dict[str, Dict]]:
    sel = [r for r in rows if r["experiment"] == "numerical"]
    out, joint = [], {}
    for sid in sorted({r["system"] for r in sel}):
        s = [r for r in sel if r["system"] == sid]
        cats = collections.Counter(category(r) for r in s)
        acc_rel = {rel: (lambda v: round(sum(bool(r["correct"]) for r in v) / len(v), 4) if v else None)(
            [r for r in s if r["relation"] == rel]) for rel in sorted({r["relation"] for r in s})}
        groups = collections.defaultdict(list)
        for r in s:
            groups[(r["bundle_id"], r["wording"])].append(bool(r["correct"]))
        j = {k: float(len(v) == 3 and all(v)) for k, v in groups.items()}
        joint[sid] = j
        jw = {w: round(np.mean([v for (b, ww), v in j.items() if ww == w]), 4) for w in sorted({k[1] for k in j})}
        out.append({"system": label(sid), "system_id": sid, "n_prompts": len(s),
                    "first": cats["first"], "second": cats["second"], "equal": cats["equal"],
                    "unparsed_or_other": cats["unparsed"] + cats["other"],
                    "largest_share": round(max(cats["first"], cats["second"], cats["equal"]) / len(s), 4),
                    "accuracy_overall": round(sum(bool(r["correct"]) for r in s) / len(s), 4),
                    **{f"accuracy_{k}": v for k, v in acc_rel.items()},
                    **{f"fully_correct_scenarios_{k}": v for k, v in jw.items()},
                    "fully_correct_scenarios_pooled": round(float(np.mean(list(j.values()))), 4)})
    return out, joint


def finetune_contrasts(joint: Dict[str, Dict], comp_acc: Dict[str, Dict]) -> List[Dict]:
    """H4: fine-tuned minus unmodified, per family x method x seed, on fully correct scenarios
    (cluster = scenario x wording) and, as a secondary outcome, on verified-comparison accuracy."""
    out = []
    for sid in sorted(joint):
        if method(sid) in ("frozen_base", "unmodified"):
            continue
        base = f"{family(sid)}|frozen_base|seed42"
        if base not in joint:
            continue
        r = paired_cluster(joint[sid], joint[base], first)
        out.append({"family": FAMILY_NAME[family(sid)], "method": METHOD_NAME[canon(method(sid))], "seed": seed_of(sid),
                    "variant": variant(sid), "outcome": "fully correct scenarios (numerical set)", **r})
        if sid in comp_acc and base in comp_acc:
            r2 = paired_cluster(comp_acc[sid], comp_acc[base], first)
            out.append({"family": FAMILY_NAME[family(sid)], "method": METHOD_NAME[canon(method(sid))], "seed": seed_of(sid),
                        "variant": variant(sid), "outcome": "accuracy from memory (154 verified comparisons)", **r2})
    for outcome in {o["outcome"] for o in out}:
        fam = [o for o in out if o["outcome"] == outcome]
        for o, adj in zip(fam, holm([o["p_signflip"] for o in fam])):
            o["p_holm_h4"] = round(adj, 6)
    return out


def method_contrasts(joint: Dict[str, Dict], comp_acc: Dict[str, Dict]) -> List[Dict]:
    """H6: attribution-guided vs plain LoRA (seeds 42-44) and both vs random placement (seed 42)."""
    out = []
    for fam in sorted({family(s) for s in joint if not s.startswith("bedrock")}):
        pairs = [(f"{fam}|graft_proposed|seed{s}", f"{fam}|reference_vanilla_lora|seed{s}") for s in (42, 43, 44)]
        for d in DRAW:   # each random-placement draw against the seed-42 adapters of the other two methods
            pairs += [(f"{fam}|graft_proposed|seed42", f"{fam}|{d}|seed42"),
                      (f"{fam}|reference_vanilla_lora|seed42", f"{fam}|{d}|seed42")]
        rows = []
        for a, b in pairs:
            for outcome, src in (("fully correct scenarios (numerical set)", joint),
                                 ("accuracy from memory (154 verified comparisons)", comp_acc)):
                if a in src and b in src:
                    rows.append({"family": FAMILY_NAME[fam], "contrast": f"{label(a)} minus {label(b)}",
                                 "outcome": outcome, **paired_cluster(src[a], src[b], first)})
        for outcome in {r["outcome"] for r in rows}:
            fr = [r for r in rows if r["outcome"] == outcome]
            for r, adj in zip(fr, holm([r["p_signflip"] for r in fr])):
                r["p_holm_h6_within_family"] = round(adj, 6)
        out += rows
    return out


def seed_summary(num_rows: List[Dict], item_rows: List[Dict]) -> List[Dict]:
    by = collections.defaultdict(list)
    num = {r["system_id"]: r for r in num_rows}
    mem = {r["system_id"]: r for r in item_rows if r["set"] == "all verified (154)"}
    for sid in num:
        if sid.startswith("bedrock") or method(sid) == "frozen_base":
            continue
        by[(family(sid), canon(method(sid)))].append(sid)
    out = []
    for (fam, meth), sids in sorted(by.items()):
        for metric, src, key in (("numerical accuracy", num, "accuracy_overall"),
                                 ("fully correct scenarios", num, "fully_correct_scenarios_pooled"),
                                 ("memory accuracy (154)", mem, "accuracy_overall"),
                                 ("share of 'roughly equal' (154)", mem, "share_roughly_equal_answers")):
            vals = [src[s][key] for s in sids if s in src and src[s][key] is not None]
            if vals:
                out.append({"family": FAMILY_NAME[fam], "method": METHOD_NAME[meth], "metric": metric,
                            "n_adapters": len(vals), "mean": round(float(np.mean(vals)), 4),
                            "min": round(min(vals), 4), "max": round(max(vals), 4)})
    return out


def ranking(rows: List[Dict]) -> Dict:
    """H5: Kendall's tau between accuracy from memory (154 verified) and numerical-set accuracy."""
    mem = collections.defaultdict(dict); num = collections.defaultdict(dict); clus = {}
    for r in rows:
        if r["experiment"] in ("verified", "extended"):
            mem[r["system"]][r["prompt_id"]] = float(bool(r["correct"])); clus[r["prompt_id"]] = r["comparison_id"]
        elif r["experiment"] == "numerical":
            num[r["system"]][r["prompt_id"]] = float(bool(r["correct"])); clus[r["prompt_id"]] = r["bundle_id"]
    systems = sorted(set(mem) & set(num))
    mids = sorted(set.intersection(*(set(mem[s]) for s in systems)))
    nids = sorted(set.intersection(*(set(num[s]) for s in systems)))
    M = np.array([[mem[s][i] for i in mids] for s in systems]); N = np.array([[num[s][i] for i in nids] for s in systems])

    def tau(x, y):
        conc = disc = 0
        for i, j in itertools.combinations(range(len(x)), 2):
            sgn = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            conc += sgn > 0; disc += sgn < 0
        n = len(x) * (len(x) - 1) / 2
        return (conc - disc) / n, int(disc)

    t, d = tau(M.mean(1), N.mean(1))
    # resample whole comparisons and whole scenarios (all prompts of a drawn cluster come together)
    mc = collections.defaultdict(list); nc = collections.defaultdict(list)
    for j, i in enumerate(mids):
        mc[clus[i]].append(j)
    for j, i in enumerate(nids):
        nc[clus[i]].append(j)
    mcl, ncl = list(mc.values()), list(nc.values())
    rng = np.random.default_rng(SEED); boots = []
    for _ in range(2000):
        mi = np.concatenate([mcl[k] for k in rng.integers(0, len(mcl), len(mcl))])
        ni = np.concatenate([ncl[k] for k in rng.integers(0, len(ncl), len(ncl))])
        boots.append(tau(M[:, mi].mean(1), N[:, ni].mean(1))[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per = [{"system": label(s), "system_id": s, "accuracy_from_memory_154": round(float(M[k].mean()), 4),
            "accuracy_supplied_numbers": round(float(N[k].mean()), 4)} for k, s in enumerate(systems)]
    return {"n_systems": len(systems), "kendall_tau": round(t, 4), "ci_lower": round(float(lo), 4),
            "ci_upper": round(float(hi), 4), "discordant_pairs": d,
            "total_pairs": len(systems) * (len(systems) - 1) // 2, "per_system": per}


def diagnostics(rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Altered minus unchanged table, per system and alteration. Scenarios B001-B012 are the published 12;
    B013-B048 were added on 4 Oct 2026. One response per scenario, so McNemar applies; Holm runs across
    systems within each alteration on the 48-scenario scope."""
    clean = {(r["system"], r["bundle_id"]): bool(r["correct"]) for r in rows if r["experiment"] == "unchanged"}
    alt, neu = [], []
    scopes = (("all 48", lambda b: True), ("published 12", lambda b: int(b[1:]) <= 12),
              ("new 36", lambda b: int(b[1:]) > 12))
    for sid in sorted({r["system"] for r in rows if r["experiment"] == "altered"}):
        for variant in sorted({r["variant"] for r in rows if r["experiment"] == "altered"}):
            for scope, keep in scopes:
                v = [r for r in rows if r["experiment"] == "altered" and r["system"] == sid and r["variant"] == variant
                     and (sid, r["bundle_id"]) in clean and keep(r["bundle_id"])]
                if not v:
                    continue
                a = {r["bundle_id"]: float(bool(r["correct"])) for r in v}
                c = {r["bundle_id"]: float(clean[(sid, r["bundle_id"])]) for r in v}
                res = paired(a, c)
                alt.append({"system": label(sid), "system_id": sid, "alteration": variant, "scope": scope,
                            "n": len(v), "unchanged_accuracy": round(np.mean(list(c.values())), 4),
                            "altered_accuracy": round(np.mean(list(a.values())), 4),
                            "altered_minus_unchanged": res["difference"], "ci_lower": res["ci_lower"],
                            "ci_upper": res["ci_upper"], "p_mcnemar": res["p_mcnemar"]})
    for variant in {r["alteration"] for r in alt}:
        fam = [r for r in alt if r["alteration"] == variant and r["scope"] == "all 48"]
        for r, adj in zip(fam, holm([r["p_mcnemar"] for r in fam])):
            r["p_holm"] = round(adj, 6)
    agri = {(r["system"], r["prompt_id"]): bool(r["correct"]) for r in rows if r["experiment"] == "numerical"}
    for sid in sorted({r["system"] for r in rows if r["experiment"] == "neutral"}):
        pairs = [(agri[(sid, r["prompt_id"].replace("v2neutral-", ""))], bool(r["correct"]))
                 for r in rows if r["experiment"] == "neutral" and r["system"] == sid
                 and (sid, r["prompt_id"].replace("v2neutral-", "")) in agri]
        if pairs:
            a = sum(x for x, _ in pairs) / len(pairs); n = sum(y for _, y in pairs) / len(pairs)
            neu.append({"system": label(sid), "system_id": sid, "n_matched": len(pairs),
                        "agricultural_accuracy": round(a, 4), "neutral_accuracy": round(n, 4),
                        "neutral_minus_agricultural": round(n - a, 4)})
    return alt, neu


def gap_axis(rows: List[Dict]) -> List[Dict]:
    """Added 4 Oct 2026 (secondary): on the 146 difference comparisons, the share of 'roughly equal'
    answers and accuracy by the size of the true gap and by axis. A default that ignores the evidence
    should not weaken as the gap grows."""
    gap = {}
    for rel in ("results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "results_submission1_tmlr/extended_panel.jsonl"):
        for it in C.read_jsonl(C.CODES_ROOT / rel):
            gap[it["fresh_id"]] = abs(float(it["share1_pct"]) - float(it["share2_pct"]))
    bins = (("gap 10-25", 10, 25), ("gap 25-50", 25, 50), ("gap 50+", 50, 101))
    out = []
    sel = [r for r in rows if r["experiment"] in ("verified", "extended") and r["condition"] == "diff"]
    for sid in sorted({r["system"] for r in sel}):
        s = [r for r in sel if r["system"] == sid]
        rec = {"system": label(sid), "system_id": sid}
        for name, lo, hi in bins:
            v = [r for r in s if lo <= gap[r["comparison_id"]] < hi]
            rec[f"{name}_n"] = len(v)
            rec[f"{name}_equal_share"] = round(np.mean([is_equal(r) for r in v]), 4) if v else None
            rec[f"{name}_accuracy"] = round(np.mean([bool(r["correct"]) for r in v]), 4) if v else None
        for ax in ("landholding", "social_group"):
            v = [r for r in s if r["axis"] == ax]
            rec[f"{ax}_equal_share"] = round(np.mean([is_equal(r) for r in v]), 4) if v else None
            rec[f"{ax}_accuracy"] = round(np.mean([bool(r["correct"]) for r in v]), 4) if v else None
        out.append(rec)
    return out


CANT_TELL = "I cannot tell from what I know"


def followups(rows: List[Dict]) -> List[Dict]:
    """Added 4 Oct 2026: why do models answer 'roughly equal'? Each condition is paired with the
    same comparison and wording under the standard prompt (verified + extended sets)."""
    std = {(r["system"], r["comparison_id"], r["wording"]): r for r in rows if r["experiment"] in ("verified", "extended")}
    out = []
    for sid in sorted({r["system"] for r in rows if r["experiment"] in ("norule", "abstain", "realtable", "cot")}):
        rec = {"system": label(sid), "system_id": sid}
        base = {k: v for k, v in std.items() if k[0] == sid}
        rec["standard_accuracy"] = round(np.mean([bool(v["correct"]) for v in base.values()]), 4) if base else None
        rec["standard_equal_share"] = round(np.mean([is_equal(v) for v in base.values()]), 4) if base else None
        for exp in ("norule", "abstain", "realtable", "cot"):
            sel = {(r["system"], r["comparison_id"], r["wording"]): r for r in rows if r["experiment"] == exp and r["system"] == sid}
            keys = [k for k in sel if k in base]
            if not keys:
                continue
            rec[f"{exp}_n"] = len(keys)
            rec[f"{exp}_accuracy"] = round(np.mean([bool(sel[k]["correct"]) for k in keys]), 4)
            rec[f"{exp}_equal_share"] = round(np.mean([is_equal(sel[k]) for k in keys]), 4)
            diff_keys = [k for k in keys if sel[k]["condition"] == "diff"]
            rec[f"{exp}_accuracy_difference_items"] = round(np.mean([bool(sel[k]["correct"]) for k in diff_keys]), 4) if diff_keys else None
            if exp == "cot":
                rec["cot_no_final_answer_share"] = round(np.mean([bool(sel[k].get("cot_no_final_answer")) for k in keys]), 4)
                done_k = [k for k in keys if not sel[k].get("cot_no_final_answer")]
                rec["cot_accuracy_answered"] = round(np.mean([bool(sel[k]["correct"]) for k in done_k]), 4) if done_k else None
                rec["cot_equal_share_answered"] = round(np.mean([is_equal(sel[k]) for k in done_k]), 4) if done_k else None
            if exp == "abstain":
                rec["abstain_cant_tell_share"] = round(np.mean([sel[k].get("picked_choice") == CANT_TELL for k in keys]), 4)
            # paired: condition minus standard; equal-answer share for norule/abstain, accuracy for realtable
            if exp in ("realtable", "cot"):
                a = {k[1:]: float(bool(sel[k]["correct"])) for k in keys}; b = {k[1:]: float(bool(base[k]["correct"])) for k in keys}
            else:
                a = {k[1:]: float(is_equal(sel[k])) for k in keys}; b = {k[1:]: float(is_equal(base[k])) for k in keys}
            res = paired_cluster(a, b, first)
            rec[f"{exp}_minus_standard"] = res["difference"]; rec[f"{exp}_ci"] = f"[{res['ci_lower']}, {res['ci_upper']}]"
            rec[f"{exp}_ci_lower"], rec[f"{exp}_ci_upper"] = res["ci_lower"], res["ci_upper"]
            rec[f"{exp}_p"] = res["p_signflip"]
        out.append(rec)
    for exp in ("norule", "abstain", "realtable", "cot"):
        have = [r for r in out if f"{exp}_p" in r]
        for r, adj in zip(have, holm([r[f"{exp}_p"] for r in have])):
            r[f"{exp}_p_holm"] = round(adj, 6)
    return out


def budget_sensitivity(rows: List[Dict]) -> List[Dict]:
    """Added 4 Oct 2026: the two GPU systems with unparsed 24-token answers, re-run with 256 tokens
    (the API models' budget). Same measures as the main analysis, on the same prompts."""
    retry = rows_of(OUT / "gpu_retry256_predictions.jsonl")
    if retry is None:
        return []
    out = []
    for sid in sorted({r["system"] for r in retry}):
        for tag, src in (("24 tokens (main)", [r for r in rows if r["system"] == sid]),
                         ("256 tokens", [r for r in retry if r["system"] == sid])):
            mem = [r for r in src if r["experiment"] in ("verified", "extended")]
            num = [r for r in src if r["experiment"] == "numerical"]
            groups = collections.defaultdict(list)
            for r in num:
                groups[(r["bundle_id"], r["wording"])].append(bool(r["correct"]))
            out.append({"system": label(sid), "system_id": sid, "budget": tag,
                        "parse_rate_memory": round(np.mean([bool(r["parse_ok"]) for r in mem]), 4),
                        "accuracy_memory_154": round(np.mean([bool(r["correct"]) for r in mem]), 4),
                        "equal_share_memory_154": round(np.mean([is_equal(r) for r in mem]), 4),
                        "defaults_to_equal": bool(np.mean([is_equal(r) for r in mem]) >= 0.8),
                        "parse_rate_numerical": round(np.mean([bool(r["parse_ok"]) for r in num]), 4),
                        "accuracy_numerical": round(np.mean([bool(r["correct"]) for r in num]), 4),
                        "fully_correct_scenarios": round(np.mean([len(v) == 3 and all(v) for v in groups.values()]), 4)})
    return out


def reproducibility() -> List[Dict]:
    """Main run (sdpa) against the published outputs, and FlashAttention-2 against the main run."""
    main_rows, flash_rows = rows_of(OUT / "gpu_main_predictions.jsonl"), rows_of(OUT / "gpu_flash_predictions.jsonl")
    if main_rows is None:
        return []
    published = {}
    for f in ("results_submission1_dke_repair_v2/v2_predictions.jsonl",
              "results_submission1_phase2/predictions/main_predictions.jsonl",
              "results_submission1_phase2/predictions/pilot_predictions.jsonl"):
        p = C.CODES_ROOT / f
        if p.exists():
            for r in C.read_jsonl(p):
                published.setdefault((r["prompt_id"], r["system"]), r)
    main = {(r["prompt_id"], r["system"]): r for r in main_rows}
    flash = {(r["prompt_id"], r["system"]): r for r in flash_rows} if flash_rows is not None else {}
    out = []
    for sid in sorted({k[1] for k in main}):
        pub = [(m, published[k]) for k, m in main.items() if k[1] == sid and k in published]
        fl = [(m, flash[k]) for k, m in main.items() if k[1] == sid and k in flash]
        if not pub and not fl:
            continue
        out.append({"system": label(sid), "system_id": sid,
                    "n_compared_with_published": len(pub),
                    "identical_raw_output_vs_published": round(sum(a["raw_output"] == b["raw_output"] for a, b in pub) / len(pub), 4) if pub else None,
                    "identical_answer_vs_published": round(sum(a.get("picked_choice") == b.get("picked_choice") for a, b in pub) / len(pub), 4) if pub else None,
                    "n_compared_sdpa_vs_flash": len(fl),
                    "answers_changed_by_flash": sum(a.get("picked_choice") != b.get("picked_choice") for a, b in fl) if fl else None,
                    "share_changed_by_flash": round(sum(a.get("picked_choice") != b.get("picked_choice") for a, b in fl) / len(fl), 4) if fl else None,
                    "accuracy_sdpa": round(sum(bool(a["correct"]) for a, _ in fl) / len(fl), 4) if fl else None,
                    "accuracy_flash": round(sum(bool(b["correct"]) for _, b in fl) / len(fl), 4) if fl else None})
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    rows = load()
    if not rows:
        raise SystemExit("no predictions found")
    ad = OUT / "analysis"; ad.mkdir(parents=True, exist_ok=True)
    comp = comparisons(rows)
    num, joint = numerical(rows)
    comp_acc = collections.defaultdict(dict)
    for r in rows:
        if r["experiment"] in ("verified", "extended"):
            comp_acc[r["system"]][(r["comparison_id"], r["wording"])] = float(bool(r["correct"]))
    ft = finetune_contrasts(joint, comp_acc)
    mc = method_contrasts(joint, comp_acc)
    alt, neu = diagnostics(rows)
    rk = ranking(rows)
    tables = {"h3_wording.csv": comp["wording"], "h2_item_type.csv": comp["itemtype"],
              "reordered_options.csv": reordered(rows), "h4_numerical.csv": num,
              "h4_finetune_contrasts.csv": ft, "h6_method_contrasts.csv": mc,
              "seed_summary.csv": seed_summary(num, comp["itemtype"]),
              "h5_ranking_per_system.csv": rk["per_system"], "altered_tables.csv": alt,
              "neutral_wording.csv": neu, "reproducibility.csv": reproducibility(),
              "followups.csv": followups(rows), "h2_gap_axis.csv": gap_axis(rows),
              "budget_sensitivity.csv": budget_sensitivity(rows)}
    for name, t in tables.items():
        if t:
            # the writer drops keys missing from the first row, so pass the union of keys in order
            cols = list(dict.fromkeys(k for r in t for k in r))
            C.write_csv(ad / name, t, columns=cols)
    summary = {"n_predictions": len(rows), "systems": len({r["system"] for r in rows}),
               "by_experiment": dict(collections.Counter(r["experiment"] for r in rows)),
               "h5_ranking": {k: v for k, v in rk.items() if k != "per_system"},
               "systems_defaulting_to_equal_154": [r["system"] for r in comp["itemtype"]
                                                   if r["set"] == "all verified (154)" and r["defaults_to_equal"]],
               "h3_significant_after_holm": [w["system"] for w in comp["wording"] if w.get("p_holm_h3", 1) < 0.05],
               "h4_significant_after_holm": [f"{o['family']}, {o['method']}, seed {o['seed']}: {o['outcome']} {o['difference']}"
                                             for o in ft if o.get("p_holm_h4", 1) < 0.05]}
    C.write_json(ad / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
