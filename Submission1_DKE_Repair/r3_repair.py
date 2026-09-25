"""M4: report what the advice ratings establish, and nothing beyond it.

    python -m Submission1_DKE_Repair.r3_repair

Four things change relative to the published analysis.

1. Naming. "Floor" and "ceiling" asserted a bound on the true rate of identity-driven advice
   change. Two readers can be wrong together, and they can miss the same real change, so an
   intersection is not a lower bound on truth without an error model or a validated reference
   standard. These are now the both-reader and either-reader proportions: summaries of reader
   labels. A mechanical intersection is also not an adjudication, and is no longer called one.

2. Resampling unit. The 128 identity pairs come from 16 cases, each contributing 8 pairs (four
   systems x two verbosity instructions). Independent-binomial intervals treat them as 128
   independent observations. Uncertainty is recomputed by resampling CASES.

3. The rubric. The analysed three-way field holds none / presentation_only / substantive.
   Whether a change was warranted is a separate yes/no/not-applicable field. The manuscript
   described the three categories as no change, unsupported change and warranted change, which
   is not the field the reported kappa was computed on.

4. The fatigue explanation is removed. The quarter kappas are 0.400, 0.199, 0.240, 0.088 -- not
   monotone -- and sheet order alone cannot separate fatigue from item composition.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from Submission1_Code_Phase2 import common as C

ADVICE = "results_submission1_phase2/advice"
OUT_DIR = "results_submission1_dke_repair_v2"


def load() -> Dict:
    base = C.CODES_ROOT / ADVICE
    key = {r["assessment_id"]: r for r in
           csv.DictReader((base / "KEEP_FROM_RATERS_r3_key.csv").open(encoding="utf-8"))}
    per = {}
    for p in sorted(base.glob("r3_ratings_scrubbed_R*.csv")):
        per[p.stem.split("_")[-1]] = {r["assessment_id"]: r
                                      for r in csv.DictReader(p.open(encoding="utf-8"))}
    return {"key": key, "per": per}


def case_cluster_ci(cases: Dict[str, List[int]], draws: int, seed: int) -> Dict:
    """Resample CASES, not pairs. Each case carries all its systems and verbosity conditions."""
    names = sorted(cases)
    if len(names) < 2:
        return {"point": None, "lower": None, "upper": None, "n_cases": len(names)}
    rng = random.Random(seed)
    flat = [v for n in names for v in cases[n]]
    point = sum(flat) / len(flat)
    stats = []
    for _ in range(draws):
        pick = [cases[names[rng.randrange(len(names))]] for _ in names]
        vals = [v for c in pick for v in c]
        if vals:
            stats.append(sum(vals) / len(vals))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": round(point, 4), "lower": round(float(lo), 4), "upper": round(float(hi), 4),
            "n_cases": len(names), "n_pairs": len(flat), "draws": draws}


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    cfg = C.load_config()
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    key, per = d["key"], d["per"]
    if len(per) < 2:
        raise SystemExit("need both scrubbed rater files")
    r1, r2 = per["R1"], per["R2"]
    ids = [i for i in r1 if i in r2]

    # ---- per-reader and agreement summaries, identity cases -------------------------------
    idn = [i for i in ids if key[i]["case_type"] == "identity_irrelevant"]
    ctx = [i for i in ids if key[i]["case_type"] == "context_control"]

    def sub(rec):  # substantive change, as the rubric field actually reads
        return rec.get("pair_change_type", "") == "substantive"

    per_reader = {}
    for tag, table in (("R1", r1), ("R2", r2)):
        per_reader[tag] = {
            "identity_substantive": sum(1 for i in idn if sub(table[i])),
            "identity_n": len(idn),
            "context_substantive": sum(1 for i in ctx if sub(table[i])),
            "context_n": len(ctx)}

    both = [i for i in idn if sub(r1[i]) and sub(r2[i])]
    either = [i for i in idn if sub(r1[i]) or sub(r2[i])]
    disputed = [i for i in idn if sub(r1[i]) != sub(r2[i])]

    # ---- justification reported separately, with missingness (M4 point 3) ------------------
    just = collections.Counter()
    missing = 0
    for i in both:
        a = r1[i].get("change_is_justified_yes_no_na", "").strip()
        b = r2[i].get("change_is_justified_yes_no_na", "").strip()
        if not a or not b:
            missing += 1
        just[(a or "blank", b or "blank")] += 1

    # ---- case-clustered uncertainty (M4 point 4) -------------------------------------------
    def by_case(pred) -> Dict[str, List[int]]:
        cases: Dict[str, List[int]] = collections.defaultdict(list)
        for i in idn:
            cases[key[i]["case_id"]].append(1 if pred(i) else 0)
        return cases

    draws = int(cfg.get("bootstrap_draws", 10000))
    seed = int(cfg.get("analysis_seed", 20260924))
    both_ci = case_cluster_ci(by_case(lambda i: sub(r1[i]) and sub(r2[i])), draws, seed)
    either_ci = case_cluster_ci(by_case(lambda i: sub(r1[i]) or sub(r2[i])), draws, seed + 1)

    per_system = []
    for sysname in sorted({key[i]["system"] for i in idn}):
        sel = [i for i in idn if key[i]["system"] == sysname]
        cases: Dict[str, List[int]] = collections.defaultdict(list)
        for i in sel:
            cases[key[i]["case_id"]].append(1 if (sub(r1[i]) and sub(r2[i])) else 0)
        ci = case_cluster_ci(cases, draws, seed + 2)
        per_system.append({"system": sysname, "n_pairs": len(sel), "n_cases": ci["n_cases"],
                           "both_reader_substantive": sum(1 for i in sel if sub(r1[i]) and sub(r2[i])),
                           "either_reader_substantive": sum(1 for i in sel if sub(r1[i]) or sub(r2[i])),
                           "both_reader_proportion": ci["point"],
                           "case_cluster_lower": ci["lower"], "case_cluster_upper": ci["upper"]})
    C.write_csv(out / "r3_per_system_reader_agreement.csv", per_system)

    res = {
        "what_these_quantities_are": (
            "Summaries of two readers' labels, not bounds on a true rate. Both readers can err in "
            "the same direction, and an intersection is not a lower bound on truth without an "
            "error model or a validated reference standard. No adjudication was performed."),
        "rubric_field": ("pair_change_type holds none / presentation_only / substantive. Whether a "
                         "change was warranted is a separate yes/no/not_applicable field. The "
                         "reported kappa is computed on the three-way change field."),
        "identity": {
            "n_pairs": len(idn), "n_cases": both_ci["n_cases"],
            "pairs_per_case": len(idn) // max(both_ci["n_cases"], 1),
            "both_reader_substantive": len(both),
            "either_reader_substantive": len(either),
            "disputed": len(disputed),
            "both_reader_proportion_case_clustered": both_ci,
            "either_reader_proportion_case_clustered": either_ci},
        "context_controls": {
            "n_pairs": len(ctx),
            "reported": False,
            "reason": "reader agreement on the change field is at chance here (kappa 0.008)"},
        "per_reader": per_reader,
        "justification_on_both_reader_substantive_pairs": {
            "pairs": len(both),
            "joint_labels": {f"{a}|{b}": n for (a, b), n in sorted(just.items())},
            "missing_either": missing,
            "note": ("Agreement that a change was unwarranted is descriptive agreement between the "
                     "same two readers; it does not remove the possibility of shared error.")},
        "fatigue_claim": ("removed: the quarter kappas are 0.400, 0.199, 0.240, 0.088, which are "
                          "not monotone, and sheet order alone cannot separate fatigue from item "
                          "composition"),
        "analysis_chronology": (
            "The intersection/union rule was written and applied after the ratings were returned "
            "and after reader agreement had been computed. It was applied uniformly to every pair "
            "and no pair was re-rated, but it was not specified before the outcomes were seen and "
            "is therefore not a prospectively declared endpoint."),
        "status": "exploratory",
    }
    C.write_json(out / "r3_reader_agreement_and_case_cluster_results.json", res)
    print(json.dumps({k: res[k] for k in ("identity", "per_reader",
                                          "justification_on_both_reader_substantive_pairs")},
                     indent=1))


if __name__ == "__main__":
    main()
