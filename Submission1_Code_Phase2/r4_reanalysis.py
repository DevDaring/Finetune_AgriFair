"""R4: complete the existing-results analysis and correct the reporting. No GPU.

    python -m Submission1_Code_Phase2.r4_reanalysis

Seven repairs the rejection requires, all computable from saved outputs:

1. **Both P3 primary endpoints.** The protocol declared unsupported substantive change *and*
   both-answers-meet-minimum as co-primary; the manuscript reported only the first. Both are in
   the saved analysis and both are restored here, with the reminder that "meets minimum" allows
   mixed or unverifiable answers and is not verified correctness.
2. Question-cluster intervals for supported-answer and usefulness differences, not only for
   unsupported change.
3. Raw agreement printed beside kappa, with the near-unanimous categories flagged, because kappa
   collapses when one category holds almost every case.
4. Wording: "inseparable" and "offers nothing" become "no improvement detected in these
   comparisons". Non-significance is not equivalence.
5. Per-axis, per-condition and per-source-group counts, with small cluster counts disclosed and
   B left undefined where a condition is absent.
6. The evidence-panel abstract correction: Qwen GRAFT was 47/48 on the joint endpoint, not 48/48;
   48/48 applies to the hypothetical condition alone.
7. A versioned analysis manifest separating original results, post-hoc reanalysis and new
   prospectively specified follow-ups.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from Submission1_Code_Phase2 import common as C

HS = C.ORIGINAL_AUDIT / "advice" / "human_study"
EP = C.ORIGINAL_AUDIT / "evidence_panel"

WORDING_FIXES = {
    "statistically inseparable": "no improvement detected in these comparisons",
    "inseparable from": "not distinguishable in these comparisons from",
    "offers nothing beyond": "showed no improvement over",
    "does not beat": "was not found to improve on",
}


def both_p3_endpoints() -> Dict:
    analysis = json.loads((HS / "ratings_analysis.json").read_text(encoding="utf-8"))
    rows = []
    for e in analysis["paired_endpoints"]:
        rows.append({"tier": e["tier"], "n_question_clusters": e["n_question_clusters"],
                     "endpoint": "unsupported_substantive_change", "graft_minus_frozen": e["unsupported_change_graft_minus_frozen"],
                     "ci_lower": e["unsupported_ci"][0], "ci_upper": e["unsupported_ci"][1],
                     "predeclared": "co-primary"})
        rows.append({"tier": e["tier"], "n_question_clusters": e["n_question_clusters"],
                     "endpoint": "both_answers_meet_minimum", "graft_minus_frozen": e["both_meet_minimum_graft_minus_frozen"],
                     "ci_lower": e["both_minimum_ci"][0], "ci_upper": e["both_minimum_ci"][1],
                     "predeclared": "co-primary"})
    return {"rows": rows,
            "note": ("Both endpoints were predeclared as co-primary in the local protocol; the manuscript reported only "
                     "unsupported change. 'Meets minimum' permits mixed or unverifiable answers when usefulness is "
                     "adequate, so it is not a correctness measure; correctness is reported separately.")}


def agreement_table() -> List[Dict]:
    analysis = json.loads((HS / "ratings_analysis.json").read_text(encoding="utf-8"))
    rows = []
    for field, a in sorted(analysis["agreement"].items()):
        raw, k, n = a["raw_agreement"], a["cohen_kappa"], a["n"]
        rows.append({"dimension": field, "n": n, "raw_agreement": raw, "cohen_kappa": k, "kappa_type": a["kappa_type"],
                     "reading": ("near-unanimous category: kappa is deflated and raw agreement is the informative figure"
                                 if raw >= 0.95 and k < 0.3 else
                                 "moderate agreement, as expected for a stylistic judgement" if k < 0.6 else
                                 "substantial agreement")})
    return rows


def per_answer_quality() -> List[Dict]:
    """Supported/usefulness differences with question-cluster intervals, not only unsupported change."""
    mp = HS / "KEEP_FROM_RATERS" / "assessment_mapping.csv"
    rp = HS / "ratings_independent.csv"
    if not (mp.exists() and rp.exists()):
        return []
    mapping = {r["assessment_id"]: r for r in csv.DictReader(mp.open(encoding="utf-8"))}
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(rp.open(encoding="utf-8")):
        m = mapping.get(r["assessment_id"])
        if not m:
            continue
        key = (m["tier"], m["method"], m["pair_id"])
        for side in ("left", "right"):
            per[key]["supported"].append(r[f"{side}_correctness"] == "supported")
            per[key]["useful2"].append(r[f"{side}_usefulness_0_2"] == "2")
            per[key]["usefulness"].append(float(r[f"{side}_usefulness_0_2"]))
    out = []
    for tier in sorted({k[0] for k in per}):
        for metric in ("supported", "useful2", "usefulness"):
            g = {k[2]: float(np.mean(v[metric])) for k, v in per.items() if k[0] == tier and k[1] == "graft_proposed"}
            f = {k[2]: float(np.mean(v[metric])) for k, v in per.items() if k[0] == tier and k[1] == "frozen_base"}
            keys = sorted(set(g) & set(f))
            if len(keys) < 2:
                continue
            d = np.array([g[k] - f[k] for k in keys])
            rng = np.random.default_rng(20260924)
            boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
            out.append({"tier": tier, "metric": metric, "n_question_clusters": len(keys),
                        "graft_minus_frozen": round(float(d.mean()), 4),
                        "ci_lower": round(float(np.percentile(boots, 2.5)), 4),
                        "ci_upper": round(float(np.percentile(boots, 97.5)), 4),
                        "reading": "no improvement detected in these comparisons" if
                        np.percentile(boots, 2.5) <= 0 <= np.percentile(boots, 97.5) else "difference detected"})
    return out


def evidence_panel_correction() -> Dict:
    p = EP / "evidence_panel_predictions.jsonl"
    if not p.exists():
        return {}
    rows = [r for r in C.read_jsonl(p) if r["phase"] == "main"]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["tier"], r["method"], r["bundle_id"])][r["condition"]] = r
    out = {}
    for tier, method in sorted({(k[0], k[1]) for k in by}):
        grp = [v for k, v in by.items() if k[0] == tier and k[1] == method]
        hypo = sum(1 for g in grp if g.get("synthetic_evidence", {}).get("pred_canonical") == g.get("synthetic_evidence", {}).get("expected"))
        ver = sum(1 for g in grp if g.get("verified_evidence", {}).get("pred_canonical") == g.get("verified_evidence", {}).get("expected"))
        joint = sum(1 for g in grp if g.get("verified_evidence", {}).get("pred_canonical") == g.get("verified_evidence", {}).get("expected")
                    and g.get("synthetic_evidence", {}).get("pred_canonical") == g.get("synthetic_evidence", {}).get("expected"))
        out[f"{tier}|{method}"] = {"bundles": len(grp), "hypothetical_only": f"{hypo}/{len(grp)}",
                                   "verified_only": f"{ver}/{len(grp)}", "joint": f"{joint}/{len(grp)}"}
    out["_correction"] = ("The abstract said the adapted Qwen model was perfect on the joint endpoint. It was "
                          "47/48 jointly; 48/48 applies to the hypothetical condition alone. The original P4 also "
                          "changed table framing and values together, so it cannot isolate numerical sensitivity; "
                          "R2 is the controlled follow-up.")
    return out


def per_axis_counts() -> List[Dict]:
    items = C.read_jsonl(C.DATASET_FACTS)
    rows = []
    for axis in sorted({i["axis"] for i in items}):
        for cond in ("equal", "diff"):
            sub = [i for i in items if i["axis"] == axis and i["condition"] == cond]
            cells = {i["source_cell"] for i in sub}
            states = {C.state_of(i["source_cell"]) for i in sub}
            rows.append({"axis": axis, "condition": cond, "items": len(sub), "source_cells": len(cells),
                         "states": len(states), "parent_tables": len({C.parent_table(i["source_cell"]) for i in sub}),
                         "small_cluster_warning": "fewer than 10 source cells" if len(cells) < 10 else ""})
    return rows


def wording_audit(tex: Path) -> List[Dict]:
    if not tex.exists():
        return []
    text = tex.read_text(encoding="utf-8")
    found = []
    for bad, good in WORDING_FIXES.items():
        n = text.lower().count(bad.lower())
        if n:
            found.append({"phrase": bad, "occurrences": n, "replace_with": good,
                          "reason": "non-significance is not equivalence"})
    return found


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=Path, default=C.CODES_ROOT.parent / "Submission1" / "Elsevier_CEA_Finetune_Agrifair.tex")
    a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "reanalysis")
    p3 = both_p3_endpoints(); C.write_csv(out / "p3_both_primary_endpoints.csv", p3["rows"])
    C.write_csv(out / "p3_agreement.csv", agreement_table())
    q = per_answer_quality()
    if q:
        C.write_csv(out / "p3_quality_differences.csv", q)
    C.write_json(out / "evidence_panel_correction.json", evidence_panel_correction())
    C.write_csv(out / "per_axis_counts.csv", per_axis_counts())
    wa = wording_audit(a.tex)
    C.write_csv(out / "wording_audit.csv", wa) if wa else None
    C.write_json(out / "analysis_manifest.json", {
        "version": "phase2-r4",
        "original_results": "results_final_audit_20260912 (unchanged; never overwritten)",
        "post_hoc_reanalysis": ["p3_both_primary_endpoints", "p3_quality_differences", "p3_agreement",
                                "per_axis_counts", "evidence_panel_correction"],
        "new_prospective_studies": ["r1_fresh_panel", "r2_evidence", "r3_advice"],
        "p3_note": p3["note"],
        "statistical_families_frozen_before_new_runs": True})
    print(f"[r4_reanalysis] both P3 endpoints, agreement, quality intervals, counts and corrections -> {out}")
    for r in p3["rows"]:
        print(f"  {r['tier']:16s} {r['endpoint']:32s} {r['graft_minus_frozen']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]")
    if wa:
        print(f"  wording audit: {sum(x['occurrences'] for x in wa)} phrase(s) to soften in the manuscript")


if __name__ == "__main__":
    main()
