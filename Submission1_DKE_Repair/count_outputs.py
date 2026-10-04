"""Audit of every generated model output behind the manuscript, by study.

    python -m Submission1_DKE_Repair.count_outputs

The abstract and Methods report one total. This file shows what it is made of, so each count is
traceable: what the unit is (a model output, not a prompt or a rated pair), which run produced it,
and whether the study it belongs to is still reported or was withdrawn.
"""
from __future__ import annotations

import collections
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"

# study key -> (description, status in the manuscript)
PHASE2 = {"r2_main": ("numerical set, 48 bundles x 3 relations x 2 wordings", "reported"),
          "r2_diagnostic": ("altered tables, 12 bundles x 3 alterations", "reported"),
          "r3_advice": ("advice answers; two per rated pair", "reported"),
          "r1_fresh": ("original wording panel, before the repair", "withdrawn")}
V2 = {"E1": ("corrected wording panel, 34 comparisons x 2 wordings", "reported"),
      "E3": ("unchanged-table control, 12 bundles", "reported"),
      "E4": ("neutral wording, 72 prompts", "reported"),
      "E5": ("permuted option order, 34 comparisons x 2 wordings", "reported")}


def phase2_rows() -> List[Dict]:
    """Main and pilot predictions, pooled and de-duplicated on (prompt, system) as every
    phase-2 analysis does."""
    seen = {}
    for name in ("pilot_predictions.jsonl", "main_predictions.jsonl"):
        path = C.CODES_ROOT / "results_submission1_phase2" / "predictions" / name
        for r in C.read_jsonl(path):
            seen[(r.get("prompt_id"), r.get("system"))] = r
    return list(seen.values())


def main() -> None:
    out = C.CODES_ROOT / OUT_DIR
    p2 = collections.Counter(r.get("study") for r in phase2_rows())
    v2 = collections.Counter(r.get("experiment") for r in
                             C.read_jsonl(out / "v2_predictions.jsonl"))
    rows = []
    for key, (desc, status) in PHASE2.items():
        rows.append({"run": "phase2", "study": key, "outputs": p2.get(key, 0),
                     "unit": "model output", "description": desc, "status": status})
    for key, (desc, status) in V2.items():
        rows.append({"run": "repair_v2", "study": key, "outputs": v2.get(key, 0),
                     "unit": "model output", "description": desc, "status": status})
    unknown = (set(p2) - set(PHASE2)) | (set(v2) - set(V2))
    if unknown:
        raise SystemExit("unrecognised study keys: %s" % sorted(unknown))
    totals = {"phase2": sum(p2.values()), "repair_v2": sum(v2.values())}
    totals["all"] = totals["phase2"] + totals["repair_v2"]
    totals["withdrawn"] = sum(r["outputs"] for r in rows if r["status"] == "withdrawn")
    totals["advice_rated_pairs"] = p2.get("r3_advice", 0) // 2
    C.write_csv(out / "output_count_audit.csv", rows)
    C.write_json(out / "output_count_audit.json", {"per_study": rows, "totals": totals})
    print(totals)


if __name__ == "__main__":
    main()
