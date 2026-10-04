"""The planted-error test of the source checks, re-run on all 155 verified comparisons
(34 original + 121 extended). Same corruptions, same valid controls, same checker.

    python -m Submission1_TMLR.checker_extended
"""
from __future__ import annotations

import collections
import json

from Submission1_Code_Phase2 import common as C
from Submission1_DKE_Repair import source_schema as S
from Submission1_DKE_Repair.checker_challenge import CONTROLS, MUTATIONS
from Submission1_DKE_Repair.prompt_checks import check_one

PANELS = {"original 34": "results_submission1_dke_repair_v2/r1_corrected_panel.jsonl",
          "extended 121": "results_submission1_tmlr/extended_panel.jsonl"}
OUT = C.CODES_ROOT / "results_submission1_tmlr" / "checker_challenge_155"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, cases = [], []
    for panel_name, rel in PANELS.items():
        for p in C.read_jsonl(C.CODES_ROOT / rel):
            spec = S.SourceSpec(**{k: p[k] for k in S.SourceSpec.__dataclass_fields__})
            for wname in ("wording_a", "wording_b"):
                for label, fn, should in MUTATIONS + CONTROLS:
                    mutated = fn(spec, p[wname])
                    if mutated is None or mutated == p[wname]:
                        continue
                    fails = check_one(spec, {wname: mutated})
                    cases.append({"panel": panel_name, "id": spec.fresh_id, "wording": wname, "transform": label,
                                  "should_be_flagged": should, "was_flagged": bool(fails)})
    for scope in ("original 34", "extended 121", "all 155"):
        sel = [c for c in cases if scope == "all 155" or c["panel"] == scope]
        by = collections.defaultdict(lambda: [0, 0, None])
        for c in sel:
            e = by[c["transform"]]; e[0] += 1; e[1] += c["was_flagged"]; e[2] = c["should_be_flagged"]
        for t, (n, f, should) in by.items():
            rows.append({"scope": scope, "transform": t, "kind": "corruption" if should else "valid control",
                         "n_cases": n, "share_flagged": round(f / n, 4)})
    summ = {}
    for scope in ("original 34", "extended 121", "all 155"):
        sel = [c for c in cases if scope == "all 155" or c["panel"] == scope]
        cor = [c for c in sel if c["should_be_flagged"]]; ctl = [c for c in sel if not c["should_be_flagged"]]
        summ[scope] = {"corrupted": len(cor), "caught": sum(c["was_flagged"] for c in cor),
                       "detection_rate": round(sum(c["was_flagged"] for c in cor) / len(cor), 4),
                       "controls": len(ctl), "false_alarms": sum(c["was_flagged"] for c in ctl)}
    C.write_csv(OUT / "by_transform.csv", rows)
    C.write_json(OUT / "summary.json", summ)
    print(json.dumps(summ, indent=1))
    for r in rows:
        if r["scope"] == "all 155": print(f"  {r['transform']:55s} {r['kind']:13s} n={r['n_cases']:4d} flagged={r['share_flagged']}")


if __name__ == "__main__":
    main()
