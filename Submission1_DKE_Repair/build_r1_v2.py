"""M1 / E1: rebuild the R1 panel from the typed source schema and freeze it.

Version 1 is never modified. This writes a version-2 panel beside it, and records the check
result for BOTH versions so that the repair is evidenced rather than asserted.

    python -m Submission1_DKE_Repair.build_r1_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_DKE_Repair import source_schema as S
from Submission1_DKE_Repair.prompt_checks import check_panel

V1_PANEL = "results_submission1_phase2/human_review/fresh_panel_verified.jsonl"
OUT_DIR = "results_submission1_dke_repair_v2"


def load_v1() -> List[Dict]:
    return [json.loads(l) for l in (C.CODES_ROOT / V1_PANEL).open(encoding="utf-8")]


def build(records: List[Dict]):
    specs = [S.parse(r) for r in records]
    rendered = {s.fresh_id: S.render(s) for s in specs}
    return specs, rendered


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    records = load_v1()

    # ---- version 1, as it was generated, through the same checker -------------------------
    v1_rendered = {r["fresh_id"]: {"wording_a": r["wordings"]["original_family"],
                                   "wording_b": r["wordings"]["new_family"]} for r in records}
    v1_specs = [S.parse(r) for r in records]
    v1 = check_panel(v1_specs, v1_rendered)

    # ---- version 2, rendered from the typed schema ----------------------------------------
    specs, rendered = build(records)
    v2 = check_panel(specs, rendered)

    panel = []
    for spec in specs:
        w = rendered[spec.fresh_id]
        panel.append({**spec.as_dict(), "population": spec.population(),
                      "denominator_text": spec.denominator(),
                      "wording_a": w["wording_a"], "wording_b": w["wording_b"],
                      "choices": [spec.entity1, spec.entity2, S.EQUAL_CHOICE],
                      "gold_choice_text": spec.gold_entity})
    C.write_jsonl(out / "r1_corrected_panel.jsonl", panel)

    report = {
        "version_1": {"n_failures": v1["n_failures"], "by_rule": v1["failures_by_rule"],
                      "verdict": "the generated panel does not describe its own source rows"},
        "version_2": {"n_failures": v2["n_failures"], "by_rule": v2["failures_by_rule"],
                      "verdict": "clean" if v2["passed"] else "still failing"},
        "n_comparisons": len(panel),
        "by_axis": {a: sum(1 for p in panel if p["axis"] == a)
                    for a in sorted({p["axis"] for p in panel})},
        "by_condition": {c: sum(1 for p in panel if p["condition"] == c)
                         for c in sorted({p["condition"] for p in panel})},
        "restrictions_restored": sum(1 for p in panel if p["social_group"] or p["size_class"]),
        "note": ("Version 1 is preserved unchanged. Version 2 is rendered from a typed schema in "
                 "which a restriction can only be absent from the question if it is absent from "
                 "the source row."),
    }
    C.write_json(out / "source_schema_and_prompt_checks.json",
                 {**report, "version_1_failures": v1["failures"]})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
