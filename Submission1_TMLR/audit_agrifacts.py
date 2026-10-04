"""Run the ten source checks on all 2,000 released AgriFacts questions (reviewer question:
"did you check your own benchmark?").

    python -m Submission1_TMLR.audit_agrifacts

Each question is joined to its validated construction record by source cell, parsed into a source
record, and checked. Failures are split into:
  semantic -- the question describes another population, place or pair of entities, or two source
              rows share one question, or the gold answer does not follow from the shares;
  policy   -- the question omits the decision rule, the total or the exact measure phrase, which the
              original resource did not state by design (the corrected wording states them).
The checker was validated on its own sentence patterns only, so every semantic flag is written out
with its question text for manual reading.
"""
from __future__ import annotations

import ast
import collections
import json

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2 import r1_fresh_panel as R
from Submission1_DKE_Repair import source_schema as S
from Submission1_DKE_Repair.prompt_checks import check_one

OUT = C.CODES_ROOT / "results_submission1_tmlr" / "agrifacts_audit"
SEMANTIC = {"P1_social_group_dropped", "P1_false_all_farmers", "P1_invented_social_group", "P1_population_negated",
            "P2_size_class_dropped", "P3_geography_missing", "P6_entity_missing", "P9_question_collides",
            "P10_gold_mismatch", "P10_rule_violation"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = {r["source_cell"]: r for r in R.load_validated()}
    items = list(C.read_jsonl(C.DATASET_FACTS))
    failures, unmatched, per_item = [], [], {}
    seen_text = {}
    for it in items:
        rec = records.get(it["source_cell"])
        if rec is None:
            unmatched.append(it["id"]); continue
        s1, s2 = float(rec["share_1"]) * 100, float(rec["share_2"]) * 100
        gold_rec = S.EQUAL_CHOICE if rec["condition"] == "equal" else (rec["entity_1"] if s1 > s2 else rec["entity_2"])
        spec = S.parse({"fresh_id": it["id"], "parent_table": C.parent_table(it["source_cell"]), "axis": it["axis"],
                        "group1": rec["entity_1"], "group2": rec["entity_2"], "share1_pct": s1, "share2_pct": s2,
                        "gap_pp": abs(s1 - s2), "condition": rec["condition"], "gold_choice_text": gold_rec,
                        "source_cell": it["source_cell"]})
        f = [x for x in check_one(spec, {"question": it["question"]}) if x["rule"] != "P8_wordings_identical"]
        # released answer must equal the answer the record implies
        if str(it["answer"]).strip().lower() != str(gold_rec).strip().lower():
            f.append({"rule": "P10_gold_mismatch", "detail": f"released {it['answer']!r} vs record {gold_rec!r}"})
        if it["condition"] != rec["condition"]:
            f.append({"rule": "P10_rule_violation", "detail": f"released condition {it['condition']} vs record {rec['condition']}"})
        q = " ".join(it["question"].lower().split())
        if q in seen_text and seen_text[q] != it["source_cell"]:
            f.append({"rule": "P9_question_collides", "detail": f"same text as {seen_text[q]}"})
        seen_text.setdefault(q, it["source_cell"])
        per_item[it["id"]] = f
        for x in f:
            failures.append({"id": it["id"], "axis": it["axis"], "rule": x["rule"],
                             "kind": "semantic" if x["rule"] in SEMANTIC else "policy",
                             "detail": x.get("detail", ""), "source_cell": it["source_cell"],
                             "question": it["question"]})
    by_rule = collections.Counter(f["rule"] for f in failures)
    sem_items = sorted({f["id"] for f in failures if f["kind"] == "semantic"})
    summary = {"items": len(items), "unmatched_to_records": len(unmatched),
               "items_with_semantic_flags": len(sem_items),
               "items_with_any_flag": sum(1 for v in per_item.values() if v),
               "flags_by_rule": dict(by_rule.most_common()),
               "semantic_flags_by_axis_rule": dict(collections.Counter(
                   (f["axis"], f["rule"]) for f in failures if f["kind"] == "semantic").most_common())}
    C.write_jsonl(OUT / "semantic_flags.jsonl", [f for f in failures if f["kind"] == "semantic"])
    C.write_json(OUT / "summary.json", {**summary, "semantic_flags_by_axis_rule":
                                        {f"{a}/{r}": n for (a, r), n in summary["semantic_flags_by_axis_rule"].items()}})
    print(json.dumps({k: v for k, v in summary.items() if k != "semantic_flags_by_axis_rule"}, indent=1))
    print({f"{a}/{r}": n for (a, r), n in summary["semantic_flags_by_axis_rule"].items()})


if __name__ == "__main__":
    main()
