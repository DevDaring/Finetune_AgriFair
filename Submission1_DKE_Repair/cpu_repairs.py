"""M2, M3, M5: the CPU-only corrections that need no new generation.

    python -m Submission1_DKE_Repair.cpu_repairs

M2  the split audit pooled training and validation and called the result "training".
M3  the R2 concentration statistic counted raw answer strings, which fragments the two
    directional categories across many group names while pooling every equality answer into one.
M5  the verbosity conditions were an instruction, not achieved length matching; report compliance.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"
EQUAL = "Roughly equal"


# ------------------------------------------------------------------ M2: split audit
def split_audit() -> Dict:
    """Recompute template-family overlap with training and validation kept apart."""
    from Submission1_Code_Phase2.common import canonical_template, template_id

    def ids(name: str) -> set:
        p = C.CODES_ROOT / "data" / name
        return {json.loads(l)["id"] for l in p.open(encoding="utf-8")} if p.exists() else set()

    train_ids, val_ids, test_ids = (ids("train_instances.jsonl"), ids("validation_instances.jsonl"),
                                    ids("test_instances_frozen.jsonl"))
    items = [json.loads(l) for l in
             (C.CODES_ROOT / "data" / "templated_all_instances.jsonl").open(encoding="utf-8")]

    fam: Dict[str, set] = {"train": set(), "validation": set(), "test": set()}
    per_item = {}
    for it in items:
        tid = template_id(canonical_template(it.get("question") or it.get("text") or ""))
        per_item[it["id"]] = tid
        if it["id"] in train_ids:
            fam["train"].add(tid)
        elif it["id"] in val_ids:
            fam["validation"].add(tid)
        elif it["id"] in test_ids:
            fam["test"].add(tid)

    shared_train = fam["test"] & fam["train"]
    shared_pooled = fam["test"] & (fam["train"] | fam["validation"])
    test_items = [i for i in items if i["id"] in test_ids]
    hit_train = sum(1 for i in test_items if per_item[i["id"]] in shared_train)
    hit_pooled = sum(1 for i in test_items if per_item[i["id"]] in shared_pooled)

    return {
        "counts": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "families": {"train_only": len(fam["train"]),
                     "train_plus_validation": len(fam["train"] | fam["validation"]),
                     "validation": len(fam["validation"]), "test": len(fam["test"]),
                     "all": len(set(per_item.values()))},
        "shared_with_test": {"train_only": len(shared_train), "pooled": len(shared_pooled)},
        "test_items_sharing_a_family": {
            "train_only": hit_train,
            "train_only_share": round(hit_train / len(test_items), 4) if test_items else None,
            "pooled": hit_pooled,
            "pooled_share": round(hit_pooled / len(test_items), 4) if test_items else None},
        "note": ("The published figure pooled training and validation and labelled the result "
                 "'training'. Train-only is the defensible number; both are reported."),
    }


def r1_family_match() -> Dict:
    """M2: do the R1 wordings match any original-dataset template family at all?"""
    from Submission1_Code_Phase2.common import canonical_template, template_id
    items = [json.loads(l) for l in
             (C.CODES_ROOT / "data" / "templated_all_instances.jsonl").open(encoding="utf-8")]
    known = {template_id(canonical_template(i.get("question") or i.get("text") or "")) for i in items}
    panel = [json.loads(l) for l in
             (C.CODES_ROOT / OUT_DIR / "r1_corrected_panel.jsonl").open(encoding="utf-8")]
    hit = {"wording_a": 0, "wording_b": 0}
    for p in panel:
        for k in hit:
            if template_id(canonical_template(p[k])) in known:
                hit[k] += 1
    return {"n_panel": len(panel), "matches_an_original_family": hit,
            "reading": ("Neither wording reproduces a template family seen in the original data "
                        "under the paper's own identifier function. The experiment is therefore a "
                        "wording-A versus wording-B sensitivity test, not a demonstration of "
                        "familiarity with a seen training template.")}


# ------------------------------------------------------------------ M3: semantic response counts
def _r2_rows() -> List[Dict]:
    seen, out = set(), []
    for name in ("main_predictions.jsonl", "pilot_predictions.jsonl"):
        p = C.CODES_ROOT / "results_submission1_phase2" / "predictions" / name
        if not p.exists():
            continue
        for r in C.read_jsonl(p):
            if r.get("study") != "r2_main":
                continue
            k = (r.get("prompt_id"), r.get("system"))
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    return out


def semantic_distribution() -> List[Dict]:
    """Map every answer to first group / second group / roughly equal BEFORE counting."""
    by = collections.defaultdict(collections.Counter)
    for r in _r2_rows():
        pick = r.get("picked_choice")
        if pick is None:
            cat = "unparsed"
        elif str(pick).strip().lower().startswith("roughly equal"):
            cat = "equal"
        elif pick == r.get("group1"):
            cat = "first"
        elif pick == r.get("group2"):
            cat = "second"
        else:
            cat = "unmapped"
        by[r["system"]][cat] += 1
    rows = []
    for sysname, c in sorted(by.items()):
        n = sum(c.values())
        top = max(c["first"], c["second"], c["equal"])
        rows.append({"system": sysname, "n": n, "first_group": c["first"],
                     "second_group": c["second"], "roughly_equal": c["equal"],
                     "unmapped": c["unmapped"] + c["unparsed"],
                     "max_semantic_category_share": round(top / n, 4) if n else None})
    return rows


# ------------------------------------------------------------------ M5: verbosity compliance
_BANDS = {"concise": (80, 100), "standard": (160, 180)}


def verbosity_compliance() -> List[Dict]:
    seen, rows = set(), []
    for name in ("main_predictions.jsonl", "pilot_predictions.jsonl"):
        p = C.CODES_ROOT / "results_submission1_phase2" / "predictions" / name
        if not p.exists():
            continue
        for r in C.read_jsonl(p):
            if r.get("study") != "r3_advice":
                continue
            k = (r.get("prompt_id"), r.get("system"))
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["system"], r["verbosity"])].append(len(str(r.get("raw_output", "")).split()))
    out = []
    for (sysname, verb), lens in sorted(by.items()):
        lo, hi = _BANDS[verb]
        inside = sum(1 for x in lens if lo <= x <= hi)
        out.append({"system": sysname, "verbosity": verb, "requested_band": f"{lo}-{hi}",
                    "n": len(lens), "mean_words": round(statistics.mean(lens), 1),
                    "median_words": int(statistics.median(lens)),
                    "min_words": min(lens), "max_words": max(lens),
                    "in_requested_band": inside,
                    "compliance_rate": round(inside / len(lens), 4)})
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    m2 = {"split_audit": split_audit(), "r1_wording_family_match": r1_family_match()}
    C.write_json(out / "split_and_template_audit.json", m2)
    with (out / "split_and_template_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "train_only", "train_plus_validation"])
        f, s, t = m2["split_audit"]["families"], m2["split_audit"]["shared_with_test"], \
            m2["split_audit"]["test_items_sharing_a_family"]
        w.writerow(["template families on the non-test side", f["train_only"], f["train_plus_validation"]])
        w.writerow(["families shared with test", s["train_only"], s["pooled"]])
        w.writerow(["test items sharing such a family", t["train_only"], t["pooled"]])
        w.writerow(["share of the 749 test items", t["train_only_share"], t["pooled_share"]])

    m3 = semantic_distribution()
    C.write_csv(out / "r2_semantic_response_distribution.csv", m3)

    m5 = verbosity_compliance()
    C.write_csv(out / "advice_length_compliance.csv", m5)

    print(json.dumps({"M2_split": m2["split_audit"]["test_items_sharing_a_family"],
                      "M2_families": m2["split_audit"]["families"],
                      "M2_r1_family_match": m2["r1_wording_family_match"]["matches_an_original_family"],
                      "M3_semantic": m3,
                      "M5_compliance": [{k: r[k] for k in
                                         ("system", "verbosity", "mean_words", "compliance_rate")}
                                        for r in m5]}, indent=1))


if __name__ == "__main__":
    main()
