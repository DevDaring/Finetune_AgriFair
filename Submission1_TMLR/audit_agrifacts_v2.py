"""Second pass of the AgriFacts audit, with matching that tolerates the released resource's phrasing.

    python -m Submission1_TMLR.audit_agrifacts_v2

The first pass (audit_agrifacts) used the corrected-wording checker, which knows only its own sentence
patterns. On the released questions it flagged 614 items, and reading them showed most flags to be
phrasing mismatches: plural group names ("Scheduled Castes"), "across all of India" for "all-India",
and a gender cell format (<group>/femaleShare/<metric>/<sizeA>vs<sizeB>) that the parser does not
model. This pass checks only the semantic content -- population restriction, place, compared
entities -- with tolerant patterns, and writes every remaining flag out for manual reading.
"""
from __future__ import annotations

import collections
import json
import re

from Submission1_Code_Phase2 import common as C

OUT = C.CODES_ROOT / "results_submission1_tmlr" / "agrifacts_audit"
GROUP = {"SC": r"scheduled[\s-]+castes?|\bsc\b|\bscs\b", "ST": r"scheduled[\s-]+tribes?|\bst\b|\bsts\b",
         "Others": r"other social groups?|others?\b|other groups?"}
SIZE = {"Marginal": r"marginal", "Small": r"small", "Semi-medium": r"semi[\s-]*medium",
        "Medium": r"(?<!semi-)(?<!semi )medium", "Large": r"\blarge"}
INDIA = r"all[\s-]*india|all of india|across india|\bindia\b|national"


def norm(s: str) -> str:
    return " ".join(s.lower().replace("—", " ").replace("–", " ").split())


def parse_cell(cell: str):
    _, table, tail = cell.split(" ", 2)
    seg = tail.split("/")
    return table, seg


def checks(item: dict) -> list:
    q = norm(item["question"])
    table, seg = parse_cell(item["source_cell"])
    flags = []
    if table == "T2-4":                                  # <state>/<stratum>/<metric>/<A>vs<B>
        state, stratum, metric, cmp_ = seg
        if norm(state.replace("&", "and")) not in q.replace("&", "and"):
            flags.append(f"state {state!r} not named")
        a, b = cmp_.split("vs")
        if item["axis"] == "landholding":                # stratum = social group, entities = size classes
            if stratum not in ("All", "All Classes") and not re.search(GROUP[stratum], q):
                flags.append(f"social group {stratum!r} dropped")
            for s in (a, b):
                if not re.search(SIZE[s], q):
                    flags.append(f"size class {s!r} missing")
        else:                                            # stratum = size class, entities = social groups
            if stratum not in ("All", "All Classes") and not re.search(SIZE[stratum], q):
                flags.append(f"size class {stratum!r} dropped")
            for g in (a, b):
                if not re.search(GROUP[g], q):
                    flags.append(f"social group {g!r} missing")
    else:                                                # T14-16, all-India gender tables
        if not re.search(INDIA, q):
            flags.append("all-India not stated")
        group, second, metric, cmp_ = seg
        if group not in ("All", "All Classes") and not re.search(GROUP[group], q):
            flags.append(f"social group {group!r} dropped")
        if second == "femaleShare":                      # women's share compared across two size classes
            a, b = cmp_.split("vs")
            for s in (a, b):
                if not re.search(SIZE[s], q):
                    flags.append(f"size class {s!r} missing")
            if not re.search(r"wom[ae]n|female", q):
                flags.append("women's share not stated")
        else:                                            # men vs women within one size class
            if second not in ("All", "All Classes") and not re.search(SIZE[second], q):
                flags.append(f"size class {second!r} dropped")
            if not (re.search(r"\bm[ae]n\b|male", q) and re.search(r"wom[ae]n|female", q)):
                flags.append("men and women not both named")
    return flags


def main() -> None:
    items = list(C.read_jsonl(C.DATASET_FACTS))
    out = []
    for it in items:
        f = checks(it)
        if f:
            out.append({"id": it["id"], "axis": it["axis"], "source_cell": it["source_cell"], "flags": f,
                        "question": it["question"]})
    C.write_jsonl(OUT / "semantic_flags_v2.jsonl", out)
    by = collections.Counter(x.split(" '")[0] if "'" in x else x for o in out for x in o["flags"])
    summary = {"items": len(items), "items_flagged": len(out), "flags": dict(by.most_common()),
               "by_axis": dict(collections.Counter(o["axis"] for o in out))}
    C.write_json(OUT / "summary_v2.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
