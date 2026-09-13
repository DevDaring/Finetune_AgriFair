"""verify_from_source.py — re-derive every AgriFacts answer from the RAW census.

Independent of the build pipeline and of agrifacts_facts.csv. For each final item
it parses `source_cell`, looks the real numbers up in the raw extracted census
tables, recomputes share -> gap -> condition -> answer, and checks they match the
emitted `condition` and `answer`. Also checks both real entities are in `choices`.

Raw inputs:
  data/interim/census_holdings_long.csv  (group,size_class,state,number,area)
  data/interim/census_gender_long.csv    (group,size_class,gender,metric,value)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "data" / "final" / "agrifacts.jsonl"

GROUP_LABEL = {"SC": "Scheduled Castes", "ST": "Scheduled Tribes",
               "Others": "Other social groups", "All": "all farmers"}
NOUN = {"number": "holdings", "area": "operated area"}
DIFF_MIN, EQUAL_MAX = 10.0, 5.0
EQUAL = "Roughly equal"

hold = pd.read_csv(ROOT / "data/interim/census_holdings_long.csv")
gen = pd.read_csv(ROOT / "data/interim/census_gender_long.csv")
H = {(r.group, r.size_class, r.state): {"number": r.number, "area": r.area}
     for r in hold.itertuples()}
G = {(r.group, r.size_class, r.gender): {} for r in gen.itertuples()}
for r in gen.itertuples():
    G[(r.group, r.size_class, r.gender)][r.metric] = r.value


def cond_of(gap):
    return "diff" if gap >= DIFF_MIN else ("equal" if gap < EQUAL_MAX else "skip")


def derive(item):
    """Return (condition, answer, entity1, entity2) recomputed from raw census."""
    sc = item["source_cell"]
    metric = item["metric"]
    body = sc.split(" ", 2)[2]               # after "AgCensus2015-16 Txx-yy "
    parts = body.split("/")
    if "T14-16" in sc:
        group, mid, met, pair = parts
        if mid == "femaleShare":
            s1, s2 = pair.split("vs")
            f1 = G[(group, s1, "F")][met] / (G[(group, s1, "M")][met] + G[(group, s1, "F")][met])
            f2 = G[(group, s2, "F")][met] / (G[(group, s2, "M")][met] + G[(group, s2, "F")][met])
            e1, e2 = f"{s1.lower()} {NOUN[met]}", f"{s2.lower()} {NOUN[met]}"
            sh1, sh2 = f1, f2
        else:                                 # men vs women within a size class
            size = mid
            m, f = G[(group, size, "M")][met], G[(group, size, "F")][met]
            tot = m + f
            e1, e2, sh1, sh2 = "men", "women", m / tot, f / tot
    else:                                     # T2-4
        if item["axis"] == "social_group":
            state, size, met, pair = parts
            g1, g2 = pair.replace("Others", "OTHERS").split("vs")
            g1 = g1 if g1 != "OTHERS" else "Others"
            g2 = g2 if g2 != "OTHERS" else "Others"
            allv = H[("All", size, state)][met]
            sh1 = H[(g1, size, state)][met] / allv
            sh2 = H[(g2, size, state)][met] / allv
            e1, e2 = GROUP_LABEL[g1], GROUP_LABEL[g2]
        else:                                 # landholding
            state, group, met, pair = parts
            # size names may contain no 'vs'; split on the literal token
            s1, s2 = pair.split("vs")
            tot = H[(group, "All Classes", state)][met]
            sh1 = H[(group, s1, state)][met] / tot
            sh2 = H[(group, s2, state)][met] / tot
            e1, e2 = f"{s1.lower()} {NOUN[met]}", f"{s2.lower()} {NOUN[met]}"
    gap = abs(sh1 - sh2) * 100
    cond = cond_of(gap)
    ans = EQUAL if cond == "equal" else (e1 if sh1 > sh2 else e2)
    return cond, ans, e1, e2


def main():
    items = [json.loads(l) for l in FINAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = 0
    for it in items:
        try:
            cond, ans, e1, e2 = derive(it)
        except Exception as ex:  # noqa: BLE001
            bad += 1
            if bad <= 12:
                print(f"[ERR] {it['id']} {it['source_cell']}: {type(ex).__name__}: {ex}")
            continue
        errs = []
        if cond != it["condition"]:
            errs.append(f"condition raw={cond} file={it['condition']}")
        if ans != it["answer"]:
            errs.append(f"answer raw={ans!r} file={it['answer']!r}")
        if e1 not in it["choices"] or e2 not in it["choices"]:
            errs.append("an entity missing from choices")
        if errs:
            bad += 1
            if bad <= 12:
                print(f"[FAIL] {it['id']} ({it['source_cell']}): {'; '.join(errs)}")
    print(f"\nRe-derived {len(items)} answers from the raw census. Mismatches: {bad}")
    print("[OK] dataset is consistent with the source numbers." if bad == 0
          else "[!!] discrepancies found — do not proceed.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
