"""verify_agrifacts.py — independent per-instance audit of every final item.

Re-checks all 2000 AgriFacts items against the deterministic census-derived facts
(agrifacts_facts.csv), which were themselves validated against the raw census
counts. For EACH item it confirms:
  * the item traces to a real fact (paraphrase_of -> fact_id)
  * emitted condition == fact condition (gap band)
  * emitted answer == the census-implied answer
        diff  -> the larger-share entity
        equal -> "Roughly equal"
  * the answer sits in the 3 choices; both compared entities survive in the
    (possibly paraphrased) question stem (answer-invariance)
Any single mismatch aborts with the offending id. Silence = every instance clean.
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
FACTS = ROOT / "data" / "interim" / "agrifacts_facts.csv"

facts = pd.read_csv(FACTS).set_index("fact_id")
items = [json.loads(l) for l in FINAL.read_text(encoding="utf-8").splitlines() if l.strip()]

bad = 0
for it in items:
    fid = it["paraphrase_of"]
    if fid not in facts.index:
        print(f"[FAIL] {it['id']}: fact {fid} not found"); bad += 1; continue
    f = facts.loc[fid]
    # census-implied answer, recomputed from the stored real shares
    implied = "Roughly equal" if f.condition == "equal" else (
        f.entity_1 if f.share_1 > f.share_2 else f.entity_2)
    errs = []
    if it["condition"] != f.condition:
        errs.append(f"condition {it['condition']}!={f.condition}")
    if it["answer"] != implied:
        errs.append(f"answer {it['answer']!r}!=implied {implied!r}")
    if it["answer"] not in it["choices"] or len(it["choices"]) != 3:
        errs.append("answer not in 3 choices")
    for e in (f.entity_1, f.entity_2):
        if str(e) not in it["question"]:
            errs.append(f"entity {e!r} missing from stem")
    # gap band sanity
    if f.condition == "diff" and not f.gap_pts >= 10:
        errs.append(f"diff gap {f.gap_pts}<10")
    if f.condition == "equal" and not f.gap_pts < 5:
        errs.append(f"equal gap {f.gap_pts}>=5")
    if errs:
        bad += 1
        if bad <= 10:
            print(f"[FAIL] {it['id']} ({fid}): {'; '.join(errs)}")

print(f"\nAudited {len(items)} items against real census facts. Mismatches: {bad}")
if bad == 0:
    print("[OK] Every AgriFacts instance's answer is consistent with the census numbers.")
sys.exit(1 if bad else 0)
