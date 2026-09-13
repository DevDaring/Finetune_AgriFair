"""a7_emit.py — balance the filtered pool and emit the final AgriFacts dataset.

Input : data/interim/agrifacts_filtered.jsonl (unanimous 3-judge keeps)
Output: data/final/agrifacts.jsonl with fields
        id, question, choices[3], answer, condition, axis, metric, source_cell, paraphrase_of

Balancing goals: ~1000 diff / ~1000 equal, every axis >= min_per_axis. Gender is
content-limited, so it is capped (not allowed to dominate) while still clearing
its floor; the plentiful social_group and landholding axes fill the remainder.
De-duplicates identical question strings first. Every emitted item is re-validated.
"""
from __future__ import annotations

import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
AF = CFG["agrifacts"]
FILT = ROOT / "data" / "interim" / "agrifacts_filtered.jsonl"
OUT = ROOT / "data" / "final" / "agrifacts.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

TARGET_PER_COND = AF["target_diff"]      # 1000
FLOOR_PER_AXIS = AF["min_per_axis"]      # 150
AXES = ["gender", "social_group", "landholding"]
GENDER_CAP_PER_COND = 90                  # keep gender present but not dominant


def _allocate(avail: dict, target: int) -> dict:
    """avail: axis->count for one condition. Return axis->take summing ~target."""
    caps = {a: (GENDER_CAP_PER_COND if a == "gender" else 10**9) for a in avail}
    floor = max(1, FLOOR_PER_AXIS // 2)   # per-condition floor -> >=150 across 2 conds
    take = {a: min(avail[a], caps[a], floor) for a in avail}
    remaining = target - sum(take.values())
    axes = sorted(avail)
    # round-robin the remainder over axes with spare capacity
    progress = True
    while remaining > 0 and progress:
        progress = False
        for a in axes:
            if remaining <= 0:
                break
            if take[a] < min(avail[a], caps[a]):
                take[a] += 1
                remaining -= 1
                progress = True
    return take


def main():
    rows = [json.loads(l) for l in FILT.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("question", keep="first").reset_index(drop=True)
    print(f"filtered (deduped): {len(df)}")
    print(df.groupby(["axis", "condition"]).size().unstack(fill_value=0))

    rng = random.Random(SEED)
    pool_by = defaultdict(list)
    for _, r in df.iterrows():
        pool_by[(r["axis"], r["condition"])].append(r.to_dict())
    for k in pool_by:
        rng.shuffle(pool_by[k])

    selected = []
    for cond in ("diff", "equal"):
        avail = {a: len(pool_by[(a, cond)]) for a in AXES}
        take = _allocate(avail, TARGET_PER_COND)
        for a in AXES:
            selected.extend(pool_by[(a, cond)][:take[a]])
    rng.shuffle(selected)

    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(selected):
            assert len(r["choices"]) == 3 and r["answer"] in r["choices"], r["candidate_id"]
            rec = {
                "id": f"agrifacts-{i:05d}",
                "question": r["question"],
                "choices": r["choices"],
                "answer": r["answer"],
                "condition": r["condition"],
                "axis": r["axis"],
                "metric": r["metric"],
                "source_cell": r["source_cell"],
                "paraphrase_of": r["paraphrase_of"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    out = pd.DataFrame(selected)
    print(f"\nWrote {OUT} ({n} items)")
    print("condition:", out.condition.value_counts().to_dict())
    print("axis:", out.axis.value_counts().to_dict())
    assert all(out.axis.value_counts() >= FLOOR_PER_AXIS), "an axis is below the floor"
    print("[OK] every axis >= floor; diff/equal balanced.")


if __name__ == "__main__":
    main()
