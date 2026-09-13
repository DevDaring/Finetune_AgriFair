"""a3_seeds.py — build seed MCQs from validated census comparisons (pure Python).

Each comparison in agrifacts_facts.csv becomes one 3-choice MCQ. The answer is
NOT decided by any model: it is the larger group/size for "diff" rows and
"Roughly equal" for "equal" rows. Choice order is shuffled with a per-seed seeded
RNG so the correct answer is not positionally predictable.

Every seed is validated before it is written; a single failure aborts the run.
Output: data/interim/agrifacts_seeds.jsonl
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED = int(yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["seed"])
FACTS = ROOT / "data" / "interim" / "agrifacts_facts.csv"
OUT = ROOT / "data" / "interim" / "agrifacts_seeds.jsonl"

EQUAL_CHOICE = "Roughly equal"


def build():
    df = pd.read_csv(FACTS)
    seeds = []
    for _, r in df.iterrows():
        rng = random.Random(f"{SEED}-{r.fact_id}")
        answer = EQUAL_CHOICE if r.condition == "equal" else r.larger
        choices = [r.entity_1, r.entity_2, EQUAL_CHOICE]
        rng.shuffle(choices)

        rec = {
            "seed_id": r.fact_id,
            "axis": r.axis,
            "metric": r.metric,
            "region": r.region,
            "condition": r.condition,
            "question": r.question,
            "choices": choices,
            "answer": answer,
            "answer_idx": choices.index(answer),
            "share_1": float(r.share_1),
            "share_2": float(r.share_2),
            "gap_pts": float(r.gap_pts),
            "source_cell": r.source_cell,
        }
        _validate(rec, r)
        seeds.append(rec)

    with OUT.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT} ({len(seeds)} seeds)")
    sm = pd.DataFrame(seeds).groupby(["axis", "condition"]).size().unstack(fill_value=0)
    print(sm)
    # answer-position balance check (should be spread across 0/1/2)
    pos = pd.Series([s["answer_idx"] for s in seeds]).value_counts().sort_index()
    print("answer position counts:", dict(pos))


def _validate(rec, r):
    assert len(rec["choices"]) == 3, f"{rec['seed_id']}: not 3 choices"
    assert len(set(rec["choices"])) == 3, f"{rec['seed_id']}: duplicate choices"
    assert rec["answer"] in rec["choices"], f"{rec['seed_id']}: answer not in choices"
    assert rec["choices"][rec["answer_idx"]] == rec["answer"], f"{rec['seed_id']}: idx mismatch"
    if r.condition == "equal":
        assert rec["answer"] == EQUAL_CHOICE, f"{rec['seed_id']}: equal must answer 'Roughly equal'"
        assert r.gap_pts < 5.0, f"{rec['seed_id']}: equal gap >= 5"
    else:
        assert rec["answer"] != EQUAL_CHOICE, f"{rec['seed_id']}: diff must name a group"
        assert rec["answer"] in (r.entity_1, r.entity_2), f"{rec['seed_id']}: answer not an entity"
        assert r.gap_pts >= 10.0, f"{rec['seed_id']}: diff gap < 10"
        bigger = r.entity_1 if r.share_1 > r.share_2 else r.entity_2
        assert rec["answer"] == bigger, f"{rec['seed_id']}: answer is not the larger share"
    # the two compared entities must appear in the question stem
    assert str(r.entity_1) in rec["question"] and str(r.entity_2) in rec["question"], \
        f"{rec['seed_id']}: entities missing from stem"


if __name__ == "__main__":
    build()
