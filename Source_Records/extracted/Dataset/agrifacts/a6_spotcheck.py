"""a6_spotcheck.py — draw a 250-item stratified sample for the author to verify.

The sample is stratified across axis x condition so the reviewer sees the full
range. For each item we print the question, the options, the gold answer (computed
from the census), and the exact source_cell so the reviewer can trace the number.
Two blank columns are left for the human: `author_agrees` (y/n) and `note`.

Output: data/interim/agrifacts_spotcheck.csv  (fill author_agrees, then read the
agreement rate printed by re-running with --score once marked).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data" / "final" / "agrifacts.jsonl"
OUT = ROOT / "data" / "interim" / "agrifacts_spotcheck.csv"
N = 250
SEED = 20260502


def make():
    rows = [json.loads(l) for l in FINAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # proportional stratified sample across axis x condition
    frac = N / len(df)
    samp = (df.groupby(["axis", "condition"], group_keys=False)
              .apply(lambda g: g.sample(max(1, round(len(g) * frac)), random_state=SEED)))
    samp = samp.sample(min(N, len(samp)), random_state=SEED).reset_index(drop=True)
    samp["choices"] = samp["choices"].apply(lambda c: " | ".join(c))
    samp["author_agrees"] = ""   # reviewer fills y / n
    samp["note"] = ""
    cols = ["id", "axis", "condition", "metric", "question", "choices",
            "answer", "source_cell", "author_agrees", "note"]
    samp[cols].to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {OUT} ({len(samp)} items) — fill the 'author_agrees' column (y/n).")
    print(samp.groupby(["axis", "condition"]).size())


def score():
    df = pd.read_csv(OUT).fillna("")
    marked = df[df.author_agrees.str.lower().isin(["y", "n", "yes", "no"])]
    if len(marked) == 0:
        print("No rows marked yet."); return
    agree = marked.author_agrees.str.lower().isin(["y", "yes"]).mean()
    print(f"Marked {len(marked)}/{len(df)}; author agreement = {agree:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="report agreement after marking")
    a = ap.parse_args()
    score() if a.score else make()
