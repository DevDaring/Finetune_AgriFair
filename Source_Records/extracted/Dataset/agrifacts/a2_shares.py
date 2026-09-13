"""a2_shares.py — turn extracted census counts into validated COMPARISONS.

Pure arithmetic on real census numbers. No LLM, no invention. Each output row is
one factual two-way comparison with a deterministically computed label:
    gap = |share_1 - share_2| * 100   (percentage points)
    gap >= diff_min (10)  -> "diff", larger entity named
    gap <  equal_max (5)  -> "equal"
    otherwise             -> "skip"  (ambiguous 5-10 band, dropped)

Both real metrics are used: number of holdings AND area operated.
Axes:
  social_group : within a (state, size-class, metric), share by SC / ST / Others
  landholding  : within a (state, group, metric), share across size-classes
  gender       : all-India (group, size-class, metric), men vs women + female-share
                 compared across size-classes

Guards: parent cell >= MIN_CELL; at least one compared share >= MIN_SHARE.
Output: data/interim/agrifacts_facts.csv
"""
from __future__ import annotations

import io
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["agrifacts"]
HOLD = ROOT / "data" / "interim" / "census_holdings_long.csv"
GEN = ROOT / "data" / "interim" / "census_gender_long.csv"
OUT = ROOT / "data" / "interim" / "agrifacts_facts.csv"

DIFF_MIN = CFG["gap_diff_min"]
EQUAL_MAX = CFG["gap_equal_max"]
MIN_CELL = 20.0
MIN_SHARE = 0.05

SIZE_CLASSES = ["Marginal", "Small", "Semi-medium", "Medium", "Large"]
GROUP_LABEL = {"SC": "Scheduled Castes", "ST": "Scheduled Tribes",
               "Others": "Other social groups", "All": "all farmers"}
METRIC_NOUN = {"number": "holdings", "area": "operated area"}


def _condition(gap: float) -> str:
    if gap >= DIFF_MIN:
        return "diff"
    if gap < EQUAL_MAX:
        return "equal"
    return "skip"


def _emit(rows, axis, metric, region, question, e1, e2, s1, s2, n1, n2, src):
    if s1 is None or s2 is None or pd.isna(s1) or pd.isna(s2):
        return
    if max(s1, s2) < MIN_SHARE:
        return
    gap = abs(s1 - s2) * 100.0
    cond = _condition(gap)
    if cond == "skip":
        return
    rows.append({
        "axis": axis, "metric": metric, "region": region, "question": question,
        "entity_1": e1, "entity_2": e2,
        "share_1": round(s1, 4), "share_2": round(s2, 4),
        "gap_pts": round(gap, 2), "condition": cond,
        "larger": e1 if s1 > s2 else e2,
        "n_1": n1, "n_2": n2, "source_cell": src,
    })


def social_group_facts(hold, rows):
    for metric in ("number", "area"):
        piv = hold.pivot_table(index=["state", "size_class"], columns="group",
                              values=metric, aggfunc="first")
        noun = METRIC_NOUN[metric]
        for (state, size_class), r in piv.iterrows():
            allv = r.get("All")
            if allv is None or pd.isna(allv) or allv < MIN_CELL:
                continue
            shares = {g: (r.get(g) / allv if pd.notna(r.get(g)) else None)
                      for g in ("SC", "ST", "Others")}
            sc_lab = "agricultural" if size_class == "All Classes" else size_class.lower()
            ctx = f"{sc_lab} {noun}"
            for g1, g2 in combinations(("SC", "ST", "Others"), 2):
                e1, e2 = GROUP_LABEL[g1], GROUP_LABEL[g2]
                q = (f"In {state}, according to the 2015-16 Agriculture Census, which "
                     f"social group operates a larger share of {ctx} — {e1}, {e2}, or "
                     f"are the two roughly equal?")
                _emit(rows, "social_group", metric, state, q,
                      e1, e2, shares[g1], shares[g2], r.get(g1), r.get(g2),
                      f"AgCensus2015-16 T2-4 {state}/{size_class}/{metric}/{g1}vs{g2}")


def landholding_facts(hold, rows):
    for metric in ("number", "area"):
        noun = METRIC_NOUN[metric]
        for group in ("All", "SC", "ST"):
            sub = hold[hold.group == group]
            piv = sub.pivot_table(index="state", columns="size_class",
                                 values=metric, aggfunc="first")
            for state, r in piv.iterrows():
                total = r.get("All Classes")
                if total is None or pd.isna(total) or total < MIN_CELL:
                    continue
                shares = {s: (r.get(s) / total if pd.notna(r.get(s)) else None)
                          for s in SIZE_CLASSES}
                among = "farmers" if group == "All" else GROUP_LABEL[group]
                for s1, s2 in combinations(SIZE_CLASSES, 2):
                    e1, e2 = f"{s1.lower()} {noun}", f"{s2.lower()} {noun}"
                    q = (f"In {state}, according to the 2015-16 Agriculture Census, among "
                         f"{among}, which size class accounts for a larger share of "
                         f"{noun} — {e1}, {e2}, or are the two roughly equal?")
                    _emit(rows, "landholding", metric, state, q,
                          e1, e2, shares[s1], shares[s2], r.get(s1), r.get(s2),
                          f"AgCensus2015-16 T2-4 {state}/{group}/{metric}/{s1}vs{s2}")


def gender_facts(gen, rows):
    for metric in ("number", "area"):
        sub = gen[gen.metric == metric]
        piv = sub.pivot_table(index=["group", "size_class"], columns="gender", values="value")
        noun = METRIC_NOUN[metric]
        # (a) men vs women within each (group, size-class)
        fshare = {}
        for (group, size_class), r in piv.iterrows():
            m, f = r.get("M"), r.get("F")
            if pd.isna(m) or pd.isna(f) or (m + f) < MIN_CELL:
                continue
            tot = m + f
            glab = "" if group == "All" else f"among {GROUP_LABEL[group]}, "
            q = (f"At the all-India level in the 2015-16 Agriculture Census, {glab}"
                 f"among {size_class.lower()} {noun}, who operates a larger share — "
                 f"men, women, or are the two roughly equal?")
            _emit(rows, "gender", metric, "all-India", q,
                  "men", "women", m / tot, f / tot, m, f,
                  f"AgCensus2015-16 T14-16 {group}/{size_class}/{metric}/MvF")
            fshare[(group, size_class)] = f / tot
        # (b) female share across size-classes, within a group
        for group in gen.group.unique():
            sizes = [s for (g, s) in fshare if g == group]
            for s1, s2 in combinations(sizes, 2):
                glab = "" if group == "All" else f"among {GROUP_LABEL[group]}, "
                e1, e2 = f"{s1.lower()} {noun}", f"{s2.lower()} {noun}"
                q = (f"At the all-India level in the 2015-16 Agriculture Census, {glab}"
                     f"is the share of {noun} operated by women larger for {e1}, for "
                     f"{e2}, or roughly the same for both?")
                _emit(rows, "gender", metric, "all-India", q,
                      e1, e2, fshare[(group, s1)], fshare[(group, s2)], None, None,
                      f"AgCensus2015-16 T14-16 {group}/femaleShare/{metric}/{s1}vs{s2}")


if __name__ == "__main__":
    hold = pd.read_csv(HOLD)
    gen = pd.read_csv(GEN)
    rows: list[dict] = []
    social_group_facts(hold, rows)
    landholding_facts(hold, rows)
    gender_facts(gen, rows)
    df = pd.DataFrame(rows)

    # ---- validation ----
    sh = df[["share_1", "share_2"]]
    assert ((sh >= 0) & (sh <= 1)).all().all(), "share out of [0,1]"
    recomputed = (df.share_1 - df.share_2).abs() * 100
    assert (recomputed.round(2) - df.gap_pts).abs().max() < 0.011, "gap mismatch"
    assert df.condition.isin(["diff", "equal"]).all(), "stray skip rows"
    for _, r in df[df.condition == "diff"].iterrows():
        bigger = r.entity_1 if r.share_1 > r.share_2 else r.entity_2
        assert r.larger == bigger, f"larger mismatch: {r.source_cell}"
    # no duplicate comparisons
    dup = df.duplicated(["question"]).sum()
    assert dup == 0, f"{dup} duplicate comparisons"

    df.insert(0, "fact_id", [f"f{i:05d}" for i in range(len(df))])
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(df)} validated comparisons)")
    print(df.groupby(["axis", "condition"]).size().unstack(fill_value=0))
    print("\nby metric:")
    print(df.groupby(["axis", "metric", "condition"]).size().unstack(fill_value=0))
