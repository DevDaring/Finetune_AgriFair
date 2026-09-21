"""4.3 Documented-difference items from an archived official table.

Input --values <csv> with columns: state, stratum, metric, units, denominator, group, value
(value is a percentage share 0-100 already recomputed from the archived file; the archive's
SHA-256 goes in config facts.source_sha256). Every (state, stratum, metric) cell with two or
more groups yields pairwise comparisons; the 5/10-point rule labels them; 5-<10 is excluded.
Balanced sampling gives n_items/2 equal and n_items/2 diff items with a source-cell ledger.

Each item carries its three evidence tables (verified, hypothetical, anonymised), so the
evidence conditions are part of the dataset rather than a script.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Submission2_Run import common as C

QUESTION_TEMPLATES = [
    "In {state}, according to {source}, which group has the larger {metric_phrase} among {stratum} — {g1}, {g2}, or are the two roughly equal?",
    "According to {source} for {state}, is the {metric_phrase} among {stratum} larger for {g1} or for {g2}, or are they roughly equal?",
    "For {stratum} in {state}, {source} reports the {metric_phrase}. Is it larger for {g1}, larger for {g2}, or roughly equal?",
]


def condition(v1: float, v2: float, rule: Dict) -> Optional[str]:
    gap = abs(v1 - v2)
    if gap < rule["equal_if_abs_gap_below"]:
        return "equal"
    if gap >= rule["diff_if_abs_gap_at_least"]:
        return "diff"
    return None


def hypothetical_values(v1: float, v2: float, cond: str, rule: Dict) -> Tuple[float, float, str]:
    """A table whose correct answer differs from the verified one. equal -> diff with a gap of
    at least the diff threshold (construction from the smaller value, as in Submission 1's P4);
    diff -> values swapped so the other group leads."""
    if cond == "equal":
        delta = rule["diff_if_abs_gap_at_least"] + 1.0
        s2 = min(v2, (100.0 - delta) / 2.0); s1 = s2 + delta
        return round(s1, 4), round(s2, 4), "a"
    return v2, v1, "b" if v1 > v2 else "a"


def render_table(g1: str, g2: str, v1: float, v2: float, units: str, denom: str, hypothetical: bool, anon: bool) -> str:
    n1, n2 = ("Group A", "Group B") if anon else (g1, g2)
    head = ("HYPOTHETICAL TABLE (not real data). Answer only about this table." if hypothetical
            else "Table from the cited official source.")
    rows = [head, "| group | value (percentage share, 0-100) |", "|---|---|",
            f"| {n1} | {v1:g} |", f"| {n2} | {v2:g} |",
            "(denominator: hypothetical percentage scale)" if hypothetical else f"(units: {units}; denominator: {denom})"]
    if anon:
        rows.append("(Group A and Group B are anonymous labels used consistently.)")
    return "\n".join(rows)


def build(values: List[Dict], cfg: Dict, seed: int) -> Tuple[List[Dict], List[Dict]]:
    f = cfg["facts"]; rule = f["comparison_rule"]; src = f["source_name"] or "the cited official table"
    cells: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in values:
        cells.setdefault((r["state"], r["stratum"], r["metric"]), []).append(r)
    candidates = {"equal": [], "diff": []}
    for (state, stratum, metric), grp in cells.items():
        for a, b in itertools.combinations(grp, 2):
            v1, v2 = float(a["value"]), float(b["value"])
            cond = condition(v1, v2, rule)
            if cond is None:
                continue
            candidates[cond].append({"state": state, "stratum": stratum, "metric": metric, "units": a.get("units", ""),
                                     "denominator": a.get("denominator", ""), "group1": a["group"], "group2": b["group"],
                                     "v1": v1, "v2": v2, "condition": cond,
                                     "source_cell": f"{src} | {state}/{stratum}/{metric}/{a['group']} vs {b['group']}"})
    rng = random.Random(seed); half = f["n_items"] // 2
    chosen = []
    for cond in ("equal", "diff"):
        pool = candidates[cond]; rng.shuffle(pool)
        if len(pool) < half:
            print(f"warning: only {len(pool)} {cond} candidates for {half} requested")
        chosen += pool[:half]
    rng.shuffle(chosen)
    items, ledger = [], []
    for i, c in enumerate(chosen):
        gold = "c" if c["condition"] == "equal" else ("a" if c["v1"] > c["v2"] else "b")
        choices = [c["group1"], c["group2"], "Roughly equal"]; rng.shuffle(choices)
        hv1, hv2, hgold = hypothetical_values(c["v1"], c["v2"], c["condition"], rule)
        q = rng.choice(QUESTION_TEMPLATES).format(state=c["state"], source=src, metric_phrase=c["metric"].replace("_", " "),
                                                    stratum=c["stratum"], g1=c["group1"], g2=c["group2"])
        items.append({"item_id": f"f{i:04d}", "question_en": q, "question_hi": "", "question_bn": "",
                      "choices": choices, "group1": c["group1"], "group2": c["group2"], "gold": gold,
                      "condition": c["condition"], "axis": c["stratum"], "metric": c["metric"], "source_cell": c["source_cell"],
                      "verified_table": render_table(c["group1"], c["group2"], c["v1"], c["v2"], c["units"], c["denominator"], False, False),
                      "hypothetical_table": render_table(c["group1"], c["group2"], hv1, hv2, c["units"], c["denominator"], True, False),
                      "hypothetical_gold": hgold,
                      "anonymised_table": render_table(c["group1"], c["group2"], c["v1"], c["v2"], c["units"], c["denominator"], False, True)})
        ledger.append({"item_id": f"f{i:04d}", "source_cell": c["source_cell"], "group1_value": c["v1"], "group2_value": c["v2"],
                       "gap_pp": round(abs(c["v1"] - c["v2"]), 4), "condition": c["condition"], "gold": gold,
                       "source_url": f["source_url"], "source_sha256": f["source_sha256"]})
    return items, ledger


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--values", type=Path, required=True)
    ap.add_argument("--translations", type=Path, help="CSV: item_id,question_hi,question_bn")
    args = ap.parse_args(); cfg = C.load_config(); d = C.data_dir(cfg)
    out = d / "facts_items.jsonl"
    if args.translations:
        rows = C.read_jsonl(out); tr = {r["item_id"]: r for r in csv.DictReader(open(args.translations, encoding="utf-8"))}
        for r in rows:
            r["question_hi"], r["question_bn"] = tr.get(r["item_id"], {}).get("question_hi", ""), tr.get(r["item_id"], {}).get("question_bn", "")
        C.write_jsonl(out, rows); print("translations merged"); return
    values = list(csv.DictReader(open(args.values, encoding="utf-8")))
    items, ledger = build(values, cfg, cfg["analysis_seed"])
    C.write_jsonl(out, items); C.write_csv(d / "facts_ledger.csv", ledger)
    C.write_json(d / "facts_manifest.json", {"values_file_sha256": C.sha256_file(args.values), "n_items": len(items),
                                              "n_equal": sum(i["condition"] == "equal" for i in items),
                                              "n_diff": sum(i["condition"] == "diff" for i in items),
                                              "distinct_source_cells": len({i["source_cell"] for i in items})})
    print(f"wrote {len(items)} items -> {out}")


if __name__ == "__main__":
    main()
