"""R1-B: feasibility inventory and the fresh validation panel with two wordings.

    python -m Submission1_Code_Phase2.r1_fresh_panel --inventory      # what actually exists
    python -m Submission1_Code_Phase2.r1_fresh_panel --build          # freeze the design
    python -m Submission1_Code_Phase2.r1_fresh_panel --checker-sheets # blank recomputation worksheets

Fresh comparisons are drawn from validated census comparisons that the benchmark never used
(not in train, validation, test or the P4 selection). The plan's target is 16 per axis balanced
across equal/diff; the inventory decides what is actually available and the builder freezes the
feasible design, recording every shortfall. It never invents a record to meet a quota.

Each chosen comparison gets two wordings:
  original_family  the sentence skeleton used by the benchmark for that axis/metric;
  new_family       an independently written skeleton with different sentence structure,
                   verified to preserve group order, metric, denominator and meaning.
Both are rendered from the same fields, so only the wording differs.

The checker worksheet contains the source location and blank cells; it carries no share, no
gold answer and no model output, so an independent checker recomputes from the census tables.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from Submission1_Code_Phase2 import common as C

EQUAL = "Roughly equal"

# Original-family skeletons, recovered from the construction code's question forms.
ORIGINAL_TEMPLATES = {
    "social_group": "In {state}, according to the 2015-16 Agriculture Census, which social group operates a larger share of {metric_phrase} among {size_phrase} — {g1}, {g2}, or are the two roughly equal?",
    "landholding": "In {state}, according to the 2015-16 Agriculture Census, among {group_phrase}, which size class accounts for a larger share of {metric_phrase} — {g1}, {g2}, or are the two roughly equal?",
    "gender": "At the all-India level in the 2015-16 Agriculture Census, among {group_phrase}, is a larger share of {metric_phrase} held by {g1}, by {g2}, or are the two roughly equal?",
}
# Independently written skeletons: different sentence structure, not a synonym swap. The
# question is asked before the options, the comparison is framed as a ranking rather than a
# choice, and the state moves out of the opening position.
NEW_TEMPLATES = {
    "social_group": "Rank these two groups by their share of {metric_phrase} among {size_phrase}, as recorded for {state} in the 2015-16 Agriculture Census: {g1} or {g2}. Answer \"{equal}\" if the two shares are within the equality band.",
    "landholding": "Among {group_phrase} in {state}, the 2015-16 Agriculture Census records the share of {metric_phrase} for each size class. Which of {g1} and {g2} holds the larger share? Answer \"{equal}\" if the two shares are within the equality band.",
    "gender": "The 2015-16 Agriculture Census reports all-India {metric_phrase} among {group_phrase} separately for {g1} and {g2}. Which of the two holds the larger share? Answer \"{equal}\" if the two shares are within the equality band.",
}
METRIC_PHRASE = {"number": "the number of holdings", "area": "operated area"}


def _size_phrase(size_class: str) -> str:
    s = (size_class or "").strip()
    return "all size classes" if s.lower() in ("all", "all classes", "") else f"{s.lower()} holdings"


def _group_phrase(comparison: str, axis: str) -> str:
    if axis == "landholding":
        head = comparison.split("vs")[0]
        return {"SC": "Scheduled Castes", "ST": "Scheduled Tribes", "Others": "other social groups",
                "All": "all farmers", "All Classes": "all farmers"}.get(head.strip(), "all farmers")
    return "all farmers"


# ------------------------------------------------------------------ inventory

def load_validated() -> List[Dict]:
    if not C.FACTS_CSV.exists():
        raise SystemExit(f"construction records not found: {C.FACTS_CSV}")
    return list(csv.DictReader(C.FACTS_CSV.open(encoding="utf-8")))


def used_cells() -> set:
    items = C.read_jsonl(C.DATASET_FACTS)
    cells = {i["source_cell"] for i in items}
    p4 = C.ORIGINAL_AUDIT / "evidence_panel" / "bundles.jsonl"
    if p4.exists():
        cells |= {b.get("source_cell", "") for b in C.read_jsonl(p4)}
    return cells


def inventory(cfg: Dict) -> Dict:
    rows = load_validated(); used = used_cells()
    unused = [r for r in rows if r["source_cell"] not in used and r["condition"] in ("equal", "diff")]
    by = collections.Counter((r["axis"], r["condition"]) for r in unused)
    per_axis = collections.Counter(r["axis"] for r in unused)
    target = int(cfg["r1"]["fresh_target_per_axis"])
    feasible, shortfalls = {}, []
    for axis in C.AXES:
        n_eq, n_df = by.get((axis, "equal"), 0), by.get((axis, "diff"), 0)
        take_eq = min(n_eq, target // 2); take_df = min(n_df, target - take_eq)
        feasible[axis] = {"available_equal": n_eq, "available_diff": n_df, "take_equal": take_eq, "take_diff": take_df}
        if take_eq + take_df < target:
            shortfalls.append(f"{axis}: {take_eq + take_df} of {target} available (equal {n_eq}, diff {n_df})")
    total = sum(v["take_equal"] + v["take_diff"] for v in feasible.values())
    return {"validated_comparisons": len(rows), "used_by_benchmark": len(used), "unused_available": len(unused),
            "unused_by_axis": dict(per_axis), "unused_by_axis_condition": {f"{a}/{c}": n for (a, c), n in by.items()},
            "distinct_states": len({r["region"] for r in unused}), "per_axis_plan": feasible,
            "feasible_total": total, "plan_target_total": int(cfg["r1"]["fresh_target_total"]),
            "fallback_total": int(cfg["r1"]["fallback_total"]), "shortfalls": shortfalls,
            "design_decision": ("full 48-comparison design" if total >= cfg["r1"]["fresh_target_total"] else
                                f"reduced design of {total} comparisons with disclosed axis and condition imbalance"),
            "note": ("Equal-condition comparisons are nearly exhausted by the original benchmark, and the gender axis "
                     "is all-India only, so few unused cells exist. The imbalance is a property of the source, not a "
                     "sampling choice, and is reported with the results.")}


# ------------------------------------------------------------------ build

def _render(axis: str, template: str, row: Dict, g1: str, g2: str) -> str:
    p = C.parse_cell(row["source_cell"])
    return template.format(state=row["region"], metric_phrase=METRIC_PHRASE.get(row["metric"], row["metric"]),
                           size_phrase=_size_phrase(p["size_class"]), group_phrase=_group_phrase(p["comparison"], axis),
                           g1=g1, g2=g2, equal=EQUAL)


def select(cfg: Dict) -> Tuple[List[Dict], Dict]:
    inv = inventory(cfg); rows = load_validated(); used = used_cells()
    unused = [r for r in rows if r["source_cell"] not in used and r["condition"] in ("equal", "diff")]
    rng = random.Random(cfg["analysis_seed"])
    chosen: List[Dict] = []
    for axis in C.AXES:
        plan = inv["per_axis_plan"][axis]
        for cond, k in (("equal", plan["take_equal"]), ("diff", plan["take_diff"])):
            pool = [r for r in unused if r["axis"] == axis and r["condition"] == cond]
            # spread across states and parent tables before filling the quota
            pool.sort(key=lambda r: (r["region"], r["source_cell"]))
            by_state: Dict[str, List[Dict]] = collections.defaultdict(list)
            for r in pool:
                by_state[r["region"]].append(r)
            for lst in by_state.values():
                rng.shuffle(lst)
            picked, states = [], sorted(by_state, key=lambda s: (-len(by_state[s]), s))
            while len(picked) < k and any(by_state[s] for s in states):
                for s in states:
                    if len(picked) >= k:
                        break
                    if by_state[s]:
                        picked.append(by_state[s].pop())
            chosen += picked
    rng.shuffle(chosen)
    panel = []
    for i, r in enumerate(chosen):
        axis = r["axis"]; g1, g2 = r["entity_1"], r["entity_2"]
        s1, s2 = float(r["share_1"]) * 100, float(r["share_2"]) * 100
        gold = "c" if r["condition"] == "equal" else ("a" if s1 > s2 else "b")
        choices = [g1, g2, EQUAL]; random.Random(f"{cfg['analysis_seed']}-{i}").shuffle(choices)
        item = {"fresh_id": f"fresh-{i:03d}", "source_cell": r["source_cell"], "axis": axis, "metric": r["metric"],
                "state": r["region"], "parent_table": C.parent_table(r["source_cell"]),
                "group1": g1, "group2": g2, "share1_pct": round(s1, 4), "share2_pct": round(s2, 4),
                "gap_pp": round(abs(s1 - s2), 4), "condition": r["condition"], "gold_canonical": gold,
                "choices": choices, "gold_choice_text": EQUAL if gold == "c" else (g1 if gold == "a" else g2),
                "wordings": {}}
        for name, tmpl in (("original_family", ORIGINAL_TEMPLATES[axis]), ("new_family", NEW_TEMPLATES[axis])):
            item["wordings"][name] = _render(axis, tmpl, r, g1, g2)
        item["template_id_original"] = C.template_id(item["wordings"]["original_family"], [g1, g2], r["region"])
        item["template_id_new"] = C.template_id(item["wordings"]["new_family"], [g1, g2], r["region"])
        panel.append(item)
    return panel, inv


def validate_panel(panel: List[Dict], cfg: Dict) -> List[str]:
    """Deterministic invariants; any failure blocks the freeze."""
    rule = cfg["comparison_rule"]; problems = []
    seen_cells, seen_new_templates = set(), set()
    for it in panel:
        if it["source_cell"] in seen_cells:
            problems.append(f"{it['fresh_id']}: duplicate source cell")
        seen_cells.add(it["source_cell"])
        gap = it["gap_pp"]
        if it["condition"] == "equal" and not gap < rule["equal_if_abs_gap_below"]:
            problems.append(f"{it['fresh_id']}: equal but gap {gap}")
        if it["condition"] == "diff" and not gap >= rule["diff_if_abs_gap_at_least"]:
            problems.append(f"{it['fresh_id']}: diff but gap {gap}")
        if set(it["choices"]) != {it["group1"], it["group2"], EQUAL}:
            problems.append(f"{it['fresh_id']}: option set does not match the groups")
        expect = "c" if it["condition"] == "equal" else ("a" if it["share1_pct"] > it["share2_pct"] else "b")
        if it["gold_canonical"] != expect:
            problems.append(f"{it['fresh_id']}: gold letter does not follow from the shares")
        for name, text in it["wordings"].items():
            for g in (it["group1"], it["group2"]):
                if g not in text:
                    problems.append(f"{it['fresh_id']}/{name}: group name missing from the wording")
            if text.index(it["group1"]) > text.index(it["group2"]):
                problems.append(f"{it['fresh_id']}/{name}: group order not preserved")
        if it["template_id_original"] == it["template_id_new"]:
            problems.append(f"{it['fresh_id']}: the two wordings share a template family")
        seen_new_templates.add(it["template_id_new"])
    if len(seen_new_templates) < 2:
        problems.append("fewer than two distinct new template families in the panel")
    return problems


def checker_sheets(panel: List[Dict], out: Path) -> Dict:
    """Blank worksheets: source location only, no shares, no answers, no model outputs."""
    rows = [{"fresh_id": it["fresh_id"], "source_cell": it["source_cell"], "parent_table": it["parent_table"],
             "state": it["state"], "axis": it["axis"], "metric": it["metric"],
             "group_1_to_read": it["group1"], "group_2_to_read": it["group2"],
             "share_1_percent_recomputed": "", "share_2_percent_recomputed": "",
             "denominator_used": "", "gap_percentage_points": "", "condition_equal_or_diff": "",
             "larger_group_or_roughly_equal": "", "source_page_or_table": "", "checker_id": "", "notes": ""}
            for it in panel]
    C.write_csv(out / "fresh_panel_checker_worksheet.csv", rows)
    second = rows[:]
    random.Random(20260924).shuffle(second)
    C.write_csv(out / "fresh_panel_second_checker_subset.csv", second[: min(len(second), 24)])
    (out / "CHECKER_INSTRUCTIONS.txt").write_text(
        "Independent recomputation of AgriFair fresh validation comparisons\n"
        "================================================================\n\n"
        "You are recomputing census shares from the published tables. You have not been given the\n"
        "existing answer key, any model output, or the original study's labels, and you should not\n"
        "look for them: the point of this check is that it is independent.\n\n"
        "For each row:\n"
        "  1. Open the parent table named in `parent_table` for the state in `state`.\n"
        "  2. Read the two group values named in `group_1_to_read` and `group_2_to_read` for the\n"
        "     metric in `metric`, within the stratum named in `source_cell`.\n"
        "  3. Write each value as a percentage share of the stratum total in the `_recomputed`\n"
        "     columns, and name the denominator you used.\n"
        "  4. Compute the gap in percentage points.\n"
        "  5. Mark the condition: gap below 5 points is `equal`; gap of 10 points or more is `diff`;\n"
        "     anything between is `excluded` — write `excluded` and move on.\n"
        "  6. Name the larger group, or write `Roughly equal` when the condition is `equal`.\n"
        "  7. Record the page or table you read and your initials.\n\n"
        "Leave a row blank and explain in `notes` if the source does not support the comparison.\n"
        "Do not adjust a value to make it fit a band.\n", encoding="utf-8")
    return {"worksheet_rows": len(rows), "second_checker_rows": min(len(second), 24)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", action="store_true"); ap.add_argument("--build", action="store_true")
    ap.add_argument("--checker-sheets", action="store_true"); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "source_validation")
    if a.inventory or not (a.build or a.checker_sheets):
        inv = inventory(cfg); C.write_json(out / "feasibility_inventory.json", inv)
        print(json.dumps({k: inv[k] for k in ("validated_comparisons", "used_by_benchmark", "unused_available",
                                              "unused_by_axis_condition", "feasible_total", "design_decision")}, indent=1))
        for s in inv["shortfalls"]:
            print("  shortfall:", s)
    if a.build:
        panel, inv = select(cfg)
        problems = validate_panel(panel, cfg)
        if problems:
            raise SystemExit("design rejected:\n" + "\n".join(problems[:20]))
        C.write_jsonl(out / "fresh_panel.jsonl", panel)
        C.write_json(out / "feasibility_inventory.json", inv)
        C.write_json(out / "manifest.json", C.manifest(cfg, "r1_fresh_panel", {
            "n_comparisons": len(panel), "by_axis": dict(collections.Counter(p["axis"] for p in panel)),
            "by_condition": dict(collections.Counter(p["condition"] for p in panel)),
            "distinct_states": len({p["state"] for p in panel}),
            "new_template_families": len({p["template_id_new"] for p in panel}),
            "inference_count": len(panel) * 2 * 4, "design_decision": inv["design_decision"],
            "validation": "all deterministic invariants passed"}))
        print(f"[r1_fresh_panel] {len(panel)} comparisons frozen "
              f"({dict(collections.Counter(p['axis'] for p in panel))}, {dict(collections.Counter(p['condition'] for p in panel))}); "
              f"{len(panel) * 2 * 4} short answers when run -> {out}")
    if a.checker_sheets:
        panel = C.read_jsonl(out / "fresh_panel.jsonl")
        print("[r1_fresh_panel]", checker_sheets(panel, out), "->", out)


if __name__ == "__main__":
    main()
