"""Extend the verified comparisons from unused, validated census cells (plan section 3).

    python -m Submission1_TMLR.build_extended

Rules fixed before any model output exists:
- A candidate must be a validated construction record whose census cell is used neither by the
  benchmark nor by the 34 existing verified comparisons.
- The unused pool holds almost only difference items (the near-equal and gender cells were
  exhausted by the benchmark), so the extension takes TARGET_PER_AXIS difference items on each of
  the landholding and social-group axes, plus every remaining near-equal or gender item.
- Items are spread across states by round-robin before any state contributes a second item.
- Each item is rendered by the existing source-record generator, must pass all ten checks, and its
  label is recomputed from the raw holding counts by a second code path.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2 import r1_fresh_panel as R
from Submission1_DKE_Repair import source_schema as S
from Submission1_DKE_Repair.prompt_checks import check_panel

OUT_DIR = "results_submission1_tmlr"
PANEL_V2 = "results_submission1_dke_repair_v2/r1_corrected_panel.jsonl"
TARGET_PER_AXIS = 60
SEED = 20261004
ANSWER_SUFFIX = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'
LOW, HIGH = 5.0, 10.0          # the decision rule stated in every prompt


def _recheck(r: Dict) -> List[str]:
    """Second code path: the label must follow from the raw counts and the stated rule."""
    s1, s2 = float(r["share_1"]), float(r["share_2"])
    gap = abs(s1 - s2) * 100
    problems = []
    if abs(gap - float(r["gap_pts"])) > 0.05:
        problems.append(f"gap {gap:.2f} != recorded {r['gap_pts']}")
    if r.get("n_1") and r.get("n_2"):               # counts are recorded for some rows only
        n1, n2 = float(r["n_1"]), float(r["n_2"])
        # Counts are rounded to whole units (thousands) and shares to 4 decimals. The share implied
        # by the counts may therefore differ from the recorded share by at most the error those two
        # roundings can produce; anything larger is a real inconsistency. A zero count gives no
        # ratio and is skipped (the gap and rule checks above still apply).
        if n1 > 0 and n2 > 0 and s1 > 0:
            implied = n2 * s1 / n1
            tol = implied * (0.5 / n1 + 0.5 / n2 + 0.00005 / s1) + 0.00005
            if abs(implied - s2) > tol:
                problems.append("recorded share does not match the share implied by the counts")
    cond = "equal" if gap < LOW else ("diff" if gap >= HIGH else "excluded")
    if cond != r["condition"]:
        problems.append(f"condition {r['condition']} but the rule gives {cond}")
    if cond == "diff" and r["larger"] != (r["entity_1"] if s1 > s2 else r["entity_2"]):
        problems.append("recorded larger entity disagrees with the shares")
    return problems


def select() -> List[Dict]:
    rows = R.load_validated()
    used = R.used_cells() | {json.loads(l)["source_cell"] for l in (C.CODES_ROOT / PANEL_V2).open()}
    pool = [r for r in rows if r["source_cell"] not in used and r["condition"] in ("equal", "diff")]
    rng = random.Random(SEED)
    chosen: List[Dict] = []
    for axis in ("landholding", "social_group"):
        cand = sorted((r for r in pool if r["axis"] == axis and r["condition"] == "diff"),
                      key=lambda r: (r["region"], r["source_cell"]))
        by_state: Dict[str, List[Dict]] = collections.defaultdict(list)
        for r in cand:
            by_state[r["region"]].append(r)
        for lst in by_state.values():
            rng.shuffle(lst)
        states = sorted(by_state)
        rng.shuffle(states)
        picked: List[Dict] = []
        while len(picked) < TARGET_PER_AXIS and any(by_state[s] for s in states):
            for s in states:
                if len(picked) < TARGET_PER_AXIS and by_state[s]:
                    picked.append(by_state[s].pop())
        chosen += picked
    chosen += [r for r in pool if r["condition"] == "equal" or r["axis"] == "gender"]
    return chosen


def to_record(r: Dict, i: int) -> Dict:
    s1, s2 = float(r["share_1"]) * 100, float(r["share_2"]) * 100
    gold = S.EQUAL_CHOICE if r["condition"] == "equal" else (r["entity_1"] if s1 > s2 else r["entity_2"])
    return {"fresh_id": f"ext-{i:03d}", "parent_table": C.parent_table(r["source_cell"]), "axis": r["axis"],
            "group1": r["entity_1"], "group2": r["entity_2"], "share1_pct": round(s1, 4),
            "share2_pct": round(s2, 4), "gap_pp": round(abs(s1 - s2), 4), "condition": r["condition"],
            "gold_choice_text": gold, "source_cell": r["source_cell"]}


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    rows = select()
    recheck = {r["source_cell"]: _recheck(r) for r in rows}
    bad = {k: v for k, v in recheck.items() if v}
    if bad:
        raise SystemExit(f"label recheck failed for {len(bad)} items: {list(bad.items())[:3]}")
    records = [to_record(r, i) for i, r in enumerate(rows)]
    specs = [S.parse(rec) for rec in records]
    rendered = {s.fresh_id: S.render(s) for s in specs}
    checks = check_panel(specs, rendered)
    if checks["n_failures"]:
        raise SystemExit(f"source checks failed: {checks['failures_by_rule']}")

    panel, prompts = [], []
    for spec in specs:
        w = rendered[spec.fresh_id]
        choices = [spec.entity1, spec.entity2, S.EQUAL_CHOICE]       # equality at (c), as in E1
        item = {**spec.as_dict(), "population": spec.population(), "denominator_text": spec.denominator(),
                "wording_a": w["wording_a"], "wording_b": w["wording_b"], "choices": choices,
                "gold_choice_text": spec.gold_entity}
        panel.append(item)
        for wording in ("wording_a", "wording_b"):
            opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", choices))
            prompts.append({"prompt_id": f"tmlr-{spec.fresh_id}-{wording}", "study": "r1_extended",
                            "comparison_id": spec.fresh_id, "source_cell": spec.source_cell, "axis": spec.axis,
                            "state": spec.geography, "parent_table": spec.parent_table, "wording": wording,
                            "condition": spec.condition, "choices": choices,
                            "gold_choice_text": spec.gold_entity, "social_group": spec.social_group,
                            "size_class": spec.size_class, "max_new_tokens": 24,
                            "prompt": f"{w[wording]}\n{opts}{ANSWER_SUFFIX}"})
    C.write_jsonl(out / "extended_panel.jsonl", panel)
    C.write_jsonl(out / "prompts_r1_extended.jsonl", prompts)
    summary = {"items": len(panel), "prompts": len(prompts),
               "by_axis_condition": dict(collections.Counter(f"{p['axis']}/{p['condition']}" for p in panel)),
               "states": len({p["geography"] for p in panel}),
               "check_failures": checks["n_failures"], "label_recheck_failures": 0,
               "selection_seed": SEED, "target_per_axis": TARGET_PER_AXIS}
    C.write_json(out / "extended_panel_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
