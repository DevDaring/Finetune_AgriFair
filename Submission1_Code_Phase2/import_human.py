"""Import the returned human work: validate it, measure agreement, apply it to the designs.

    python -m Submission1_Code_Phase2.import_human --dir Submission1/phase2_human_pack/Human_Reviewed

Privacy: the identity columns (`checker_initials`, `prepared_by`, `checked_by`, and the whole
`Reader` sheet) are counted for completeness and then dropped. They are never written into any
output, and no name, initial, employer or contact appears in the results tree. Checkers are
referred to as "checker 1" and "checker 2", readers as E1 and E2, positionally.

Task A is the one that can change the study: where the independent recomputation disagrees with
the frozen panel, the comparison is **excluded from the fresh panel**, not silently corrected.
The panel that goes to inference is written as `fresh_panel_verified.jsonl`; the original stays.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openpyxl import load_workbook

from Submission1_Code_Phase2 import common as C

IDENTITY_COLUMNS = {"checker_initials", "prepared_by", "checked_by", "reader_id", "date_completed"}
IDENTITY_SHEETS = {"Reader"}


def read_sheet(path: Path, sheet: str) -> Tuple[List[str], List[Dict]]:
    """Rows as dicts with every identity column removed before the data leaves this function."""
    wb = load_workbook(path, data_only=True)
    if sheet not in wb.sheetnames:
        return [], []
    ws = wb[sheet]
    hdr = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    keep = [i for i, h in enumerate(hdr) if h not in IDENTITY_COLUMNS]
    cols = [hdr[i] for i in keep]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if all(v in (None, "") for v in r):
            continue
        rows.append({cols[j]: ("" if r[i] is None else str(r[i]).strip()) for j, i in enumerate(keep)})
    return cols, rows


def _num(x) -> Optional[float]:
    try:
        return float(str(x).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ Task A

def import_task_a(cfg: Dict, path: Path, out: Path) -> Dict:
    panel = {p["fresh_id"]: p for p in C.read_jsonl(C.CODES_ROOT / cfg["output_directory"] / "source_validation" / "fresh_panel.jsonl")}
    rule = cfg["comparison_rule"]
    _, c1 = read_sheet(path, "Checker 1 (all rows)")
    _, c2 = read_sheet(path, "Checker 2 (subset)")
    _, disagreements = read_sheet(path, "Disagreements")
    by1 = {r["fresh_id"]: r for r in c1}
    by2 = {r["fresh_id"]: r for r in c2}

    rows, arithmetic_problems = [], []
    for fid, p in panel.items():
        r = by1.get(fid)
        if not r:
            rows.append({"fresh_id": fid, "status": "not_checked"}); continue
        s1, s2, gap = _num(r.get("share_1_percent")), _num(r.get("share_2_percent")), _num(r.get("gap_percentage_points"))
        cond = (r.get("condition") or "").lower()
        # the checker's own arithmetic must be internally consistent
        if s1 is not None and s2 is not None:
            if gap is None or abs(abs(s1 - s2) - gap) > 0.15:
                arithmetic_problems.append(f"{fid}: gap {gap} does not follow from {s1} and {s2}")
            implied = ("equal" if abs(s1 - s2) < rule["equal_if_abs_gap_below"]
                       else "diff" if abs(s1 - s2) >= rule["diff_if_abs_gap_at_least"] else "excluded")
            if cond and cond != implied:
                arithmetic_problems.append(f"{fid}: condition '{cond}' does not follow from a gap of {abs(s1 - s2):.2f}")
        agrees_condition = cond == p["condition"]
        larger = (r.get("larger_group_or_roughly_equal") or "").strip().lower()
        agrees_answer = larger == str(p["gold_choice_text"]).strip().lower()
        rows.append({"fresh_id": fid, "axis": p["axis"], "panel_condition": p["condition"],
                     "checker_condition": cond, "panel_answer": p["gold_choice_text"],
                     "checker_answer": r.get("larger_group_or_roughly_equal", ""),
                     "checker_gap_pp": gap, "panel_gap_pp": p["gap_pp"],
                     "condition_agrees": agrees_condition, "answer_agrees": agrees_answer,
                     "double_checked": fid in by2,
                     "status": "agree" if (agrees_condition and agrees_answer) else
                               ("excluded_by_checker" if cond == "excluded" else "disagreement")})
    # checker 1 vs checker 2 on the overlapping subset
    both = sorted(set(by1) & set(by2))
    inter = [{"fresh_id": f,
              "condition_match": (by1[f].get("condition") or "").lower() == (by2[f].get("condition") or "").lower(),
              "answer_match": (by1[f].get("larger_group_or_roughly_equal") or "").strip().lower() ==
                              (by2[f].get("larger_group_or_roughly_equal") or "").strip().lower()} for f in both]
    agreed = [r for r in rows if r["status"] == "agree"]
    excluded = [r for r in rows if r["status"] in ("excluded_by_checker", "disagreement")]
    verified = [panel[r["fresh_id"]] for r in agreed]
    C.write_jsonl(out / "fresh_panel_verified.jsonl", verified)
    C.write_csv(out / "taskA_check_results.csv", rows)
    if disagreements:
        # the team titled their columns with initials; rename positionally so no identity is stored
        import re as _re
        def _anon(col: str) -> str:
            m = _re.match(r"\s*Checker\s*(\d)", col, _re.I)
            return f"checker_{m.group(1)}" if m else _re.sub(r"\s*\([A-Z]{1,4}\)\s*$", "", col).strip().lower().replace(" / ", "_").replace(" ", "_")
        C.write_csv(out / "taskA_reported_disagreements.csv",
                    [{_anon(k): v for k, v in row.items()} for row in disagreements])
    summary = {
        "panel_comparisons": len(panel), "checked_by_checker_1": len(by1), "double_checked_by_checker_2": len(by2),
        "independent_recomputation_agrees": len(agreed),
        "agreement_rate": round(len(agreed) / max(1, len(by1)), 4),
        "excluded_after_checking": [{"fresh_id": r["fresh_id"], "reason": r["status"],
                                     "checker_condition": r["checker_condition"], "panel_condition": r["panel_condition"]}
                                    for r in excluded],
        "checker1_vs_checker2": {"overlap": len(inter),
                                 "condition_agreement": round(sum(r["condition_match"] for r in inter) / max(1, len(inter)), 4),
                                 "answer_agreement": round(sum(r["answer_match"] for r in inter) / max(1, len(inter)), 4),
                                 "rows_differing": [r["fresh_id"] for r in inter if not (r["condition_match"] and r["answer_match"])]},
        "checker_arithmetic_problems": arithmetic_problems,
        "team_reported_disagreements": len(disagreements),
        "verified_panel_size": len(verified),
        "verified_by_axis": dict(collections.Counter(p["axis"] for p in verified)),
        "verified_by_condition": dict(collections.Counter(p["condition"] for p in verified)),
        "note": ("Comparisons the independent checker did not confirm are removed from the panel that goes to "
                 "inference; they are not corrected to match the original labels. The original fresh_panel.jsonl "
                 "is unchanged.")}
    C.write_json(out / "taskA_summary.json", summary)
    return summary


# ------------------------------------------------------------------ Task B

def _kappa(a: List[str], b: List[str]) -> float:
    import numpy as np
    cats = sorted(set(a) | set(b)); idx = {c: i for i, c in enumerate(cats)}; k = len(cats)
    if k < 2:
        return 0.0
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum(); E = np.outer(O.sum(1), O.sum(0)) / n
    W = 1 - np.eye(k)
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else 0.0


def import_task_b(cfg: Dict, paths: List[Path], out: Path) -> Dict:
    readers = {}
    for i, p in enumerate(sorted(paths)):
        _, rows = read_sheet(p, "Eligibility")
        readers[f"E{i + 1}"] = {r["review_id"]: r for r in rows}
    ids = sorted(set().union(*[set(v) for v in readers.values()]))
    key = json.loads((C.CODES_ROOT / cfg["output_directory"] / "constructs" / "eligibility_key.json").read_text(encoding="utf-8"))
    axis = {r["review_id"]: r["axis"] for r in key["rows"]}
    pair = {r["review_id"]: r["pair_id"] for r in key["rows"]}
    long_rows, per_id = [], {}
    for rid in ids:
        vals = {r: readers[r].get(rid, {}).get("eligibility", "") for r in readers}
        per_id[rid] = vals
        for r, v in vals.items():
            long_rows.append({"reader": r, "review_id": rid, "pair_id": pair.get(rid, ""), "axis": axis.get(rid, ""),
                              "eligibility": v,
                              "relevant_context_changed_C": readers[r].get(rid, {}).get("relevant_context_changed_C", ""),
                              "presentation_need_changed_R": readers[r].get(rid, {}).get("presentation_need_changed_R", ""),
                              "justification": readers[r].get(rid, {}).get("task_based_justification", "")})
    C.write_csv(out / "taskB_eligibility_long.csv", long_rows)
    names = sorted(readers)
    a = [per_id[i][names[0]] for i in ids]; b = [per_id[i][names[1]] for i in ids]
    unanimous = [i for i in ids if len(set(per_id[i].values())) == 1]
    invariant = [i for i in unanimous if per_id[i][names[0]] == "substantive_advice_should_be_invariant"]
    disagreed = [i for i in ids if i not in unanimous]
    # binary collapse: is substantive invariance expected at all?
    bin_a = ["invariant" if x == "substantive_advice_should_be_invariant" else "not_invariant" for x in a]
    bin_b = ["invariant" if x == "substantive_advice_should_be_invariant" else "not_invariant" for x in b]
    summary = {
        "questions": len(ids), "readers": len(readers),
        "distribution": {r: dict(collections.Counter(readers[r].get(i, {}).get("eligibility", "") for i in ids)) for r in names},
        "raw_agreement_four_way": round(sum(x == y for x, y in zip(a, b)) / len(ids), 4),
        "cohen_kappa_four_way": round(_kappa(a, b), 4),
        "raw_agreement_invariance_binary": round(sum(x == y for x, y in zip(bin_a, bin_b)) / len(ids), 4),
        "cohen_kappa_invariance_binary": round(_kappa(bin_a, bin_b), 4),
        "unanimous": len(unanimous), "disagreements_to_adjudicate": len(disagreed),
        "eligible_for_invariance_analysis": len(invariant),
        "by_axis_invariance_eligible": dict(collections.Counter(axis.get(i, "") for i in invariant)),
        "note": ("The invariance-eligible set is a post-hoc stratum for a sensitivity analysis. Report it beside the "
                 "original all-axis result with its reduced denominator; it does not replace the original analysis, "
                 "and the disagreements stay in the record.")}
    C.write_json(out / "taskB_eligibility_summary.json", summary)
    C.write_json(out / "taskB_eligibility_subsets.json",
                 {"invariance_eligible_review_ids": invariant, "invariance_eligible_pair_ids": [pair[i] for i in invariant],
                  "disagreement_review_ids": disagreed, "all_review_ids": ids})
    return summary


# ------------------------------------------------------------------ Task C

def import_task_c(cfg: Dict, path: Path, out: Path) -> Dict:
    _, rows = read_sheet(path, "Cases")
    problems = []
    for r in rows:
        if not r.get("base_question"):
            problems.append(f"{r.get('case_id')}: no question")
        if not r.get("essential_points_semicolon_separated"):
            problems.append(f"{r.get('case_id')}: no essential points to score against")
        if not r.get("reference_url"):
            problems.append(f"{r.get('case_id')}: no reference")
        if r.get("case_type") == "identity_irrelevant" and not (r.get("variant_A_description") and r.get("variant_B_description")):
            problems.append(f"{r.get('case_id')}: identity case missing a variant description")
    adv = C.out_dir(cfg, "advice")
    C.write_csv(adv / "reference_packets.csv", rows)          # what r3_advice --build consumes
    ess = [len([e for e in r.get("essential_points_semicolon_separated", "").split(";") if e.strip()]) for r in rows]
    summary = {"cases": len(rows), "usable": len(rows) - len({p.split(':')[0] for p in problems}),
               "by_type": dict(collections.Counter(r.get("case_type", "") for r in rows)),
               "by_axis": dict(collections.Counter(r.get("toggle_axis", "") for r in rows)),
               "essential_points_per_case": {"min": min(ess) if ess else 0, "max": max(ess) if ess else 0,
                                             "mean": round(sum(ess) / max(1, len(ess)), 1)},
               "distinct_references": len({r.get("reference_url", "") for r in rows if r.get("reference_url")}),
               "problems": problems,
               "written_to": str(adv / "reference_packets.csv")}
    C.write_json(out / "taskC_reference_summary.json", summary)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args(argv)
    cfg = C.load_config()
    d = a.dir if a.dir.is_absolute() else (C.CODES_ROOT.parent / a.dir)
    out = C.out_dir(cfg, "human_review")
    res = {}
    ta = d / "TaskA_source_checking.xlsx"
    if ta.exists():
        res["task_a"] = import_task_a(cfg, ta, out)
    tb = sorted(d.glob("TaskB_prompt_eligibility_*.xlsx"))
    if len(tb) >= 2:
        res["task_b"] = import_task_b(cfg, tb, out)
    tc = d / "TaskC_reference_packets.xlsx"
    if tc.exists():
        res["task_c"] = import_task_c(cfg, tc, out)
    C.write_json(out / "human_review_manifest.json",
                 {"source_directory": str(d), "identity_columns_dropped": sorted(IDENTITY_COLUMNS),
                  "identity_sheets_ignored": sorted(IDENTITY_SHEETS), **{k: v for k, v in res.items()}})
    for k, v in res.items():
        print(f"\n=== {k}")
        print(json.dumps({kk: vv for kk, vv in v.items() if kk != "note"}, indent=1)[:1400])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
