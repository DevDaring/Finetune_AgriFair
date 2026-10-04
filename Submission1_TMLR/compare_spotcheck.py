"""Compare the returned human spot-check sheets with the study's source records.

    python -m Submission1_TMLR.compare_spotcheck <checker1.xlsx> <checker2.xlsx>

Reads only the value columns (shares, total, gap, condition, larger group, notes) and the Task B
judgements. Checker initials are never read, and the outputs hold aggregates and per-item
comparisons only. Outputs stay in results_submission1_tmlr/spotcheck/, which is not published.

Agreement rules, fixed before reading the sheets:
  share agreement: |human - study| <= 0.15 percentage points (the sheets ask for one decimal)
  condition and larger group: exact match after normalising case and "Roughly equal"
"""
from __future__ import annotations

import collections
import json
import sys

import openpyxl

from Submission1_Code_Phase2 import common as C

OUT = C.CODES_ROOT / "results_submission1_tmlr" / "spotcheck"
TOL = 0.15


def norm(x):
    return " ".join(str(x or "").strip().lower().replace("roughly equal", "roughly equal").split())


def fnum(x):
    try:
        return float(str(x).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def task_a(path: str, sheet_prefix: str = "Task A"):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = next(wb[n] for n in wb.sheetnames if n.startswith(sheet_prefix))
    head = [c.value for c in ws[1]]
    rows = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        d = dict(zip(head, row))
        d.pop("checker_initials", None)                       # never read or stored
        if d.get("item_id"):
            rows[d["item_id"]] = d
    return rows, wb


def task_b(wb):
    if "Task B - questions" not in wb.sheetnames:
        return []
    ws = wb["Task B - questions"]
    head = [c.value for c in ws[1]]
    return [dict(zip(head, r)) for r in ws.iter_rows(min_row=2, values_only=True) if r[0]]


def compare(human: dict, study: dict) -> dict:
    s1, s2 = fnum(human.get("share_1_percent")), fnum(human.get("share_2_percent"))
    cond = norm(human.get("condition")); larger = norm(human.get("larger_group_or_roughly_equal"))
    st_larger = norm(study["gold_choice_text"])
    entity_alias = {norm(study["entity1"]): norm(study["entity1"]), norm(study["entity2"]): norm(study["entity2"])}
    # the sheets name size classes as "marginal holdings"; the study's entity may read "marginal operated area"
    def same_group(h, g):
        h = h.replace(" holdings", "").replace(" operated area", ""); g = g.replace(" holdings", "").replace(" operated area", "")
        return h == g
    return {"filled": s1 is not None and s2 is not None,
            "share1_human": s1, "share2_human": s2,
            "share1_study": round(study["share1_pct"], 2), "share2_study": round(study["share2_pct"], 2),
            "share1_agree": s1 is not None and abs(s1 - study["share1_pct"]) <= TOL,
            "share2_agree": s2 is not None and abs(s2 - study["share2_pct"]) <= TOL,
            "abs_diff_max": round(max(abs(s1 - study["share1_pct"]), abs(s2 - study["share2_pct"])), 3) if s1 is not None and s2 is not None else None,
            "condition_human": cond, "condition_study": study["condition"], "condition_agree": cond == study["condition"],
            "larger_human": larger, "larger_study": st_larger, "larger_agree": same_group(larger, st_larger),
            "has_note": bool(human.get("notes"))}


def main() -> None:
    c1_path, c2_path = sys.argv[1], sys.argv[2]
    key = json.loads((OUT / "sample_key.json").read_text())
    panel = {it["fresh_id"]: it for it in C.read_jsonl(C.CODES_ROOT / "results_submission1_tmlr" / "extended_panel.jsonl")}
    for it in panel.values():
        it["entity1"], it["entity2"] = it.get("entity1") or it["group1"], it.get("entity2") or it["group2"]
    c1, wb1 = task_a(c1_path)
    c2, _ = task_a(c2_path)
    per_item, summary = [], {}
    for checker, rows in (("checker1", c1), ("checker2", c2)):
        res = []
        for sid, h in sorted(rows.items()):
            fid = key["items"][sid]
            r = {"checker": checker, "item": sid, "comparison_id": fid, **compare(h, panel[fid])}
            res.append(r)
        per_item += res
        filled = [r for r in res if r["filled"]]
        summary[checker] = {"rows": len(res), "filled": len(filled),
                            "both_shares_agree": sum(r["share1_agree"] and r["share2_agree"] for r in filled),
                            "condition_agree": sum(r["condition_agree"] for r in filled),
                            "larger_agree": sum(r["larger_agree"] for r in filled),
                            "max_abs_diff_pp": max((r["abs_diff_max"] for r in filled), default=None),
                            "rows_with_notes": sum(r["has_note"] for r in res)}
    # checker 1 vs checker 2 on the shared rows
    shared = sorted(set(c1) & set(c2))
    inter = {"shared_rows": len(shared),
             "condition_agree": sum(norm(c1[s].get("condition")) == norm(c2[s].get("condition")) for s in shared),
             "larger_agree": sum(norm(c1[s].get("larger_group_or_roughly_equal")) == norm(c2[s].get("larger_group_or_roughly_equal")) for s in shared),
             "shares_within_tol": sum(all(fnum(c1[s].get(k)) is not None and fnum(c2[s].get(k)) is not None
                                          and abs(fnum(c1[s].get(k)) - fnum(c2[s].get(k))) <= TOL
                                          for k in ("share_1_percent", "share_2_percent")) for s in shared)}
    # Task B
    tb = task_b(wb1)
    cols = ["place_correct", "population_correct", "measure_correct", "groups_correct", "total_stated", "rule_stated", "same_meaning"]
    tb_sum = {c: dict(collections.Counter(str(r.get(c) or "").strip().upper() or "blank" for r in tb)) for c in cols}
    tb_flags = [{"item": r["item_id"], "wording": r["wording"], "comparison_id": key["items"][r["item_id"]],
                 "no": [c for c in cols if str(r.get(c) or "").strip().upper() == "N"], "note": r.get("notes")}
                for r in tb if any(str(r.get(c) or "").strip().upper() == "N" for c in cols) or r.get("notes")]
    notes_a = [{"checker": ch, "item": sid, "comparison_id": key["items"][sid], "note": h.get("notes")}
               for ch, rows in (("checker1", c1), ("checker2", c2)) for sid, h in sorted(rows.items()) if h.get("notes")]
    C.write_csv(OUT / "spotcheck_per_item.csv", per_item, columns=list(per_item[0]))
    C.write_json(OUT / "spotcheck_summary.json", {"tolerance_pp": TOL, "task_a": summary, "checker1_vs_checker2": inter,
                                                  "task_b_counts": tb_sum, "task_b_rows": len(tb)})
    C.write_json(OUT / "spotcheck_notes.json", {"task_a_notes": notes_a, "task_b_flags": tb_flags})
    print(json.dumps({"task_a": summary, "checker1_vs_checker2": inter, "task_b_counts": tb_sum}, indent=1))
    bad = [r for r in per_item if r["filled"] and not (r["share1_agree"] and r["share2_agree"] and r["condition_agree"] and r["larger_agree"])]
    print(f"\nrows with any disagreement: {len(bad)}")
    for r in bad:
        print(f"  {r['checker']} {r['item']} {r['comparison_id']}: human {r['share1_human']}/{r['share2_human']} {r['condition_human']} '{r['larger_human']}'"
              f" | study {r['share1_study']}/{r['share2_study']} {r['condition_study']} '{r['larger_study']}' | max diff {r['abs_diff_max']}")
    print(f"\nTask B rows flagged N or with a note: {len(tb_flags)}")


if __name__ == "__main__":
    main()
