"""Import the returned rater protocol pack (pack 1), research content only.

    python -m Submission1_Code_Phase2.import_rater_protocol --dir Submission1/phase2_rater_pack/Reviewed_Doc

What is read: the rubric sign-off decisions, the four relevance checks on each case, the
calibration scores, and any free-text that describes the *task* (a proposed anchor change, a
reason a case is unsafe to score).

What is never read: `reader_initials_optional`, any name, employer, contact or date-of-work cell,
and anything on a sheet whose purpose is to identify who did the work. Readers are referred to as
reader A and reader B, assigned by filename order only.

The output decides three things before any model runs:
  - whether the rubric is frozen as written, or what must change first;
  - which of the 24 cases survive review, and why any were dropped;
  - whether the two readers are calibrated closely enough to start.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openpyxl import load_workbook

from Submission1_Code_Phase2 import common as C

IDENTITY_COLUMNS = {"reader_initials_optional", "reader_id", "date", "date_completed", "name",
                    "initials", "email", "affiliation", "organisation", "organization", "employer"}
IDENTITY_PATTERNS = re.compile(r"(name|initial|email|affili|organi|employer|contact|phone|signature)", re.I)
CHECKS = ("question_is_answerable", "essentials_are_checkable", "expectation_is_right", "safe_to_score")


_INITIALS = re.compile(r"\b[A-Z]{2,3}\b")
_KEEP_ACRONYMS = {"SC", "ST", "OBC", "AWD", "FYM", "NPK", "GPU", "CPU", "JSON", "URL", "PDF", "LLM",
                  "T2", "T4", "T20", "AP", "MP", "UP", "TN", "WB", "HP", "JK", "IND", "ENG", "R1", "R2", "R3", "R4"}


def scrub_identity(text: str) -> str:
    """Remove person initials from free text. Domain acronyms are kept; a capitalised token that
    is not one of them is replaced, because reviewer initials must not enter the results tree."""
    if not text:
        return text
    def sub(m):
        tok = m.group(0)
        return tok if tok in _KEEP_ACRONYMS else "[reviewer]"
    out = _INITIALS.sub(sub, text)
    return re.sub(r"(\[reviewer\]\W+){2,}", "[reviewers] ", out)


def read_sheet(path: Path, sheet: str) -> List[Dict]:
    """Rows as dicts. Identity columns are dropped here, before any value leaves the function."""
    wb = load_workbook(path, data_only=True)
    if sheet not in wb.sheetnames:
        return []
    ws = wb[sheet]
    hdr = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    keep = [i for i, h in enumerate(hdr)
            if h and h.lower() not in IDENTITY_COLUMNS and not IDENTITY_PATTERNS.search(h)]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if all(v in (None, "") for v in r):
            continue
        rows.append({hdr[i]: ("" if r[i] is None else scrub_identity(str(r[i]).strip())) for i in keep})
    return rows


def read_overall(path: Path) -> Dict:
    """The two-question sign-off sheet, with any date or identity row skipped."""
    wb = load_workbook(path, data_only=True)
    if "Overall" not in wb.sheetnames:
        return {}
    out = {}
    for row in wb["Overall"].iter_rows(values_only=True):
        if not row or row[0] in (None, ""):
            continue
        k = str(row[0]).strip()
        if IDENTITY_PATTERNS.search(k) or k.lower() in IDENTITY_COLUMNS:
            continue
        out[k] = "" if len(row) < 2 or row[1] is None else scrub_identity(str(row[1]).strip())
    return out


def _pair(paths: List[Path]) -> List[Tuple[str, Path]]:
    return [(f"reader_{chr(65 + i)}", p) for i, p in enumerate(sorted(paths))]


# ------------------------------------------------------------------ rubric

def import_rubric(paths: List[Path], out: Path) -> Dict:
    per_reader, rows = {}, []
    for label, p in _pair(paths):
        sheet = read_sheet(p, "Rubric sign-off")
        per_reader[label] = {r["dimension"]: r for r in sheet if r.get("dimension")}
        for r in sheet:
            rows.append({"reader": label, "dimension": r.get("dimension", ""),
                         "anchors_are_clear": r.get("anchors_are_clear", ""),
                         "proposed_change": r.get("proposed_change", "")})
        per_reader[label]["_overall"] = read_overall(p)
    C.write_csv(out / "rubric_signoff_long.csv", rows)
    dims = sorted({r["dimension"] for r in rows if r["dimension"]})
    needs_change = {d: [lab for lab in per_reader
                        if per_reader[lab].get(d, {}).get("anchors_are_clear", "").lower() == "needs_change"]
                    for d in dims}
    changes = [{"dimension": d, "raised_by_n_readers": len(v),
                "proposed_changes": [per_reader[lab][d].get("proposed_change", "") for lab in v]}
               for d, v in needs_change.items() if v]
    overall = {lab: per_reader[lab].get("_overall", {}) for lab in per_reader}
    frozen_votes = []
    for lab, o in overall.items():
        val = next((v for k, v in o.items() if "frozen" in k.lower()), "")
        frozen_votes.append(val.strip().lower())
    summary = {"readers": len(paths), "dimensions": len(dims),
               "dimensions_flagged": [c["dimension"] for c in changes],
               "requested_changes": changes,
               "freeze_votes": frozen_votes,
               "rubric_frozen": all(v.startswith("y") for v in frozen_votes if v),
               "blocking_conditions": [next((v for k, v in o.items() if "must change" in k.lower()), "")
                                       for o in overall.values()
                                       if next((v for k, v in o.items() if "must change" in k.lower()), "")]}
    C.write_json(out / "rubric_signoff_summary.json", summary)
    return summary


# ------------------------------------------------------------------ case relevance

def import_cases(cfg: Dict, paths: List[Path], out: Path) -> Dict:
    per_reader = {}
    for label, p in _pair(paths):
        per_reader[label] = {r["case_id"]: r for r in read_sheet(p, "Cases") if r.get("case_id")}
    ids = sorted(set().union(*[set(v) for v in per_reader.values()]))
    long_rows, verdict = [], {}
    for cid in ids:
        flags, comments = {}, []
        for lab, d in per_reader.items():
            r = d.get(cid, {})
            row = {"reader": lab, "case_id": cid, "case_type": r.get("case_type", "")}
            for c in CHECKS:
                row[c] = (r.get(c, "") or "").strip().lower()
                flags.setdefault(c, []).append(row[c])
            row["comment"] = r.get("comment", "")
            if row["comment"]:
                comments.append(f"{lab}: {row['comment']}")
            long_rows.append(row)
        any_no = {c: any(v == "no" for v in vals) for c, vals in flags.items()}
        verdict[cid] = {"case_id": cid,
                        "case_type": next((per_reader[l][cid].get("case_type", "") for l in per_reader if cid in per_reader[l]), ""),
                        **{f"{c}_any_no": any_no[c] for c in CHECKS},
                        "status": "hold" if any(any_no.values()) else "accepted",
                        "reasons": "; ".join(comments)}
    C.write_csv(out / "case_relevance_long.csv", long_rows)
    C.write_csv(out / "case_relevance_verdicts.csv", list(verdict.values()))
    held = [v for v in verdict.values() if v["status"] == "hold"]
    accepted = [v["case_id"] for v in verdict.values() if v["status"] == "accepted"]
    summary = {"cases_reviewed": len(ids), "accepted": len(accepted), "held_for_repair_or_drop": len(held),
               "held_cases": [{"case_id": h["case_id"], "case_type": h["case_type"],
                               "failed_checks": [c for c in CHECKS if h[f"{c}_any_no"]],
                               "reasons": h["reasons"]} for h in held],
               "accepted_by_type": dict(collections.Counter(
                   v["case_type"] for v in verdict.values() if v["status"] == "accepted")),
               "agreement_per_check": {c: round(sum(
                   1 for cid in ids
                   if len({(per_reader[l].get(cid, {}).get(c, "") or "").strip().lower()
                           for l in per_reader if cid in per_reader[l]}) == 1) / max(1, len(ids)), 4) for c in CHECKS},
               "note": ("A case is held if either reader answered 'no' to any check. Held cases are repaired or "
                        "dropped now, before generation; nothing is dropped after answers exist.")}
    C.write_json(out / "case_relevance_summary.json", summary)
    C.write_json(out / "accepted_case_ids.json", {"accepted": accepted, "held": [h["case_id"] for h in held]})
    return summary


# ------------------------------------------------------------------ calibration

def import_practice(paths: List[Path], out: Path) -> Dict:
    per_reader = {}
    for label, p in _pair(paths):
        per_reader[label] = {r["practice_id"]: r for r in read_sheet(p, "Practice") if r.get("practice_id")}
    ids = sorted(set().union(*[set(v) for v in per_reader.values()]))
    labs = sorted(per_reader)
    rows, diffs = [], []
    for pid in ids:
        rec = {"practice_id": pid}
        for lab in labs:
            r = per_reader[lab].get(pid, {})
            for side in ("left", "right"):
                for field, key in (("essentials", f"{side}_essentials_covered_count"),
                                   ("unsupported", f"{side}_unsupported_or_incorrect_claims"),
                                   ("unsafe", f"{side}_unsafe_yes_no"),
                                   ("useful", f"{side}_usefulness_0_2")):
                    rec[f"{lab}_{side}_{field}"] = r.get(key, "")
            rec[f"{lab}_change_type"] = r.get("pair_change_type", "")
            rec[f"{lab}_justified"] = r.get("change_is_justified_yes_no_na", "")
        rows.append(rec)
        if len(labs) == 2:
            for side in ("left", "right"):
                for field in ("essentials", "unsupported", "useful"):
                    a, b = rec.get(f"{labs[0]}_{side}_{field}", ""), rec.get(f"{labs[1]}_{side}_{field}", "")
                    try:
                        gap = abs(float(a) - float(b))
                    except (TypeError, ValueError):
                        continue
                    if gap > 1:
                        diffs.append({"practice_id": pid, "side": side, "field": field, "gap": gap})
    C.write_csv(out / "practice_calibration.csv", rows)
    agree = {}
    if len(labs) == 2:
        for field in ("change_type", "justified"):
            vals = [(r.get(f"{labs[0]}_{field}", ""), r.get(f"{labs[1]}_{field}", "")) for r in rows]
            vals = [(a, b) for a, b in vals if a and b]
            agree[field] = round(sum(a == b for a, b in vals) / max(1, len(vals)), 4)
    summary = {"practice_items": len(ids), "readers": len(labs),
               "categorical_agreement": agree,
               "count_gaps_over_one_point": diffs,
               "calibrated": not diffs,
               "note": ("Calibration is judged on whether the two readers' counts are within one point and their "
                        "categorical judgements match. Practice answers were hand-written and never enter a result.")}
    C.write_json(out / "practice_calibration_summary.json", summary)
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, required=True); a = ap.parse_args(argv)
    cfg = C.load_config()
    d = a.dir if a.dir.is_absolute() else (C.CODES_ROOT.parent / a.dir)
    out = C.out_dir(cfg, "rater_protocol")
    res = {}
    rub = sorted(d.glob("Rubric_review*.xlsx"))
    if rub:
        res["rubric"] = import_rubric(rub, out)
    cas = sorted(d.glob("Case_relevance_review*.xlsx"))
    if cas:
        res["cases"] = import_cases(cfg, cas, out)
    pra = sorted(d.glob("Practice_set*.xlsx"))
    if pra:
        res["calibration"] = import_practice(pra, out)
    C.write_json(out / "rater_protocol_manifest.json",
                 {"source_directory": str(d), "identity_columns_dropped": sorted(IDENTITY_COLUMNS),
                  "identity_pattern": IDENTITY_PATTERNS.pattern,
                  "readers_labelled": "positionally by filename (reader_A, reader_B); no identity stored",
                  **res})
    for k, v in res.items():
        print(f"\n=== {k}")
        print(json.dumps({kk: vv for kk, vv in v.items() if kk != "note"}, indent=1)[:1600])
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
