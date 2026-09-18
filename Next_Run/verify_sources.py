"""P1 - source ledger: establish that the agricultural ground truth is trustworthy.

The released benchmark contains qualitative rationales only.  If the separately preserved
construction archive is unavailable, this stage cannot fill the ledger and must not invent
values.  Codes/Next_Run/reconcile_source_records.py can import the recovered numeric records
into a provenance-rich filled ledger.  This stage deterministically

  1. build the ledger SKELETON - one row per distinct source_cell across all splits, with
     the key decomposed into edition / table / state / size class / metric / comparison,
     the items it anchors, and blank value columns for the authors to fill from their
     construction records;
  2. freeze the audit SAMPLE - 60 then 120 distinct comparisons, stratified by axis x
     condition, drawn by the analysis seed BEFORE any model error is inspected;
  3. VALIDATE a filled ledger and RECOMPUTE gold from the declared comparison rule,
     producing a discrepancy report against the frozen labels.

Step 3 refuses to run unless both the rule in config and the numeric values are present.
It never modifies legacy data; corrected labels live only in the audit output directory
with a change log, as the plan requires.
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Next_Run import common as C

LEDGER_COLUMNS = [
    # identity
    "source_cell", "edition", "table", "state", "size_class", "metric", "comparison", "axis",
    "group1", "group2", "n_items_total", "n_items_train", "n_items_validation", "n_items_test",
    "frozen_condition", "frozen_gold_letter",
    # to be filled from primary records - NEVER auto-filled
    "page_or_sheet", "source_url_or_doc_hash", "extraction_method", "units", "denominator_or_stratum",
    "group1_value", "group2_value", "notes",
    # verification bookkeeping
    "verification_status", "checker", "checked_date", "second_checker", "adjudicated_date",
    "discrepancy_type",
]


def build_skeleton() -> List[Dict]:
    """One ledger row per distinct source_cell across train/validation/test."""
    by_cell: Dict[str, Dict] = {}
    counts = defaultdict(Counter)
    for split in ("train", "validation", "test"):
        for r in C.load_split(split):
            cell = r["source_cell"]
            counts[cell][split] += 1
            if cell not in by_cell:
                parts = C.parse_source_cell(cell)
                by_cell[cell] = {
                    "source_cell": cell, "edition": parts["edition"], "table": parts["table"],
                    "state": parts["state"], "size_class": parts["size_class"], "metric": parts["metric"],
                    "comparison": parts["comparison"], "axis": r["category"],
                    "group1": r["group1"], "group2": r["group2"],
                    "frozen_condition": r["condition"], "frozen_gold_letter": r["correct_answer"],
                    "verification_status": "unverified",
                }
            elif by_cell[cell]["frozen_condition"] != r["condition"]:
                # the same cell with two conditions is a construction inconsistency worth seeing
                by_cell[cell]["notes"] = "INCONSISTENT condition across paraphrases"
    rows = []
    for cell, row in sorted(by_cell.items()):
        c = counts[cell]
        row.update({"n_items_total": sum(c.values()), "n_items_train": c["train"],
                    "n_items_validation": c["validation"], "n_items_test": c["test"]})
        rows.append({k: row.get(k, "") for k in LEDGER_COLUMNS})
    return rows


def freeze_audit_sample(rows: List[Dict], seed: int, stage1: int, stage2_total: int) -> List[Dict]:
    """Stratified sample of distinct comparisons: axis x frozen_condition strata, drawn once by
    seed. Stage 2 is a superset of stage 1 (120 total, not 60 + 120). Test-split cells are
    preferred so that at least half the audited comparisons bear on the frozen test set."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["axis"], r["frozen_condition"])].append(r["source_cell"])
    for k in strata:
        # prefer cells with test items; shuffle within preference groups
        with_test = [c for c in strata[k] if next(x for x in rows if x["source_cell"] == c)["n_items_test"] > 0]
        without = [c for c in strata[k] if c not in set(with_test)]
        rng.shuffle(with_test); rng.shuffle(without)
        strata[k] = with_test + without
    n_strata = len(strata)
    per1 = max(1, stage1 // n_strata)
    per2 = max(per1, stage2_total // n_strata)
    sample = []
    shortfall = []
    for k, cells in sorted(strata.items()):
        take = cells[:per2]
        if len(take) < per2:
            shortfall.append({"stratum": f"{k[0]}/{k[1]}", "wanted": per2, "available": len(cells)})
        for i, cell in enumerate(take):
            sample.append({"source_cell": cell, "axis": k[0], "frozen_condition": k[1],
                           "audit_stage": 1 if i < per1 else 2, "order_in_stratum": i + 1})
    if shortfall:
        print("[verify_sources] audit sample shortfall (document, do not fill):", shortfall)
    return sample


def _rule_ready(cfg: Dict) -> Tuple[bool, str]:
    rule = cfg.get("comparison_rule") or {}
    equal_cut = rule.get("equal_if_abs_gap_below")
    if equal_cut is None:
        equal_cut = rule.get("equal_if_abs_gap_at_most")  # legacy smoke/config compatibility
    if equal_cut is None or rule.get("diff_if_abs_gap_at_least") is None:
        return False, "comparison_rule thresholds are null in config - declare them from the construction records"
    if float(equal_cut) > float(rule["diff_if_abs_gap_at_least"]):
        return False, "equal threshold exceeds diff threshold; the excluded band would be negative"
    return True, ""


def derive_condition(g1: float, g2: float, rule: Dict) -> str:
    """Apply the declared rule. Returns 'equal', 'diff', or 'excluded_band'."""
    gap = abs(float(g1) - float(g2))
    if rule.get("metric_unit") == "relative_percent":
        base = max(abs(float(g1)), abs(float(g2)), 1e-12)
        gap = 100.0 * gap / base
    if rule.get("equal_if_abs_gap_below") is not None:
        is_equal = gap < float(rule["equal_if_abs_gap_below"])
    else:  # explicit legacy semantics
        is_equal = gap <= float(rule["equal_if_abs_gap_at_most"])
    if is_equal:
        return "equal"
    if gap >= float(rule["diff_if_abs_gap_at_least"]):
        return "diff"
    return "excluded_band"


def expected_gold(g1: float, g2: float, condition: str, group1_is_choice: str = "a") -> str:
    """Canonical letter implied by the values: c for equal; otherwise the larger group's letter.
    group1_is_choice says which canonical letter group1 maps to (a by construction here)."""
    if condition != "diff":
        return C.EQUAL_LETTER
    other = "b" if group1_is_choice == "a" else "a"
    return group1_is_choice if float(g1) > float(g2) else other


def validate_and_recompute(ledger: List[Dict], cfg: Dict) -> Tuple[List[Dict], Dict]:
    """Recompute condition and gold for every row with both values present. Report agreement
    with the frozen labels; never rewrite them."""
    ok, why = _rule_ready(cfg)
    if not ok:
        return [], {"status": "blocked", "reason": why}
    rule = cfg["comparison_rule"]
    report, agree, disagree, band, empty = [], 0, 0, 0, 0
    for r in ledger:
        v1, v2 = str(r.get("group1_value", "")).strip(), str(r.get("group2_value", "")).strip()
        if not v1 or not v2:
            empty += 1
            continue
        try:
            g1, g2 = float(v1), float(v2)
        except ValueError:
            report.append({**r, "recomputed_condition": "unparseable_value", "recomputed_gold": "",
                           "agrees_with_frozen": False, "discrepancy_type": "unparseable_value"})
            disagree += 1
            continue
        cond = derive_condition(g1, g2, rule)
        gold = expected_gold(g1, g2, cond)
        if cond == "excluded_band":
            band += 1
            dt = "frozen_item_falls_in_excluded_band"
            agrees = False
        else:
            agrees = (cond == r["frozen_condition"]) and (gold == r["frozen_gold_letter"])
            dt = "" if agrees else ("condition_mismatch" if cond != r["frozen_condition"] else "group_direction_mismatch")
        agree += agrees; disagree += (not agrees) and cond != "excluded_band"
        report.append({**r, "recomputed_condition": cond, "recomputed_gold": gold,
                       "agrees_with_frozen": agrees, "discrepancy_type": dt})
    missing_provenance = sum(
        not r.get("page_or_sheet") or not r.get("source_url_or_doc_hash")
        or not r.get("denominator_or_stratum") or not r.get("verification_status")
        or r.get("verification_status") == "unverified"
        for r in report
    )
    summary = {"status": "ok", "rows_with_values": len(report), "rows_without_values": empty,
               "agree": agree, "disagree": disagree, "excluded_band": band,
               "rows_missing_provenance_or_status": missing_provenance,
               "by_stratum": {}}
    strat = defaultdict(lambda: Counter())
    for x in report:
        strat[f"{x['axis']}/{x['frozen_condition']}"][("agree" if x["agrees_with_frozen"] else "disagree")] += 1
    summary["by_stratum"] = {k: dict(v) for k, v in strat.items()}
    return report, summary


def main(cfg: Dict) -> Dict:
    out = C.output_dir(cfg) / "sources"
    out.mkdir(parents=True, exist_ok=True)
    skeleton_path = out / "evidence_ledger.csv"
    filled_path = out / "evidence_ledger_filled.csv"

    skeleton = build_skeleton()
    if not skeleton_path.exists():
        C.write_csv(skeleton_path, skeleton, LEDGER_COLUMNS)
        print(f"[verify_sources] ledger skeleton: {len(skeleton)} distinct source cells -> {skeleton_path}")
    else:
        print(f"[verify_sources] ledger skeleton already present ({skeleton_path}); not overwritten")

    sample = freeze_audit_sample(skeleton, cfg["analysis_seed"], cfg["audit_sample_stage1"], cfg["audit_sample_stage2_total"])
    sample_path = out / "audit_sample_frozen.csv"
    if not sample_path.exists():
        C.write_csv(sample_path, sample)
        print(f"[verify_sources] audit sample frozen: {sum(1 for s in sample if s['audit_stage']==1)} stage-1, {len(sample)} total")

    if C.smoke(cfg).get("synthetic_ledger") and not filled_path.exists():
        # SMOKE ONLY: fabricate values consistent with each row's frozen label so the recompute
        # and the evidence panel have something to chew on. Written to the smoke directory,
        # labelled in every row, and never to the real ledger.
        import random as _r
        rng = _r.Random(cfg["analysis_seed"]); rule = cfg["comparison_rule"]
        fake = []
        for r in skeleton:
            base = rng.uniform(5, 35)   # shares of one total: base + gap stays under 100 so the synthetic swap is feasible
            equal_cut = rule.get("equal_if_abs_gap_below", rule.get("equal_if_abs_gap_at_most"))
            gap = rng.uniform(0, equal_cut * 0.999) if r["frozen_condition"] == "equal" \
                  else rng.uniform(rule["diff_if_abs_gap_at_least"], rule["diff_if_abs_gap_at_least"] + 15)
            g1 = base + gap if r["frozen_gold_letter"] == "a" else base
            g2 = base if r["frozen_gold_letter"] == "a" else base + gap
            fake.append({**r, "group1_value": round(g1, 2), "group2_value": round(g2, 2), "units": "percent",
                         "denominator_or_stratum": "SYNTHETIC", "verification_status": "SYNTHETIC_SMOKE_VALUE",
                         "notes": "smoke-test fabrication; not a census value"})
        C.write_csv(filled_path, fake, LEDGER_COLUMNS)
        print(f"[verify_sources] SMOKE: synthetic ledger written to {filled_path.name} (not real data)")
    result = {"skeleton_rows": len(skeleton), "audit_sample": len(sample), "recompute": {"status": "not_run"}}
    if filled_path.exists():
        with open(filled_path, encoding="utf-8") as f:
            ledger = list(csv.DictReader(f))
        report, summary = validate_and_recompute(ledger, cfg)
        result["recompute"] = summary
        if report:
            C.write_csv(out / "discrepancy_report.csv", report)
            print(f"[verify_sources] recompute: {summary}")
        else:
            print(f"[verify_sources] recompute BLOCKED: {summary.get('reason')}")
    else:
        print(f"[verify_sources] no filled ledger at {filled_path.name}; recompute not run. "
              "Gate (plan section 6.2): without recoverable source values, stop optional GPU spending.")
    C.write_json(out / "verify_sources_summary.json", result)
    return result


if __name__ == "__main__":
    main(C.load_config())
