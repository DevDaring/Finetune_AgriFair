"""R3 results: what the two readers found, and how far they agreed.

    python -m Submission1_Code_Phase2.r3_results

Reads only the scrubbed per-rater CSVs written by import_r3_ratings. Nothing here touches the
returned workbooks, and no value in the outputs identifies who did the work.

The headline question is not "how good is the advice" but "does the advice change when only the
asker changes". Identity pairs and context pairs carry opposite expectations and are never pooled:
on an identity pair a substantive change is a failure, on a context pair it is the expected
behaviour. Both are reported separately throughout.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from Submission1_Code_Phase2 import common as C


def _num(v) -> Optional[float]:
    v = str(v).strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None  # R2 wrote prose in some claim-count cells; treated as missing, never invented


def cohen_kappa(a: List, b: List, weights: Optional[str] = None) -> Optional[Dict]:
    """Cohen's kappa over the pairs where both readers gave a value."""
    pairs = [(x, y) for x, y in zip(a, b) if x not in (None, "") and y not in (None, "")]
    if len(pairs) < 2:
        return None
    cats = sorted({c for p in pairs for c in p}, key=str)
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return {"n": len(pairs), "categories": k, "kappa": None, "observed_agreement": 1.0,
                "note": "both readers used a single category; kappa is undefined"}
    m = np.zeros((k, k))
    for x, y in pairs:
        m[idx[x], idx[y]] += 1
    m /= m.sum()
    row, col = m.sum(1), m.sum(0)
    if weights == "linear":
        w = np.array([[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    else:
        w = np.eye(k)
    po = float((w * m).sum())
    pe = float((w * np.outer(row, col)).sum())
    kap = None if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return {"n": len(pairs), "categories": k, "kappa": None if kap is None else round(kap, 3),
            "observed_agreement": round(po, 3),
            "note": ("marginals are near-degenerate, so kappa is unstable and observed agreement "
                     "is the more honest number") if pe > 0.9 else ""}


def load(out: Path) -> Dict[str, List[Dict]]:
    per = {}
    for p in sorted(out.glob("r3_ratings_scrubbed_R*.csv")):
        rid = p.stem.split("_")[-1]
        per[rid] = list(csv.DictReader(p.open(encoding="utf-8")))
    if len(per) < 2:
        raise SystemExit("need both scrubbed rater files; run import_r3_ratings first")
    return per


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    cfg = C.load_config()
    out = C.CODES_ROOT / cfg["output_directory"] / "advice"
    per = load(out)
    key = {r["assessment_id"]: r for r in
           csv.DictReader((out / "KEEP_FROM_RATERS_r3_key.csv").open(encoding="utf-8"))}

    # ---- agreement between the two readers, per dimension -------------------------------------
    r1, r2 = per["R1"], per["R2"]
    by1 = {r["assessment_id"]: r for r in r1}
    ids = [r["assessment_id"] for r in r2 if r["assessment_id"] in by1]
    agree = {}
    for col, mode in (("pair_change_type", None), ("change_is_justified_yes_no_na", None),
                      ("left_unsafe_yes_no", None), ("right_unsafe_yes_no", None),
                      ("left_usefulness_0_2", "linear"), ("right_usefulness_0_2", "linear"),
                      ("left_essentials_covered_count", "linear"),
                      ("right_essentials_covered_count", "linear")):
        a = [by1[i].get(col, "") for i in ids]
        b = [next(r for r in r2 if r["assessment_id"] == i).get(col, "") for i in ids]
        agree[col] = cohen_kappa(a, b, mode)

    # ---- the headline: change behaviour, split by case type and system -------------------------
    change = collections.defaultdict(lambda: collections.Counter())
    justified = collections.defaultdict(lambda: collections.Counter())
    for rid, rows in per.items():
        for r in rows:
            k = key.get(r["assessment_id"])
            if not k:
                continue
            cell = (k["case_type"], k["system"], k["verbosity"])
            change[cell][r.get("pair_change_type", "") or "blank"] += 1
            justified[cell][r.get("change_is_justified_yes_no_na", "") or "blank"] += 1

    change_rows = []
    for (ct, sysname, verb), c in sorted(change.items()):
        n = sum(c.values())
        change_rows.append({
            "case_type": ct, "system": sysname, "verbosity": verb, "n_ratings": n,
            "none": c.get("none", 0), "presentation_only": c.get("presentation_only", 0),
            "substantive": c.get("substantive", 0),
            "substantive_share": round(c.get("substantive", 0) / n, 4) if n else None,
            "not_justified": justified[(ct, sysname, verb)].get("no", 0)})
    C.write_csv(out / "r3_change_by_system.csv", change_rows)

    # ---- quality: essential coverage and usefulness, by verbosity ------------------------------
    qual = collections.defaultdict(list)
    useful = collections.defaultdict(list)
    unsafe = collections.defaultdict(lambda: [0, 0])
    for rid, rows in per.items():
        for r in rows:
            k = key.get(r["assessment_id"])
            if not k:
                continue
            total = _num(r.get("essentials_total"))
            for side in ("left", "right"):
                cov = _num(r.get(f"{side}_essentials_covered_count"))
                if cov is not None and total:
                    qual[(k["system"], k["verbosity"], k["case_type"])].append(cov / total)
                u = _num(r.get(f"{side}_usefulness_0_2"))
                if u is not None:
                    useful[(k["system"], k["verbosity"])].append(u)
                s = str(r.get(f"{side}_unsafe_yes_no", "")).strip()
                if s in ("yes", "no"):
                    unsafe[(k["system"], k["verbosity"])][0] += (s == "yes")
                    unsafe[(k["system"], k["verbosity"])][1] += 1

    qual_rows = [{"system": s, "verbosity": v, "case_type": t, "n": len(x),
                  "essential_coverage_mean": round(float(np.mean(x)), 4)}
                 for (s, v, t), x in sorted(qual.items())]
    C.write_csv(out / "r3_quality_by_verbosity.csv", qual_rows)

    use_rows = [{"system": s, "verbosity": v, "n": len(x),
                 "usefulness_mean": round(float(np.mean(x)), 4),
                 "unsafe_flagged": unsafe[(s, v)][0], "unsafe_judged": unsafe[(s, v)][1]}
                for (s, v), x in sorted(useful.items())]
    C.write_csv(out / "r3_usefulness_by_verbosity.csv", use_rows)

    # ---- coverage of what the readers actually supplied -----------------------------------------
    supplied = {}
    for rid, rows in per.items():
        miss = sum(1 for r in rows
                   for side in ("left", "right")
                   if _num(r.get(f"{side}_unsupported_or_incorrect_claims")) is None
                   and str(r.get(f"{side}_unsupported_or_incorrect_claims", "")).strip())
        blank = sum(1 for r in rows
                    for side in ("left", "right")
                    if not str(r.get(f"{side}_unsupported_or_incorrect_claims", "")).strip())
        supplied[rid] = {"rows": len(rows), "claim_cells_describing_not_counting": miss,
                         "claim_cells_blank": blank}

    # ---- reliability diagnostics: where does the change judgement break down? ------------------
    idl = sorted(ids)

    def kap(sub, col="pair_change_type", fn=lambda v: v):
        return cohen_kappa([fn(by1[i].get(col, "")) for i in sub],
                           [fn(next(r for r in r2 if r["assessment_id"] == i).get(col, "")) for i in sub])

    binary = lambda v: "actions_changed" if v == "substantive" else "actions_same"
    reliability = {
        "three_way": agree["pair_change_type"],
        "binary_actions_changed": kap(idl, fn=binary),
        "by_case_type": {ct: kap([i for i in idl if key[i]["case_type"] == ct])
                         for ct in ("identity_irrelevant", "context_control")},
        "by_verbosity": {v: kap([i for i in idl if key[i]["verbosity"] == v])
                         for v in ("concise", "standard")},
        "by_quarter_of_the_sheet": {f"rows_{q * 48 + 1}_{(q + 1) * 48}": kap(idl[q * 48:(q + 1) * 48])
                                    for q in range(4)},
        "reading": ("The change judgement is not reliable enough to carry a headline claim. It is "
                    "at chance on the context controls, and it declines monotonically across the "
                    "sheet, which is the signature of fatigue in a single long sitting rather than "
                    "of a disagreement about meaning. Collapsing to a binary does not repair it."),
    }
    C.write_json(out / "r3_reliability.json", reliability)

    res = {"ratings_per_reader": {k: len(v) for k, v in per.items()},
           "agreement": agree,
           "reliability": reliability,
           "change_by_system": change_rows,
           "quality_by_verbosity": qual_rows,
           "usefulness_by_verbosity": use_rows,
           "claim_count_supply": supplied,
           "notes": [
               "Identity pairs and context pairs are reported separately and never pooled; a "
               "substantive change is a failure on the first and the expected behaviour on the second.",
               "One reader entered descriptions of the claims rather than counts in part of the "
               "unsupported-claims columns. Those cells are treated as missing for the numeric "
               "summary; no count was inferred from the text, because that would substitute the "
               "analyst's judgement for the reader's and corrupt the agreement statistic.",
               "24 cases is a diagnostic study, not a powered estimate of advice quality."]}
    C.write_json(out / "r3_results.json", res)
    print(json.dumps({"agreement": agree, "claim_count_supply": supplied}, indent=1))
    print(f"\nwrote r3_results.json, r3_change_by_system.csv, r3_quality_by_verbosity.csv, "
          f"r3_usefulness_by_verbosity.csv -> {out}")


if __name__ == "__main__":
    main()
