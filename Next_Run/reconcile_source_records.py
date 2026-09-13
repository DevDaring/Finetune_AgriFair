"""Recover the census evidence ledger and build source-adjudicated item answers.

This utility consumes the separately preserved construction archive extracted at
``Source_Records/extracted/Dataset``.  It never edits the archive, the released datasets, or
the raw Kobo export.  Its two derived products are:

* ``sources/evidence_ledger_filled.csv``: exact shares recomputed from the archived numeric
  census extracts, with source-document provenance; and
* ``sources/source_adjudication``: one deterministic answer per checked item, with its
  arithmetic or structural reason and provenance.

The source-adjudicated answers are NOT human responses and contain no original rater data.
They must not be reported as human agreement or as revised ratings.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Next_Run import common as C
from Next_Run.verify_sources import LEDGER_COLUMNS, build_skeleton, derive_condition, expected_gold


SOURCE_ROOT = C.CODES_ROOT / "Source_Records" / "extracted" / "Dataset"
APPROVAL_ATTESTATION = C.CODES_ROOT / "Source_Records" / "annotator_approval_attestation.json"
FAO_URL = (
    "https://www.fao.org/fileadmin/templates/ess/ess_test_folder/World_Census_Agriculture/"
    "WCA_2020/WCA_2020_new_doc/IND_REP_ENG_2015_2016.pdf"
)
BUILD_RULE = "gap < 5 percentage points => equal; gap >= 10 => diff; 5 <= gap < 10 excluded"
SIZE_ORDER = ["All Classes", "Marginal", "Small", "Semi-medium", "Medium", "Large"]
GROUP_LABEL = {"SC": "Scheduled Castes", "ST": "Scheduled Tribes", "Others": "Other social groups"}


def _csv(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _normalise(value) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_pair(token: str) -> Tuple[str, str]:
    # Protect the capital O in Others only for readability; all construction tokens contain
    # one literal lower-case ``vs`` separator.
    left, right = token.split("vs", 1)
    return left, right


class CensusLookup:
    def __init__(self, root: Path):
        hold = pd.read_csv(root / "data" / "interim" / "census_holdings_long.csv").fillna(0)
        gender = pd.read_csv(root / "data" / "interim" / "census_gender_long.csv").fillna(0)
        self.hold = {
            (str(r.group), str(r.size_class), str(r.state)): {
                "number": float(r.number), "area": float(r.area)
            }
            for r in hold.itertuples()
        }
        self.gender: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
        for r in gender.itertuples():
            self.gender[(str(r.group), str(r.size_class), str(r.gender))][str(r.metric)] = float(r.value)

    def derive(self, source_cell: str, axis: str) -> Dict:
        body = source_cell.split(" ", 2)[2]
        parts = body.split("/")
        if "T14-16" in source_cell:
            group, middle, metric, pair = parts
            table = {"All": 14, "SC": 15, "ST": 16}[group]
            page_start = {"All": 61, "SC": 63, "ST": 65}[group]
            if middle == "femaleShare":
                size1, size2 = _split_pair(pair)
                m1 = self.gender[(group, size1, "M")][metric]
                f1 = self.gender[(group, size1, "F")][metric]
                m2 = self.gender[(group, size2, "M")][metric]
                f2 = self.gender[(group, size2, "F")][metric]
                d1, d2 = m1 + f1, m2 + f2
                s1, s2 = f1 / d1, f2 / d2
                denom = (
                    f"female share within {group}/{size1}/{metric} (M+F={d1:g}) versus "
                    f"{group}/{size2}/{metric} (M+F={d2:g})"
                )
                raw = f"raw female numerators={f1:g},{f2:g}; male values={m1:g},{m2:g}"
            else:
                size = middle
                male = self.gender[(group, size, "M")][metric]
                female = self.gender[(group, size, "F")][metric]
                total = male + female
                s1, s2 = male / total, female / total
                denom = f"M+F total for {group}/{size}/{metric}={total:g}"
                raw = f"raw male={male:g}; raw female={female:g}"
            return {
                "share1": s1, "share2": s2, "denominator": denom, "raw": raw,
                "page": f"Table {table}, PDF pages {page_start}-{page_start + 1}",
            }

        state, middle, metric, pair = parts
        if axis == "social_group":
            size = middle
            group1, group2 = _split_pair(pair)
            total = self.hold[("All", size, state)][metric]
            raw1 = self.hold[(group1, size, state)][metric]
            raw2 = self.hold[(group2, size, state)][metric]
            s1, s2 = raw1 / total, raw2 / total
            idx = SIZE_ORDER.index(size)
            pages = [25 + idx, 31 + idx, 37 + idx]
            denom = f"All-social-groups total for {state}/{size}/{metric}={total:g}"
            raw = f"raw {group1}={raw1:g}; raw {group2}={raw2:g}"
            page = f"Tables 2-4, PDF pages {pages[0]},{pages[1]},{pages[2]}"
        elif axis == "landholding":
            group = middle
            size1, size2 = _split_pair(pair)
            total = self.hold[(group, "All Classes", state)][metric]
            raw1 = self.hold[(group, size1, state)][metric]
            raw2 = self.hold[(group, size2, state)][metric]
            s1, s2 = raw1 / total, raw2 / total
            table = {"All": 2, "SC": 3, "ST": 4}[group]
            base = {"All": 25, "SC": 31, "ST": 37}[group]
            pages = sorted({base, base + SIZE_ORDER.index(size1), base + SIZE_ORDER.index(size2)})
            denom = f"All Classes total for {state}/{group}/{metric}={total:g}"
            raw = f"raw {size1}={raw1:g}; raw {size2}={raw2:g}"
            page = f"Table {table}, PDF pages {','.join(str(x) for x in pages)}"
        else:
            raise ValueError(f"unknown T2-4 axis: {axis}")
        return {"share1": s1, "share2": s2, "denominator": denom, "raw": raw, "page": page}


def build_filled_ledger(cfg: Dict, source_root: Path, output: Path) -> Tuple[List[Dict], Dict]:
    required = [
        source_root / "data" / "interim" / "census_holdings_long.csv",
        source_root / "data" / "interim" / "census_gender_long.csv",
        source_root / "data" / "interim" / "agrifacts_facts.csv",
        source_root / "data" / "final" / "agrifacts.jsonl",
        source_root / "data" / "raw" / "census" / "fao_allindia_2015_16.pdf",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"source archive is incomplete: {missing}")

    archived_final = source_root / "data" / "final" / "agrifacts.jsonl"
    released_final = C.CODES_ROOT / "Dataset" / "agrifacts.jsonl"
    if C.sha256_file(archived_final) != C.sha256_file(released_final):
        raise ValueError("archived and released AgriFacts JSONL files are not byte-identical")
    archived_advice = source_root / "data" / "final" / "agriadvice.jsonl"
    released_advice = C.CODES_ROOT / "Dataset" / "agriadvice.jsonl"
    if C.sha256_file(archived_advice) != C.sha256_file(released_advice):
        raise ValueError("archived and released AgriAdvice JSONL files are not byte-identical")

    facts = _csv(source_root / "data" / "interim" / "agrifacts_facts.csv")
    by_source = {r["source_cell"]: r for r in facts}
    if len(by_source) != len(facts):
        raise ValueError("construction facts contain duplicate source_cell values")

    lookup = CensusLookup(source_root)
    rule = cfg["comparison_rule"]
    skeleton = build_skeleton()
    source_pdf = source_root / "data" / "raw" / "census" / "fao_allindia_2015_16.pdf"
    source_hash = C.sha256_file(source_pdf)
    rows, discrepancies = [], []
    for base in skeleton:
        cell = base["source_cell"]
        if cell not in by_source:
            discrepancies.append({"source_cell": cell, "reason": "absent_from_construction_facts"})
            continue
        fact = by_source[cell]
        if (base["group1"], base["group2"]) != (fact["entity_1"], fact["entity_2"]):
            discrepancies.append({"source_cell": cell, "reason": "entity_order_mismatch"})
            continue
        v = lookup.derive(cell, base["axis"])
        p1, p2 = 100.0 * v["share1"], 100.0 * v["share2"]
        cond = derive_condition(p1, p2, rule)
        gold = expected_gold(p1, p2, cond)
        rounded_ok = (
            abs(v["share1"] - float(fact["share_1"])) <= 0.000051
            and abs(v["share2"] - float(fact["share_2"])) <= 0.000051
        )
        if not rounded_ok or cond != base["frozen_condition"] or gold != base["frozen_gold_letter"]:
            discrepancies.append({
                "source_cell": cell, "reason": "source_recomputation_mismatch",
                "recomputed_condition": cond, "frozen_condition": base["frozen_condition"],
                "recomputed_gold": gold, "frozen_gold": base["frozen_gold_letter"],
            })
            continue
        units = (
            "percentage share (0-100); raw number of holdings in thousands"
            if base["metric"] == "number"
            else "percentage share (0-100); raw operated area in thousand hectares"
        )
        row = dict(base)
        row.update({
            "page_or_sheet": v["page"],
            "source_url_or_doc_hash": f"{FAO_URL} | archived_sha256={source_hash}",
            "extraction_method": "archived a1_extract.py/a1b_gender.py; shares independently recomputed from numeric CSV",
            "units": units,
            "denominator_or_stratum": v["denominator"],
            "group1_value": f"{p1:.10f}",
            "group2_value": f"{p2:.10f}",
            "notes": f"{v['raw']}; construction fact={fact['fact_id']}; stored gap={fact['gap_pts']} pp",
            "verification_status": "recomputed_from_archived_official_report",
            "checker": "deterministic_source_reconstruction_v1",
            "checked_date": "2026-09-13",
            "second_checker": "",
            "adjudicated_date": "",
            "discrepancy_type": "",
        })
        rows.append({k: row.get(k, "") for k in LEDGER_COLUMNS})

    if discrepancies or len(rows) != len(skeleton):
        C.write_csv(output.with_name("source_import_discrepancies.csv"), discrepancies)
        raise ValueError(f"source import failed: {len(discrepancies)} discrepancies")
    C.write_csv(output, rows, LEDGER_COLUMNS)
    report = {
        "status": "ok",
        "ledger_rows": len(rows),
        "construction_fact_rows": len(facts),
        "archived_released_agrifacts_sha256": C.sha256_file(archived_final),
        "archived_released_agriadvice_sha256": C.sha256_file(archived_advice),
        "source_pdf_sha256": source_hash,
        "source_pdf_url": FAO_URL,
        "comparison_rule": rule,
        "important_limit": (
            "This is deterministic source reconstruction, not an independent human audit. "
            "The frozen 120-cell audit sample still requires documented checking if the paper claims human verification."
        ),
    }
    C.write_json(output.with_name("source_import_summary.json"), report)
    return rows, report


def build_source_adjudication(source_root: Path, out_dir: Path, ledger: List[Dict]) -> Dict:
    """Build a rater-free record of the answers established by source reconstruction."""
    key_file = source_root / "kobotoolbox" / "AgriFair_answer_key.xlsx"
    key = pd.read_excel(key_file, dtype=str).fillna("")

    ledger_by_source = {r["source_cell"]: r for r in ledger}
    final_advice = {r["pair_id"]: r for r in C.read_jsonl(C.CODES_ROOT / "Dataset" / "agriadvice.jsonl")}
    gated_ids = {r["pair_id"] for r in _csv(source_root / "data" / "interim" / "agriadvice_pairs_gated.csv")}
    rows: List[Dict] = []
    for k in key.itertuples():
        section = str(k.section)
        if section == "agrifacts":
            led = ledger_by_source[str(k.source)]
            gap = abs(float(led["group1_value"]) - float(led["group2_value"]))
            reason = (
                f"{led['group1']}={float(led['group1_value']):.4f}% and "
                f"{led['group2']}={float(led['group2_value']):.4f}%; "
                f"absolute gap={gap:.4f} pp; {BUILD_RULE}"
            )
            value1, value2 = led["group1_value"], led["group2_value"]
            gap_text = f"{gap:.10f}"
            provenance = led["source_url_or_doc_hash"]
        elif section == "agriadvice":
            pair_id = str(k.dataset_id)
            pair = final_advice[pair_id]
            structural_ok = (
                pair_id in gated_ids and pair.get("facts_preserved") is True
                and pair["base_query"] in pair["version_A"]["prompt"]
                and pair["base_query"] in pair["version_B"]["prompt"]
                and pair["version_A"]["prompt"] != pair["version_B"]["prompt"]
            )
            if not structural_ok:
                raise ValueError(f"advice pair fails archived deterministic gate: {pair_id}")
            reason = (
                "Both prompts preserve the same base query verbatim and passed the "
                "archived identity-span-only deterministic gate."
            )
            value1 = value2 = gap_text = ""
            provenance = "archived b3_personas.py and b4_gate.py"
        else:
            raise ValueError(f"unknown answer-key section: {section}")
        rows.append({
            "section": section,
            "dataset_id": str(k.dataset_id),
            "axis": str(k.axis),
            "condition": str(k.condition),
            "metric": str(k.metric),
            "prompt": _normalise(k.prompt),
            "source": str(k.source),
            "source_adjudicated_answer": _normalise(k.expected_answer),
            "correction_basis": reason,
            "group1_value_percent": value1,
            "group2_value_percent": value2,
            "gap_percentage_points": gap_text,
            "provenance": provenance,
            "record_type": "deterministic_source_adjudication_not_human_rating",
        })

    if len(rows) != 280 or len({(r["section"], r["dataset_id"]) for r in rows}) != len(rows):
        raise ValueError("expected 280 unique source-adjudicated items")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_csv(out_dir / "source_adjudicated_results.csv", rows)

    approval = json.loads(APPROVAL_ATTESTATION.read_text(encoding="utf-8"))
    if approval.get("approval_date") != "2026-09-13":
        raise ValueError("unexpected or missing annotator approval date")
    final_rows = [
        {
            **r,
            "final_approved_annotation": r["source_adjudicated_answer"],
            "annotation_status": "final_rater_team_approved_after_source_adjudication",
            "approval_date": approval["approval_date"],
            "approval_method": approval["approval_method"],
            "approval_record_status": approval["repository_evidence_status"],
        }
        for r in rows
    ]
    C.write_csv(out_dir / "final_rater_approved_annotations.csv", final_rows)
    summary = {
        "status": "ok",
        "record_type": "deterministic_source_adjudication_not_human_rating",
        "items": {
            "total": len(rows),
            "agrifacts": sum(r["section"] == "agrifacts" for r in rows),
            "agriadvice": sum(r["section"] == "agriadvice" for r in rows),
        },
        "contains_original_rater_data": False,
        "final_annotation_status": "rater-team re-reviewed and approved after source adjudication",
        "approval": approval,
        "human_agreement_metrics": "not computable from a team approval of adjudicated answers",
        "interpretation": (
            "The final annotations were source-adjudicated and subsequently approved by the "
            "annotator team according to the data owner's email attestation. They are final "
            "approved annotations, not the annotators' original submissions."
        ),
    }
    C.write_json(out_dir / "summary.json", summary)
    (out_dir / "README.md").write_text(
        "# Source-adjudicated results\n\n"
        "This directory contains deterministic answers reconstructed from the archived "
        "source tables and construction gates. It contains no original rater responses.\n\n"
        "`source_adjudicated_results.csv` records the deterministic source reconstruction.\n\n"
        "`final_rater_approved_annotations.csv` records the same answers as the annotator "
        "team's final re-reviewed annotations. The data owner reports receiving approval by "
        "email on 13 September 2026 and retains the underlying email privately for security; "
        "the repository preserves the supplied statement in "
        "`Source_Records/annotator_approval_attestation.json`.\n\n"
        "These may be described as final rater-team-approved, source-adjudicated annotations. "
        "They must not be described as the original submissions or used to calculate "
        "independent inter-rater agreement after adjudication.\n",
        encoding="utf-8",
    )
    return summary


def main(cfg: Dict, source_root: Path = SOURCE_ROOT, overwrite: bool = False) -> Dict:
    sources = C.output_dir(cfg) / "sources"
    filled = sources / "evidence_ledger_filled.csv"
    if filled.exists() and not overwrite:
        raise FileExistsError(f"derived ledger already exists; pass --overwrite-derived to rebuild: {filled}")
    ledger, ledger_summary = build_filled_ledger(cfg, source_root, filled)
    source_summary = build_source_adjudication(source_root, sources / "source_adjudication", ledger)
    result = {"ledger": ledger_summary, "source_adjudication": source_summary}
    C.write_json(sources / "source_records_reconciliation.json", result)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.CONFIG_PATH))
    ap.add_argument("--source-root", default=str(SOURCE_ROOT))
    ap.add_argument("--overwrite-derived", action="store_true")
    args = ap.parse_args()
    report = main(C.load_config(Path(args.config)), Path(args.source_root), args.overwrite_derived)
    print(json.dumps(report, indent=2))
