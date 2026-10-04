"""Human spot-check pack for the 120 new verified comparisons (TMLR study).

    python -m Submission1_TMLR.make_spotcheck_pack

Draws 30 of the 120 new comparisons (15 per axis, seeded, each axis with at least 3 items counted in
holdings), and writes a blind pack for two checkers:

  Checker1_spotcheck.xlsx  Task A: recompute all 30 from the census report.
                           Task B: check that both question wordings match the row's source fields.
  Checker2_spotcheck.xlsx  Task A only, on 12 of the 30, done without seeing Checker 1's file.
  INSTRUCTIONS.txt         the same instructions as the workbooks' first sheet.

No file contains a share, gap, label, model answer or earlier rating. Items carry neutral ids
(S01-S30); the mapping to comparison ids is kept outside the pack, in
results_submission1_tmlr/spotcheck/sample_key.json, for comparing the returned sheets with the study.
"""
from __future__ import annotations

import random
import zipfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from Submission1_Code_Phase2 import common as C
from Submission1_TMLR.exclusions import EXCLUDED_COMPARISONS

SEED = 20261005
PANEL = C.CODES_ROOT / "results_submission1_tmlr" / "extended_panel.jsonl"
KEY = C.CODES_ROOT / "results_submission1_tmlr" / "spotcheck" / "sample_key.json"
DEST = C.CODES_ROOT.parent / "Submission1" / "tmlr_spotcheck_pack"
ZIP = C.CODES_ROOT.parent / "Submission1" / "AgriFair_TMLR_spotcheck.zip"

GROUP = {"SC": "Scheduled Castes", "ST": "Scheduled Tribes", "Others": "Other social groups", "All": "All social groups"}
METRIC = {"area": "operated area (hectares)", "number": "number of operational holdings"}
SOURCE = [
    ("Published source", "Agriculture Census 2015-16 (Phase I), All India Report on Number and Area of Operational Holdings"),
    ("Publisher", "Agriculture Census Division, Department of Agriculture, Co-operation and Farmers Welfare, Government of India, 2019"),
    ("Copy to use", "https://www.fao.org/fileadmin/templates/ess/ess_test_folder/World_Census_Agriculture/WCA_2020/WCA_2020_new_doc/IND_REP_ENG_2015_2016.pdf"),
    ("SHA-256 of that file", "fb01bcb922a38ef964a2ca2fddada5540426bf5b06317d931425ae8d699d6405"),
    ("Tables", "T2-4: number and area of operational holdings by size class and social group, for each state"),
    ("Decision rule", "gap below 5 percentage points = equal; gap of 10 points or more = diff; anything between = excluded"),
]

INSTRUCTIONS = """AgriFair spot-check of new census comparisons

Why this is needed
We added 120 new census comparisons to the study. A computer program built them and a second program
re-checked them, but no person has checked them yet. This pack asks two of you to recompute 30 of
them by hand from the published census report, independently of the study. Your answers will be
compared with the study's values afterwards.

None of the files contains the study's shares, labels or any model answer. Please do not look for
them while you work.

What is in the pack
  Checker1_spotcheck.xlsx   Task A (30 rows) and Task B (30 rows). For Checker 1.
  Checker2_spotcheck.xlsx   Task A only (12 of the same rows). For Checker 2.
  INSTRUCTIONS.txt          This text.
Checker 2 must not see Checker 1's file, and the two of you should not discuss rows until both
files are returned.

Rough time: Task A about 6 minutes a row (3 hours for 30); Task B about 2 minutes a row (1 hour).

The source
Agriculture Census 2015-16 (Phase I), All India Report on Number and Area of Operational Holdings,
Government of India, 2019. Use the copy linked on the "Source" sheet, so that everyone reads the
same document. Every row uses Table T2-4 for the state named in the row.

TASK A - recompute each comparison (both checkers)
For each row:
 1. Open Table T2-4 for the state in "state".
 2. Find the population in "population". It is either one social group (for example Scheduled
    Tribes, all size classes) or one size class (for example small holdings, all social groups).
 3. Read the measure in "measure": operated area, or number of operational holdings.
 4. Read the value for "group_1_to_read" and for "group_2_to_read" within that population.
 5. Divide each by the population's total for the same measure, and write the results as
    percentages (0-100, one decimal) in "share_1_percent" and "share_2_percent".
    Write the total you divided by, with its value, in "denominator_used".
 6. Write the absolute difference between the two percentages in "gap_percentage_points".
 7. Set "condition": equal if the gap is below 5 points, diff if it is 10 points or more,
    excluded if it is in between. Excluded rows are expected sometimes; do not force a row.
 8. In "larger_group_or_roughly_equal", name the larger group, or write Roughly equal.
 9. Record the page and table you read in "source_page_or_table", and your initials.
Rules: if the report does not support the comparison, leave the numbers blank and explain in
"notes". Never adjust a value to make it fit a band. Disagreements are kept and reported, not
quietly reconciled.

TASK B - check the questions (Checker 1, after finishing Task A)
Each row shows the two wordings of one question, labelled with the same item_id as in Task A.
For each wording, answer Y or N:
  place_correct          the question names the state in the Task A row
  population_correct     the question restricts to the Task A population, and to nothing else
  measure_correct        the question asks about the Task A measure
  groups_correct         the question compares exactly the two Task A groups
  total_stated           the question says what the shares are measured against
  rule_stated            the question states the 5-point and 10-point rule
  same_meaning           wordings A and B ask the same thing
Write anything odd in "notes": a garbled phrase, a wrong word, a group or population that does not
exist in the census. One earlier item was removed because its wording named a population the census
does not have, so please read each question as a farmer or official would.

Returning the files
Fill in the workbook, save it with your initials in the file name, and return it. Do not change the
item_id column or the order of rows.
"""

A_COLS = ["item_id", "parent_table", "state", "population", "measure", "group_1_to_read", "group_2_to_read",
          "share_1_percent", "share_2_percent", "denominator_used", "gap_percentage_points", "condition",
          "larger_group_or_roughly_equal", "source_page_or_table", "checker_initials", "notes"]
B_COLS = ["item_id", "wording", "question_text", "place_correct", "population_correct", "measure_correct",
          "groups_correct", "total_stated", "rule_stated", "same_meaning", "notes"]


def population(it):
    if it["axis"] == "landholding":    # one social group, entities are size classes
        g = it["social_group"]
        return f"{GROUP.get(g, g)}, all size classes" if g else "All social groups, all size classes"
    if not it["size_class"]:                                               # all size classes together
        return "all holdings (all size classes), all social groups"
    return f"{it['size_class'].lower()} holdings, all social groups"   # one size class, entities are social groups


def entity(name, it):
    """Size-class entities appear as 'marginal operated area' / 'marginal holdings'; name the size class."""
    if it["axis"] == "landholding":
        return name.replace(" operated area", " holdings").replace("holdings holdings", "holdings")
    return name


def sample():
    items = [it for it in C.read_jsonl(PANEL) if it["fresh_id"] not in EXCLUDED_COMPARISONS]
    rng = random.Random(SEED)
    chosen = []
    for axis in ("landholding", "social_group"):
        pool = [it for it in items if it["axis"] == axis]
        num = [it for it in pool if it["metric"] == "number"]
        pick = rng.sample(num, min(3, len(num)))
        rest = [it for it in pool if it not in pick]
        pick += rng.sample(rest, 15 - len(pick))
        chosen += pick
    rng.shuffle(chosen)
    return chosen


def style(ws, widths):
    head = PatternFill("solid", fgColor="DDE6F0")
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = head
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"


def add_instructions(wb):
    ws = wb.active
    ws.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS.splitlines(), start=1):
        ws.cell(i, 1, line)
        if line and not line.startswith(" ") and (line.isupper() or line.startswith("TASK") or i == 1):
            ws.cell(i, 1).font = Font(bold=True)
    ws.column_dimensions["A"].width = 110
    src = wb.create_sheet("Source")
    for k, v in SOURCE:
        src.append([k, v])
    src.column_dimensions["A"].width = 22; src.column_dimensions["B"].width = 120
    for c in src["A"]:
        c.font = Font(bold=True)


def task_a(wb, rows, title):
    ws = wb.create_sheet(title)
    ws.append(A_COLS)
    for sid, it in rows:
        ws.append([sid, "T2-4", it["geography"], population(it), METRIC[it["metric"]],
                   entity(it["entity1"], it), entity(it["entity2"], it)] + [None] * 9)
    dv = DataValidation(type="list", formula1='"equal,diff,excluded"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"L2:L{len(rows) + 1}")
    style(ws, [8, 10, 16, 30, 26, 24, 24, 12, 12, 30, 12, 11, 24, 18, 10, 40])


def task_b(wb, rows):
    ws = wb.create_sheet("Task B - questions")
    ws.append(B_COLS)
    for sid, it in rows:
        for w, key in (("A", "wording_a"), ("B", "wording_b")):
            ws.append([sid, w, it[key]] + [None] * 8)
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"D2:J{2 * len(rows) + 1}")
    style(ws, [8, 8, 90, 10, 12, 10, 10, 10, 10, 10, 40])
    for r in range(2, 2 * len(rows) + 2):
        ws.row_dimensions[r].height = 75


def main():
    chosen = sample()
    rows = [(f"S{i:02d}", it) for i, it in enumerate(chosen, start=1)]
    sub = sorted(random.Random(SEED + 1).sample(rows, 12), key=lambda r: r[0])
    DEST.mkdir(parents=True, exist_ok=True)
    wb1 = Workbook(); add_instructions(wb1); task_a(wb1, rows, "Task A - Checker 1"); task_b(wb1, rows)
    wb1.save(DEST / "Checker1_spotcheck.xlsx")
    wb2 = Workbook(); add_instructions(wb2); task_a(wb2, sub, "Task A - Checker 2")
    wb2.save(DEST / "Checker2_spotcheck.xlsx")
    (DEST / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("INSTRUCTIONS.txt", "Checker1_spotcheck.xlsx", "Checker2_spotcheck.xlsx"):
            z.write(DEST / name, f"AgriFair_TMLR_spotcheck/{name}")
    KEY.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(KEY, {"seed": SEED, "items": {sid: it["fresh_id"] for sid, it in rows},
                       "checker2_subset": [sid for sid, _ in sub]})
    import collections
    print(ZIP, "| rows", len(rows), "| checker 2:", len(sub),
          "| axis", dict(collections.Counter(it["axis"] for _, it in rows)),
          "| measure", dict(collections.Counter(it["metric"] for _, it in rows)),
          "| states", len({it["geography"] for _, it in rows}))


if __name__ == "__main__":
    main()
