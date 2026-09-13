"""a1_extract.py — extract REAL Agriculture Census 2015-16 holding tables.

Source (downloaded, provenance in DATASHEET.md):
  data/raw/census/fao_allindia_2015_16.pdf  (FAO mirror of the official
  "All India Report on Agriculture Census 2015-16"; identical content to
  agcensus.da.gov.in/document/agcen1516/ac_1516_report_final-220221.pdf).

Tables parsed (one size-class per page):
  Table 2 (pp.25-30): All Social Groups
  Table 3 (pp.31-36): Scheduled Castes
  Table 4 (pp.37-42): Scheduled Tribes
  Page order in each table: All Size Classes, Marginal, Small, Semi-medium,
  Medium, Large. Columns: Sl.No | State | 2015-16 Number | Area | 2010-11 ...

NO number is invented. "Others" = All - SC - ST, computed per (state,size).
Output: data/interim/census_holdings_long.csv
  columns: group, size_class, state, number, area   (number in '000, area in '000 ha)
A consistency report is printed: SC+ST must not exceed All in any cell.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "census" / "fao_allindia_2015_16.pdf"
OUT = ROOT / "data" / "interim" / "census_holdings_long.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# (group, 1-based page numbers in PDF order: All, Marginal, Small, Semi-medium, Medium, Large)
GROUP_PAGES = {
    "All": [25, 26, 27, 28, 29, 30],
    "SC":  [31, 32, 33, 34, 35, 36],
    "ST":  [37, 38, 39, 40, 41, 42],
}
SIZE_RX = re.compile(r"(?i)(All Size Classes|Marginal|Semi-medium|Small|Medium|Large)\b")


def _num(x: str):
    """Parse a census cell: '16413' -> 16413; 'Neg.' -> 0 (negligible); '' -> None."""
    if x is None:
        return None
    s = x.strip().replace(",", "")
    if s in ("", "-", "—"):
        return None
    if s.lower().startswith("neg"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def _page_size_class(page) -> str | None:
    txt = page.extract_text() or ""
    for line in txt.split("\n"):
        m = SIZE_RX.search(line)
        if m:
            lab = m.group(1).title().replace("Semi-Medium", "Semi-medium")
            return "All Classes" if lab.lower().startswith("all size") else lab
    return None


def extract() -> pd.DataFrame:
    rows = []
    with pdfplumber.open(PDF) as pdf:
        for group, pages in GROUP_PAGES.items():
            for pno in pages:
                page = pdf.pages[pno - 1]
                size_class = _page_size_class(page)
                if size_class is None:
                    print(f"[warn] no size-class label on p{pno} ({group})")
                    continue
                tables = page.extract_tables()
                if not tables:
                    print(f"[warn] no table on p{pno}")
                    continue
                for r in tables[0]:
                    if len(r) < 4:
                        continue
                    slno = (r[0] or "").strip()
                    state = (r[1] or "").strip().replace("\n", " ")
                    if not re.match(r"^\d+$", slno):  # skip header / TOTAL rows
                        continue
                    number = _num(r[2])
                    area = _num(r[3])
                    rows.append((group, size_class, state, number, area))
    df = pd.DataFrame(rows, columns=["group", "size_class", "state", "number", "area"])
    return df


def add_others(df: pd.DataFrame) -> pd.DataFrame:
    """Others = All - SC - ST, per (state, size_class), for number and area."""
    piv = df.pivot_table(index=["state", "size_class"], columns="group",
                         values=["number", "area"], aggfunc="first")
    out_rows = []
    for (state, size_class), row in piv.iterrows():
        for metric in ("number", "area"):
            allv = row.get((metric, "All"))
            sc = row.get((metric, "SC")) or 0
            st = row.get((metric, "ST")) or 0
            if allv is None:
                continue
            others = allv - sc - st
            # tiny negative from rounding in '000 -> clamp to 0
            if -2 <= others < 0:
                others = 0.0
            out_rows.append((state, size_class, metric, others))
    od = pd.DataFrame(out_rows, columns=["state", "size_class", "metric", "value"])
    od = od.pivot_table(index=["state", "size_class"], columns="metric",
                        values="value", aggfunc="first").reset_index()
    od["group"] = "Others"
    od = od.rename(columns={})
    others_long = od[["group", "size_class", "state", "number", "area"]]
    return pd.concat([df, others_long], ignore_index=True)


def consistency_report(df: pd.DataFrame):
    piv = df.pivot_table(index=["state", "size_class"], columns="group",
                        values="number", aggfunc="first")
    bad = 0
    neg = 0
    for (state, size_class), row in piv.iterrows():
        allv, sc, st = row.get("All"), row.get("SC") or 0, row.get("ST") or 0
        if allv is None:
            continue
        if sc + st > allv + 2:  # +2 tolerance for '000 rounding
            bad += 1
            print(f"[CHECK] SC+ST>All: {state}/{size_class}: SC={sc} ST={st} All={allv}")
        oth = row.get("Others")
        if oth is not None and oth < -2:
            neg += 1
            print(f"[CHECK] Others<0: {state}/{size_class}: {oth}")
    print(f"[consistency] cells with SC+ST>All: {bad}; Others<0: {neg}")


if __name__ == "__main__":
    df = extract()
    print(f"Extracted {len(df)} (group,size,state) cells across {df.state.nunique()} states.")
    df = add_others(df)
    df = df.sort_values(["group", "size_class", "state"]).reset_index(drop=True)
    consistency_report(df)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(df)} rows)")
    print(df.groupby("group").size())
    # quick peek
    print(df[df.state == "Bihar"].pivot_table(index="size_class", columns="group", values="number"))
