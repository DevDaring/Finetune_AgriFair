"""a1b_gender.py — extract All-India gender x size-class holdings (number & area).

Source: census report PDF, Tables 14/15/16 (pp.61-66):
  Table 14 -> All Social Groups, Table 15 -> Scheduled Castes, Table 16 -> Scheduled Tribes.
  Each: Size Class x Gender (M/F/T) x {No. of Holdings, Area Operated}.
  Number-of-holdings 'Total' and Area 'Total' columns are taken. Units: '000 / '000 ha.

Output: data/interim/census_gender_long.csv
  columns: group(All/SC/ST), size_class, gender(M/F), metric(number/area), value
Validation printed: female share by size-class for each group.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "census" / "fao_allindia_2015_16.pdf"
OUT = ROOT / "data" / "interim" / "census_gender_long.csv"

# group -> pages (each table spans two pages)
GROUP_PAGES = {"All": [61, 62], "SC": [63, 64], "ST": [65, 66]}
SIZE_NAMES = {"marginal", "small", "semi-medium", "medium", "large"}


def _num(x):
    if x is None:
        return None
    s = str(x).strip().replace(",", "")
    if s.lower().startswith("neg"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def extract() -> pd.DataFrame:
    rows = []
    with pdfplumber.open(PDF) as pdf:
        for group, pages in GROUP_PAGES.items():
            for pno in pages:
                tbls = pdf.pages[pno - 1].extract_tables()
                if not tbls:
                    continue
                tbl = tbls[0]
                ncol = max(len(r) for r in tbl)
                # 11-col layout: num Total idx 6, area Total idx 10
                # 8-col layout (no institutional): num Total idx 5, area Total idx 8
                if ncol >= 11:
                    num_idx, area_idx = 6, 10
                else:
                    num_idx, area_idx = 5, 8
                block = None
                for r in tbl:
                    cells = [(c or "").strip() for c in r]
                    label = cells[1].lower() if len(cells) > 1 else ""
                    gender = cells[2] if len(cells) > 2 else ""
                    nval = _num(cells[num_idx]) if len(cells) > num_idx else None
                    aval = _num(cells[area_idx]) if len(cells) > area_idx else None
                    if gender == "M" and label in SIZE_NAMES:
                        block = label.title().replace("Semi-Medium", "Semi-medium")
                        if nval is not None:
                            rows.append((group, block, "M", "number", nval))
                        if aval is not None:
                            rows.append((group, block, "M", "area", aval))
                    elif gender == "F" and block is not None:
                        if nval is not None:
                            rows.append((group, block, "F", "number", nval))
                        if aval is not None:
                            rows.append((group, block, "F", "area", aval))
                    elif gender == "T":
                        block = None
    df = pd.DataFrame(rows, columns=["group", "size_class", "gender", "metric", "value"])
    df = df.drop_duplicates(["group", "size_class", "gender", "metric"], keep="first")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = extract()
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {OUT} ({len(df)} rows)")
    num = df[df.metric == "number"]
    piv = num.pivot_table(index=["group", "size_class"], columns="gender", values="value")
    piv["female_share_%"] = (piv["F"] / (piv["M"] + piv["F"]) * 100).round(1)
    print(piv)
