"""build_kobo.py — build ONE combined KoboToolbox XLSForm that lets human
reviewers verify the WHOLE of both datasets, plus a private answer/expectation key.

Form: kobotoolbox/AgriFair_human_verification.xlsx  (XLSForm: survey/choices/settings)
  Section A - AgriFacts (all 2,000): each item is a select_one MCQ. The reviewer
              picks the single best option. Agreement with the census-computed gold
              answer verifies the item. (Gold answer is NOT in the form.)
  Section B - AgriAdvice (all 800): each pair is shown (version A vs version B) and
              the reviewer answers Yes / No / Unsure to "do both ask the same farming
              question, differing only in the farmer's identity?". This human-checks
              the facts_preserved claim. Expected answer is Yes for every pair.

  10 questions per page (field-list groups + settings style=pages). Questions are
  NOT required, so reviewers can submit partial work and the load can be split.

Key: kobotoolbox/AgriFair_answer_key.xlsx  (PRIVATE - do not upload): the gold MCQ
  answer for each AgriFacts item and the expected "yes" for each AgriAdvice pair.
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
SEED = 20260502
PER_PAGE = 10
FACTS = ROOT / "data" / "final" / "agrifacts.jsonl"
ADVICE = ROOT / "data" / "final" / "agriadvice.jsonl"
OUTDIR = ROOT / "kobotoolbox"
OUTDIR.mkdir(exist_ok=True)
FORM = OUTDIR / "AgriFair_human_verification.xlsx"
KEY = OUTDIR / "AgriFair_answer_key.xlsx"


def slug(text, used):
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not s or not s[0].isalpha():
        s = "opt_" + s
    base, i = s, 2
    while s in used:
        s = f"{base}_{i}"; i += 1
    used.add(s)
    return s


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stratified_10pct(records, by_keys, per_page=PER_PAGE):
    """Take ~10% stratified by `by_keys`, sized to an exact multiple of per_page
    (largest-remainder rounding across strata), then shuffle."""
    df = pd.DataFrame(records)
    strata = list(df.groupby(by_keys))
    target = (round(len(df) * 0.10) // per_page) * per_page
    quotas = {k: len(g) * 0.10 for k, g in strata}
    alloc = {k: int(q) for k, q in quotas.items()}
    leftover = target - sum(alloc.values())
    for k, _ in sorted(quotas.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:leftover]:
        alloc[k] += 1
    picks = [g.sample(n=alloc[k], random_state=SEED) for k, g in strata]
    return pd.concat(picks).sample(frac=1.0, random_state=SEED).to_dict("records")


def build():
    pd.options.mode.chained_assignment = None
    # 10% stratified samples: AgriFacts by axis x condition, AgriAdvice by axis.
    facts = stratified_10pct(load(FACTS), ["axis", "condition"])
    advice = stratified_10pct(load(ADVICE), ["toggle_axis"])

    n_fact_pages = (len(facts) + PER_PAGE - 1) // PER_PAGE
    n_adv_pages = (len(advice) + PER_PAGE - 1) // PER_PAGE
    total_pages = n_fact_pages + n_adv_pages

    wb = Workbook()
    sv = wb.active
    sv.title = "survey"
    sv.append(["type", "name", "label", "required", "hint", "appearance"])
    ch = wb.create_sheet("choices")
    ch.append(["list_name", "name", "label"])
    # shared Yes/No/Unsure list for AgriAdvice
    for nm, lb in [("yes", "Yes - same question, only identity differs"),
                   ("no", "No - the two versions differ in more than identity"),
                   ("unsure", "Unsure")]:
        ch.append(["yesno", nm, lb])

    # cover page
    sv.append(["begin_group", "intro_page", "Welcome - dataset verification", "", "", "field-list"])
    sv.append(["text", "reviewer_name", "Your name or initials", "no", "", ""])
    sv.append(["note", "intro",
               "This survey has two parts. PART A (questions Q1 onward): pick the single "
               "best answer to each agricultural fact question (one option is best; it may "
               "be 'Roughly equal'). PART B (pair checks): for each pair, say whether both "
               "versions ask the same farming question, differing only in the farmer's "
               "identity. Each page has 10 items. You may stop and submit at any time.",
               "no", "", ""])
    sv.append(["end_group", "intro_page", "", "", "", ""])

    key_rows = []
    page = 0

    # ---------------- PART A: AgriFacts MCQs ----------------
    qn = 0
    open_pg = None
    for it in facts:
        qn += 1
        pg = (qn - 1) // PER_PAGE + 1
        if pg != open_pg:
            if open_pg is not None:
                sv.append(["end_group", f"a_page{open_pg:03d}", "", "", "", ""])
            page += 1
            sv.append(["begin_group", f"a_page{pg:03d}",
                       f"Part A - Facts - page {pg} of {n_fact_pages}", "", "", "field-list"])
            open_pg = pg
        qname = it["id"].replace("-", "_")
        listname = f"opts_{qname}"
        used, opt_code = set(), {}
        for opt in it["choices"]:
            code = slug(opt, used)
            opt_code[opt] = code
            ch.append([listname, code, opt])
        sv.append([f"select_one {listname}", qname, f"Q{qn}. {it['question']}", "no", "", ""])
        key_rows.append({"section": "agrifacts", "page": page, "kobo_name": qname,
                         "dataset_id": it["id"], "axis": it["axis"], "condition": it["condition"],
                         "metric": it["metric"], "prompt": it["question"],
                         "options": " | ".join(it["choices"]),
                         "expected_answer": it["answer"], "expected_code": opt_code[it["answer"]],
                         "source": it["source_cell"]})
    if open_pg is not None:
        sv.append(["end_group", f"a_page{open_pg:03d}", "", "", "", ""])

    # ---------------- PART B: AgriAdvice pair checks ----------------
    pn = 0
    open_pg = None
    for it in advice:
        pn += 1
        pg = (pn - 1) // PER_PAGE + 1
        if pg != open_pg:
            if open_pg is not None:
                sv.append(["end_group", f"b_page{open_pg:03d}", "", "", "", ""])
            page += 1
            sv.append(["begin_group", f"b_page{pg:03d}",
                       f"Part B - Pair checks - page {pg} of {n_adv_pages}", "", "", "field-list"])
            open_pg = pg
        qname = "adv_" + str(it["pair_id"])
        a = it["version_A"]["prompt"]
        b = it["version_B"]["prompt"]
        label = (f"P{pn}. Read the two versions, then answer below.\n\n"
                 f"Version A: {a}\n\nVersion B: {b}\n\n"
                 f"Do BOTH versions ask the same farming question, differing only in the "
                 f"farmer's identity?")
        sv.append(["select_one yesno", qname, label, "no", "", ""])
        key_rows.append({"section": "agriadvice", "page": page, "kobo_name": qname,
                         "dataset_id": it["pair_id"], "axis": it["toggle_axis"], "condition": "",
                         "metric": "", "prompt": it["base_query"],
                         "options": "Yes | No | Unsure",
                         "expected_answer": "Yes", "expected_code": "yes",
                         "source": it["source_dataset"]})
    if open_pg is not None:
        sv.append(["end_group", f"b_page{open_pg:03d}", "", "", "", ""])

    st = wb.create_sheet("settings")
    st.append(["form_title", "form_id", "version", "style"])
    st.append(["AgriFair - human verification (10% stratified sample)",
               "agrifair_verify_10pct", "20260611", "pages"])
    wb.save(FORM)

    pd.DataFrame(key_rows).to_excel(KEY, index=False)

    kdf = pd.DataFrame(key_rows)
    print(f"Wrote {FORM}")
    print(f"  AgriFacts MCQs : {len(facts)}  ({n_fact_pages} pages)")
    print(f"  AgriAdvice pairs: {len(advice)} ({n_adv_pages} pages)")
    print(f"  total items: {len(facts)+len(advice)}  | total pages: {total_pages} (10 per page)")
    print(f"Wrote {KEY}  (PRIVATE gold/expected answers - do NOT upload)")
    print("\nkey rows by section:", dict(kdf.section.value_counts()))


if __name__ == "__main__":
    build()
