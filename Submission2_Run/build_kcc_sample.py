"""4.1 Real farmer queries from the Kisan Call Centre (data.gov.in / AIKosh).

Input: either --csv <export.csv> (columns are normalised case-insensitively; expected fields
include state, district, crop or category, query text and the Farm Tele-Advisor answer) or
the data.gov.in API when kcc.resource_id and DATA_GOV_IN_API_KEY are set.

Output: data_submission2/kcc_queries.jsonl with n_queries rows, stratified by state x
query_type, personal data stripped, the FTA answer kept as reference_answer, and empty
question_hi / question_bn fields to be filled by --translations.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Dict, List

import requests

from Submission2_Run import common as C

TYPE_RULES = {   # query_type from free-text category / query, first match wins
    "plant_protection": r"pest|disease|insect|fungus|blight|rot|attack|control|spray|weed",
    "nutrient_management": r"fertili[sz]er|urea|dap|nutrient|manure|dose|npk|zinc|deficien",
    "weather": r"weather|rain|forecast|temperature|frost|humidity|monsoon",
    "government_schemes": r"scheme|subsid|pm[- ]?kisan|insurance|yojana|loan|kcc card|registration",
    "market": r"price|market|mandi|rate|sell|msp|procure",
}
_QUERY_KEYS = ("querytext", "query_text", "query", "question", "farmer_query")
_ANSWER_KEYS = ("kccans", "kcc_ans", "answer", "response", "fta_answer")
_STATE_KEYS = ("statename", "state_name", "state")
_CROP_KEYS = ("crop", "crop_name")
_CAT_KEYS = ("querytype", "query_type", "category", "sector")


def _norm(row: Dict) -> Dict:
    r = {k.lower().replace(" ", "_"): (v or "").strip() for k, v in row.items()}
    pick = lambda keys: next((r[k] for k in keys if k in r and r[k]), "")
    return {"state": pick(_STATE_KEYS), "crop": pick(_CROP_KEYS), "category": pick(_CAT_KEYS),
            "query": pick(_QUERY_KEYS), "answer": pick(_ANSWER_KEYS)}


def query_type(category: str, query: str) -> str:
    text = f"{category} {query}".lower()
    for qt, pat in TYPE_RULES.items():
        if re.search(pat, text):
            return qt
    return "other"


def fetch_api(resource_id: str, api_key: str, limit: int = 5000, max_rows: int = 200000) -> List[Dict]:
    rows, offset = [], 0
    while offset < max_rows:
        r = requests.get(f"https://api.data.gov.in/resource/{resource_id}",
                         params={"api-key": api_key, "format": "json", "limit": limit, "offset": offset}, timeout=120)
        r.raise_for_status()
        batch = r.json().get("records", [])
        if not batch:
            break
        rows += batch; offset += limit
    return rows


def stratified_sample(rows: List[Dict], cfg: Dict, seed: int) -> List[Dict]:
    k = cfg["kcc"]; states = set(k["states"]); types = list(k["query_types"])
    pool = [r for r in rows if r["state"] in states and len(r["query"].split()) >= 4 and r["answer"]]
    for r in pool:
        r["query_type"] = query_type(r["category"], r["query"])
    pool = [r for r in pool if r["query_type"] in types]
    rng = random.Random(seed)
    per_cell = max(1, k["n_queries"] // (len(states) * len(types)))
    out = []
    for s in sorted(states):
        for t in types:
            cell = [r for r in pool if r["state"] == s and r["query_type"] == t]
            rng.shuffle(cell); out += cell[:per_cell]
    rng.shuffle(out)
    return out[: k["n_queries"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, help="local KCC export")
    ap.add_argument("--translations", type=Path, help="CSV: query_id,question_hi,question_bn")
    args = ap.parse_args()
    cfg = C.load_config(); out = C.data_dir(cfg) / "kcc_queries.jsonl"
    if args.translations:
        rows = C.read_jsonl(out)
        tr = {r["query_id"]: r for r in csv.DictReader(open(args.translations, encoding="utf-8"))}
        for r in rows:
            t = tr.get(r["query_id"], {})
            r["question_hi"], r["question_bn"] = t.get("question_hi", ""), t.get("question_bn", "")
        C.write_jsonl(out, rows); print(f"translations merged into {out}"); return
    if args.csv:
        raw = [_norm(r) for r in csv.DictReader(open(args.csv, encoding="utf-8", errors="replace"))]
        source = {"kind": "csv", "path": str(args.csv), "sha256": C.sha256_file(args.csv)}
    else:
        rid, key = cfg["kcc"]["resource_id"], os.environ.get("DATA_GOV_IN_API_KEY", "")
        if not (rid and key):
            raise SystemExit("give --csv, or set kcc.resource_id in config.yaml and DATA_GOV_IN_API_KEY")
        raw = [_norm(r) for r in fetch_api(rid, key)]
        source = {"kind": "data.gov.in", "resource_id": rid}
    sample = stratified_sample(raw, cfg, cfg["analysis_seed"])
    rows = [{"query_id": f"q{i:04d}", "state": r["state"], "crop": r["crop"], "query_type": r["query_type"],
             "question_en": C.strip_personal_data(r["query"]), "reference_answer": C.strip_personal_data(r["answer"]),
             "question_hi": "", "question_bn": ""} for i, r in enumerate(sample)]
    C.write_jsonl(out, rows)
    C.write_json(out.with_suffix(".manifest.json"), {"source": source, "raw_rows": len(raw), "sampled": len(rows),
                                                     "by_state_type": {f"{r['state']}|{r['query_type']}": 0 for r in rows}})
    print(f"wrote {len(rows)} queries -> {out}")


if __name__ == "__main__":
    main()
