"""b1_acquire.py — download REAL farmer / agronomic-question corpora.

Sources (verified live on 2026-06-08):
  * KisanVaani/agriculture-qa-english-only  (Apache-2.0, open, 22.6k QA pairs)
  * bharatgenai/BhashaBench-Krishi           (CC-BY-4.0, GATED, 15.4k MCQ items
                                              from 55+ Indian govt agri exams;
                                              EN 12.6k + HI 2.8k). arXiv:2510.25409

Auth: HUGGINGFACE_TOKEN from .env (never printed). The gated BBK requires that
the HF account has accepted the dataset's terms on the website first.

Output: data/raw/*.parquet plus a normalised data/interim/agri_queries_raw.csv
with columns [source, qid, text, lang, meta]. We do NOT invent any text here —
every row traces to a downloaded file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
RAW.mkdir(parents=True, exist_ok=True)
INTERIM.mkdir(parents=True, exist_ok=True)

TOKEN = os.getenv("HUGGINGFACE_TOKEN")

SOURCES = [
    # (repo_id, filename_in_repo, local_name, lang, gated)
    ("KisanVaani/agriculture-qa-english-only",
     "default/train/0000.parquet", "kisanvaani_qa.parquet", "en", False),
    ("bharatgenai/BhashaBench-Krishi",
     "English/test/0000.parquet", "bbk_english.parquet", "en", True),
    ("bharatgenai/BhashaBench-Krishi",
     "Hindi/test/0000.parquet", "bbk_hindi.parquet", "hi", True),
]


def _download(repo_id: str, filename: str, local_name: str, gated: bool) -> Path | None:
    dest = RAW / local_name
    if dest.exists():
        print(f"[cached] {local_name}")
        return dest
    try:
        p = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset",
            revision="refs/convert/parquet", token=TOKEN,
            local_dir=RAW / "_hf_tmp",
        )
        Path(p).replace(dest)
        print(f"[ok] {repo_id} :: {filename} -> {local_name}")
        return dest
    except Exception as e:  # noqa: BLE001
        tag = "GATED — accept terms on HF website first" if gated else ""
        print(f"[FAIL] {repo_id} :: {filename}: {type(e).__name__}: {str(e)[:160]} {tag}")
        return None


def normalise() -> pd.DataFrame:
    rows = []
    kv = RAW / "kisanvaani_qa.parquet"
    if kv.exists():
        df = pd.read_parquet(kv)
        for i, r in df.iterrows():
            q = str(r.get("question", "")).strip()
            if q:
                rows.append(("kisanvaani", f"kv-{i}", q, "en",
                             str(r.get("answers", ""))[:400]))
    for name, lang in [("bbk_english.parquet", "en"), ("bbk_hindi.parquet", "hi")]:
        f = RAW / name
        if f.exists():
            df = pd.read_parquet(f)
            for _, r in df.iterrows():
                q = str(r.get("question", "")).strip()
                if q:
                    rows.append(("bbk", str(r.get("id", "")), q, lang,
                                 str(r.get("subject_domain", ""))))
    out = pd.DataFrame(rows, columns=["source", "qid", "text", "lang", "meta"])
    out.to_csv(INTERIM / "agri_queries_raw.csv", index=False, encoding="utf-8")
    return out


if __name__ == "__main__":
    if not TOKEN:
        print("[warn] HUGGINGFACE_TOKEN not set — gated BBK will fail.")
    for repo, fn, local, lang, gated in SOURCES:
        _download(repo, fn, local, gated)
    df = normalise()
    print(f"\nNormalised {len(df)} raw queries -> data/interim/agri_queries_raw.csv")
    if len(df):
        print(df.groupby(['source', 'lang']).size())
