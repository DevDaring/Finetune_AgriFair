"""b2_select.py — select clean, single-topic, advice-relevant base queries.

Source: data/interim/agri_queries_raw.csv (real queries; KisanVaani QA used here
because its items are genuine farmer-style agronomic questions). NO query is
invented or edited — bases are kept VERBATIM and later wrapped with an identity
framing that toggles between version_A and version_B. Material facts therefore
stay byte-stable by construction.

Selection rules (single-topic, answerable, agronomic):
  * a real question, 25-160 chars, exactly one '?'
  * mentions a concrete crop OR an agronomic problem keyword (material content)
  * advice-relevant phrasing preferred (how/control/manage/best/improve/prevent)
  * de-duplicated
Each base is tagged with the detected crop/problem (the facts to FREEZE) and
assigned to one toggle axis. Output: data/interim/agriadvice_base.csv
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SEED = int(CFG["seed"])
AXES = list(CFG["agriadvice"]["toggle_axes"].keys())            # 4 axes
PER_AXIS = max(CFG["agriadvice"]["toggle_axes"].values())       # 200 -> select buffer
RAW = ROOT / "data" / "interim" / "agri_queries_raw.csv"
OUT = ROOT / "data" / "interim" / "agriadvice_base.csv"

CROPS = [
    "rice", "paddy", "wheat", "maize", "corn", "cotton", "sugarcane", "potato",
    "tomato", "onion", "cassava", "groundnut", "peanut", "mustard", "soybean",
    "soyabean", "chilli", "chili", "pepper", "banana", "mango", "millet", "sorghum",
    "barley", "gram", "chickpea", "lentil", "pigeon pea", "pulses", "tea", "coffee",
    "coconut", "cabbage", "brinjal", "eggplant", "okra", "cucumber", "guava",
    "papaya", "turmeric", "ginger", "garlic", "jute", "tobacco", "sunflower",
]
PROBLEMS = [
    "pest", "disease", "blight", "borer", "weed", "fungus", "fungal", "rot",
    "deficiency", "yield", "irrigation", "fertiliz", "fertilis", "nutrient",
    "wilt", "mildew", "aphid", "whitefly", "insect", "drought", "soil",
    "germination", "spacing", "manure", "compost", "mulch", "harvest",
]
ADVICE = re.compile(r"(?i)\b(how|control|manage|treat|best|improve|increase|"
                    r"prevent|reduce|protect|apply|when|which|should|recommend)\b")


def _find(text, vocab):
    low = text.lower()
    for w in vocab:
        if w in low:
            return w
    return None


def select() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    df = df[(df.source == "kisanvaani") & (df.lang == "en")].copy()
    df["text"] = df["text"].astype(str).str.strip()

    rows = []
    seen = set()
    for _, r in df.iterrows():
        q = r["text"]
        n = len(q)
        if n < 25 or n > 160:
            continue
        if q.count("?") != 1:
            continue
        crop = _find(q, CROPS)
        problem = _find(q, PROBLEMS)
        if crop is None and problem is None:
            continue
        key = re.sub(r"[^a-z0-9 ]", "", q.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        advice = bool(ADVICE.search(q))
        rows.append({"source_query": q, "crop": crop or "", "problem": problem or "",
                     "advice": advice, "has_crop": crop is not None})
    pool = pd.DataFrame(rows)
    # rank: advice-phrased + crop-bearing first (richer material content)
    pool["score"] = pool.advice.astype(int) * 2 + pool.has_crop.astype(int)
    pool = pool.sort_values(["score"], ascending=False, kind="stable").reset_index(drop=True)

    need = PER_AXIS * len(AXES)
    buffer = int(need * 1.3)
    chosen = pool.head(buffer).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    chosen = chosen.head(need).reset_index(drop=True)
    chosen["axis"] = [AXES[i % len(AXES)] for i in range(len(chosen))]
    chosen.insert(0, "base_id", [f"b{i:04d}" for i in range(len(chosen))])
    return chosen[["base_id", "axis", "source_query", "crop", "problem"]]


if __name__ == "__main__":
    df = select()
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Wrote {OUT} ({len(df)} base queries)")
    print(df.groupby("axis").size())
    print(f"with crop: {(df.crop != '').sum()} | with problem: {(df.problem != '').sum()}")
    print("\nsamples:")
    for _, r in df.groupby("axis").head(1).iterrows():
        print(f"  [{r.axis}] crop={r.crop!r} prob={r.problem!r} :: {r.source_query}")
