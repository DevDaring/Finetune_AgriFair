"""b4_gate.py — content-preservation gate for AgriAdvice pairs.

Because injection is deterministic (base query kept verbatim, only the identity
span changes), this gate is a per-instance PROOF of content preservation rather
than a noisy estimate. Each pair must pass ALL checks:

  1. base_query appears verbatim in both prompt_A and prompt_B
  2. prompt_A != prompt_B (the identity actually toggled)
  3. removing the persona leaves an identical residual in A and B
     (i.e. nothing but the identity differs)
  4. any detected crop/problem appears in both prompts

Discard rate is logged. Per the brief, if it exceeds 20% the run pauses; here it
should be ~0. Output: data/interim/agriadvice_pairs_gated.csv
"""
from __future__ import annotations

import io
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
THRESH = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["agriadvice"]["discard_pause_threshold"]
IN = ROOT / "data" / "interim" / "agriadvice_pairs_raw.csv"
OUT = ROOT / "data" / "interim" / "agriadvice_pairs_gated.csv"


def _only_identity_differs(a: str, b: str, pa: str, pb: str) -> bool:
    """Word-level diff between the two prompts; every differing word must belong
    to the respective identity clause. Word-level avoids substring traps such as
    'man' inside 'woman'/'manage'."""
    aw, bw = a.split(), b.split()
    sm = SequenceMatcher(None, aw, bw, autojunk=False)
    da, db = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            da += aw[i1:i2]
            db += bw[j1:j2]
    strip = lambda w: w.strip(".,:;()?!").lower()
    pa_l, pb_l = pa.lower(), pb.lower()
    a_ok = all(strip(w) in pa_l for w in da if strip(w))
    b_ok = all(strip(w) in pb_l for w in db if strip(w))
    return a_ok and b_ok


def gate(df: pd.DataFrame):
    kept, dropped = [], []
    for _, r in df.iterrows():
        q = str(r.base_query).strip()
        a, b = str(r.prompt_A), str(r.prompt_B)
        reasons = []
        if q not in a or q not in b:
            reasons.append("base_query not verbatim in both")
        if a == b:
            reasons.append("versions identical")
        if not _only_identity_differs(a, b, str(r.persona_A), str(r.persona_B)):
            reasons.append("difference is not solely the identity")
        for fact in (str(r.crop), str(r.problem)):
            if fact and (fact.lower() not in a.lower() or fact.lower() not in b.lower()):
                reasons.append(f"material fact '{fact}' missing")
        if reasons:
            dropped.append({"pair_id": r.pair_id, "reasons": "; ".join(reasons)})
        else:
            row = r.to_dict()
            row["facts_preserved"] = True
            kept.append(row)
    return pd.DataFrame(kept), pd.DataFrame(dropped)


if __name__ == "__main__":
    df = pd.read_csv(IN).fillna("")
    kept, dropped = gate(df)
    rate = len(dropped) / max(1, len(df))
    kept.to_csv(OUT, index=False, encoding="utf-8")
    print(f"pairs={len(df)}  kept={len(kept)}  dropped={len(dropped)}  discard_rate={rate:.2%}")
    if len(dropped):
        print(dropped.head(10).to_string(index=False))
    if rate > THRESH:
        print(f"\n[PAUSE] discard rate {rate:.1%} exceeds {THRESH:.0%} — investigate before continuing.")
    else:
        print(f"\n[OK] discard rate within tolerance -> {OUT}")
    print("per-axis kept:")
    print(kept.groupby("toggle_axis").size())
