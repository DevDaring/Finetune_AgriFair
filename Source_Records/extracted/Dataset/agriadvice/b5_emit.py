"""b5_emit.py — emit the final AgriAdvice dataset (the pairs, not model outputs).

Reads the gated pairs and writes data/final/agriadvice.jsonl with the schema from
the brief (section B5):
  pair_id, base_query, toggle_axis,
  version_A{persona, prompt}, version_B{persona, prompt},
  facts_preserved=true, source_query

Balanced to the per-axis targets in config (>=150 each; here 200 each = 800 pairs).
Every emitted record is re-validated (base query verbatim in both versions).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["agriadvice"]
IN = ROOT / "data" / "interim" / "agriadvice_pairs_gated.csv"
OUT = ROOT / "data" / "final" / "agriadvice.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(IN).fillna("")
    targets = CFG["toggle_axes"]
    floor = CFG["min_per_axis"]

    chosen = []
    for axis, tgt in targets.items():
        sub = df[df.toggle_axis == axis]
        take = sub.head(tgt)
        assert len(take) >= floor, f"{axis}: only {len(take)} < floor {floor}"
        chosen.append(take)
    out = pd.concat(chosen, ignore_index=True)

    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for _, r in out.iterrows():
            # final re-validation
            assert str(r.base_query) in str(r.prompt_A), r.pair_id
            assert str(r.base_query) in str(r.prompt_B), r.pair_id
            assert r.prompt_A != r.prompt_B, r.pair_id
            rec = {
                "pair_id": r.pair_id,
                "base_query": r.base_query,
                "toggle_axis": r.toggle_axis,
                "version_A": {"persona": r.persona_A, "prompt": r.prompt_A},
                "version_B": {"persona": r.persona_B, "prompt": r.prompt_B},
                "facts_preserved": True,
                "source_query": r.base_query,
                "source_dataset": "KisanVaani/agriculture-qa-english-only",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {OUT} ({n} pairs)")
    print(out.groupby("toggle_axis").size())


if __name__ == "__main__":
    main()
