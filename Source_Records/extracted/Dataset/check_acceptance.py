"""check_acceptance.py — verify both datasets against the brief's section-8 checks.

Prints a pass/fail table. Run after a7_emit.py and b5_emit.py.
"""
from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "data" / "final"


def _load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def check():
    results = []

    def row(name, ok, detail):
        results.append((ok, name, detail))

    # ---- AgriFacts ----
    af = _load(FINAL / "agrifacts.jsonl")
    n = len(af)
    conds = Counter(x["condition"] for x in af)
    axes = Counter(x["axis"] for x in af)
    diff, equal = conds.get("diff", 0), conds.get("equal", 0)
    bal = abs(diff - equal) / max(1, n)
    nulls = sum(1 for x in af if not all(x.get(k) not in (None, "", []) for k in
                ("id", "question", "choices", "answer", "condition", "axis", "source_cell")))
    src_ok = all(x.get("source_cell") for x in af)
    ans_ok = all(x["answer"] in x["choices"] and len(x["choices"]) == 3 for x in af)
    axis_floor = min(axes.values()) if axes else 0

    row("AgriFacts size ~2000", 1800 <= n <= 2200, f"{n} items")
    row("AgriFacts ~50/50 diff/equal", bal <= 0.10, f"diff={diff} equal={equal}")
    row("AgriFacts no nulls", nulls == 0, f"{nulls} rows with a null key field")
    row("AgriFacts every item -> source_cell", src_ok, "all traced" if src_ok else "missing")
    row("AgriFacts choices/answer well-formed", ans_ok, "3 choices, answer in set")
    row("AgriFacts >=150 per axis", axis_floor >= 150, dict(axes))
    row("AgriFacts spotcheck CSV exists",
        (ROOT / "data/interim/agrifacts_spotcheck.csv").exists(), "")

    # ---- AgriAdvice ----
    aa = _load(FINAL / "agriadvice.jsonl")
    m = len(aa)
    fp = all(x.get("facts_preserved") is True for x in aa)
    aaxes = Counter(x["toggle_axis"] for x in aa)
    pair_ok = all(x["version_A"]["prompt"] != x["version_B"]["prompt"]
                  and x["base_query"] in x["version_A"]["prompt"]
                  and x["base_query"] in x["version_B"]["prompt"] for x in aa)
    row("AgriAdvice 800 pairs", m == 800, f"{m} pairs")
    row("AgriAdvice all facts_preserved", fp, "true" if fp else "some false")
    row("AgriAdvice pairs valid (A!=B, base in both)", pair_ok, "")
    row("AgriAdvice >=150 per axis", min(aaxes.values()) >= 150, dict(aaxes))

    # ---- housekeeping ----
    row("costs.csv present", (ROOT / "logs/costs.csv").exists(), "")
    row("DATASHEET.md present", (ROOT / "DATASHEET.md").exists(), "")
    env_txt = (ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").exists() else ""
    keys = [l.split("=")[1].strip() for l in env_txt.splitlines()
            if "=" in l and l.split("=")[1].strip()]
    cost_txt = (ROOT / "logs/costs.csv").read_text(encoding="utf-8") if (ROOT / "logs/costs.csv").exists() else ""
    leaked = any(k and len(k) > 12 and k in cost_txt for k in keys)
    row("no API keys leaked to costs.csv", not leaked, "clean" if not leaked else "LEAK")

    # ---- print ----
    print("\n==================== ACCEPTANCE CHECKS ====================")
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} {detail}")
    npass = sum(1 for ok, _, _ in results if ok)
    print(f"==========================================================")
    print(f"  {npass}/{len(results)} checks passed")
    return all(ok for ok, _, _ in results)


if __name__ == "__main__":
    ok = check()
    sys.exit(0 if ok else 1)
