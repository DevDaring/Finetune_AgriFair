"""E2: a deterministic table executor, and the seven invariants it is used to test.

    python -m Submission1_DKE_Repair.table_executor

This is a transparent parser and calculator, not a learning method. It reads the rendered table
out of the prompt text, extracts the two queried shares, applies the operational rule, and
abstains when a value is absent. It has no access to a gold field and cannot recover an answer
from a prompt id: `execute()` is given the prompt STRING and the option list, nothing else.

What a perfect score here would and would not mean: it shows the synthetic task is mechanically
solvable under the declared rule, so a model that fails it is not defeated by an ill-posed item.
It is not evidence of reasoning, and it is not a fair comparison with a closed-book model.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from typing import Dict, List, Optional, Tuple

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"
EQUAL_BELOW, DIFF_AT_LEAST = 5.0, 10.0
ABSTAIN = "insufficient evidence"

_ROW = re.compile(r"^\s*([A-Za-z][^|\n]*?)\s*\|\s*([0-9]+(?:\.[0-9]+)?|--|n/a|not reported)\s*$",
                  re.I | re.M)


def parse_table(prompt: str) -> Dict[str, Optional[float]]:
    """Every 'label | value' row in the prompt. A missing value parses to None, not to zero."""
    out: Dict[str, Optional[float]] = {}
    for label, value in _ROW.findall(prompt):
        key = label.strip().lower()
        out[key] = None if re.match(r"--|n/a|not reported", value, re.I) else float(value)
    return out


def decide(v1: Optional[float], v2: Optional[float], e1: str, e2: str) -> str:
    """The operational rule, applied to two shares. Abstains when either value is absent."""
    if v1 is None or v2 is None:
        return ABSTAIN
    gap = abs(v1 - v2)
    if gap < EQUAL_BELOW:
        return "Roughly equal"
    if gap >= DIFF_AT_LEAST:
        return e1 if v1 > v2 else e2
    return ABSTAIN          # the excluded 5-10 band is not a decidable item


def execute(prompt: str, entity1: str, entity2: str) -> str:
    table = parse_table(prompt)
    return decide(table.get(entity1.lower()), table.get(entity2.lower()), entity1, entity2)


# ------------------------------------------------------------------ the seven invariants
def _prompt(rows: List[Tuple[str, object]], question: str = "Which is larger?") -> str:
    body = "\n".join(f"{k} | {v}" for k, v in rows)
    return f"{question}\n\n{body}\n"


def invariants(seed: int = 20260925) -> List[Dict]:
    rng = random.Random(seed)
    results: List[Dict] = []

    def case(name: str, ok: bool, detail: str = ""):
        results.append({"invariant": name, "passed": bool(ok), "detail": detail})

    trials = 200
    # 1 row reversal preserves the semantic answer
    bad = 0
    for _ in range(trials):
        a, b = rng.uniform(0, 100), rng.uniform(0, 100)
        p1 = _prompt([("alpha", round(a, 2)), ("beta", round(b, 2))])
        p2 = _prompt([("beta", round(b, 2)), ("alpha", round(a, 2))])
        bad += execute(p1, "alpha", "beta") != execute(p2, "alpha", "beta")
    case("1_row_reversal_preserves_answer", bad == 0, f"{bad}/{trials} mismatches")

    # 2 swapping the values changes the winner when the relation flips
    bad = 0
    for _ in range(trials):
        a = rng.uniform(0, 45); b = a + rng.uniform(DIFF_AT_LEAST, 50)
        f = execute(_prompt([("alpha", round(a, 2)), ("beta", round(b, 2))]), "alpha", "beta")
        r = execute(_prompt([("alpha", round(b, 2)), ("beta", round(a, 2))]), "alpha", "beta")
        bad += not (f == "beta" and r == "alpha")
    case("2_value_swap_flips_winner", bad == 0, f"{bad}/{trials} mismatches")

    # 3 values inside the equality band map to equality
    bad = 0
    for _ in range(trials):
        a = rng.uniform(0, 95); b = a + rng.uniform(0, EQUAL_BELOW - 0.01)
        bad += execute(_prompt([("alpha", round(a, 3)), ("beta", round(b, 3))]),
                       "alpha", "beta") != "Roughly equal"
    case("3_equality_band_maps_to_equal", bad == 0, f"{bad}/{trials} mismatches")

    # 4 an unrelated column does not change the queried relation
    bad = 0
    for _ in range(trials):
        a, b, c = rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(0, 100)
        base = execute(_prompt([("alpha", round(a, 2)), ("beta", round(b, 2))]), "alpha", "beta")
        more = execute(_prompt([("alpha", round(a, 2)), ("gamma", round(c, 2)),
                                ("beta", round(b, 2))]), "alpha", "beta")
        bad += base != more
    case("4_irrelevant_row_is_ignored", bad == 0, f"{bad}/{trials} mismatches")

    # 5 a missing required value causes abstention
    bad = 0
    for token in ("--", "n/a", "not reported"):
        for _ in range(trials // 3):
            a = rng.uniform(0, 100)
            bad += execute(_prompt([("alpha", round(a, 2)), ("beta", token)]),
                           "alpha", "beta") != ABSTAIN
    case("5_missing_value_abstains", bad == 0, f"{bad} failures")

    # 6 shuffling displayed options does not change semantic scoring
    bad = 0
    for _ in range(trials):
        a = rng.uniform(0, 45); b = a + rng.uniform(DIFF_AT_LEAST, 50)
        p = _prompt([("alpha", round(a, 2)), ("beta", round(b, 2))])
        opts = ["alpha", "beta", "Roughly equal"]; rng.shuffle(opts)
        bad += execute(p, "alpha", "beta") not in opts
    case("6_option_shuffle_preserves_scoring", bad == 0, f"{bad}/{trials} mismatches")

    # 7 every retained R1 question matches its source specification
    from Submission1_DKE_Repair import source_schema as S
    from Submission1_DKE_Repair.prompt_checks import check_panel
    panel = [json.loads(l) for l in
             (C.CODES_ROOT / OUT_DIR / "r1_corrected_panel.jsonl").open(encoding="utf-8")]
    specs, rendered = [], {}
    for p in panel:
        specs.append(S.SourceSpec(**{k: p[k] for k in S.SourceSpec.__dataclass_fields__}))
        rendered[p["fresh_id"]] = {"wording_a": p["wording_a"], "wording_b": p["wording_b"]}
    rep = check_panel(specs, rendered)
    case("7_questions_match_their_source", rep["passed"],
         f"{rep['n_failures']} failures over {rep['n_specs']} comparisons")
    return results


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    res = invariants()
    payload = {"invariants": res,
               "passed": sum(1 for r in res if r["passed"]), "total": len(res),
               "rule": {"equal_if_gap_below_pp": EQUAL_BELOW,
                        "diff_if_gap_at_least_pp": DIFF_AT_LEAST,
                        "intermediate_band": "excluded; the executor abstains"},
               "what_a_perfect_score_means": (
                   "The synthetic task is mechanically solvable under the declared rule from the "
                   "rendered table alone, so a model that fails an item is not defeated by an "
                   "ill-posed question. It is not evidence of reasoning, and the executor is not "
                   "a fair comparator for a closed-book model: it is given the table.")}
    C.write_json(out / "table_executor_validation.json", payload)
    for r in res:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['invariant']:42s} {r['detail']}")
    print(f"\n  {payload['passed']}/{payload['total']} invariants pass")


if __name__ == "__main__":
    main()
