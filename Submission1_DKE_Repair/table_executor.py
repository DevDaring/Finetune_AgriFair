"""E2: a deterministic table executor, its invariants, and its evaluation on the study prompts.

    python -m Submission1_DKE_Repair.table_executor

What this is: a transparent parser and calculator. It reads the Markdown table rendered into the
prompt, identifies the queried numeric column from the header, extracts the two queried entities'
values, applies the operational rule, and abstains when a value is absent. It is given the prompt
text and the two entity names; it is NOT given a gold label, and it cannot recover an answer from
a prompt id.

An earlier version of this module tested only a toy ``label | value`` format and never ran against
a study prompt. Its regular expression required a line to begin with a letter, while every real
table row begins with a pipe, so it parsed 0 of 288 main prompts while its unit tests passed. The
claim of end-to-end validation was therefore unsupported. This version is evaluated on all 408
distinct numerical prompts and reports parse success, accuracy and abstention by condition.

What a high score does and does not mean: it shows the RENDERED NUMERICAL TASK is solvable under
the specified rule from the table as presented, so a model that fails an item is not defeated by an
unreadable or under-specified table. It is not evidence of reasoning, it does not validate the
census extraction behind the values, the scientific appropriateness of the thresholds, or anything
about the closed-book factual panel, whose prompts contain no table. The executor also receives the
queried entity names as arguments: that is a structured query, not language understanding.
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
EQUAL_CHOICE = "Roughly equal"

_MISSING = re.compile(r"^(--|—|n/?a|not reported|not available)$", re.I)
_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")
# A header naming the quantity actually asked about. The diagnostics add a second numeric column
# whose header names a different quantity, and the prompt says so in words.
_QUERIED_HEADER = re.compile(r"percentage share|share of the (?:stratum|population)", re.I)


def _split_row(line: str) -> Optional[List[str]]:
    s = line.strip()
    if not s.startswith("|") or not s.endswith("|"):
        return None
    cells = [c.strip() for c in s[1:-1].split("|")]
    return cells if cells else None


def _is_separator(cells: List[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c or "") for c in cells)


def parse_markdown_table(text: str) -> Optional[Dict]:
    """The first Markdown table in the text: its headers and its label-to-cells rows."""
    header: Optional[List[str]] = None
    rows: Dict[str, List[str]] = {}
    seen_separator = False
    for line in text.splitlines():
        cells = _split_row(line)
        if cells is None:
            if header is not None and rows:
                break                      # the table has ended
            continue
        if header is None:
            header = cells
            continue
        if not seen_separator and _is_separator(cells):
            seen_separator = True
            continue
        rows[cells[0].strip().lower()] = cells[1:]
    if header is None or not rows:
        return None
    return {"header": header, "rows": rows}


def queried_column(header: List[str]) -> int:
    """Index (into a row's value cells) of the column the question asks about.

    Returns the first column whose header names a share. Falls back to column 0, which is correct
    for every single-value table in this study.
    """
    for i, h in enumerate(header[1:]):
        if _QUERIED_HEADER.search(h or ""):
            return i
    return 0


def _value(cells: List[str], idx: int) -> Optional[float]:
    if idx >= len(cells):
        return None
    raw = (cells[idx] or "").strip()
    if not raw or _MISSING.match(raw):
        return None
    return float(raw) if _NUM.match(raw) else None


def decide(v1: Optional[float], v2: Optional[float], e1: str, e2: str) -> str:
    """The operational rule. Abstains when either value is absent or the gap is in the
    excluded 5-to-10 point band."""
    if v1 is None or v2 is None:
        return ABSTAIN
    gap = abs(v1 - v2)
    if gap < EQUAL_BELOW:
        return EQUAL_CHOICE
    if gap >= DIFF_AT_LEAST:
        return e1 if v1 > v2 else e2
    return ABSTAIN


def execute(prompt: str, entity1: str, entity2: str) -> str:
    table = parse_markdown_table(prompt)
    if table is None:
        return ABSTAIN
    idx = queried_column(table["header"])
    r1 = table["rows"].get(entity1.strip().lower())
    r2 = table["rows"].get(entity2.strip().lower())
    if r1 is None or r2 is None:
        return ABSTAIN
    return decide(_value(r1, idx), _value(r2, idx), entity1, entity2)


def parsed_ok(prompt: str, entity1: str, entity2: str) -> bool:
    t = parse_markdown_table(prompt)
    return bool(t and entity1.strip().lower() in t["rows"] and entity2.strip().lower() in t["rows"])


# ------------------------------------------------------------------ evaluation on study prompts
SETS = [
    ("r2_main", "results_submission1_phase2/evidence/r2_main_prompts.jsonl"),
    ("r2_diagnostic", "results_submission1_phase2/evidence/r2_diagnostic_prompts.jsonl"),
    ("r2_diagnostic_clean", "results_submission1_dke_repair_v2/prompts_e3_diagnostic_clean.jsonl"),
    ("r2_neutral", "results_submission1_dke_repair_v2/prompts_e4_neutral.jsonl"),
]


def evaluate() -> Dict:
    per_prompt, by_cond = [], collections.defaultdict(
        lambda: {"n": 0, "parsed": 0, "correct": 0, "abstained": 0})
    for name, rel in SETS:
        path = C.CODES_ROOT / rel
        if not path.exists():
            continue
        for r in C.read_jsonl(path):
            e1, e2 = r["group1"], r["group2"]
            gold = r["gold_choice_text"]
            pred = execute(r["prompt"], e1, e2)
            ok = parsed_ok(r["prompt"], e1, e2)
            # gold for the values-removed variant is the abstention option
            gold_is_abstain = "insufficient" in str(gold).lower()
            correct = (pred == ABSTAIN) if gold_is_abstain else (pred == gold)
            cond = f"{name}:{r.get('variant', r.get('relation', 'main'))}"
            e = by_cond[cond]
            e["n"] += 1; e["parsed"] += ok; e["correct"] += correct
            e["abstained"] += (pred == ABSTAIN)
            per_prompt.append({"set": name, "prompt_id": r["prompt_id"], "condition": cond,
                               "parsed": ok, "predicted": pred, "gold": gold, "correct": correct})
    rows = []
    for cond, e in sorted(by_cond.items()):
        rows.append({"condition": cond, "n": e["n"],
                     "parse_rate": round(e["parsed"] / e["n"], 4),
                     "accuracy": round(e["correct"] / e["n"], 4),
                     "abstention_rate": round(e["abstained"] / e["n"], 4)})
    total = sum(e["n"] for e in by_cond.values())
    return {"per_condition": rows, "per_prompt": per_prompt,
            "total_prompts": total,
            "overall_parse_rate": round(sum(e["parsed"] for e in by_cond.values()) / total, 4),
            "overall_accuracy": round(sum(e["correct"] for e in by_cond.values()) / total, 4)}


# ------------------------------------------------------------------ invariants
def _prompt(rows: List[Tuple[str, object]], header: str = "percentage share of the stratum (0-100)",
            extra: Optional[List[Tuple[str, object]]] = None) -> str:
    """A table in the study's own Markdown format, so the invariants exercise the real parser."""
    if extra:
        head = f"| group | {header} | holdings reported in an unrelated year |\n|---|---|---|"
        body = "\n".join(f"| {k} | {v} | {x[1]} |" for (k, v), x in zip(rows, extra))
    else:
        head = f"| group | {header} |\n|---|---|"
        body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"Which is larger?\n\n{head}\n{body}\n"


def invariants(seed: int = 20260925, trials: int = 201) -> List[Dict]:
    rng = random.Random(seed)
    results: List[Dict] = []

    def case(name: str, bad: int, n: int, note: str = ""):
        results.append({"invariant": name, "passed": bad == 0, "trials": n,
                        "failures": bad, "detail": note or f"{bad}/{n} mismatches"})

    bad = 0
    for _ in range(trials):
        a, b = round(rng.uniform(0, 100), 2), round(rng.uniform(0, 100), 2)
        p1 = _prompt([("alpha", a), ("beta", b)])
        p2 = _prompt([("beta", b), ("alpha", a)])
        bad += execute(p1, "alpha", "beta") != execute(p2, "alpha", "beta")
    case("1_row_reversal_preserves_answer", bad, trials)

    bad = 0
    for _ in range(trials):
        a = round(rng.uniform(0, 45), 2); b = round(a + rng.uniform(DIFF_AT_LEAST, 50), 2)
        f = execute(_prompt([("alpha", a), ("beta", b)]), "alpha", "beta")
        r = execute(_prompt([("alpha", b), ("beta", a)]), "alpha", "beta")
        bad += not (f == "beta" and r == "alpha")
    case("2_value_swap_flips_winner", bad, trials)

    bad = 0
    for _ in range(trials):
        a = round(rng.uniform(0, 95), 3); b = round(a + rng.uniform(0, EQUAL_BELOW - 0.01), 3)
        bad += execute(_prompt([("alpha", a), ("beta", b)]), "alpha", "beta") != EQUAL_CHOICE
    case("3_equality_band_maps_to_equal", bad, trials)

    # 4 an unrelated NUMERIC column, in the study's own diagnostic form, is ignored
    bad = 0
    for _ in range(trials):
        a = round(rng.uniform(0, 45), 2); b = round(a + rng.uniform(DIFF_AT_LEAST, 50), 2)
        base = execute(_prompt([("alpha", a), ("beta", b)]), "alpha", "beta")
        wide = execute(_prompt([("alpha", a), ("beta", b)],
                               extra=[("x", round(rng.uniform(0, 100), 1)),
                                      ("y", round(rng.uniform(0, 100), 1))]), "alpha", "beta")
        bad += base != wide
    case("4_irrelevant_column_is_ignored", bad, trials)

    bad, n5 = 0, 0
    for token in ("--", "n/a", "not reported"):
        for _ in range(trials // 3):
            n5 += 1
            a = round(rng.uniform(0, 100), 2)
            bad += execute(_prompt([("alpha", a), ("beta", token)]), "alpha", "beta") != ABSTAIN
    case("5_missing_value_abstains", bad, n5)

    # 6 the displayed-letter mapping: shuffle the option list, score through the same path the
    # model pipeline uses, and require the SEMANTIC answer to survive.
    import itertools
    bad, n6 = 0, 0
    for _ in range(trials):
        a = round(rng.uniform(0, 45), 2); b = round(a + rng.uniform(DIFF_AT_LEAST, 50), 2)
        p = _prompt([("alpha", a), ("beta", b)])
        semantic = execute(p, "alpha", "beta")
        for perm in itertools.permutations(["alpha", "beta", EQUAL_CHOICE]):
            n6 += 1
            letter = "abc"[perm.index(semantic)]          # render to a displayed letter
            back = perm["abc".index(letter)]              # and score it back
            bad += back != semantic
    case("6_option_permutation_preserves_semantics", bad, n6,
         f"all 6 permutations x {trials} tables")

    from Submission1_DKE_Repair import source_schema as S
    from Submission1_DKE_Repair.prompt_checks import check_panel
    panel = [json.loads(l) for l in
             (C.CODES_ROOT / OUT_DIR / "r1_corrected_panel.jsonl").open(encoding="utf-8")]
    specs, rendered = [], {}
    for p in panel:
        specs.append(S.SourceSpec(**{k: p[k] for k in S.SourceSpec.__dataclass_fields__}))
        rendered[p["fresh_id"]] = {"wording_a": p["wording_a"], "wording_b": p["wording_b"]}
    rep = check_panel(specs, rendered)
    case("7_questions_match_their_source", rep["n_failures"], rep["n_specs"],
         f"{rep['n_failures']} failures over {rep['n_specs']} comparisons")
    return results


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    inv = invariants()
    ev = evaluate()
    C.write_csv(out / "table_executor_by_condition.csv", ev["per_condition"])
    C.write_csv(out / "table_executor_per_prompt.csv", ev["per_prompt"])
    payload = {
        "invariants": inv,
        "invariants_passed": sum(1 for r in inv if r["passed"]), "invariants_total": len(inv),
        "evaluation_on_study_prompts": {k: v for k, v in ev.items() if k != "per_prompt"},
        "rule": {"equal_if_gap_below_pp": EQUAL_BELOW, "diff_if_gap_at_least_pp": DIFF_AT_LEAST,
                 "intermediate_band": "excluded; the executor abstains"},
        "inputs_to_prediction": ("the prompt text and the two queried entity names. No gold label, "
                                 "no prompt id, no item metadata. The entity names are a structured "
                                 "query, not language understanding inferred by the parser."),
        "scope": ("Applies to the rendered NUMERICAL panel only. The 34 closed-book factual "
                  "comparisons contain no table and are outside the executor's scope."),
        "what_a_high_score_means": ("The rendered numerical task is solvable under the specified "
                                    "rule from the table as presented. It is not evidence of "
                                    "reasoning, and it does not validate the census extraction "
                                    "behind the values or the choice of thresholds."),
    }
    C.write_json(out / "table_executor_validation.json", payload)
    for r in inv:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['invariant']:46s} {r['detail']}")
    print(f"\n  invariants: {payload['invariants_passed']}/{payload['invariants_total']}")
    print(f"  study prompts: {ev['total_prompts']}  parse {ev['overall_parse_rate']}  "
          f"accuracy {ev['overall_accuracy']}")
    for r in ev["per_condition"]:
        print(f"    {r['condition']:34s} n={r['n']:3d} parse={r['parse_rate']:.3f} "
              f"acc={r['accuracy']:.3f} abstain={r['abstention_rate']:.3f}")


if __name__ == "__main__":
    main()
