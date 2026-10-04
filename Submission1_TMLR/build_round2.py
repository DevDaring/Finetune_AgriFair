"""Round-2 prompt files (Future_PLan.md E3, E4, E6), built from the verified panels before any output.

    python -m Submission1_TMLR.build_round2

prompts_rulevariants.jsonl   E3: the 154 comparisons x 2 wordings x 3 rule variants (experiment tags
                             rule_2_20, rule_reversed, rule_first). Items whose label would change
                             under the 2/20 thresholds are dropped from that variant and counted.
prompts_fewshot.jsonl        E4: 3 fixed demonstrations (first entity, second entity, roughly equal),
                             rendered by the published generator from training-split census cells,
                             then the standard prompt (experiment tag fewshot).
prompts_reordered_ext.jsonl  E6: the 120 new comparisons with "Roughly equal" moved to (a) for even
                             item numbers and (b) for odd ones, as for the original 34 (experiment
                             tag reordered).
"""
from __future__ import annotations

import collections
import json
import random
import re
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2 import r1_fresh_panel as R
from Submission1_DKE_Repair import source_schema as S
from Submission1_TMLR.build_extended import to_record
from Submission1_TMLR.exclusions import EXCLUDED_COMPARISONS

OUT = C.CODES_ROOT / "results_submission1_tmlr"
PANELS = [("results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "verified"),
          ("results_submission1_tmlr/extended_panel.jsonl", "extended")]
SUFFIX = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'
SEED = 20261004
RULE_2_20 = ("Treat the two as roughly equal if their shares differ by less than 2 percentage points, "
             "and treat one as larger only if it leads by at least 20 percentage points.")
RULE_REVERSED = ("Treat one as larger only if it leads by at least 10 percentage points; "
                 "otherwise treat the two as roughly equal.")


def items() -> List[Dict]:
    out = []
    for rel, origin in PANELS:
        for it in C.read_jsonl(C.CODES_ROOT / rel):
            if it["fresh_id"] not in EXCLUDED_COMPARISONS:
                out.append({**it, "origin": origin})
    return out


def common(it: Dict, wording: str) -> Dict:
    return {"comparison_id": it["fresh_id"], "origin": it["origin"], "source_cell": it["source_cell"],
            "axis": it["axis"], "state": it["geography"], "parent_table": it["parent_table"], "wording": wording,
            "condition": it["condition"], "gold_choice_text": it["gold_choice_text"], "max_new_tokens": 24}


def options(choices: List[str]) -> str:
    return "\n".join(f"({d}) {c}" for d, c in zip("abc", choices))


# ---------------------------------------------------------------- E3
def rule_variants(its: List[Dict]) -> List[Dict]:
    out, dropped = [], 0
    for it in its:
        choices = [it["entity1"], it["entity2"], S.EQUAL_CHOICE]
        gap = abs(float(it["share1_pct"]) - float(it["share2_pct"]))
        label_holds_2_20 = (gap < 2) if it["condition"] == "equal" else (gap >= 20)
        for wording in ("wording_a", "wording_b"):
            text = it[wording]
            assert text.endswith(" " + S.RULE) and text.count(S.RULE) == 1, it["fresh_id"]
            stem = text[: -len(" " + S.RULE)]
            variants = {"rule_reversed": f"{stem} {RULE_REVERSED}", "rule_first": f"{S.RULE} {stem}"}
            if label_holds_2_20:
                variants["rule_2_20"] = f"{stem} {RULE_2_20}"
            else:
                dropped += 1
            for tag, prompt in variants.items():
                out.append({**common(it, wording), "prompt_id": f"r2-{tag}-{it['fresh_id']}-{wording}",
                            "study": "r1_rule_variants", "experiment": tag, "choices": choices,
                            "prompt": f"{prompt}\n{options(choices)}{SUFFIX}"})
    print(f"rule variants: {len(out)} prompts; 2/20 variant drops {dropped} prompts whose label would change")
    return out


# ---------------------------------------------------------------- E4
def demonstrations() -> List[Dict]:
    """One training-split cell per answer type, chosen with a fixed seed; rendered as wording A."""
    train_cells = {json.loads(l)["source_cell"] for l in (C.CODES_ROOT / "data" / "train_instances.jsonl").open()}
    rows = [r for r in R.load_validated() if r["source_cell"] in train_cells and r["axis"] != "gender"
            and r["condition"] in ("equal", "diff")]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    want = {"first": lambda r: r["condition"] == "diff" and float(r["share_1"]) > float(r["share_2"]) and r["axis"] == "landholding",
            "second": lambda r: r["condition"] == "diff" and float(r["share_2"]) > float(r["share_1"]) and r["axis"] == "social_group",
            "equal": lambda r: r["condition"] == "equal"}
    demos = []
    for kind, ok in want.items():
        r = next(x for x in rows if ok(x))
        spec = S.parse(to_record(r, 900 + len(demos)))
        choices = [spec.entity1, spec.entity2, S.EQUAL_CHOICE]
        letter = "abc"[choices.index(spec.gold_entity)]
        demos.append({"kind": kind, "source_cell": r["source_cell"], "letter": letter,
                      "text": f"{S.render(spec)['wording_a']}\n{options(choices)}\n"
                              f'Answer: {{"answer_choice_letter": "{letter}"}}'})
    rng.shuffle(demos)
    return demos


def fewshot(its: List[Dict], demos: List[Dict]) -> List[Dict]:
    head = "Here are three answered examples.\n\n" + "\n\n".join(f"Example {i + 1}:\n{d['text']}" for i, d in enumerate(demos))
    out = []
    for it in its:
        choices = [it["entity1"], it["entity2"], S.EQUAL_CHOICE]
        for wording in ("wording_a", "wording_b"):
            out.append({**common(it, wording), "prompt_id": f"r2-fewshot-{it['fresh_id']}-{wording}",
                        "study": "r1_fewshot", "experiment": "fewshot", "choices": choices,
                        "prompt": f"{head}\n\nNow answer this question.\n{it[wording]}\n{options(choices)}{SUFFIX}"})
    return out


# ---------------------------------------------------------------- E6
def reordered_ext(its: List[Dict]) -> List[Dict]:
    out = []
    for it in its:
        if it["origin"] != "extended":
            continue
        n = int(re.sub(r"\D", "", it["fresh_id"]))
        choices = [S.EQUAL_CHOICE, it["entity1"], it["entity2"]] if n % 2 == 0 else [it["entity1"], S.EQUAL_CHOICE, it["entity2"]]
        for wording in ("wording_a", "wording_b"):
            out.append({**common(it, wording), "prompt_id": f"r2-perm-{it['fresh_id']}-{wording}",
                        "study": "r1_option_permuted_ext", "experiment": "reordered", "choices": choices,
                        "equality_position": "abc"[choices.index(S.EQUAL_CHOICE)],
                        "prompt": f"{it[wording]}\n{options(choices)}{SUFFIX}"})
    return out


def main() -> None:
    its = items()
    assert len(its) == 154, len(its)
    rv = rule_variants(its)
    demos = demonstrations()
    fs = fewshot(its, demos)
    ro = reordered_ext(its)
    for name, rows in (("prompts_rulevariants.jsonl", rv), ("prompts_fewshot.jsonl", fs), ("prompts_reordered_ext.jsonl", ro)):
        C.write_jsonl(OUT / name, rows)
    C.write_json(OUT / "fewshot_demonstrations.json", demos)
    test_cells = {it["source_cell"] for it in its}
    assert not any(d["source_cell"] in test_cells for d in demos), "a demonstration cell is a test cell"
    print({n: len(r) for n, r in (("rulevariants", rv), ("fewshot", fs), ("reordered_ext", ro))},
          dict(collections.Counter(r["experiment"] for r in rv)), [d["kind"] + ":" + d["letter"] for d in demos])


if __name__ == "__main__":
    main()
