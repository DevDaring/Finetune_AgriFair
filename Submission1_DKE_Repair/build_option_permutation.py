"""Priority 3, Route B: an option-permutation control for the factual panel.

    python -m Submission1_DKE_Repair.build_option_permutation

Every prompt in the corrected factual panel lists its options as [entity1, entity2, Roughly equal],
so ``roughly equal`` is always option (c). Three of the four systems answered ``roughly equal`` to
almost every item, which means they also answered (c) to almost every item. Those two explanations
are not separable in that design.

This builds one additional, predeclared permutation per comparison, deciding placement from the
comparison identifier alone:

    even-numbered comparisons  -> equality at (a)
    odd-numbered comparisons   -> equality at (b)

The assignment is fixed by identifier, not by any observed model behaviour, and both wordings of a
comparison receive the same permutation so the paired wording contrast stays intact. Text, values,
decision rule, gold answer, decoding and checkpoints are otherwise unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"
EQUAL = "Roughly equal"
ANSWER_SUFFIX = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'


def permuted_choices(panel_row: Dict) -> List[str]:
    """Equality at (a) for even comparison numbers, at (b) for odd ones."""
    n = int(re.sub(r"\D", "", panel_row["fresh_id"]) or 0)
    e1, e2 = panel_row["entity1"], panel_row["entity2"]
    return [EQUAL, e1, e2] if n % 2 == 0 else [e1, EQUAL, e2]


def build(cfg: Dict) -> List[Dict]:
    panel = [json.loads(l) for l in
             (C.CODES_ROOT / OUT_DIR / "r1_corrected_panel.jsonl").open(encoding="utf-8")]
    out = []
    for it in panel:
        choices = permuted_choices(it)
        for wording in ("wording_a", "wording_b"):
            opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", choices))
            out.append({"prompt_id": f"v2perm-{it['fresh_id']}-{wording}",
                        "study": "r1_option_permuted",
                        "comparison_id": it["fresh_id"], "source_cell": it["source_cell"],
                        "axis": it["axis"], "state": it["geography"],
                        "parent_table": it["parent_table"], "wording": wording,
                        "condition": it["condition"], "choices": choices,
                        "gold_choice_text": it["gold_choice_text"],
                        "equality_position": "abc"[choices.index(EQUAL)],
                        "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]),
                        "prompt": f"{it[wording]}\n{opts}{ANSWER_SUFFIX}"})
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    cfg = C.load_config()
    out = C.CODES_ROOT / OUT_DIR
    rows = build(cfg)
    C.write_jsonl(out / "prompts_e5_option_permuted.jsonl", rows)
    by_pos: Dict[str, int] = {}
    for r in rows:
        by_pos[r["equality_position"]] = by_pos.get(r["equality_position"], 0) + 1
    # the paired wordings of a comparison must share a permutation
    paired = {}
    for r in rows:
        paired.setdefault(r["comparison_id"], set()).add(r["equality_position"])
    summary = {"prompts": len(rows), "responses_at_4_systems": len(rows) * 4,
               "equality_position_counts": by_pos,
               "comparisons_with_one_position": sum(1 for v in paired.values() if len(v) == 1),
               "comparisons": len(paired),
               "assignment_rule": ("equality at (a) for even comparison numbers and at (b) for odd "
                                   "ones, fixed by identifier before any output was seen")}
    C.write_json(out / "option_permutation_manifest.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
