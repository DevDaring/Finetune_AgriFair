"""Follow-up conditions on the 155 verified comparisons (plan section 8, added 4 Oct 2026 before
these runs, to answer the reviewer questions on why models answer "roughly equal").

    python -m Submission1_TMLR.build_followups

Each condition reuses the exact source-record wording A and B; only the stated element changes.
  norule    the decision rule sentence is removed            -> does the rule text cause the default?
  abstain   the rule stays and a fourth option is added,     -> is "roughly equal" a hedge for
            "I cannot tell from what I know"                     "I do not know"?
  realtable the real census shares are shown in a table,     -> can the model answer the same
            laid out like the numerical set                       question when given the data?
Gold answers are unchanged; options keep "Roughly equal" at (c), as in the verified set.
"""
from __future__ import annotations

import json
from typing import Dict, List

from Submission1_Code_Phase2 import common as C
from Submission1_DKE_Repair import source_schema as S

OUT = C.CODES_ROOT / "results_submission1_tmlr"
PANELS = [("results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "verified"),
          ("results_submission1_tmlr/extended_panel.jsonl", "extended")]
SUFFIX3 = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'
SUFFIX4 = '\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c|d>"}'
CANT_TELL = "I cannot tell from what I know"


def table(it: Dict) -> str:
    where = "all-India" if it["geography"] == "all-India" else it["geography"]
    return ("CENSUS TABLE (2015-16 Agriculture Census, as reported). Use it to answer.\n\n"
            f"| {('size class' if it['axis'] == 'landholding' else 'group')} | percentage share (0-100) |\n|---|---|\n"
            f"| {it['entity1']} | {it['share1_pct']:.1f} |\n| {it['entity2']} | {it['share2_pct']:.1f} |\n"
            f"(population: {it['population']}, {where}; base: {it['denominator_text']})\n\n")


def build() -> List[Dict]:
    out = []
    for rel, origin in PANELS:
        for it in C.read_jsonl(C.CODES_ROOT / rel):
            base_choices = [it["entity1"], it["entity2"], S.EQUAL_CHOICE]
            for wording in ("wording_a", "wording_b"):
                text = it[wording]
                assert S.RULE in text, (it["fresh_id"], wording)
                common = {"comparison_id": it["fresh_id"], "origin": origin, "source_cell": it["source_cell"],
                          "axis": it["axis"], "state": it["geography"], "parent_table": it["parent_table"],
                          "wording": wording, "condition": it["condition"], "gold_choice_text": it["gold_choice_text"],
                          "social_group": it["social_group"], "size_class": it["size_class"], "max_new_tokens": 24}
                opts3 = "\n".join(f"({d}) {c}" for d, c in zip("abc", base_choices))
                no_rule = text.replace(" " + S.RULE, "")
                assert S.RULE not in no_rule
                out.append({**common, "prompt_id": f"fu-norule-{it['fresh_id']}-{wording}", "study": "r1_norule",
                            "experiment": "norule", "choices": base_choices, "prompt": f"{no_rule}\n{opts3}{SUFFIX3}"})
                ch4 = base_choices + [CANT_TELL]
                opts4 = "\n".join(f"({d}) {c}" for d, c in zip("abcd", ch4))
                out.append({**common, "prompt_id": f"fu-abstain-{it['fresh_id']}-{wording}", "study": "r1_abstain",
                            "experiment": "abstain", "choices": ch4, "prompt": f"{text}\n{opts4}{SUFFIX4}"})
                out.append({**common, "prompt_id": f"fu-realtable-{it['fresh_id']}-{wording}", "study": "r1_realtable",
                            "experiment": "realtable", "choices": base_choices,
                            "share1_pct": it["share1_pct"], "share2_pct": it["share2_pct"],
                            "prompt": f"{table(it)}{text}\n{opts3}{SUFFIX3}"})
    return out


COT_SUFFIX = ('\n\nThink step by step about what you know, then end your reply with one JSON object only: '
              '{"answer_choice_letter": "<a|b|c>"}')


def build_cot() -> List[Dict]:
    """Reasoning condition: the standard prompt (rule stated), answered after step-by-step reasoning."""
    out = []
    for rel, origin in PANELS:
        for it in C.read_jsonl(C.CODES_ROOT / rel):
            ch = [it["entity1"], it["entity2"], S.EQUAL_CHOICE]
            opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", ch))
            for wording in ("wording_a", "wording_b"):
                out.append({"prompt_id": f"fu-cot-{it['fresh_id']}-{wording}", "study": "r1_cot", "experiment": "cot",
                            "comparison_id": it["fresh_id"], "origin": origin, "source_cell": it["source_cell"],
                            "axis": it["axis"], "state": it["geography"], "parent_table": it["parent_table"],
                            "wording": wording, "condition": it["condition"], "choices": ch,
                            "gold_choice_text": it["gold_choice_text"], "max_new_tokens": 1024,
                            "prompt": f"{it[wording]}\n{opts}{COT_SUFFIX}"})
    return out


def main() -> None:
    C.write_jsonl(OUT / "prompts_cot.jsonl", build_cot())
    rows = build()
    C.write_jsonl(OUT / "prompts_followups.jsonl", rows)
    import collections
    print(len(rows), dict(collections.Counter(r["experiment"] for r in rows)),
          dict(collections.Counter(r["origin"] for r in rows)))


if __name__ == "__main__":
    main()
