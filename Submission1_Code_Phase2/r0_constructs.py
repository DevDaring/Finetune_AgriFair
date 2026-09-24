"""R0: replace the two-part fairness definition, and review prompt eligibility blind.

    python -m Submission1_Code_Phase2.r0_constructs --spec        # construct specification
    python -m Submission1_Code_Phase2.r0_constructs --packets     # blinded eligibility packets
    python -m Submission1_Code_Phase2.r0_constructs --import R1.csv R2.csv

The manuscript claimed fair behaviour has exactly two parts. It does not. This module writes
the four-construct specification the revision uses, and builds the blinded packet that lets
domain readers judge, for each of the 48 human-study question pairs, whether substantive advice
*should* be invariant — without seeing any model answer, any rating, or which system produced
what. Their judgement is a new validity assessment; it never edits the original ratings.

The advice decomposition used throughout: agronomic context C, demographic description I,
presentation needs R, substantive actions A, explanation style S. Invariance of A is a
hypothesis only where C is fixed and the task makes I irrelevant; a change in R may legitimately
change S; a change in relevant C may require a change in A.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

ELIGIBILITY = ["substantive_advice_should_be_invariant", "presentation_adaptation_warranted",
               "substantive_adaptation_may_be_warranted", "insufficient_context"]

SPEC = """# AgriFair construct specification (Phase 2)

The earlier manuscript defined fair behaviour as two things: reporting documented group
differences and keeping advice invariant under identity changes. That definition is too broad
for what this resource measures, and it treats every identity change as irrelevant. This
specification replaces it with four separate constructs, each with its own instrument and its
own explicit limits. They are reported separately; they are never combined into one score.

| Construct | Instrument here | What a score establishes | What it does not establish |
|---|---|---|---|
| Factual comparison accuracy | Closed-book three-option questions over published census comparisons | Whether a model reports a specified aggregate numerical relation correctly | Fair treatment of an individual farmer, or knowledge of present-day conditions |
| Numerical-evidence sensitivity | Hypothetical tables where only the numbers change (R2) | Whether a prediction follows controlled changes in supplied numbers | Comprehension, causal reasoning, or safe deployment |
| Context-appropriate advice consistency | Paired prompts whose eligibility is reviewed first (R0, R3) | Whether an irrelevant identity change alters substantive advice, and whether a relevant context change receives warranted adaptation | Unconditional demographic invariance, or comprehensive social fairness |
| Advice quality | Reference-checklist scoring at controlled verbosity (R3) | Reference-supported claims, coverage of essentials, usefulness, safety | Field effectiveness, farmer benefit, or equal access to services |

## The advice decomposition

For an advice prompt, separate:

- **C** agronomic context: crop, stage, symptom, season, water, soil, location where it bears on agronomy.
- **I** demographic description: gender, social group, and similar attributes of the asker.
- **R** presentation needs: reading level, language, length preference.
- **A** substantive actions: what the farmer is told to do.
- **S** explanation style: wording, ordering, amount of background.

Invariance of **A** is a hypothesis only where **C** is held fixed and the reviewed task makes
**I** irrelevant. A change in **R** may legitimately change **S** without changing **A**. A
change in a relevant part of **C** may require **A** itself to change; treating that as unfairness
would be an error.

## Consequences for the original pairs

The construction code changes **Bihar to Punjab**, which are different agro-climatic regions with
different crop calendars and different state services: that is a change in **C**, not a register
change, and advice may legitimately differ. It also changes limited schooling to an agriculture
degree, which is a change in **R**: style may legitimately change while actions should not.
Gender and social group are irrelevant to most agronomic questions, but not to questions about
eligibility, entitlement or access. Eligibility is therefore decided per task, by readers, before
any outcome is examined.

## The comparison rule is not a fairness threshold

The 5 and 10 percentage-point thresholds are an operational rule for turning census shares into
three answer categories. They carry no claim about when a difference matters to a farmer, and
aggregate holding shares say nothing about any individual's resources or needs.

## Measurement framing

This separation follows the distinction between a construct and its measurement instrument in
Jacobs and Wallach, *Measurement and Fairness* (arXiv:1912.05511). The behavioural-testing design
follows CheckList (Ribeiro et al., ACL 2020); what this resource adds is source-traceable
numerical evidence with a published denominator, not the idea of invariance testing itself.
"""


def build_packets(cfg: Dict, out: Path) -> Dict:
    """One blinded row per human-study question: the two prompts only, no answers, no systems."""
    mapping_p = C.ORIGINAL_AUDIT / "advice" / "human_study" / "KEEP_FROM_RATERS" / "assessment_mapping.csv"
    pairs_p = C.DATASET_ADVICE
    if not mapping_p.exists():
        raise SystemExit(f"assessment mapping not found: {mapping_p}")
    mapping = list(csv.DictReader(mapping_p.open(encoding="utf-8")))
    pair_ids = sorted({m["pair_id"] for m in mapping if m.get("kind") == "main"})
    pairs = {p["pair_id"]: p for p in C.read_jsonl(pairs_p)}
    rng = random.Random(cfg["analysis_seed"])
    rows = []
    for pid in pair_ids:
        p = pairs.get(pid)
        if not p:
            continue
        a, b = p["version_A"], p["version_B"]
        flip = rng.random() < 0.5                       # hide which side was version A
        left, right = (b, a) if flip else (a, b)
        rows.append({"review_id": f"elig-{len(rows):03d}", "pair_id": pid, "toggle_axis": p["toggle_axis"],
                     "base_question": p["base_query"], "prompt_left": left["prompt"], "prompt_right": right["prompt"],
                     "what_differs_between_left_and_right": "", "eligibility": "", "task_based_justification": "",
                     "relevant_context_changed_C_yes_no": "", "presentation_need_changed_R_yes_no": ""})
    C.write_csv(out / "eligibility_packet.csv", rows)
    C.write_json(out / "eligibility_key.json",
                 {"note": "kept out of the reviewer packet; maps review_id back to the pair and side order",
                  "rows": [{"review_id": r["review_id"], "pair_id": r["pair_id"], "axis": r["toggle_axis"]} for r in rows]})
    (out / "ELIGIBILITY_INSTRUCTIONS.txt").write_text(
        "AgriFair: which advice questions should be invariant?\n"
        "=====================================================\n\n"
        "You are judging QUESTIONS, not answers. No model output is included, and you are not being\n"
        "asked to check anyone's earlier ratings. Nothing here is a correction of previous work.\n\n"
        "Each row shows two versions of one farmer question. They differ in how the asker is described.\n"
        "For each row, decide what SHOULD happen to the substance of good agronomic advice:\n\n"
        "  substantive_advice_should_be_invariant\n"
        "      the difference between the two versions has no bearing on what the farmer should do;\n"
        "  presentation_adaptation_warranted\n"
        "      the actions should be the same, but the explanation may reasonably be pitched differently;\n"
        "  substantive_adaptation_may_be_warranted\n"
        "      the difference touches something agronomic or procedural (region, season, water, scheme\n"
        "      eligibility), so the recommended actions may legitimately differ;\n"
        "  insufficient_context\n"
        "      the question does not give you enough to decide.\n\n"
        "Write one short task-based justification: what about THIS question makes the difference\n"
        "relevant or irrelevant. Also mark whether a relevant agronomic context changed (C) and\n"
        "whether a presentation need changed (R).\n\n"
        "Two readers complete this independently. Disagreements are kept and reported; only the\n"
        "eligibility decision is adjudicated afterwards, with the reasoning recorded.\n", encoding="utf-8")
    return {"questions": len(rows), "axes": dict(collections.Counter(r["toggle_axis"] for r in rows))}


def import_reviews(cfg: Dict, out: Path, paths: List[Path]) -> Dict:
    readers, rows = [], []
    for i, p in enumerate(paths):
        rid = f"E{i + 1}"; readers.append(rid)
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r.get("eligibility") and r["eligibility"] not in ELIGIBILITY:
                raise SystemExit(f"{p.name}:{r.get('review_id')}: eligibility '{r['eligibility']}' is not one of {ELIGIBILITY}")
            rows.append({"reader": rid, **{k: r.get(k, "") for k in
                                           ("review_id", "pair_id", "toggle_axis", "eligibility",
                                            "task_based_justification", "relevant_context_changed_C_yes_no",
                                            "presentation_need_changed_R_yes_no")}})
    C.write_csv(out / "eligibility_reviews.csv", rows)
    by_id = collections.defaultdict(dict)
    for r in rows:
        by_id[r["review_id"]][r["reader"]] = r["eligibility"]
    agreed = {k: v for k, v in by_id.items() if len(set(v.values())) == 1 and len(v) == len(readers)}
    disagreed = {k: v for k, v in by_id.items() if k not in agreed}
    invariant = [k for k, v in agreed.items() if list(v.values())[0] == "substantive_advice_should_be_invariant"]
    summary = {"readers": readers, "questions_reviewed": len(by_id), "unanimous": len(agreed),
               "disagreements_to_adjudicate": len(disagreed),
               "eligible_for_invariance_analysis": len(invariant),
               "distribution": {rdr: dict(collections.Counter(r["eligibility"] for r in rows if r["reader"] == rdr)) for rdr in readers},
               "note": ("The invariance subset is a post-hoc eligibility stratum. Report it beside the original "
                        "all-axis result with its reduced denominator; do not replace the original analysis with it.")}
    C.write_json(out / "eligibility_summary.json", summary)
    C.write_json(out / "eligibility_subsets.json",
                 {"invariance_eligible_review_ids": sorted(invariant),
                  "disagreement_review_ids": sorted(disagreed), "all_review_ids": sorted(by_id)})
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="store_true"); ap.add_argument("--packets", action="store_true")
    ap.add_argument("--import", dest="imp", nargs="+", type=Path); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "constructs")
    if a.spec or not (a.packets or a.imp):
        (out / "construct_specification.md").write_text(SPEC, encoding="utf-8")
        print(f"[r0_constructs] construct specification -> {out / 'construct_specification.md'}")
    if a.packets:
        print("[r0_constructs]", build_packets(cfg, out), "->", out)
    if a.imp:
        print("[r0_constructs]", json.dumps(import_reviews(cfg, out, a.imp), indent=1))


if __name__ == "__main__":
    main()
