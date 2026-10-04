"""E1: answer policies that need no model, scored like a system (Future_PLan.md, E1).

    python -m Submission1_TMLR.baselines

Policies on the 154 verified comparisons (both wordings):
  constant_equal, constant_first, constant_second
  cross_state_prior: for each comparison, the answer that is correct in the majority of the OTHER
    states' validated census cells with the same population restriction, measure and entity pair
    (leave-one-state-out). Ties and comparison types with no other state fall back to "roughly
    equal", and the count of such fallbacks is reported. The prior reads census data only.
On the 288 numerical prompts the three constant policies score 1/3 by design; they are listed for
completeness.
"""
from __future__ import annotations

import collections
import json

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2 import r1_fresh_panel as R
from Submission1_TMLR.exclusions import EXCLUDED_COMPARISONS

OUT = C.CODES_ROOT / "results_submission1_tmlr" / "analysis"
PANELS = ["results_submission1_dke_repair_v2/r1_corrected_panel.jsonl", "results_submission1_tmlr/extended_panel.jsonl"]
EQUAL = "Roughly equal"


def comparison_type(cell: str):
    """'T2-4 Karnataka/ST/area/MarginalvsSmall' -> ('T2-4', 'ST', 'area', 'MarginalvsSmall'); state dropped."""
    _, table, tail = cell.split(" ", 2)
    state, stratum, metric, cmp_ = tail.split("/")
    return (table, stratum, metric, cmp_), state


def cross_state_prior():
    """comparison type -> majority answer over validated cells, keyed so one state can be left out."""
    by_type = collections.defaultdict(list)
    for r in R.load_validated():
        key, state = comparison_type(r["source_cell"])
        ans = EQUAL if r["condition"] == "equal" else r["larger"]
        if r["condition"] in ("equal", "diff"):
            by_type[key].append((state, ans))
    return by_type


def main() -> None:
    items = [it for rel in PANELS for it in C.read_jsonl(C.CODES_ROOT / rel) if it["fresh_id"] not in EXCLUDED_COMPARISONS]
    prior = cross_state_prior()
    rows, fallbacks = [], 0
    answers = collections.defaultdict(dict)
    for it in items:
        key, state = comparison_type(it["source_cell"])
        others = [a for s, a in prior.get(key, []) if s != state]
        if others:
            cnt = collections.Counter(others).most_common()
            pick = cnt[0][0] if len(cnt) == 1 or cnt[0][1] > cnt[1][1] else None
        else:
            pick = None
        if pick is None:
            fallbacks += 1; pick = EQUAL
        answers["cross_state_prior"][it["fresh_id"]] = pick
        answers["constant_equal"][it["fresh_id"]] = EQUAL
        answers["constant_first"][it["fresh_id"]] = it["entity1"]
        answers["constant_second"][it["fresh_id"]] = it["entity2"]
    for policy, ans in answers.items():
        corr = [ans[it["fresh_id"]] == it["gold_choice_text"] for it in items]
        diff = [ans[it["fresh_id"]] == it["gold_choice_text"] for it in items if it["condition"] == "diff"]
        eq = [ans[it["fresh_id"]] == it["gold_choice_text"] for it in items if it["condition"] == "equal"]
        rows.append({"policy": policy, "n_comparisons": len(items),
                     "accuracy_154": round(sum(corr) / len(corr), 4),
                     "accuracy_difference_items": round(sum(diff) / len(diff), 4),
                     "accuracy_equal_items": round(sum(eq) / len(eq), 4),
                     "share_roughly_equal_answers": round(sum(a == EQUAL for a in ans.values()) / len(ans), 4),
                     "accuracy_numerical_288": round(1 / 3, 4) if policy.startswith("constant") else None,
                     "fallbacks_to_equal": fallbacks if policy == "cross_state_prior" else None})
    OUT.mkdir(parents=True, exist_ok=True)
    C.write_csv(OUT / "baselines.csv", rows)
    C.write_json(OUT / "baseline_cross_state_prior_answers.json", answers["cross_state_prior"])
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
