"""Extend the altered-table check from 12 to all 48 numerical scenarios (added 4 Oct 2026: the
irrelevant-column effect was large on 12 scenarios, so it is measured on all 48).

    python -m Submission1_TMLR.build_altered48

The 36 new scenarios use the published generator (Submission1_Code_Phase2.r2_evidence): the same
table renderer, stem, decision rule, options and insufficient-evidence option. Baseline relations are
balanced (12 per relation) by a fixed seed. Each scenario gets the three alterations and the unchanged
table. Before writing, the generator is checked by rebuilding the 12 published unchanged tables and
requiring them to equal the published ones.
"""
from __future__ import annotations

import collections
import random

from Submission1_Code_Phase2 import common as C
from Submission1_Code_Phase2 import r2_evidence as E

OUT = C.CODES_ROOT / "results_submission1_tmlr" / "prompts_altered48.jsonl"
SEED = 20261004
VARIANTS = ["row_order_reversed", "irrelevant_column", "values_removed", "unchanged"]


def render(b, relation, v1, v2, choices, variant, cfg):
    p = C.parse_cell(b["source_cell"])
    stratum = "" if p["size_class"].lower() in ("all", "all classes", "") else f" among {p['size_class'].lower()} holdings"
    opts, _ = E.options_block(choices, with_insufficient=True)
    kw = {"reverse_rows": variant == "row_order_reversed",
          "irrelevant": ("holdings reported in an unrelated year", round(min(99.0, max(v1, v2) + 20), 1),
                         round(min(99.0, max(v1, v2) + 25), 1)) if variant == "irrelevant_column" else None,
          "omit_values": variant == "values_removed"}
    table = E.render_table(b["group1"], b["group2"], v1, v2, p["metric"], b.get("denominator", "the stratum total"), **kw)
    return E.DIAG_STEM.format(warning=E.WARNING, table=table, rule=E.RULE_TEXT, state=p["state"],
                              metric_phrase=E.METRIC_PHRASE.get(p["metric"], p["metric"]), stratum=stratum,
                              g1=b["group1"], g2=b["group2"], options=opts, schema=E.DIAG_SCHEMA)


def main() -> None:
    cfg = C.load_config()
    bundles = E.source_bundles(cfg)
    n_old = int(cfg["r2"]["diagnostic_bundles"])
    published = {r["bundle_id"]: r for r in C.read_jsonl(C.CODES_ROOT / "results_submission1_phase2/evidence/r2_diagnostic_prompts.jsonl")}
    clean = {r["bundle_id"]: r for r in C.read_jsonl(C.CODES_ROOT / "results_submission1_dke_repair_v2/prompts_e3_diagnostic_clean.jsonl")}

    # ---- fidelity check on the 12 published scenarios
    mismatches = 0
    for b in bundles[:n_old]:
        pub = published[b["bundle_id"]]
        choices = pub["choices"][:3]
        text = render(b, pub["baseline_relation"], pub["value1"], pub["value2"], choices, "unchanged", cfg)
        if b["bundle_id"] in clean and " ".join(text.split()) != " ".join(clean[b["bundle_id"]]["prompt"].split()):
            mismatches += 1
    if mismatches:
        raise SystemExit(f"generator does not reproduce {mismatches} published unchanged tables")

    # ---- the 36 new scenarios
    new = bundles[n_old:]
    rels = [E.RELATIONS[i % 3] for i in range(len(new))]
    random.Random(SEED).shuffle(rels)
    rows = []
    for b, relation in zip(new, rels):
        v1, v2 = E.make_values(relation, random.Random(f"{SEED}-diag48-{b['bundle_id']}"), cfg)
        choices = [b["group1"], b["group2"], E.EQUAL]
        random.Random(f"{SEED}-diag48order-{b['bundle_id']}").shuffle(choices)
        for variant in VARIANTS:
            gold = (E.INSUFFICIENT if variant == "values_removed" else
                    (E.EQUAL if relation == "approximately_equal" else (b["group1"] if relation == "first_higher" else b["group2"])))
            p = C.parse_cell(b["source_cell"])
            rows.append({"prompt_id": f"tmlr-diag-{b['bundle_id']}-{variant}", "study": "r2_diagnostic_ext",
                         "experiment": "unchanged" if variant == "unchanged" else "altered",
                         "bundle_id": b["bundle_id"], "source_cell": b["source_cell"], "axis": b["axis"],
                         "state": p["state"], "parent_table": C.parent_table(b["source_cell"]), "variant": variant,
                         "baseline_relation": relation, "group1": b["group1"], "group2": b["group2"],
                         "value1": v1, "value2": v2, "choices": choices + ["The table does not contain the values needed"],
                         "gold_choice_text": gold, "allows_insufficient": True, "max_new_tokens": 24,
                         "prompt": render(b, relation, v1, v2, choices, variant, cfg)})
    C.write_jsonl(OUT, rows)
    print(f"fidelity: rebuilt {n_old} published unchanged tables, mismatches = {mismatches}")
    print(len(rows), "prompts;", dict(collections.Counter(r["variant"] for r in rows)),
          "relations:", dict(collections.Counter(r["baseline_relation"] for r in rows if r["variant"] == "unchanged")))


if __name__ == "__main__":
    main()
