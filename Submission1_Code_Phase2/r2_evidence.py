"""R2: controlled numerical-evidence factorial on the original 48 P4 comparisons.

    python -m Submission1_Code_Phase2.r2_evidence --build

The original P4 changed the table header *and* the values at once, so "follows the verified
table but not the hypothetical one" could be a reaction to either. Here **every** table is
hypothetical, with identical warning, framing, units, threshold text, layout and output schema.
Within one wording, only the numeric cells move. A difference between relations is therefore a
response to numbers, not to framing.

Design (config r2):
  48 source comparisons x 3 numerical relations x 2 wordings x 4 systems = 1,152 short answers
  relations: first_higher, second_higher, approximately_equal  (balanced within each bundle)
  wordings:  context_rich (original-style stem), plain_new (independently written stem)

Diagnostics, on a frozen 12-bundle subset, one baseline relation per bundle, 4 per relation,
one fixed wording, three variants under a shared wrapper that allows `insufficient_evidence`:
  row_order_reversed   values and identities preserved -> the answer must not change
  irrelevant_column    an unrelated column carries a larger number -> the queried metric decides
  values_removed       the queried values are absent -> `insufficient_evidence` is the answer
  = 12 x 3 x 4 = 144 outputs, scored separately (their wrapper differs from the main task).

Value generation respects percentage bounds and the stratum denominator: two mutually exclusive
group shares never sum above 100. Equality-band cases use nonzero unequal values, and different
cases are not all extreme, so a model cannot pass by reacting to 100-versus-0.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from Submission1_Code_Phase2 import common as C

RELATIONS = ("first_higher", "second_higher", "approximately_equal")
GOLD = {"first_higher": "a", "second_higher": "b", "approximately_equal": "c"}
EQUAL = "Roughly equal"
INSUFFICIENT = "insufficient_evidence"

RULE_TEXT = ("Decision rule: if the two shares differ by less than 5 percentage points, they are roughly equal; "
             "if they differ by 10 percentage points or more, the larger share wins. Values between those are "
             "excluded and do not appear here.")
WARNING = "HYPOTHETICAL TABLE (illustrative values, not census observations). Answer only about this table."
SCHEMA = 'Reply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}'
DIAG_SCHEMA = ('Reply with one JSON object only: {"answer_choice_letter": "<a|b|c|d>"}, '
               'where d means the table does not contain the values needed to answer.')

STEMS = {
    "context_rich": ("{warning}\n\n{table}\n\n{rule}\n\nIn {state}, according to this table, which has the larger "
                     "share of {metric_phrase}{stratum} — {g1}, {g2}, or are the two roughly equal?\n{options}\n\n{schema}"),
    "plain_new": ("{warning}\n\n{table}\n\n{rule}\n\nUsing only the table above, rank {g1} and {g2} by their share of "
                  "{metric_phrase}{stratum} for {state}. If the two shares fall inside the equality band, the answer is "
                  "\"{equal}\".\n{options}\n\n{schema}"),
}
DIAG_STEM = ("{warning}\n\n{table}\n\n{rule}\n\nUsing only the table above, which has the larger share of "
             "{metric_phrase}{stratum} in {state} — {g1}, {g2}, or are the two roughly equal? If the table does not "
             "contain the values needed to answer, choose the insufficient-evidence option.\n{options}\n\n{schema}")

METRIC_PHRASE = {"number": "the number of holdings", "area": "operated area"}


# ------------------------------------------------------------------ value generation

def make_values(relation: str, rng: random.Random, cfg: Dict) -> Tuple[float, float]:
    """Two shares of one stratum: mutually exclusive, so their sum stays within 100."""
    r = cfg["r2"]; eq_max, diff_min, cap = r["equal_band_max_gap"], r["diff_band_min_gap"], r["max_share_sum"]
    for _ in range(200):
        if relation == "approximately_equal":
            gap = round(rng.uniform(0.3, eq_max - 0.2), 1)          # nonzero: never two identical numbers
            base = round(rng.uniform(8.0, min(45.0, (cap - gap) / 2)), 1)
            v1, v2 = base + gap, base
            if rng.random() < 0.5:
                v1, v2 = v2, v1
        else:
            gap = round(rng.uniform(diff_min + 0.5, 55.0), 1)       # spread, not all extreme
            low = round(rng.uniform(2.0, max(2.1, min(35.0, cap - gap - 2))), 1)
            v1, v2 = low + gap, low
            if relation == "second_higher":
                v1, v2 = v2, v1
        if v1 + v2 <= cap and min(v1, v2) >= 0.1 and max(v1, v2) <= 99.0:
            g = abs(v1 - v2)
            ok = (g < cfg["comparison_rule"]["equal_if_abs_gap_below"]) if relation == "approximately_equal" \
                else (g >= cfg["comparison_rule"]["diff_if_abs_gap_at_least"])
            if ok:
                return round(v1, 1), round(v2, 1)
    raise RuntimeError(f"could not generate values for {relation}")


def render_table(g1: str, g2: str, v1, v2, metric: str, denominator: str,
                 reverse_rows: bool = False, irrelevant: Tuple[str, float, float] = None, omit_values: bool = False) -> str:
    unit = "percentage share of the stratum (0-100)"
    head = ["| group | " + unit + (" | " + irrelevant[0] if irrelevant else "") + " |",
            "|---|---|" + ("---|" if irrelevant else "")]
    def cell(v, extra):
        val = "not reported" if omit_values else f"{v:g}"
        return f"| {extra[0]} | {val} |" + (f" {extra[1]:g} |" if irrelevant else "")
    rows = [(g1, v1, (irrelevant[1] if irrelevant else None)), (g2, v2, (irrelevant[2] if irrelevant else None))]
    if reverse_rows:
        rows = rows[::-1]
    body = [cell(v, (name, extra)) for name, v, extra in rows]
    tail = [f"(denominator: {denominator})"]
    if irrelevant:
        tail.append(f"(the {irrelevant[0]} column is not the quantity asked about)")
    return "\n".join(head + body + tail)


def options_block(choices: List[str], with_insufficient: bool = False) -> Tuple[str, Dict[str, str]]:
    letters = "abcd"
    opts = list(choices) + (["The table does not contain the values needed"] if with_insufficient else [])
    lines, d2c = [], {}
    for i, text in enumerate(opts):
        lines.append(f"({letters[i]}) {text}")
        d2c[letters[i]] = INSUFFICIENT if text.startswith("The table does not") else text
    return "\n".join(lines), d2c


# ------------------------------------------------------------------ bundles

def source_bundles(cfg: Dict) -> List[Dict]:
    """The original 48 P4 main comparisons, read from the preserved bundle file."""
    p = C.ORIGINAL_AUDIT / "evidence_panel" / "bundles.jsonl"
    if not p.exists():
        raise SystemExit(f"original P4 bundles not found: {p}")
    main = [b for b in C.read_jsonl(p) if b.get("panel_role") == "main"]
    want = int(cfg["r2"]["bundles"])
    if len(main) < want:
        print(f"  note: {len(main)} main bundles available, design asked for {want}")
    return main[:want]


def build(cfg: Dict) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(cfg["analysis_seed"])
    bundles = source_bundles(cfg)
    prompts: List[Dict] = []
    for bi, b in enumerate(bundles):
        g1, g2 = b["group1"], b["group2"]
        p = C.parse_cell(b["source_cell"])
        stratum = "" if p["size_class"].lower() in ("all", "all classes", "") else f" among {p['size_class'].lower()} holdings"
        metric_phrase = METRIC_PHRASE.get(b.get("units", "").startswith("percentage") and p["metric"] or p["metric"], p["metric"])
        # one option order per bundle, counterbalanced across bundles
        choices = [g1, g2, EQUAL]
        random.Random(f"{cfg['analysis_seed']}-order-{bi}").shuffle(choices)
        opts, d2c_text = options_block(choices)
        for relation in RELATIONS:
            v1, v2 = make_values(relation, random.Random(f"{cfg['analysis_seed']}-{b['bundle_id']}-{relation}"), cfg)
            table = render_table(g1, g2, v1, v2, p["metric"], b.get("denominator", "the stratum total"))
            for wording, stem in STEMS.items():
                text = stem.format(warning=WARNING, table=table, rule=RULE_TEXT, state=p["state"],
                                   metric_phrase=metric_phrase, stratum=stratum, g1=g1, g2=g2,
                                   options=opts, schema=SCHEMA, equal=EQUAL)
                gold_text = EQUAL if relation == "approximately_equal" else (g1 if relation == "first_higher" else g2)
                prompts.append({"prompt_id": f"r2-{b['bundle_id']}-{relation}-{wording}", "study": "r2_main",
                                "bundle_id": b["bundle_id"], "source_cell": b["source_cell"], "axis": b["axis"],
                                "state": p["state"], "parent_table": C.parent_table(b["source_cell"]),
                                "relation": relation, "wording": wording, "group1": g1, "group2": g2,
                                "value1": v1, "value2": v2, "gap_pp": round(abs(v1 - v2), 2),
                                "choices": choices, "gold_choice_text": gold_text,
                                "gold_canonical": GOLD[relation], "display_to_text": d2c_text,
                                "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]), "prompt": text})
    # ---- diagnostics: frozen 12-bundle subset, 4 per baseline relation
    diag: List[Dict] = []
    sub = bundles[: int(cfg["r2"]["diagnostic_bundles"])]
    base_relations = [RELATIONS[i % 3] for i in range(len(sub))]
    rng.shuffle(base_relations)
    for b, relation in zip(sub, base_relations):
        g1, g2 = b["group1"], b["group2"]; p = C.parse_cell(b["source_cell"])
        stratum = "" if p["size_class"].lower() in ("all", "all classes", "") else f" among {p['size_class'].lower()} holdings"
        v1, v2 = make_values(relation, random.Random(f"{cfg['analysis_seed']}-diag-{b['bundle_id']}"), cfg)
        choices = [g1, g2, EQUAL]
        random.Random(f"{cfg['analysis_seed']}-diagorder-{b['bundle_id']}").shuffle(choices)
        opts, _ = options_block(choices, with_insufficient=True)
        for variant in cfg["r2"]["diagnostic_variants"]:
            kw = {"reverse_rows": variant == "row_order_reversed",
                  "irrelevant": ("holdings reported in an unrelated year", round(min(99.0, max(v1, v2) + 20), 1), round(min(99.0, max(v1, v2) + 25), 1)) if variant == "irrelevant_column" else None,
                  "omit_values": variant == "values_removed"}
            table = render_table(g1, g2, v1, v2, p["metric"], b.get("denominator", "the stratum total"), **kw)
            text = DIAG_STEM.format(warning=WARNING, table=table, rule=RULE_TEXT, state=p["state"],
                                    metric_phrase=METRIC_PHRASE.get(p["metric"], p["metric"]), stratum=stratum,
                                    g1=g1, g2=g2, options=opts, schema=DIAG_SCHEMA)
            gold_text = INSUFFICIENT if variant == "values_removed" else (EQUAL if relation == "approximately_equal" else (g1 if relation == "first_higher" else g2))
            diag.append({"prompt_id": f"r2diag-{b['bundle_id']}-{variant}", "study": "r2_diagnostic",
                         "bundle_id": b["bundle_id"], "source_cell": b["source_cell"], "axis": b["axis"],
                         "state": p["state"], "parent_table": C.parent_table(b["source_cell"]),
                         "variant": variant, "baseline_relation": relation, "group1": g1, "group2": g2,
                         "value1": v1, "value2": v2, "choices": choices + ["The table does not contain the values needed"],
                         "gold_choice_text": gold_text, "allows_insufficient": True,
                         "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]), "prompt": text})
    return prompts, diag


# ------------------------------------------------------------------ validation

def validate(prompts: List[Dict], diag: List[Dict], cfg: Dict) -> List[str]:
    problems, rule = [], cfg["comparison_rule"]
    by_bundle = collections.defaultdict(set)
    for p in prompts:
        by_bundle[(p["bundle_id"], p["wording"])].add(p["relation"])
        gap = p["gap_pp"]
        if p["relation"] == "approximately_equal":
            if not 0 < gap < rule["equal_if_abs_gap_below"]:
                problems.append(f"{p['prompt_id']}: equal-band gap {gap} outside (0, {rule['equal_if_abs_gap_below']})")
            if p["value1"] == p["value2"]:
                problems.append(f"{p['prompt_id']}: identical values in the equality band")
        else:
            if gap < rule["diff_if_abs_gap_at_least"]:
                problems.append(f"{p['prompt_id']}: different-relation gap {gap} below threshold")
            higher = p["group1"] if p["value1"] > p["value2"] else p["group2"]
            want = p["group1"] if p["relation"] == "first_higher" else p["group2"]
            if higher != want:
                problems.append(f"{p['prompt_id']}: values do not realise {p['relation']}")
        if p["value1"] + p["value2"] > cfg["r2"]["max_share_sum"]:
            problems.append(f"{p['prompt_id']}: shares sum above the stratum total")
        if p["gold_choice_text"] not in p["choices"]:
            problems.append(f"{p['prompt_id']}: gold text is not among the options")
        if WARNING not in p["prompt"] or RULE_TEXT not in p["prompt"]:
            problems.append(f"{p['prompt_id']}: missing hypothetical warning or decision rule")
        if "not reported" in p["prompt"]:
            problems.append(f"{p['prompt_id']}: main-task prompt must contain both values")
    for key, rels in by_bundle.items():
        if set(rels) != set(RELATIONS):
            problems.append(f"{key}: bundle/wording does not carry all three relations")
    # each wording must render the same numbers for the same bundle+relation
    pair = collections.defaultdict(dict)
    for p in prompts:
        pair[(p["bundle_id"], p["relation"])][p["wording"]] = (p["value1"], p["value2"])
    for k, d in pair.items():
        if len(set(d.values())) > 1:
            problems.append(f"{k}: the two wordings show different numbers")
    seen = collections.Counter()
    for d in diag:
        seen[(d["bundle_id"], d["variant"])] += 1
        if d["variant"] == "values_removed":
            if "not reported" not in d["prompt"] or d["gold_choice_text"] != INSUFFICIENT:
                problems.append(f"{d['prompt_id']}: missing-evidence variant is not set up correctly")
        elif d["gold_choice_text"] == INSUFFICIENT:
            problems.append(f"{d['prompt_id']}: insufficient_evidence must only be correct when values are absent")
        if "insufficient" not in d["prompt"].lower():
            problems.append(f"{d['prompt_id']}: diagnostic wrapper must offer the insufficient-evidence option")
    if diag:
        rel_counts = collections.Counter(d["baseline_relation"] for d in diag if d["variant"] == diag[0]["variant"])
        if len(set(rel_counts.values())) > 1:
            problems.append(f"diagnostic baseline relations not balanced: {dict(rel_counts)}")
    return problems


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--build", action="store_true"); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "evidence")
    prompts, diag = build(cfg)
    problems = validate(prompts, diag, cfg)
    if problems:
        raise SystemExit("design rejected, nothing written:\n" + "\n".join(problems[:25]))
    if not a.build:
        print(f"[r2_evidence] validation only: {len(prompts)} main + {len(diag)} diagnostic prompts pass every invariant")
        return
    C.write_jsonl(out / "r2_main_prompts.jsonl", prompts)
    C.write_jsonl(out / "r2_diagnostic_prompts.jsonl", diag)
    n_sys = len(C.systems(cfg))
    C.write_json(out / "manifest.json", C.manifest(cfg, "r2_evidence", {
        "bundles": len({p["bundle_id"] for p in prompts}), "relations": list(RELATIONS),
        "wordings": list(STEMS), "main_prompts": len(prompts), "diagnostic_prompts": len(diag),
        "systems": n_sys, "main_outputs": len(prompts) * n_sys, "diagnostic_outputs": len(diag) * n_sys,
        "constant_label_baseline": {"accuracy": 1 / 3, "all_three_relations_correct": 0.0},
        "framing": "every table is hypothetical with identical warning, rule text, units and layout; only the numbers move",
        "validation": "all deterministic invariants passed"}))
    print(f"[r2_evidence] {len(prompts)} main prompts x {n_sys} systems = {len(prompts) * n_sys} outputs; "
          f"{len(diag)} diagnostics x {n_sys} = {len(diag) * n_sys} -> {out}")


if __name__ == "__main__":
    main()
