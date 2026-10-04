"""Build the version-2 generation sets: E1, E3 and E4.

    python -m Submission1_DKE_Repair.build_prompts_v2

E1  the repaired wording study: 34 corrected comparisons x 2 wordings = 68 prompts.
E3  a matched CLEAN control for the diagnostics: the same 12 bundles, same values, same wrapper
    and same insufficient-evidence option, with the table left unaltered. Without this, a
    perturbation cannot be said to have caused anything, because the three existing variants have
    no unchanged counterpart under the same wrapper. 12 prompts.
E4  neutral-entity transfer: 12 bundles chosen by a declared rule (the first twelve by sorted
    bundle id), with agricultural group names replaced by neutral labels and every value and
    structure preserved. 3 relations x 2 wordings x 12 = 72 prompts.

Each prompt carries its own scoring fields, in the shape run_inference already consumes.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"
PHASE2 = "results_submission1_phase2"
ANSWER_SUFFIX = ('\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c>"}')
ANSWER_SUFFIX_4 = ('\n\nReply with one JSON object only: {"answer_choice_letter": "<a|b|c|d>"}')
NEUTRAL = {"group": ("Group P", "Group Q"), "size": ("Class P", "Class Q")}


# ------------------------------------------------------------------ E1
def build_e1(cfg: Dict) -> List[Dict]:
    panel = [json.loads(l) for l in
             (C.CODES_ROOT / OUT_DIR / "r1_corrected_panel.jsonl").open(encoding="utf-8")]
    out = []
    for it in panel:
        for wording in ("wording_a", "wording_b"):
            opts = "\n".join(f"({d}) {c}" for d, c in zip("abc", it["choices"]))
            out.append({"prompt_id": f"v2r1-{it['fresh_id']}-{wording}", "study": "r1_corrected",
                        "comparison_id": it["fresh_id"], "source_cell": it["source_cell"],
                        "axis": it["axis"], "state": it["geography"],
                        "parent_table": it["parent_table"], "wording": wording,
                        "condition": it["condition"], "choices": it["choices"],
                        "gold_choice_text": it["gold_choice_text"],
                        "social_group": it["social_group"], "size_class": it["size_class"],
                        "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]),
                        "prompt": f"{it[wording]}\n{opts}{ANSWER_SUFFIX}"})
    return out


# ------------------------------------------------------------------ E3
def build_e3(cfg: Dict) -> List[Dict]:
    """The unchanged-table counterpart of each diagnostic bundle."""
    diags = [json.loads(l) for l in
             (C.CODES_ROOT / PHASE2 / "evidence" / "r2_diagnostic_prompts.jsonl").open(encoding="utf-8")]
    by_bundle = {}
    for d in diags:
        by_bundle.setdefault(d["bundle_id"], {})[d["variant"]] = d

    out = []
    for bundle_id, variants in sorted(by_bundle.items()):
        src = variants.get("row_order_reversed") or next(iter(variants.values()))
        g1, g2 = src["group1"], src["group2"]
        v1, v2 = src["value1"], src["value2"]
        # rebuild the reversed variant's prompt with the rows in canonical order and nothing else
        # changed: same header, denominator line, rule, question and options.
        text = src["prompt"]
        rows_re = re.compile(r"\|\s*" + re.escape(g2) + r"\s*\|\s*" + re.escape(str(v2)) +
                             r"\s*\|\n\|\s*" + re.escape(g1) + r"\s*\|\s*" + re.escape(str(v1)) + r"\s*\|")
        clean = rows_re.sub(f"| {g1} | {v1} |\n| {g2} | {v2} |", text)
        if clean == text:                      # the reversal was not in the expected form
            continue
        out.append({**{k: src[k] for k in
                       ("bundle_id", "axis", "state", "parent_table", "source_cell",
                        "group1", "group2", "value1", "value2", "baseline_relation",
                        "choices", "gold_choice_text", "allows_insufficient")},
                    "prompt_id": f"v2diag-{bundle_id}-clean", "study": "r2_diagnostic_clean",
                    "variant": "clean_control",
                    "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]),
                    "prompt": clean})
    return out


# ------------------------------------------------------------------ E4
def _neutralise(text: str, mapping: Dict[str, str]) -> str:
    for old, new in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        text = re.sub(re.escape(old), new, text)
    return text


def build_e4(cfg: Dict) -> List[Dict]:
    """Twelve R2 bundles with neutral entity labels; values and structure untouched."""
    main = [json.loads(l) for l in
            (C.CODES_ROOT / PHASE2 / "evidence" / "r2_main_prompts.jsonl").open(encoding="utf-8")]
    chosen = sorted({r["bundle_id"] for r in main})[:12]      # declared rule: first twelve by id
    out = []
    for r in main:
        if r["bundle_id"] not in chosen:
            continue
        g1, g2 = r["group1"], r["group2"]
        kind = "size" if ("holding" in g1.lower() or "operated area" in g1.lower()) else "group"
        n1, n2 = NEUTRAL[kind]
        mapping = {g1: n1, g2: n2}
        prompt = _neutralise(r["prompt"], mapping)
        # the state name and the denominator descriptor also carry domain signal
        if r.get("state"):
            prompt = prompt.replace(r["state"], "Region R")
        prompt = re.sub(r"All-social-groups total", "Population total", prompt)
        prompt = re.sub(r"All Classes", "All Segments", prompt)
        prompt = re.sub(r"percentage share of the stratum", "percentage share of the population",
                        prompt)
        prompt = re.sub(r"\boperated area\b", "measured quantity", prompt)
        prompt = re.sub(r"\bholdings\b", "units", prompt)
        prompt = re.sub(r"not census observations", "not real observations", prompt)
        choices = [_neutralise(c, mapping) for c in r["choices"]]
        out.append({**{k: r[k] for k in ("bundle_id", "axis", "parent_table", "relation",
                                         "wording", "value1", "value2")
                       if k in r},
                    "prompt_id": f"v2neutral-{r['prompt_id']}", "study": "r2_neutral",
                    "state": "Region R", "source_cell": r.get("source_cell", ""),
                    "group1": n1, "group2": n2, "choices": choices,
                    "gold_choice_text": _neutralise(r["gold_choice_text"], mapping),
                    "original_group1": g1, "original_group2": g2,
                    "max_new_tokens": int(cfg["budget"]["answer_max_new_tokens"]),
                    "prompt": prompt})
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    cfg = C.load_config()
    out = C.CODES_ROOT / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    e1, e3, e4 = build_e1(cfg), build_e3(cfg), build_e4(cfg)
    C.write_jsonl(out / "prompts_e1_r1_corrected.jsonl", e1)
    C.write_jsonl(out / "prompts_e3_diagnostic_clean.jsonl", e3)
    C.write_jsonl(out / "prompts_e4_neutral.jsonl", e4)
    n_sys = 4
    summary = {"E1_prompts": len(e1), "E3_prompts": len(e3), "E4_prompts": len(e4),
               "total_prompts": len(e1) + len(e3) + len(e4),
               "responses_at_4_systems": (len(e1) + len(e3) + len(e4)) * n_sys,
               "expected": {"E1": 68, "E3": 12, "E4": 72}}
    C.write_json(out / "prompt_build_manifest.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
