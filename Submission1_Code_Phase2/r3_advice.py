"""R3: a small advice study that separates quality from verbosity.

    python -m Submission1_Code_Phase2.r3_advice --template   # blank reference-packet workbook
    python -m Submission1_Code_Phase2.r3_advice --build      # prompts, once the packets are filled
    python -m Submission1_Code_Phase2.r3_advice --rater-forms
    python -m Submission1_Code_Phase2.r3_advice --import R1.csv R2.csv

The original advice study cannot separate "shorter" from "worse": the answers came from one
uncapped condition, and the truncation rate is a heuristic over cached text. Here every system
answers the same case twice, once asked for ~80-100 words and once for ~160-180, under the same
generous 512-token ceiling, and actual length is recorded. Quality is scored against a reference
checklist written before any model output exists.

24 cases: 16 where identity is irrelevant (8 gender, 8 social group) and 8 positive controls
where a relevant agronomic constraint changes while identity is held fixed. The two are scored
with opposite expectations and are never pooled into one invariance rate.

Humans supply the reference packets and the ratings; this module only builds and scores.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

from Submission1_Code_Phase2 import common as C

PACKET_COLS = ["case_id", "case_type", "toggle_axis", "base_question", "fixed_context",
               "variant_A_description", "variant_B_description", "reference_url", "reference_publisher",
               "reference_date", "reference_applicability", "essential_points_semicolon_separated",
               "permitted_alternatives_semicolon_separated", "known_errors_semicolon_separated",
               "expected_relation_between_A_and_B", "prepared_by", "checked_by"]

CASE_TYPES = {"identity_irrelevant": "substantive advice should not change between A and B",
              "context_control": "a relevant constraint changes, so substantive advice is expected to adapt"}


def template_workbook(cfg: Dict, out: Path) -> Dict:
    r3 = cfg["r3"]; rows = []
    for i in range(r3["identity_pairs"]):
        axis = "gender" if i < r3["identity_pairs"] // 2 else "social_group"
        rows.append({c: "" for c in PACKET_COLS} | {"case_id": f"adv-{i:02d}", "case_type": "identity_irrelevant",
                                                     "toggle_axis": axis,
                                                     "expected_relation_between_A_and_B": "same substantive actions"})
    for j in range(r3["context_pairs"]):
        i = r3["identity_pairs"] + j
        rows.append({c: "" for c in PACKET_COLS} | {"case_id": f"adv-{i:02d}", "case_type": "context_control",
                                                     "toggle_axis": "agronomic_constraint",
                                                     "expected_relation_between_A_and_B": "advice should adapt to the changed constraint"})
    C.write_csv(out / "reference_packet_template.csv", rows)
    (out / "REFERENCE_PACKET_INSTRUCTIONS.txt").write_text(
        "AgriFair R3: preparing the reference packets\n"
        "============================================\n\n"
        f"{len(rows)} cases: {cfg['r3']['identity_pairs']} where the asker's identity is irrelevant to the\n"
        f"agronomy, and {cfg['r3']['context_pairs']} positive controls where a real agronomic constraint changes.\n\n"
        "Fill one row per case, BEFORE any model is run:\n\n"
        "  base_question        the farmer's question, with crop, stage and symptom explicit.\n"
        "  fixed_context        everything held constant across the two variants (soil, season, area,\n"
        "                       irrigation). Write it out; do not leave it implied.\n"
        "  variant_A/B          how the two versions differ. For identity cases this is only the\n"
        "                       description of who is asking. For context cases it is the constraint\n"
        "                       that changes (for example: canal water available vs bore-well only).\n"
        "  reference_*          a public extension, university or government source that answers this\n"
        "                       question, with its date and the region it applies to.\n"
        "  essential_points     the claims a good answer must contain, separated by semicolons. These\n"
        "                       are what gets scored, so write them as checkable statements.\n"
        "  permitted_alternatives  equally acceptable actions that should not be marked wrong.\n"
        "  known_errors         common wrong or unsafe advice for this case.\n\n"
        "Choose stable, low-risk agronomic questions. Avoid pesticide dosages and anything whose\n"
        "correct answer depends on current regulations or entitlement rules. A dataset QA answer is\n"
        "not automatically authoritative; cite the source you actually checked.\n\n"
        "Two readers review case relevance and the scoring anchors before any output is generated;\n"
        "at least one should be a domain specialist.\n", encoding="utf-8")
    return {"cases": len(rows), "identity": cfg["r3"]["identity_pairs"], "context": cfg["r3"]["context_pairs"]}


def build_prompts(cfg: Dict, out: Path) -> Dict:
    p = out / "reference_packets.csv"
    if not p.exists():
        raise SystemExit(f"fill and save the reference packets first: {out / 'reference_packet_template.csv'} -> {p}")
    cases = [r for r in csv.DictReader(p.open(encoding="utf-8")) if r.get("base_question")]
    problems = []
    for c in cases:
        if c["case_type"] not in CASE_TYPES:
            problems.append(f"{c['case_id']}: unknown case_type")
        if not c.get("essential_points_semicolon_separated"):
            problems.append(f"{c['case_id']}: no essential points to score against")
        if not c.get("reference_url"):
            problems.append(f"{c['case_id']}: no reference")
    if problems:
        raise SystemExit("reference packets incomplete:\n" + "\n".join(problems[:20]))
    prompts = []
    for c in cases:
        for side in ("A", "B"):
            body = f"{c['fixed_context'].strip()} {c[f'variant_{side}_description'].strip()} {c['base_question'].strip()}".strip()
            for level, spec in cfg["r3"]["verbosity"].items():
                prompts.append({"prompt_id": f"r3-{c['case_id']}-{side}-{level}", "study": "r3_advice",
                                "case_id": c["case_id"], "case_type": c["case_type"], "toggle_axis": c["toggle_axis"],
                                "side": side, "verbosity": level, "requested_words": spec["words"],
                                "choices": [], "gold_choice_text": "",
                                "max_new_tokens": int(cfg["r3"]["max_new_tokens"]),
                                "prompt": f"{spec['instruction']}\n\n{body}"})
    C.write_jsonl(out / "r3_prompts.jsonl", prompts)
    n_sys = len(C.systems(cfg))
    C.write_json(out / "manifest.json", C.manifest(cfg, "r3_advice", {
        "cases": len(cases), "identity_cases": sum(1 for c in cases if c["case_type"] == "identity_irrelevant"),
        "context_cases": sum(1 for c in cases if c["case_type"] == "context_control"),
        "prompts": len(prompts), "systems": n_sys, "generations": len(prompts) * n_sys,
        "pair_assessments_per_rater": len(prompts) * n_sys // 2,
        "ceiling_tokens": cfg["r3"]["max_new_tokens"],
        "note": "both verbosity conditions share one generous ceiling; actual length is recorded per answer"}))
    return {"cases": len(cases), "prompts": len(prompts), "generations": len(prompts) * n_sys}


def rater_forms(cfg: Dict, out: Path) -> Dict:
    preds = C.CODES_ROOT / cfg["output_directory"] / "predictions" / "main_predictions.jsonl"
    rows = [r for r in C.read_jsonl(preds) if r.get("study") == "r3_advice"] if preds.exists() else []
    if not rows:
        raise SystemExit("no R3 generations found; run inference first")
    packets = {c["case_id"]: c for c in csv.DictReader((out / "reference_packets.csv").open(encoding="utf-8"))}
    # The two sides of a pair are answers to DIFFERENT questions, so the rater has to read both
    # questions to judge whether a change between the answers is justified. Carry the prompt text.
    prompt_text = {(p["case_id"], p["side"], p["verbosity"]): p["prompt"]
                   for p in C.read_jsonl(out / "r3_prompts.jsonl")}
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["case_id"], r["system"], r["verbosity"])][r["side"]] = r
    rng = random.Random(cfg["analysis_seed"])
    forms, key = [], []
    for (case_id, system, verbosity), sides in sorted(by.items()):
        if "A" not in sides or "B" not in sides:
            continue
        flip = rng.random() < 0.5
        left, right = (sides["B"], sides["A"]) if flip else (sides["A"], sides["B"])
        aid = f"r3a-{len(forms):04d}"
        pk = packets.get(case_id, {})
        # case_type IS shown to the rater. It is a known demand characteristic -- it signals which
        # way rule 6 should come out -- but the frozen rubric states "The sheet tells you which type
        # each pair is", and rule 6 is unanswerable without it, because identity and context-control
        # cases carry opposite expectations. The rubric was signed off by both readers, so the
        # instrument is not changed here; the limitation is reported with the R3 results instead.
        forms.append({"assessment_id": aid, "case_type": pk.get("case_type", ""),
                      "question_left": prompt_text.get((case_id, left["side"], verbosity), ""),
                      "question_right": prompt_text.get((case_id, right["side"], verbosity), ""),
                      "answer_left": left.get("raw_output", ""),
                      "answer_right": right.get("raw_output", ""),
                      "essential_points": pk.get("essential_points_semicolon_separated", ""),
                      "permitted_alternatives": pk.get("permitted_alternatives_semicolon_separated", ""),
                      "known_errors": pk.get("known_errors_semicolon_separated", ""),
                      "reference_url": pk.get("reference_url", ""),
                      "left_essentials_covered_count": "", "right_essentials_covered_count": "",
                      "left_unsupported_or_incorrect_claims": "", "right_unsupported_or_incorrect_claims": "",
                      "left_unsafe_yes_no": "", "right_unsafe_yes_no": "",
                      "left_usefulness_0_2": "", "right_usefulness_0_2": "",
                      "pair_change_type": "", "change_is_justified_yes_no_na": "", "comment": ""})
        key.append({"assessment_id": aid, "case_id": case_id, "system": system, "verbosity": verbosity,
                    "case_type": pk.get("case_type", ""), "left_is_A": not flip})
    for rater in ("R1", "R2"):
        C.write_csv(out / f"r3_rating_sheet_{rater}.csv", forms)
    C.write_csv(out / "KEEP_FROM_RATERS_r3_key.csv", key)
    return {"assessments_per_rater": len(forms), "raters": 2,
            "estimated_rater_hours_each": round(len(forms) * 2.5 / 60, 1)}


def analyse(cfg: Dict, out: Path, paths: List[Path]) -> Dict:
    import numpy as np
    from Next_Run import stats as S
    key = {r["assessment_id"]: r for r in csv.DictReader((out / "KEEP_FROM_RATERS_r3_key.csv").open(encoding="utf-8"))}
    packets = {c["case_id"]: c for c in csv.DictReader((out / "reference_packets.csv").open(encoding="utf-8"))}
    rows = []
    for i, p in enumerate(paths):
        rid = f"R{i + 1}"
        for r in csv.DictReader(p.open(encoding="utf-8")):
            k = key.get(r["assessment_id"])
            if not k:
                continue
            ess = [e for e in packets.get(k["case_id"], {}).get("essential_points_semicolon_separated", "").split(";") if e.strip()]
            for side in ("left", "right"):
                is_A = (side == "left") == (str(k["left_is_A"]).lower() == "true")
                cov = r.get(f"{side}_essentials_covered_count", "")
                rows.append({"rater": rid, "assessment_id": r["assessment_id"], "case_id": k["case_id"],
                             "system": k["system"], "verbosity": k["verbosity"], "side": "A" if is_A else "B",
                             "case_type": packets.get(k["case_id"], {}).get("case_type", ""),
                             "essentials_total": len(ess),
                             "essentials_covered": float(cov) if str(cov).strip() else float("nan"),
                             "unsupported_claims": r.get(f"{side}_unsupported_or_incorrect_claims", ""),
                             "unsafe": r.get(f"{side}_unsafe_yes_no", ""),
                             "usefulness": r.get(f"{side}_usefulness_0_2", "")})
            rows[-1]["pair_change_type"] = r.get("pair_change_type", "")
            rows[-1]["change_justified"] = r.get("change_is_justified_yes_no_na", "")
    C.write_csv(out / "r3_ratings_long.csv", rows)
    summ = collections.defaultdict(list)
    for r in rows:
        if r["essentials_total"] and r["essentials_covered"] == r["essentials_covered"]:
            summ[(r["system"], r["verbosity"], r["case_type"])].append(r["essentials_covered"] / r["essentials_total"])
    out_rows = [{"system": s, "verbosity": v, "case_type": t, "n": len(vals),
                 "essential_coverage_mean": round(float(np.mean(vals)), 4)} for (s, v, t), vals in sorted(summ.items())]
    C.write_csv(out / "r3_quality_by_verbosity.csv", out_rows)
    res = {"rows": len(rows), "summary": out_rows,
           "note": ("Identity cases and context controls carry opposite expectations and are reported separately; "
                    "they are never pooled into a single invariance rate. 24 cases is a diagnostic study.")}
    C.write_json(out / "r3_analysis.json", res)
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", action="store_true"); ap.add_argument("--build", action="store_true")
    ap.add_argument("--rater-forms", action="store_true"); ap.add_argument("--import", dest="imp", nargs="+", type=Path)
    a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "advice")
    if a.template or not any((a.build, a.rater_forms, a.imp)):
        print("[r3_advice]", template_workbook(cfg, out), "->", out)
    if a.build:
        print("[r3_advice]", build_prompts(cfg, out))
    if a.rater_forms:
        print("[r3_advice]", rater_forms(cfg, out))
    if a.imp:
        print("[r3_advice]", json.dumps(analyse(cfg, out, a.imp)["summary"], indent=1))


if __name__ == "__main__":
    main()
