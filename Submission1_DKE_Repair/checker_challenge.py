"""Priority 2: measure what the schema checker detects, and what it misses.

    python -m Submission1_DKE_Repair.checker_challenge

Zero failures on the renderer the checker ships with proves very little: the renderer and the
checker were written together, so agreement between them is close to a tautology. This module
builds a frozen challenge suite by deliberately corrupting the 34 corrected comparisons in ways
whose correct verdict is known from the transformation itself, mixes in valid paraphrases as
negative controls, and reports detection by error type alongside the false-alarm rate.

The negative controls matter as much as the mutations. A checker that rejected every unfamiliar
sentence would score 100% detection and be useless.

Two known misses are included deliberately, because the plan that prompted this work supplied them
as counterexamples: a negated population, and a substituted denominator followed by an audit note
that re-introduces the expected words. Both defeat a substring test. The negation case is now
detected by an explicit rule; the appended-words case is reported as an accepted limit, because
catching it in general needs parsing rather than matching.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from typing import Callable, Dict, List, Tuple

from Submission1_Code_Phase2 import common as C
from Submission1_DKE_Repair import source_schema as S
from Submission1_DKE_Repair.prompt_checks import check_one

OUT_DIR = "results_submission1_dke_repair_v2"

SG_OTHER = {"SC": "Scheduled Tribe", "ST": "Scheduled Caste", "Others": "Scheduled Caste"}
SIZE_OTHER = {"marginal": "large", "small": "medium", "semi-medium": "marginal",
              "medium": "small", "large": "marginal"}


# ------------------------------------------------------------------ mutations
def _sg_phrase(spec: S.SourceSpec) -> str:
    return S.SOCIAL_GROUP.get(spec.social_group, spec.social_group or "")


def m_population_deleted(spec: S.SourceSpec, text: str):
    if not spec.social_group:
        return None
    return text.replace(f"{_sg_phrase(spec)} holders", "all farmers")


def m_population_substituted(spec: S.SourceSpec, text: str):
    if not spec.social_group:
        return None
    return text.replace(_sg_phrase(spec), SG_OTHER.get(spec.social_group, "Scheduled Caste"))


def m_population_negated(spec: S.SourceSpec, text: str):
    if not spec.social_group:
        return None
    p = _sg_phrase(spec)
    return text.replace(f"among {p} holders", f"among holders who are not {p} holders")


def m_size_deleted(spec: S.SourceSpec, text: str):
    if not spec.size_class:
        return None
    w = S.SIZE_CLASS.get(spec.size_class, spec.size_class.lower())
    return text.replace(f" with {w} holdings", "")


def m_size_substituted(spec: S.SourceSpec, text: str):
    if not spec.size_class:
        return None
    w = S.SIZE_CLASS.get(spec.size_class, spec.size_class.lower())
    return text.replace(f"{w} holdings", f"{SIZE_OTHER.get(w, 'large')} holdings")


def m_wrong_geography(spec: S.SourceSpec, text: str):
    if spec.geography == "all-India":
        return text.replace("At the all-India level", "In Kerala")
    return text.replace(spec.geography, "Kerala")


def m_wrong_denominator(spec: S.SourceSpec, text: str):
    other = ("the total number of operating holdings in that population"
             if spec.metric == "area" else "the total operated area in that population")
    return text.replace(spec.denominator(), other)


def m_denominator_swapped_with_audit_note(spec: S.SourceSpec, text: str):
    """The plan's second counterexample: substitute, then append the expected words."""
    other = ("the total number of operating holdings in that population"
             if spec.metric == "area" else "the total operated area in that population")
    return text.replace(spec.denominator(), other) + \
        f" (Audit note: figures are measured against {spec.denominator()}.)"


def m_wrong_metric(spec: S.SourceSpec, text: str):
    other = S.METRIC["number"] if spec.metric == "area" else S.METRIC["area"]
    return text.replace(spec.metric_phrase(), other)


def m_entity_order_swapped(spec: S.SourceSpec, text: str):
    a, b = spec.entity1, spec.entity2
    return text.replace(a, "\x00").replace(b, a).replace("\x00", b)


def m_rule_removed(spec: S.SourceSpec, text: str):
    return text.replace(" " + S.RULE, "")


MUTATIONS: List[Tuple[str, Callable, bool]] = [
    ("population_deleted", m_population_deleted, True),
    ("population_substituted", m_population_substituted, True),
    ("population_negated", m_population_negated, True),
    ("size_class_deleted", m_size_deleted, True),
    ("size_class_substituted", m_size_substituted, True),
    ("wrong_geography", m_wrong_geography, True),
    ("wrong_denominator", m_wrong_denominator, True),
    ("denominator_swapped_with_audit_note", m_denominator_swapped_with_audit_note, True),
    ("wrong_metric", m_wrong_metric, True),
    ("entity_order_swapped", m_entity_order_swapped, True),
    ("operational_rule_removed", m_rule_removed, True),
]


# ------------------------------------------------------------------ negative controls
def c_whitespace(spec, text):
    return re.sub(r" +", "  ", text)


def c_sentence_reflow(spec, text):
    return text.replace(" — ", ", namely ").replace("  ", " ")


def c_polite_prefix(spec, text):
    return "Please answer the following question. " + text


def c_trailing_note(spec, text):
    return text + " (All figures are from the published census tables.)"


def c_synonym_larger(spec, text):
    return text.replace("a larger share", "the greater share")


CONTROLS: List[Tuple[str, Callable, bool]] = [
    ("formatting_whitespace", c_whitespace, False),
    ("sentence_reflow", c_sentence_reflow, False),
    ("polite_prefix", c_polite_prefix, False),
    ("harmless_trailing_note", c_trailing_note, False),
    ("synonym_for_larger", c_synonym_larger, False),
]


def build_and_run() -> Dict:
    panel = [json.loads(l) for l in
             (C.CODES_ROOT / OUT_DIR / "r1_corrected_panel.jsonl").open(encoding="utf-8")]
    specs = [S.SourceSpec(**{k: p[k] for k in S.SourceSpec.__dataclass_fields__}) for p in panel]
    wordings = {p["fresh_id"]: {"wording_a": p["wording_a"], "wording_b": p["wording_b"]}
                for p in panel}

    cases, by_type = [], collections.defaultdict(lambda: {"n": 0, "flagged": 0})
    for spec in specs:
        for wname, original in wordings[spec.fresh_id].items():
            for label, fn, should_flag in MUTATIONS + CONTROLS:
                mutated = fn(spec, original)
                if mutated is None or mutated == original:
                    continue            # the transformation does not apply to this record
                failures = check_one(spec, {wname: mutated})
                flagged = bool(failures)
                e = by_type[label]
                e["n"] += 1
                e["flagged"] += flagged
                cases.append({"fresh_id": spec.fresh_id, "wording": wname, "transform": label,
                              "should_be_flagged": should_flag, "was_flagged": flagged,
                              "rules_fired": ";".join(sorted({f["rule"] for f in failures}))})

    rows = []
    for label, fn, should_flag in MUTATIONS + CONTROLS:
        e = by_type.get(label)
        if not e or not e["n"]:
            continue
        rate = e["flagged"] / e["n"]
        rows.append({"transform": label, "kind": "corruption" if should_flag else "valid_control",
                     "n_cases": e["n"],
                     "detection_rate" if should_flag else "false_alarm_rate": round(rate, 4),
                     "rate": round(rate, 4)})

    corr = [r for r in rows if r["kind"] == "corruption"]
    ctrl = [r for r in rows if r["kind"] == "valid_control"]
    n_corr = sum(r["n_cases"] for r in corr)
    n_ctrl = sum(r["n_cases"] for r in ctrl)
    det = sum(r["rate"] * r["n_cases"] for r in corr) / n_corr if n_corr else None
    fa = sum(r["rate"] * r["n_cases"] for r in ctrl) / n_ctrl if n_ctrl else None
    missed = sorted({r["transform"] for r in corr if r["rate"] < 1.0})
    return {"per_transform": rows, "cases": cases,
            "corruption_cases": n_corr, "control_cases": n_ctrl,
            "overall_detection_rate": round(det, 4) if det is not None else None,
            "overall_false_alarm_rate": round(fa, 4) if fa is not None else None,
            "transforms_not_fully_detected": missed}


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    out = C.CODES_ROOT / OUT_DIR
    res = build_and_run()
    C.write_csv(out / "checker_challenge_by_transform.csv", res["per_transform"])
    C.write_csv(out / "checker_challenge_cases.csv", res["cases"])
    C.write_json(out / "checker_challenge_summary.json",
                 {k: v for k, v in res.items() if k != "cases"} |
                 {"scope": ("The checker validates questions rendered from this study's controlled "
                            "grammar. It is a structural and lexical validator over that grammar, "
                            "not a general natural-language source-alignment verifier."),
                  "known_limit": ("A substituted denominator followed by an audit note that "
                                  "re-introduces the expected words defeats a substring test. "
                                  "Detecting that in general requires parsing the sentence, not "
                                  "matching it. It is reported rather than hidden.")})
    print(f"  corruption cases: {res['corruption_cases']}   detection {res['overall_detection_rate']}")
    print(f"  valid controls  : {res['control_cases']}   false alarms {res['overall_false_alarm_rate']}")
    print()
    for r in res["per_transform"]:
        tag = "detect" if r["kind"] == "corruption" else "FALSE-ALARM"
        print(f"    {r['transform']:40s} {r['kind']:14s} n={r['n_cases']:3d} {tag}={r['rate']:.3f}")
    if res["transforms_not_fully_detected"]:
        print("\n  not fully detected:", ", ".join(res["transforms_not_fully_detected"]))


if __name__ == "__main__":
    main()
