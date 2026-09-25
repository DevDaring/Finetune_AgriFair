"""M1 step 4 and E2 invariant 7: check rendered questions against their source specification.

These checks read the source record and the rendered question. They never read a model output,
because the defect they exist to catch -- a question that describes a different population from
the row it was scored against -- is invisible in the outputs. Every one of them would have
failed on the version-1 panel.
"""
from __future__ import annotations

import re
from typing import Dict, List

from Submission1_DKE_Repair.source_schema import (SourceSpec, SOCIAL_GROUP, SIZE_CLASS, METRIC,
                                                  RULE, EQUAL_CHOICE)


def _fail(spec: SourceSpec, rule: str, detail: str) -> Dict:
    return {"fresh_id": spec.fresh_id, "rule": rule, "detail": detail,
            "source_cell": spec.source_cell}


_WS = re.compile(r"\s+")


def _has_word(haystack: str, needle: str) -> bool:
    """Whole-word containment. Plain substring matching reports the size class 'large' as
    present inside 'a larger share', which silently hid six substituted-size-class cases."""
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def _norm(text: str) -> str:
    """Collapse runs of whitespace. A reflowed or double-spaced sentence is a formatting
    variant, not a source mismatch, and must not raise a flag."""
    return _WS.sub(" ", text).strip().lower()


def check_one(spec: SourceSpec, wordings: Dict[str, str]) -> List[Dict]:
    out: List[Dict] = []
    texts = list(wordings.values())

    for name, text in wordings.items():
        low = _norm(text)

        # P1 the social-group restriction appears whenever the source has one, and never otherwise
        if spec.social_group:
            want = _norm(SOCIAL_GROUP.get(spec.social_group, spec.social_group))
            if not _has_word(low, want):
                out.append(_fail(spec, "P1_social_group_dropped",
                                 f"{name}: source restricts to {spec.social_group!r}; question omits it"))
            if "all farmers" in low or "all operational holdings" in low:
                out.append(_fail(spec, "P1_false_all_farmers",
                                 f"{name}: restricted row rendered as an unrestricted population"))
        else:
            # On the social-group axis the compared ENTITIES are themselves social groups, so the
            # words legitimately appear as entity names. Only a mention outside the entity names
            # would be an invented population restriction, so blank the entities before looking.
            outside = low.replace(_norm(spec.entity1), " ").replace(_norm(spec.entity2), " ")
            for token in ("scheduled caste", "scheduled tribe"):
                if token in outside:
                    out.append(_fail(spec, "P1_invented_social_group",
                                     f"{name}: question claims {token!r}; source has no such restriction"))

        # P1b a NEGATED population keeps the expected substring while inverting its meaning
        # ("holders who are not Scheduled Tribe holders"). A substring test cannot see that,
        # so negation immediately before the population phrase is checked explicitly.
        if spec.social_group:
            want = _norm(SOCIAL_GROUP.get(spec.social_group, spec.social_group))
            if re.search(r"\b(?:not|other than|excluding|apart from)\s+(?:\w+\s+){0,2}"
                         + re.escape(want), low):
                out.append(_fail(spec, "P1_population_negated",
                                 f"{name}: the population phrase appears under a negation"))

        # P2 the size-class restriction, same rule
        if spec.size_class:
            want = _norm(SIZE_CLASS.get(spec.size_class, spec.size_class))
            if not _has_word(low, want):
                out.append(_fail(spec, "P2_size_class_dropped",
                                 f"{name}: source restricts to {spec.size_class!r}; question omits it"))

        # P3 geography
        if spec.geography == "all-India":
            if "all-india" not in low:
                out.append(_fail(spec, "P3_geography_missing", f"{name}: all-India not stated"))
        elif _norm(spec.geography) not in low:
            out.append(_fail(spec, "P3_geography_missing",
                             f"{name}: state {spec.geography!r} not stated"))

        # P4 metric and denominator are both explicit
        if _norm(METRIC[spec.metric]) not in low:
            out.append(_fail(spec, "P4_metric_missing", f"{name}: metric not stated"))
        if _norm(spec.denominator()) not in low:
            out.append(_fail(spec, "P5_denominator_missing", f"{name}: denominator not stated"))

        # P6 both compared entities appear, in the recorded order
        i1, i2 = low.find(_norm(spec.entity1)), low.find(_norm(spec.entity2))
        if i1 < 0 or i2 < 0:
            out.append(_fail(spec, "P6_entity_missing", f"{name}: a compared entity is absent"))
        elif i1 > i2:
            out.append(_fail(spec, "P6_entity_order", f"{name}: entities appear in reversed order"))

        # P7 the operational rule is stated, identically, in every wording
        if _norm(RULE) not in low:
            out.append(_fail(spec, "P7_rule_missing", f"{name}: the operational rule is not stated"))
        if "equality band" in low:
            out.append(_fail(spec, "P7_undefined_term",
                             f"{name}: refers to an 'equality band' the prompt never defines"))

    # P8 the two wordings must differ from each other
    if len(set(texts)) != len(texts):
        out.append(_fail(spec, "P8_wordings_identical", "the two wordings are byte-identical"))
    return out


def check_panel(specs: List[SourceSpec], rendered: Dict[str, Dict[str, str]]) -> Dict:
    """Panel-level checks, plus every per-item check."""
    failures: List[Dict] = []
    for spec in specs:
        failures += check_one(spec, rendered[spec.fresh_id])

    # P9 two different source rows must not share a question. This is the gender defect: the two
    # retained gender comparisons (ST/Small and ST/Medium) had byte-identical wordings.
    for key in ("wording_a", "wording_b"):
        seen: Dict[str, str] = {}
        for spec in specs:
            text = rendered[spec.fresh_id][key]
            if text in seen and seen[text] != spec.source_cell:
                failures.append(_fail(spec, "P9_question_collides",
                                      f"{key} is identical to that of {seen[text]}"))
            seen[text] = spec.source_cell

    # P10 the recorded gold answer must follow from the shares under the operational rule
    for spec in specs:
        gap = abs(spec.share1_pct - spec.share2_pct)
        if spec.condition == "equal" and gap >= 5.0:
            failures.append(_fail(spec, "P10_rule_violation",
                                  f"labelled equal but the gap is {gap:.2f} pp"))
        if spec.condition == "diff" and gap < 10.0:
            failures.append(_fail(spec, "P10_rule_violation",
                                  f"labelled diff but the gap is {gap:.2f} pp"))
        if spec.condition == "diff":
            leader = spec.entity1 if spec.share1_pct > spec.share2_pct else spec.entity2
            if spec.gold_entity != leader:
                failures.append(_fail(spec, "P10_gold_mismatch",
                                      f"gold {spec.gold_entity!r} is not the leading entity"))
        elif spec.gold_entity != EQUAL_CHOICE:
            failures.append(_fail(spec, "P10_gold_mismatch",
                                  f"equal row has gold {spec.gold_entity!r}"))

    by_rule: Dict[str, int] = {}
    for f in failures:
        by_rule[f["rule"]] = by_rule.get(f["rule"], 0) + 1
    return {"n_specs": len(specs), "n_failures": len(failures),
            "failures_by_rule": by_rule, "failures": failures,
            "passed": not failures}
