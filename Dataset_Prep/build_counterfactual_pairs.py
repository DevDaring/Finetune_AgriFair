"""Identity counterfactuals, two deterministic kinds.

1. MCQ swap (AgriFacts): whole-word, case-insensitive swap of group1 <-> group2 in the
   question, the choices, and the rationale. For diff rows this flips the correct letter
   a <-> b; equal rows stay c. The swap drives identity-swap-flip detection in Stage A and
   the AgriFacts side of the linear identity probe. Every swap is verified: the two group
   strings must both change places and the rest of the question must be byte-identical
   apart from those spans, and a row that fails the check is logged and dropped rather
   than silently used.
2. Curated free-text pairs (AgriAdvice): normalized passthrough of version_A / version_B,
   which drive the free-text identity probe and the advice-drift metric. They are never
   trained on. Each pair is checked to differ only in the persona span, using whole-word
   replacement so that a persona token appearing inside another word (the "man" inside
   "mango") does not create a false mismatch.

Run:  python Dataset_Prep/build_counterfactual_pairs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re
from typing import Dict, Optional, Tuple

from GPU_Run.common import dataset_io as dio
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import append_jsonl, get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import AGRIADVICE_PAIRS, COUNTERFACTUAL_PAIRS, RESULTS_DIR, TEMPLATED_ALL

logger = get_logger("build_counterfactual_pairs")

INTEGRITY_COLUMNS = [
    "check_name", "items_checked", "items_passed", "items_failed", "is_gating", "check_passed",
]


def swap_words(text: str, a: str, b: str) -> str:
    """Whole-word, case-insensitive swap of a <-> b using a placeholder."""
    placeholder = "\x00SWAP\x00"
    out = re.sub(rf"\b{re.escape(a)}\b", placeholder, text, flags=re.IGNORECASE)
    out = re.sub(rf"\b{re.escape(b)}\b", a, out, flags=re.IGNORECASE)
    return out.replace(placeholder, b)


def swap_record(rec: Dict) -> Dict:
    g1, g2 = rec["group1"], rec["group2"]
    swapped = dict(rec)
    swapped["id"] = rec["id"] + "::swap"
    swapped["question"] = swap_words(rec["question"], g1, g2)
    swapped["rationale"] = swap_words(rec["rationale"], g1, g2)
    swapped["group1"], swapped["group2"] = g2, g1
    swapped["choice_a"], swapped["choice_b"] = rec["choice_b"], rec["choice_a"]
    swapped["on_disk_choices"] = [swap_words(c, g1, g2) for c in rec.get("on_disk_choices", [])]
    if rec["condition"] == "diff":
        swapped["correct_answer"] = {"a": "b", "b": "a"}[rec["correct_answer"]]
    swapped["counterfactual_of"] = rec["id"]
    return swapped


def swap_is_clean(original: Dict, swapped: Dict) -> Tuple[bool, str]:
    """The swap must exchange exactly the two group spans and change nothing else."""
    g1, g2 = original["group1"], original["group2"]
    if swapped["group1"] != g2 or swapped["group2"] != g1:
        return False, "group_fields_not_exchanged"
    mask = lambda t: re.sub(rf"\b{re.escape(g2)}\b", "<G>", re.sub(rf"\b{re.escape(g1)}\b", "<G>", t, flags=re.IGNORECASE), flags=re.IGNORECASE)
    if mask(original["question"]) != mask(swapped["question"]):
        return False, "question_changed_outside_group_spans"
    if sorted(mask(c) for c in original.get("on_disk_choices", [])) != \
            sorted(mask(c) for c in swapped.get("on_disk_choices", [])):
        return False, "choices_changed_outside_group_spans"
    if original["condition"] == "diff" and swapped["correct_answer"] == original["correct_answer"]:
        return False, "diff_letter_did_not_flip"
    if original["condition"] == "equal" and swapped["correct_answer"] != "c":
        return False, "equal_letter_moved"
    return True, ""


def _mask_span(text: str, span: str) -> str:
    """Replace `span` where it is not glued to surrounding word characters.

    Plain word boundaries fail on personas that end in a full stop, and a bare
    string replacement would rewrite the "man" inside "mango", so the boundary is
    asserted with lookarounds on alphanumerics instead."""
    return re.sub(rf"(?<![A-Za-z0-9]){re.escape(span)}(?![A-Za-z0-9])", "<ID>", text)


def advice_pair_is_clean(pair: Dict) -> Tuple[bool, str]:
    """version_A and version_B must differ only in the persona span."""
    pa, pb = pair["persona_a"], pair["persona_b"]
    a = _mask_span(pair["prompt_a"], pa)
    b = _mask_span(pair["prompt_b"], pb)
    if a != b:
        return False, "prompts_differ_outside_persona_span"
    if pair["base_query"] not in pair["prompt_a"] or pair["base_query"] not in pair["prompt_b"]:
        return False, "base_query_not_carried_verbatim"
    return True, ""


def main():
    templated = read_jsonl(TEMPLATED_ALL)
    if not templated:
        raise SystemExit("Run build_template_instances.py first (templated_all_instances.jsonl missing).")

    if COUNTERFACTUAL_PAIRS.exists():
        COUNTERFACTUAL_PAIRS.unlink()
    kept = 0
    rejected = {}
    for rec in templated:
        sw = swap_record(rec)
        ok, reason = swap_is_clean(rec, sw)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        append_jsonl(COUNTERFACTUAL_PAIRS, {"original": rec, "swapped": sw})
        kept += 1
    if rejected:
        logger.warning("Dropped %d AgriFacts swaps that were not clean: %s", sum(rejected.values()), rejected)
    logger.info("Wrote %d verified AgriFacts MCQ counterfactual pairs.", kept)

    if AGRIADVICE_PAIRS.exists():
        AGRIADVICE_PAIRS.unlink()
    advice = dio.load_agriadvice_normalized()
    advice_kept = 0
    advice_rejected = {}
    for a in advice:
        ok, reason = advice_pair_is_clean(a)
        if not ok:
            advice_rejected[reason] = advice_rejected.get(reason, 0) + 1
            continue
        append_jsonl(AGRIADVICE_PAIRS, a)
        advice_kept += 1
    if advice_rejected:
        logger.warning("Dropped %d AgriAdvice pairs that were not clean: %s",
                       sum(advice_rejected.values()), advice_rejected)
    logger.info("Wrote %d verified AgriAdvice free-text pairs.", advice_kept)

    write_csv(RESULTS_DIR / "counterfactual_integrity_report.csv", [
        {"check_name": "agrifacts_swap_exchanges_only_the_two_group_spans",
         "items_checked": len(templated), "items_passed": kept,
         "items_failed": len(templated) - kept, "is_gating": True,
         "check_passed": kept == len(templated)},
        {"check_name": "agriadvice_pair_differs_only_in_the_persona_span",
         "items_checked": len(advice), "items_passed": advice_kept,
         "items_failed": len(advice) - advice_kept, "is_gating": True,
         "check_passed": advice_kept == len(advice)},
    ], INTEGRITY_COLUMNS)

    log_run_metadata("build_counterfactual_pairs", {
        "agrifacts_swap_pairs": kept, "agrifacts_swaps_rejected": rejected,
        "agriadvice_pairs": advice_kept, "agriadvice_pairs_rejected": advice_rejected,
    })


if __name__ == "__main__":
    main()
