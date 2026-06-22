"""Identity counterfactuals, two deterministic kinds (coding_prompt.md Section 5.2).

1. MCQ swap (AgriFacts): deterministic whole-word, case-insensitive swap of group1<->group2
   in the question, choices, and rationale. For neq (diff) rows this flips the correct
   letter a<->b; eq rows stay c. Drives identity-swap-flip detection (Stage A) and the MCQ
   side of the linear probe (Stage D).
2. Curated free-text pairs (AgriAdvice): normalized passthrough of version_A / version_B.
   Drives the free-text identity probe and the advice-drift metric. Never trained on.

Run:  python Dataset_Prep/build_counterfactual_pairs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GPU_Run.common import dataset_io as dio
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import append_jsonl, get_logger, log_run_metadata
from GPU_Run.common.paths import AGRIADVICE_PAIRS, COUNTERFACTUAL_PAIRS, TEMPLATED_ALL

logger = get_logger("build_counterfactual_pairs")


def _swap_words(text: str, a: str, b: str) -> str:
    """Whole-word, case-insensitive swap of a<->b using a placeholder."""
    placeholder = "\x00SWAP\x00"
    out = re.sub(rf"\b{re.escape(a)}\b", placeholder, text, flags=re.IGNORECASE)
    out = re.sub(rf"\b{re.escape(b)}\b", a, out, flags=re.IGNORECASE)
    out = out.replace(placeholder, b)
    return out


def _swap_record(rec: dict) -> dict:
    g1, g2 = rec["group1"], rec["group2"]
    swapped = dict(rec)
    swapped["id"] = rec["id"] + "::swap"
    swapped["question"] = _swap_words(rec["question"], g1, g2)
    swapped["rationale"] = _swap_words(rec["rationale"], g1, g2)
    swapped["group1"], swapped["group2"] = g2, g1
    swapped["choice_a"], swapped["choice_b"] = rec["choice_b"], rec["choice_a"]
    swapped["on_disk_choices"] = [_swap_words(c, g1, g2) for c in rec.get("on_disk_choices", [])]
    # neq: flip a<->b; eq stays c
    if rec["condition"] == "neq":
        swapped["correct_answer"] = {"a": "b", "b": "a"}[rec["correct_answer"]]
    swapped["counterfactual_of"] = rec["id"]
    return swapped


def main():
    templated = read_jsonl(TEMPLATED_ALL)
    if not templated:
        raise SystemExit("Run build_template_instances.py first (templated_all_instances.jsonl missing).")

    if COUNTERFACTUAL_PAIRS.exists():
        COUNTERFACTUAL_PAIRS.unlink()
    n_pairs = 0
    for rec in templated:
        sw = _swap_record(rec)
        append_jsonl(COUNTERFACTUAL_PAIRS, {"original": rec, "swapped": sw})
        n_pairs += 1
    logger.info("Wrote %d AgriFacts MCQ counterfactual pairs.", n_pairs)

    # AgriAdvice passthrough (never trained on)
    if AGRIADVICE_PAIRS.exists():
        AGRIADVICE_PAIRS.unlink()
    advice = dio.load_agriadvice_normalized()
    for a in advice:
        append_jsonl(AGRIADVICE_PAIRS, a)
    logger.info("Wrote %d AgriAdvice free-text pairs.", len(advice))
    log_run_metadata(
        "build_counterfactual_pairs",
        {"mcq_swap_pairs": n_pairs, "agriadvice_pairs": len(advice)},
    )


if __name__ == "__main__":
    main()
