"""Dry run for Dataset_Prep: fast, offline integrity checks before expensive work.

Validates the normalization invariants, the group-disjoint split and the structure-novel
slice, the counterfactual swap and the AgriAdvice persona check, the surface-cue ceiling
audit, data-hygiene deduplication, and the hardcoded-key self-check. Writes a pass/fail
report and exits non-zero on any failure.

Run:  python Dry_Run/dry_run_dataset_prep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common import dataset_io as dio
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.hygiene import scan_repo_for_hardcoded_keys, validate_and_dedup
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import (
    AGRIADVICE_PAIRS,
    COUNTERFACTUAL_PAIRS,
    DRY_RUN_DIR,
    TEST_INSTANCES_FROZEN,
    TRAIN_INSTANCES,
)

logger = get_logger("dry_run_dataset_prep")


def main():
    checks = {}

    facts = dio.load_agrifacts_normalized()
    checks["normalization_invariants"] = all(
        (r["choice_c"] == "Roughly equal")
        and (r["condition"] in ("diff", "equal"))
        and ((r["condition"] == "equal") == (r["correct_answer"] == "c"))
        and r["group1"] in r["on_disk_choices"] and r["group2"] in r["on_disk_choices"]
        for r in facts
    )
    checks["every_group_token_resolved_from_the_census_cell"] = all(
        r["group_order_source"] == "census_token" for r in facts)
    checks["state_blind_key_present_on_every_row"] = all(r.get("state_blind_key") for r in facts)

    train = read_jsonl(TRAIN_INSTANCES)
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    if train and test:
        checks["split_has_zero_fact_level_overlap"] = len(
            {r["source_cell"] for r in train} & {r["source_cell"] for r in test}) == 0
        novel = [r for r in test if r.get("test_slice") == "structure_novel"]
        train_keys = {r["state_blind_key"] for r in train}
        checks["structure_novel_slice_is_non_empty"] = len(novel) > 0
        checks["structure_novel_slice_shares_no_state_blind_key_with_train"] = all(
            r["state_blind_key"] not in train_keys for r in novel)
    else:
        checks["split_has_zero_fact_level_overlap"] = None
        checks["structure_novel_slice_is_non_empty"] = None
        checks["structure_novel_slice_shares_no_state_blind_key_with_train"] = None

    from Dataset_Prep.build_counterfactual_pairs import advice_pair_is_clean, swap_is_clean, swap_record

    diff_row = next(r for r in facts if r["condition"] == "diff")
    swapped = swap_record(diff_row)
    ok, reason = swap_is_clean(diff_row, swapped)
    checks["counterfactual_swap_is_clean_and_flips_the_diff_letter"] = ok and \
        swapped["correct_answer"] != diff_row["correct_answer"]
    equal_row = next(r for r in facts if r["condition"] == "equal")
    eq_swapped = swap_record(equal_row)
    checks["counterfactual_swap_leaves_the_equal_letter_alone"] = eq_swapped["correct_answer"] == "c"

    advice = dio.load_agriadvice_normalized()
    checks["every_agriadvice_pair_differs_only_in_the_persona_span"] = all(
        advice_pair_is_clean(p)[0] for p in advice)

    from Dataset_Prep.build_template_instances import surface_cue_ceiling

    ceiling = surface_cue_ceiling(facts)
    checks["surface_cue_ceiling_is_computable"] = 0.0 <= ceiling["condition_and_answer"] <= 1.0

    sample = facts[:5]
    deduped = validate_and_dedup(sample + [dict(sample[0])], ["id", "question", "correct_answer"], "dry_run_dedup")
    checks["dedup_catches_planted_duplicate"] = len(deduped) == len(sample)

    checks["no_hardcoded_keys_in_source"] = len(scan_repo_for_hardcoded_keys()) == 0

    if COUNTERFACTUAL_PAIRS.exists() and AGRIADVICE_PAIRS.exists():
        checks["counterfactual_files_written"] = bool(read_jsonl(COUNTERFACTUAL_PAIRS)) and \
            bool(read_jsonl(AGRIADVICE_PAIRS))

    report = {"checks": checks, "surface_cue_ceiling": ceiling,
              "all_passed": all(v is not False for v in checks.values())}
    (DRY_RUN_DIR / "dry_run_dataset_prep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in checks.items():
        logger.info("  %-62s %s", k, v)
    logger.info("dry_run_dataset_prep all_passed=%s", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit("dry_run_dataset_prep FAILED: " +
                         json.dumps({k: v for k, v in checks.items() if v is False}))


if __name__ == "__main__":
    main()
