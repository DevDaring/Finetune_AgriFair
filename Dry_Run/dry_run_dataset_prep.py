"""Dry run for Dataset_Prep: fast, offline integrity checks before expensive work.

Validates: normalization invariants (diff->neq/equal->eq, c="Roughly equal"), the
group-disjoint split (0 fact-level overlap), counterfactual swap flips the neq letter,
data-hygiene dedup catches a planted duplicate, and the key/hardcoded-key self-checks
(Instruction.md Section 11; coding_prompt.md Section 12). Writes a pass/fail report.

Run:  python Dry_Run/dry_run_dataset_prep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common import dataset_io as dio
from GPU_Run.common.hygiene import scan_repo_for_hardcoded_keys, validate_and_dedup
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import DRY_RUN_DIR, TEST_INSTANCES_FROZEN, TRAIN_INSTANCES
from GPU_Run.common.checkpointing import read_jsonl

logger = get_logger("dry_run_dataset_prep")


def main():
    checks = {}

    facts = dio.load_agrifacts_normalized()
    checks["normalization_invariants"] = all(
        (r["choice_c"] == "Roughly equal")
        and (r["condition"] in ("neq", "eq"))
        and ((r["condition"] == "eq") == (r["correct_answer"] == "c"))
        for r in facts
    )

    train = read_jsonl(TRAIN_INSTANCES)
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    if train and test:
        tr = set(r["source_cell"] for r in train)
        te = set(r["source_cell"] for r in test)
        checks["group_disjoint_split_zero_fact_overlap"] = len(tr & te) == 0
    else:
        checks["group_disjoint_split_zero_fact_overlap"] = None

    # counterfactual swap flips the neq letter
    from Dataset_Prep.build_counterfactual_pairs import _swap_record

    neq = next(r for r in facts if r["condition"] == "neq")
    sw = _swap_record(neq)
    checks["counterfactual_swap_flips_neq_letter"] = (
        sw["correct_answer"] != neq["correct_answer"]
        and sw["group1"] == neq["group2"]
    )

    # dedup catches a planted duplicate
    sample = facts[:5]
    planted = sample + [dict(sample[0])]
    deduped = validate_and_dedup(planted, ["id", "question", "correct_answer"], "dry_run_dedup")
    checks["dedup_catches_planted_duplicate"] = len(deduped) == len(sample)

    # hardcoded-key self-check finds nothing in tracked source
    checks["no_hardcoded_keys_in_source"] = len(scan_repo_for_hardcoded_keys()) == 0

    report = {"checks": checks, "all_passed": all(v is not False for v in checks.values())}
    out = DRY_RUN_DIR / "dry_run_dataset_prep_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in checks.items():
        logger.info("  %-45s %s", k, v)
    logger.info("dry_run_dataset_prep all_passed=%s", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit("dry_run_dataset_prep FAILED: " + json.dumps(checks))


if __name__ == "__main__":
    main()
