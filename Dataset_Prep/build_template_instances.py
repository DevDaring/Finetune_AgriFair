"""Deterministic templating, group-disjoint split, frozen test set + SHA256.

No synthetic instances: every record is a deterministic templating of a real AgriFacts
row (Instruction.md Section 4.2; coding_prompt.md Sections 3-5). The split is group-
disjoint at the scenario_type (= paraphrase_of) level, stratified by axis, seed 42:
20 percent of clusters -> frozen test, 10 percent of the remainder -> validation, the
rest -> train. The frozen test file is hashed (SHA256) into results/.

Run:  python Dataset_Prep/build_template_instances.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
import hashlib
import json
import random

from GPU_Run.common import dataset_io as dio
from GPU_Run.common import prompts as P
from GPU_Run.common.hygiene import validate_and_dedup
from GPU_Run.common.logging_utils import append_jsonl, get_logger, log_run_metadata
from GPU_Run.common.paths import (
    FROZEN_TEST_SHA256,
    TEMPLATED_ALL,
    TEST_INSTANCES_FROZEN,
    TRAIN_INSTANCES,
    VALIDATION_INSTANCES,
)
from GPU_Run.common.seeds import GLOBAL_SEED

logger = get_logger("build_template_instances")

TEST_CLUSTER_FRACTION = 0.20
VAL_CLUSTER_FRACTION = 0.10  # of the remaining clusters


def _templated_record(rec: dict) -> dict:
    prompt, target = P.build_sft_example(rec, rationale_mode=True)
    answer_prompt, answer_target = P.build_sft_example(rec, rationale_mode=False)
    out = dict(rec)
    out["sft_prompt"] = prompt
    out["sft_target_rationale_mode"] = target
    out["sft_target_answer_mode"] = answer_target
    return out


def _split_by_cluster(records):
    """Group-disjoint split by scenario_type, stratified by axis, seed 42."""
    rng = random.Random(GLOBAL_SEED)
    # cluster -> rows; cluster -> axis (majority)
    by_cluster = collections.defaultdict(list)
    for r in records:
        by_cluster[r["scenario_type"]].append(r)
    cluster_axis = {}
    for c, rows in by_cluster.items():
        axis = collections.Counter(r["category"] for r in rows).most_common(1)[0][0]
        cluster_axis[c] = axis

    test_clusters, val_clusters, train_clusters = set(), set(), set()
    by_axis = collections.defaultdict(list)
    for c, axis in cluster_axis.items():
        by_axis[axis].append(c)
    for axis, clusters in by_axis.items():
        clusters = sorted(clusters)
        rng.shuffle(clusters)
        n = len(clusters)
        n_test = max(1, int(round(TEST_CLUSTER_FRACTION * n)))
        test = clusters[:n_test]
        rest = clusters[n_test:]
        n_val = max(1, int(round(VAL_CLUSTER_FRACTION * len(rest)))) if rest else 0
        val = rest[:n_val]
        train = rest[n_val:]
        test_clusters.update(test)
        val_clusters.update(val)
        train_clusters.update(train)

    splits = {"train": [], "validation": [], "test": []}
    for r in records:
        c = r["scenario_type"]
        if c in test_clusters:
            r["split"] = "test"
            splits["test"].append(r)
        elif c in val_clusters:
            r["split"] = "validation"
            splits["validation"].append(r)
        else:
            r["split"] = "train"
            splits["train"].append(r)
    return splits


def _write_jsonl(path, rows):
    if path.exists():
        path.unlink()
    for r in rows:
        append_jsonl(path, r)


def main():
    raw = dio.load_agrifacts_normalized()
    clean = validate_and_dedup(
        raw,
        required_fields=["id", "question", "correct_answer", "condition", "rationale"],
        source_file="agrifacts_normalized",
    )
    templated = [_templated_record(r) for r in clean]
    _write_jsonl(TEMPLATED_ALL, templated)

    splits = _split_by_cluster(templated)
    _write_jsonl(TRAIN_INSTANCES, splits["train"])
    _write_jsonl(VALIDATION_INSTANCES, splits["validation"])
    _write_jsonl(TEST_INSTANCES_FROZEN, splits["test"])

    # freeze hash of the test file
    digest = hashlib.sha256(TEST_INSTANCES_FROZEN.read_bytes()).hexdigest()
    FROZEN_TEST_SHA256.write_text(digest + "\n", encoding="utf-8")

    summary = {k: len(v) for k, v in splits.items()}
    cond = {
        k: dict(collections.Counter(r["condition"] for r in v)) for k, v in splits.items()
    }
    axis = {
        k: dict(collections.Counter(r["category"] for r in v)) for k, v in splits.items()
    }
    logger.info("Split sizes: %s", summary)
    logger.info("Per-split condition mix: %s", cond)
    logger.info("Per-split axis mix: %s", axis)
    logger.info("Frozen test SHA256: %s", digest)
    log_run_metadata(
        "build_template_instances",
        {"split_sizes": summary, "condition_mix": cond, "axis_mix": axis, "frozen_test_sha256": digest},
    )


if __name__ == "__main__":
    main()
