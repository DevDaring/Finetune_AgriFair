"""Deterministic templating, the split, the frozen test set and its SHA256, and the
surface-cue ceiling audit.

Every record is a deterministic templating of a real AgriFacts row; nothing is generated.

The split has to control two different kinds of leakage, because AgriFacts lives in a small
combinatorial space. The first is paraphrase leakage: several items are rewordings of the
same census cell, so the split is group-disjoint at the scenario_type (= paraphrase_of,
one-to-one with source_cell) level. The second is structural leakage, which the paraphrase
split does not touch. Each item also carries a state_blind_key of (axis, metric, size class,
comparison token): its answer-relevant structure with the state name removed. Two items
that share it differ only in which state they name, and a majority-class predictor that
sees nothing but that key reaches 0.87 on the corpus. Under a paraphrase-only split, 97
percent of test items share a state-blind key with training, so a model can score highly
without ever reading the state.

The split therefore reserves a fraction of state-blind keys, stratified by axis, that
appear ONLY in the frozen test set. The frozen test set then carries two labelled slices:

  structure_familiar  the item's (axis, metric, class, comparison) combination is also in
                      training, only the state differs. This is the preregistered
                      in-distribution condition, comparable with a paraphrase-only split.
  structure_novel     that combination is absent from training. Answering it needs the
                      census fact, not the structural prior.

Training and validation are drawn only from the familiar keys, so nothing about the novel
keys reaches any adapter. Both slices are evaluated and reported separately; the headline
number is reported on each rather than pooled, and the surface-cue ceiling written to
results/surface_cue_ceiling_audit.csv is the accuracy each slice concedes to a state-blind
prior. Set STRUCTURE_NOVEL_KEY_FRACTION=0 to recover the original paraphrase-only split.

Run:  python Dataset_Prep/build_template_instances.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
import hashlib
import json
import os
import random
from typing import Dict, List

from GPU_Run.common import dataset_io as dio
from GPU_Run.common import prompts as P
from GPU_Run.common.hygiene import validate_and_dedup
from GPU_Run.common.logging_utils import append_jsonl, get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import (
    FROZEN_TEST_SHA256,
    RESULTS_DIR,
    TEMPLATED_ALL,
    TEST_INSTANCES_FROZEN,
    TRAIN_INSTANCES,
    VALIDATION_INSTANCES,
)
from GPU_Run.common.seeds import GLOBAL_SEED

logger = get_logger("build_template_instances")

TEST_CLUSTER_FRACTION = float(os.environ.get("TEST_CLUSTER_FRACTION", "0.20"))
VAL_CLUSTER_FRACTION = float(os.environ.get("VALIDATION_CLUSTER_FRACTION", "0.10"))
STRUCTURE_NOVEL_KEY_FRACTION = float(os.environ.get("STRUCTURE_NOVEL_KEY_FRACTION", "0.20"))
SPLIT_KEY = "scenario_type"

CEILING_COLUMNS = [
    "split_name", "number_of_items", "number_of_distinct_state_blind_keys",
    "surface_cue_majority_ceiling_for_condition_and_answer",
    "surface_cue_majority_ceiling_for_condition_only",
    "share_of_items_whose_state_blind_key_also_appears_in_train",
]


def _templated_record(rec: dict) -> dict:
    prompt, target = P.build_sft_example(rec, rationale_mode=True)
    _, answer_target = P.build_sft_example(rec, rationale_mode=False)
    out = dict(rec)
    out["sft_prompt"] = prompt
    out["sft_target_rationale_mode"] = target
    out["sft_target_answer_mode"] = answer_target
    return out


def _stratified_key_holdout(records: List[Dict], fraction: float, rng: random.Random) -> set:
    """Reserve a fraction of state-blind keys per axis for the structure-novel slice."""
    if fraction <= 0:
        return set()
    by_axis = collections.defaultdict(set)
    for r in records:
        by_axis[r["category"]].add(r["state_blind_key"])
    held = set()
    for axis, keys in sorted(by_axis.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n = max(1, int(round(fraction * len(keys)))) if len(keys) > 1 else 0
        held.update(keys[:n])
    return held


def split_records(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Structure-novel keys go to the frozen test only; the rest is split group-disjoint
    by paraphrase cluster, stratified by axis, seed 42."""
    rng = random.Random(GLOBAL_SEED)
    novel_keys = _stratified_key_holdout(records, STRUCTURE_NOVEL_KEY_FRACTION, rng)
    familiar = [r for r in records if r["state_blind_key"] not in novel_keys]
    novel = [r for r in records if r["state_blind_key"] in novel_keys]

    by_cluster = collections.defaultdict(list)
    for r in familiar:
        by_cluster[r[SPLIT_KEY]].append(r)
    cluster_axis = {c: collections.Counter(x["category"] for x in rows).most_common(1)[0][0]
                    for c, rows in by_cluster.items()}
    by_axis = collections.defaultdict(list)
    for c, axis in cluster_axis.items():
        by_axis[axis].append(c)

    test_clusters, val_clusters = set(), set()
    for axis, clusters in sorted(by_axis.items()):
        clusters = sorted(clusters)
        rng.shuffle(clusters)
        n = len(clusters)
        n_test = max(1, int(round(TEST_CLUSTER_FRACTION * n)))
        test_clusters.update(clusters[:n_test])
        rest = clusters[n_test:]
        n_val = max(1, int(round(VAL_CLUSTER_FRACTION * len(rest)))) if rest else 0
        val_clusters.update(rest[:n_val])

    splits = {"train": [], "validation": [], "test": []}
    for r in familiar:
        c = r[SPLIT_KEY]
        name = "test" if c in test_clusters else ("validation" if c in val_clusters else "train")
        r["split"] = name
        r["test_slice"] = "structure_familiar" if name == "test" else ""
        splits[name].append(r)
    for r in novel:
        r["split"] = "test"
        r["test_slice"] = "structure_novel"
        splits["test"].append(r)
    splits["test"].sort(key=lambda r: r["id"])
    return splits


def surface_cue_ceiling(records: List[Dict]) -> Dict[str, float]:
    """Accuracy of a majority-class predictor that sees only the state_blind_key."""
    if not records:
        return {"condition_and_answer": float("nan"), "condition_only": float("nan"), "keys": 0}
    both = collections.defaultdict(collections.Counter)
    cond = collections.defaultdict(collections.Counter)
    for r in records:
        k = r["state_blind_key"]
        both[k][(r["condition"], r["correct_answer"])] += 1
        cond[k][r["condition"]] += 1
    n = len(records)
    return {
        "condition_and_answer": sum(c.most_common(1)[0][1] for c in both.values()) / n,
        "condition_only": sum(c.most_common(1)[0][1] for c in cond.values()) / n,
        "keys": len(both),
    }


def _write_jsonl(path, rows):
    if path.exists():
        path.unlink()
    for r in rows:
        append_jsonl(path, r)


def _ceiling_rows(splits: Dict[str, List[Dict]], all_rows: List[Dict]) -> List[Dict]:
    train_keys = {r["state_blind_key"] for r in splits["train"]}
    groups = [("all", all_rows), ("train", splits["train"]), ("validation", splits["validation"]),
              ("test_all", splits["test"])]
    for slice_name in ("structure_familiar", "structure_novel"):
        groups.append((f"test_{slice_name}", [r for r in splits["test"] if r.get("test_slice") == slice_name]))
    rows = []
    for name, rows_in in groups:
        c = surface_cue_ceiling(rows_in)
        shared = round(sum(1 for r in rows_in if r["state_blind_key"] in train_keys) / len(rows_in), 4) \
            if rows_in and name != "train" else ""
        rows.append({
            "split_name": name, "number_of_items": len(rows_in),
            "number_of_distinct_state_blind_keys": c["keys"],
            "surface_cue_majority_ceiling_for_condition_and_answer": round(c["condition_and_answer"], 4)
            if c["condition_and_answer"] == c["condition_and_answer"] else "",
            "surface_cue_majority_ceiling_for_condition_only": round(c["condition_only"], 4)
            if c["condition_only"] == c["condition_only"] else "",
            "share_of_items_whose_state_blind_key_also_appears_in_train": shared,
        })
    return rows


def main():
    raw = dio.load_agrifacts_normalized()
    clean = validate_and_dedup(
        raw,
        required_fields=["id", "question", "correct_answer", "condition", "rationale"],
        source_file="agrifacts_normalized",
    )
    templated = [_templated_record(r) for r in clean]
    _write_jsonl(TEMPLATED_ALL, templated)

    splits = split_records(templated)
    _write_jsonl(TRAIN_INSTANCES, splits["train"])
    _write_jsonl(VALIDATION_INSTANCES, splits["validation"])
    _write_jsonl(TEST_INSTANCES_FROZEN, splits["test"])
    digest = hashlib.sha256(TEST_INSTANCES_FROZEN.read_bytes()).hexdigest()
    FROZEN_TEST_SHA256.write_text(digest + "\n", encoding="utf-8")

    ceiling_rows = _ceiling_rows(splits, templated)
    write_csv(RESULTS_DIR / "surface_cue_ceiling_audit.csv", ceiling_rows, CEILING_COLUMNS)

    summary = {k: len(v) for k, v in splits.items()}
    slice_sizes = dict(collections.Counter(r.get("test_slice", "") for r in splits["test"]))
    cond = {k: dict(collections.Counter(r["condition"] for r in v)) for k, v in splits.items()}
    axis = {k: dict(collections.Counter(r["category"] for r in v)) for k, v in splits.items()}
    logger.info("Split sizes=%s test slices=%s", summary, slice_sizes)
    logger.info("Per-split condition mix: %s", cond)
    logger.info("Per-split axis mix: %s", axis)
    logger.info("Frozen test SHA256: %s", digest)
    for row in ceiling_rows:
        if row["split_name"].startswith("test"):
            logger.info("%s: %d items, surface-cue ceiling %s, %s share a state-blind key with train.",
                        row["split_name"], row["number_of_items"],
                        row["surface_cue_majority_ceiling_for_condition_and_answer"],
                        row["share_of_items_whose_state_blind_key_also_appears_in_train"])
    log_run_metadata("build_template_instances", {
        "split_key": SPLIT_KEY, "structure_novel_key_fraction": STRUCTURE_NOVEL_KEY_FRACTION,
        "split_sizes": summary, "test_slice_sizes": slice_sizes, "condition_mix": cond,
        "axis_mix": axis, "frozen_test_sha256": digest,
    })


if __name__ == "__main__":
    main()
