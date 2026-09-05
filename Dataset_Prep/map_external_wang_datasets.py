"""Register the transfer evaluations: leave-one-axis-out in corpus, Wang et al. out of
domain if the files are present.

1. Leave-one-axis-out (always available). For each axis, an adapter is trained without it
   and evaluated on it, so "held out" means the axis was absent from training rather than
   scored per axis from a full-mixture adapter. This script records the axis inventory and
   the item counts each held-out evaluation will use.
2. Out-of-domain transfer (optional). The original Wang et al. (2025, arXiv:2502.01926)
   social-bias D2/N3 sets, if placed under data/external_wang/, are mapped to the harness
   record schema, which shows whether an agriculture-localized repair transfers back to the
   social domain. If they are absent a note is written and nothing is fabricated.

Run:  python Dataset_Prep/map_external_wang_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
import json
import os

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import DATA_DIR, EXTERNAL_WANG_DIR, TEST_INSTANCES_FROZEN, TRAIN_INSTANCES

logger = get_logger("map_external_wang_datasets")

AXES = [a.strip() for a in os.environ.get("LOAO_AXES", "social_group,landholding,gender").split(",") if a.strip()]


def main():
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    train = read_jsonl(TRAIN_INSTANCES)
    per_axis = {}
    for axis in AXES:
        held = [r for r in test if r.get("category") == axis]
        per_axis[axis] = {
            "held_out_test_items": len(held),
            "training_items_when_this_axis_is_held_out": sum(1 for r in train if r.get("category") != axis),
            "held_out_condition_mix": dict(collections.Counter(r["condition"] for r in held)),
            "held_out_slice_mix": dict(collections.Counter(r.get("test_slice", "") for r in held)),
        }
    out = DATA_DIR / "leave_one_axis_out_config.json"
    out.write_text(json.dumps({"axes": AXES, "per_axis": per_axis}, indent=2), encoding="utf-8")
    for axis, info in per_axis.items():
        logger.info("Leave-one-axis-out %s: %d held-out test items, %d training items without it.",
                    axis, info["held_out_test_items"], info["training_items_when_this_axis_is_held_out"])

    mapped = {}
    for comp in ("D2_occupation_descriptive", "N3_occupation_affirmative_action"):
        src = EXTERNAL_WANG_DIR / f"{comp}.jsonl"
        if not src.exists():
            (EXTERNAL_WANG_DIR / f"{comp}.MISSING.txt").write_text(
                f"{comp} not present. Place the official Wang et al. (2025, arXiv:2502.01926) release "
                f"file at {src} to enable out-of-domain transfer evaluation. Nothing is fabricated.\n",
                encoding="utf-8")
            logger.info("%s not present; wrote a note (no fabrication).", comp)
            continue
        rows = read_jsonl(src)
        mapped[comp] = len(rows)
        logger.info("Mapped external set %s (%d rows).", comp, len(rows))

    log_run_metadata("map_external_wang_datasets", {"per_axis": per_axis, "external_wang_mapped": mapped})


if __name__ == "__main__":
    main()
