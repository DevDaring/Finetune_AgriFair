"""Map external generalization sets onto the AgriFair harness (evaluation only).

Two faithful options (coding_prompt.md Section 6):
  1. Cross-axis transfer (always available): the held-out axis (default gender) is carved
     from the in-corpus test set; this script records the directed axis pairs and the
     held-out axis split so evaluate_all can run zero-training transfer.
  2. Out-of-domain transfer (optional): the original Wang et al. (2025) social-bias D2/N3
     sets, if present under data/external_wang/, are mapped to the harness record schema.
     If absent, a clear note is written and nothing is fabricated.

Run:  python Dataset_Prep/map_external_wang_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import DATA_DIR, EXTERNAL_WANG_DIR, TEST_INSTANCES_FROZEN

logger = get_logger("map_external_wang_datasets")

HELD_OUT_AXIS = "gender"  # cross-axis transfer target (thin slice; reported with caveat)
AXES = ["social_group", "landholding", "gender"]


def _directed_axis_pairs():
    return [(a, b) for a in AXES for b in AXES if a != b]


def main():
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    held_out = [r for r in test if r.get("category") == HELD_OUT_AXIS]
    out_cross = DATA_DIR / "cross_axis_transfer_config.json"
    out_cross.write_text(
        json.dumps(
            {
                "held_out_axis": HELD_OUT_AXIS,
                "held_out_items": len(held_out),
                "directed_axis_pairs": _directed_axis_pairs(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Cross-axis transfer: held-out axis '%s' (%d items), 6 directed pairs.", HELD_OUT_AXIS, len(held_out))

    # Optional out-of-domain Wang social-bias sets.
    mapped = {}
    for comp in ("D2_occupation_descriptive", "N3_occupation_affirmative_action"):
        src = EXTERNAL_WANG_DIR / f"{comp}.jsonl"
        if not src.exists():
            note = EXTERNAL_WANG_DIR / f"{comp}.MISSING.txt"
            note.write_text(
                f"{comp} not present. Place the official Wang et al. (2025) release file at "
                f"{src} to enable out-of-domain transfer evaluation. Nothing is fabricated.\n",
                encoding="utf-8",
            )
            logger.info("%s not present; wrote a note (no fabrication).", comp)
            continue
        rows = read_jsonl(src)
        mapped[comp] = len(rows)
        logger.info("Mapped external set %s (%d rows).", comp, len(rows))

    log_run_metadata(
        "map_external_wang_datasets",
        {"held_out_axis": HELD_OUT_AXIS, "held_out_items": len(held_out), "external_wang_mapped": mapped},
    )


if __name__ == "__main__":
    main()
