"""Build the external capability probe used for "preserve the rest".

Capability retention has to be measured off the training distribution, otherwise a repair
that overfits the benchmark looks like a repair that preserved general ability. This script
draws a fixed, seed-stratified subset of MMLU (Hendrycks et al., ICLR 2021, arXiv:2009.03300)
across all subjects and writes it to data/external_capability_probe.jsonl in the harness's
four-option schema. The same item set and order are reused for every model and method.

Scoring (in GPU_Run/common/inference.py) is exact match on the parsed option letter with no
judge fallback; an output from which no letter can be parsed is scored incorrect and stays
in the denominator, so the probe measures capability and output-format compliance together.
That is stated as a limitation rather than hidden.

Network is touched only here and in download_models_and_data.py. If MMLU cannot be
fetched, the script writes nothing and logs why; evaluation then falls back to the
in-domain validation split and records that it did.

Run:  python Dataset_Prep/build_capability_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
import json
import os
import random

from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import EXTERNAL_CAPABILITY_PROBE
from GPU_Run.common.seeds import GLOBAL_SEED

logger = get_logger("build_capability_probe")

PROBE_SIZE = int(os.environ.get("CAPABILITY_PROBE_SIZE", "500"))
PROBE_REPO = os.environ.get("CAPABILITY_PROBE_REPO") or env_loader.get("CAPABILITY_PROBE_REPO", "cais/mmlu")
PROBE_CONFIG = os.environ.get("CAPABILITY_PROBE_CONFIG", "all")
PROBE_SPLIT = os.environ.get("CAPABILITY_PROBE_SPLIT", "test")


def _normalize(row, index):
    """Return the harness item, or None if the row is not a clean four-option question."""
    question = row.get("question")
    choices = row.get("choices") or row.get("options")
    answer = row.get("answer")
    if not question or not choices or len(choices) < 4:
        return None
    if isinstance(answer, str):
        answer = answer.strip().lower()
        answer = "abcd".index(answer) if answer in "abcd" and len(answer) == 1 else None
    if answer is None or not isinstance(answer, int) or not 0 <= answer < 4:
        return None
    return {
        "id": f"capability-{index:05d}",
        "subject": row.get("subject", ""),
        "question": question,
        "choices": [str(c) for c in choices[:4]],
        "answer_index": int(answer),
        "source_name": f"external_probe:{PROBE_REPO}/{PROBE_CONFIG}/{PROBE_SPLIT}",
    }


def main():
    if EXTERNAL_CAPABILITY_PROBE.exists() and os.environ.get("FORCE_REBUILD_CAPABILITY_PROBE", "0") != "1":
        logger.info("Capability probe already present at %s; skipping.", EXTERNAL_CAPABILITY_PROBE)
        return
    try:
        from datasets import load_dataset

        ds = load_dataset(PROBE_REPO, PROBE_CONFIG, split=PROBE_SPLIT, token=env_loader.hf_token())
    except Exception as e:
        logger.warning("Could not load the capability probe %s/%s (%s). Evaluation will fall back to "
                       "the in-domain validation split and will record that it did.", PROBE_REPO, PROBE_CONFIG, e)
        return

    by_subject = collections.defaultdict(list)
    for i, row in enumerate(ds):
        item = _normalize(row, i)
        if item is not None:
            by_subject[item["subject"]].append(item)
    if not by_subject:
        logger.warning("No usable rows in %s; nothing written.", PROBE_REPO)
        return

    rng = random.Random(GLOBAL_SEED)
    total = sum(len(v) for v in by_subject.values())
    selected = []
    for subject in sorted(by_subject):
        rows = sorted(by_subject[subject], key=lambda r: r["id"])
        rng.shuffle(rows)
        take = max(1, round(PROBE_SIZE * len(by_subject[subject]) / total))
        selected.extend(rows[:take])
    selected = sorted(selected, key=lambda r: r["id"])[:PROBE_SIZE]
    for i, item in enumerate(selected):
        item["id"] = f"capability-{i:05d}"

    with open(EXTERNAL_CAPABILITY_PROBE, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Wrote %d capability-probe items from %s across %d subjects.",
                len(selected), PROBE_REPO, len(by_subject))
    log_run_metadata("build_capability_probe", {
        "probe_repo": PROBE_REPO, "items": len(selected), "subjects": len(by_subject),
    })


if __name__ == "__main__":
    main()
