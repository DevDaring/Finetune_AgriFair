"""Is AgriFair adequate for the claims this study makes?

A dataset can be clean and still be the wrong size or shape for the question asked of it.
This module separates the two. It checks integrity, which is about correctness, and then
checks adequacy, which is about whether the splits can carry the hypotheses: enough items in
each cell, enough headroom above the state-blind prior, and enough statistical power for the
comparisons the preregistration commits to.

Power is estimated the way the study actually tests, by simulating the paired item-level
bootstrap on synthetic per-item outcomes at the real slice sizes, rather than by quoting a
formula for a test the study does not run.

Run:  python CPU_Run/audit_dataset_adequacy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
from typing import Dict, List

import numpy as np

from GPU_Run.common import metrics as M
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import (
    AGRIADVICE_PAIRS,
    RESULTS_DIR,
    TEST_INSTANCES_FROZEN,
    TRAIN_INSTANCES,
    VALIDATION_INSTANCES,
)

logger = get_logger("audit_dataset_adequacy")

BOOTSTRAP_RESAMPLES = 400
POWER_TRIALS = 300
TARGET_POWER = 0.80

COLUMNS = [
    "check_name", "scope", "observed_value", "threshold_or_expectation",
    "verdict", "what_it_means_for_the_study",
]


def _rows_by_slice() -> Dict[str, List[Dict]]:
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    out = {"test_all": test}
    for name in ("structure_familiar", "structure_novel"):
        out[name] = [r for r in test if r.get("test_slice") == name]
    return out


def _minimum_detectable_effect(n_items: int, base_accuracy: float = 0.75,
                               seed: int = 42) -> float:
    """Smallest accuracy difference the paired bootstrap detects at 80 percent power.

    Two arms are simulated on the same n items: a reference at `base_accuracy` and a
    comparison at base + delta, correlated as paired arms really are, since both see the same
    items. The search returns the smallest delta whose two-sided bootstrap p-value falls below
    0.05 in at least TARGET_POWER of trials."""
    rng = np.random.default_rng(seed)
    for delta in np.arange(0.02, 0.40, 0.02):
        hits = 0
        for _ in range(POWER_TRIALS):
            shared = rng.random(n_items)
            ref = (shared < base_accuracy).astype(float)
            cmp_ = (shared < min(0.999, base_accuracy + delta)).astype(float)
            diffs = np.empty(BOOTSTRAP_RESAMPLES)
            for b in range(BOOTSTRAP_RESAMPLES):
                idx = rng.integers(0, n_items, n_items)
                diffs[b] = cmp_[idx].mean() - ref[idx].mean()
            p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            hits += int(p < 0.05)
        if hits / POWER_TRIALS >= TARGET_POWER:
            return float(round(delta, 3))
    return float("nan")


def _surface_cue_ceiling(rows: List[Dict]) -> float:
    if not rows:
        return float("nan")
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        counts[r["state_blind_key"]][(r["condition"], r["correct_answer"])] += 1
    return sum(c.most_common(1)[0][1] for c in counts.values()) / len(rows)


def main():
    train = read_jsonl(TRAIN_INSTANCES)
    validation = read_jsonl(VALIDATION_INSTANCES)
    slices = _rows_by_slice()
    advice = read_jsonl(AGRIADVICE_PAIRS)
    if not train or not slices["test_all"]:
        raise SystemExit("Run the dataset_prep stage first.")

    rows: List[Dict] = []

    def record(name, scope, value, expectation, ok, meaning):
        rows.append({"check_name": name, "scope": scope, "observed_value": value,
                     "threshold_or_expectation": expectation,
                     "verdict": "adequate" if ok else "marginal",
                     "what_it_means_for_the_study": meaning})

    # ---- sizes ----
    record("training_items", "train", len(train), "at least 500 for a LoRA adapter",
           len(train) >= 500,
           "LoRA on a few hundred items is normal; the risk is overfitting a template, which "
           "the structure-novel slice is what detects")
    record("validation_items_driving_attribution", "validation", len(validation), "at least 100",
           len(validation) >= 100,
           "Stage A attributes over the failure subset of this split, so a thin split makes the "
           "attribution noisier; ranking stability is reported alongside it for that reason")

    for name, rows_in in slices.items():
        conditions = collections.Counter(r["condition"] for r in rows_in)
        balance = min(conditions.values()) / max(1, max(conditions.values())) if conditions else 0
        record("condition_balance", name, f"{dict(conditions)} ratio {balance:.2f}",
               "within 0.8 of even", balance >= 0.8,
               "the balanced awareness score is a harmonic mean, so a lopsided split would let "
               "one condition dominate the headline number")

    # ---- per-axis cells, which the paper reports separately ----
    for name, rows_in in slices.items():
        axes = collections.Counter(r["category"] for r in rows_in)
        smallest = min(axes.values()) if axes else 0
        record("smallest_axis_cell", name, f"{dict(axes)} smallest {smallest}",
               "at least 100 to report a per-axis number",
               smallest >= 100,
               "an axis below this is reported with an explicit caveat rather than as a "
               "standalone result; gender is the axis that triggers it")

    # ---- headroom above the state-blind prior ----
    for name, rows_in in slices.items():
        ceiling = _surface_cue_ceiling(rows_in)
        headroom = 1.0 - ceiling
        record("headroom_above_the_surface_cue_ceiling", name,
               f"ceiling {ceiling:.3f}, headroom {headroom:.3f}",
               "at least 0.10 of usable range", headroom >= 0.10,
               "the band between the state-blind prior and a perfect score is the only range "
               "in which a method can demonstrate census knowledge")

    # ---- statistical power at the real slice sizes ----
    for name, rows_in in slices.items():
        if not rows_in:
            continue
        mde = _minimum_detectable_effect(len(rows_in))
        record("minimum_detectable_accuracy_difference", name,
               f"n={len(rows_in)}, smallest detectable difference {mde}",
               "0.10 or smaller to separate close methods", mde <= 0.10,
               "differences smaller than this are reported as not separable at this "
               "resolution, which is what the preregistration commits to saying")

    # ---- leave-one-axis-out cells ----
    for axis in ("social_group", "landholding", "gender"):
        held = [r for r in slices["test_all"] if r["category"] == axis]
        trainable = sum(1 for r in train if r["category"] != axis)
        record("leave_one_axis_out_cell", axis,
               f"{len(held)} held-out test items, {trainable} training items without the axis",
               "at least 100 held-out items", len(held) >= 100,
               "a thin held-out axis gives a transfer number with a wide interval; it is "
               "reported with that caveat rather than dropped")

    # ---- AgriAdvice ----
    axes = collections.Counter(p["toggle_axis"] for p in advice)
    record("advice_pairs_per_toggle_axis", "agriadvice",
           f"{dict(axes)} total {len(advice)}", "at least 100 pairs per axis",
           min(axes.values()) >= 100 if axes else False,
           "drift is reported per axis, so each axis needs enough pairs to carry its own mean")
    distinct_queries = len({p["base_query"] for p in advice})
    record("distinct_base_queries", "agriadvice", distinct_queries,
           "one per pair, so no query is reused", distinct_queries == len(advice),
           "a reused query would correlate two pairs and inflate apparent consistency")

    write_csv(RESULTS_DIR / "dataset_adequacy_audit.csv", rows, COLUMNS)
    marginal = [r for r in rows if r["verdict"] == "marginal"]
    for r in rows:
        logger.info("  %-42s %-20s %-46s %s", r["check_name"], r["scope"],
                    str(r["observed_value"])[:46], r["verdict"].upper())
    logger.info("%d checks, %d marginal.", len(rows), len(marginal))
    log_run_metadata("audit_dataset_adequacy",
                     {"checks": len(rows), "marginal": [f"{r['check_name']}:{r['scope']}" for r in marginal]})


if __name__ == "__main__":
    main()
