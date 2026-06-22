"""Collect the failure set for Stage A attribution.

On the validation split, run the frozen model and collect: wrongly answered items, with
special attention to neq (diff) items where the model wrongly chose c (the over-equalization
failure), and identity-swap flips (the MCQ counterfactual where the predicted letter changes
under a deterministic group swap). Writes data/probe_failures_<tier>.jsonl and a summary
(Instruction.md Section 2 Stage A; coding_prompt.md Section 7).

Run:  python GPU_Run/probe_subject_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.inference import run_mcq_eval
from GPU_Run.common.logging_utils import append_jsonl, get_logger, write_csv
from GPU_Run.common.paths import COUNTERFACTUAL_PAIRS, DATA_DIR, RESULTS_DIR, VALIDATION_INSTANCES
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("probe_subject_models")

SUMMARY_COLUMNS = [
    "tier",
    "validation_items",
    "wrong_items",
    "over_equalization_failures_neq_chose_c",
    "identity_swap_flips",
]


def main(smoke: bool = False):
    set_global_determinism()
    val = read_jsonl(VALIDATION_INSTANCES)
    cf = read_jsonl(COUNTERFACTUAL_PAIRS)
    val_ids = {r["id"] for r in val}
    # counterfactuals whose original is in validation
    cf_val = [c for c in cf if c["original"]["id"] in val_ids]

    rows = []
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        try:
            model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
        except Exception as e:
            logger.warning("Could not load %s (%s); skipping.", tier, e)
            continue

        preds = run_mcq_eval(model, tok, val, rationale_mode=False)
        pred_by_id = {p["id"]: p for p in preds}

        # identity-swap flips: predict on originals and swapped, compare letters
        originals = [c["original"] for c in cf_val]
        swapped = [c["swapped"] for c in cf_val]
        po = {p["id"]: p["pred_canonical"] for p in run_mcq_eval(model, tok, originals, rationale_mode=False)}
        ps = {p["id"]: p["pred_canonical"] for p in run_mcq_eval(model, tok, swapped, rationale_mode=False)}

        failures = []
        over_eq = 0
        for p in preds:
            wrong = p["pred_canonical"] != p["gold_canonical"]
            is_over_eq = p["condition"] == "neq" and p["pred_canonical"] == "c"
            if is_over_eq:
                over_eq += 1
            if wrong:
                failures.append({**p, "failure_type": "over_equalization" if is_over_eq else "other_wrong"})

        flips = 0
        for c in cf_val:
            oid = c["original"]["id"]
            sid = c["swapped"]["id"]
            if oid in po and sid in ps and po[oid] != ps[sid]:
                flips += 1
                failures.append(
                    {
                        "id": oid,
                        "failure_type": "identity_swap_flip",
                        "pred_original": po[oid],
                        "pred_swapped": ps[sid],
                        "condition": c["original"]["condition"],
                    }
                )

        out_path = DATA_DIR / f"probe_failures_{meta['tier']}.jsonl"
        if out_path.exists():
            out_path.unlink()
        for f in failures:
            append_jsonl(out_path, f)

        rows.append(
            {
                "tier": meta["tier"],
                "validation_items": len(val),
                "wrong_items": sum(1 for p in preds if p["pred_canonical"] != p["gold_canonical"]),
                "over_equalization_failures_neq_chose_c": over_eq,
                "identity_swap_flips": flips,
            }
        )
        logger.info("tier=%s over-eq failures=%d swap-flips=%d", meta["tier"], over_eq, flips)
        del model

    write_csv(RESULTS_DIR / "probe_failure_summary.csv", rows, SUMMARY_COLUMNS)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
