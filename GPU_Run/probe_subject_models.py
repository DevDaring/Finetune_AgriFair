"""Collect the failure set that drives Stage A attribution.

On the validation split, run the frozen model and collect: wrongly answered items, with
special attention to diff items answered "Roughly equal" (gap erasure), equal items
answered with a group name (gap fabrication), and identity-swap flips (the MCQ
counterfactual whose predicted letter moves in an unlicensed direction under a
deterministic group swap).

The failure records carry the item's axis so the leave-one-axis-out placements can be
built from failure sets that exclude the held-out axis.

Run:  python GPU_Run/probe_subject_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from GPU_Run.common import metrics as M
from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.inference import run_mcq_eval
from GPU_Run.common.logging_utils import append_jsonl, get_logger, write_csv
from GPU_Run.common.paths import COUNTERFACTUAL_PAIRS, DATA_DIR, RESULTS_DIR, VALIDATION_INSTANCES
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("probe_subject_models")

SUMMARY_COLUMNS = [
    "tier",
    "validation_items",
    "wrong_items",
    "gap_erasure_failures_diff_answered_roughly_equal",
    "gap_fabrication_failures_equal_answered_with_a_group",
    "identity_swap_unlicensed_flips",
    "answers_recovered_by_judge",
    "answers_left_unresolved",
    "failure_items_by_axis",
]

SWAP_LIMIT = int(os.environ.get("PROBE_SWAP_SUBSET_SIZE", "200"))


def main(smoke: bool = False):
    set_global_determinism()
    # The failure set built here is what Stage A attributes over. Without the judge, an
    # output the parser could not read is filed as a wrong answer, so the attribution would
    # be aimed partly at output formatting rather than at the fairness failure.
    judge = JudgeChain()
    if judge.available():
        logger.info("Answer-extraction judge chain: %s", " -> ".join(judge.route_names()))
    val = read_jsonl(VALIDATION_INSTANCES)
    cf = read_jsonl(COUNTERFACTUAL_PAIRS)
    val_ids = {r["id"] for r in val}
    cf_val = [c for c in cf if c["original"]["id"] in val_ids][:SWAP_LIMIT]

    rows = []
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        try:
            model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
        except Exception as e:
            logger.warning("Could not load %s (%s); skipping.", tier, e)
            continue
        try:

            preds = run_mcq_eval(model, tok, val, rationale_mode=False, judge=judge)
            by_id = {r["id"]: r for r in val}

            originals = [c["original"] for c in cf_val]
            swapped = [c["swapped"] for c in cf_val]
            po = run_mcq_eval(model, tok, originals, rationale_mode=False, judge=judge) if originals else []
            ps = run_mcq_eval(model, tok, swapped, rationale_mode=False, judge=judge) if swapped else []
            ps_by_id = {p["id"]: p for p in ps}

            failures = []
            erasure = fabrication = 0
            for p in preds:
                rec = by_id.get(p["id"], {})
                wrong = p["pred_canonical"] != p["gold_canonical"]
                is_erasure = p["condition"] == "diff" and p["pred_canonical"] == "c"
                is_fabrication = p["condition"] == "equal" and p["pred_canonical"] in ("a", "b")
                erasure += int(is_erasure)
                fabrication += int(is_fabrication)
                if wrong:
                    failures.append({
                        "id": p["id"], "condition": p["condition"], "axis": rec.get("category", ""),
                        "failure_type": "gap_erasure" if is_erasure else ("gap_fabrication" if is_fabrication else "other_wrong"),
                        "pred_canonical": p["pred_canonical"], "gold_canonical": p["gold_canonical"],
                    })

            flips = 0
            licensed = {"a": "b", "b": "a", "c": "c"}
            for p in po:
                q = ps_by_id.get(str(p["id"]) + "::swap")
                if q is None:
                    continue
                if q["pred_canonical"] != licensed.get(p["pred_canonical"], p["pred_canonical"]):
                    flips += 1
                    rec = by_id.get(p["id"], {})
                    failures.append({
                        "id": p["id"], "condition": p["condition"], "axis": rec.get("category", ""),
                        "failure_type": "identity_swap_flip",
                        "pred_canonical": p["pred_canonical"], "pred_swapped_canonical": q["pred_canonical"],
                    })

            out_path = DATA_DIR / f"probe_failures_{meta['tier']}.jsonl"
            if out_path.exists():
                out_path.unlink()
            seen = set()
            for f in failures:
                key = (f["id"], f["failure_type"])
                if key in seen:
                    continue
                seen.add(key)
                append_jsonl(out_path, f)

            by_axis = {}
            for f in failures:
                by_axis[f.get("axis", "")] = by_axis.get(f.get("axis", ""), 0) + 1
            rows.append({
                "tier": meta["tier"],
                "validation_items": len(val),
                "wrong_items": sum(1 for p in preds if p["pred_canonical"] != p["gold_canonical"]),
                "gap_erasure_failures_diff_answered_roughly_equal": erasure,
                "gap_fabrication_failures_equal_answered_with_a_group": fabrication,
                "identity_swap_unlicensed_flips": flips,
                "answers_recovered_by_judge": preds[0].get("answers_recovered_by_judge", 0) if preds else 0,
                "answers_left_unresolved": sum(1 for p in preds if p["pred_canonical"] == "unparsed"),
                "failure_items_by_axis": ";".join(f"{k}={v}" for k, v in sorted(by_axis.items()) if k),
            })
            logger.info("tier=%s erasure=%d fabrication=%d swap-flips=%d", meta["tier"], erasure, fabrication, flips)
        except Exception as e:
            # One tier failing must not discard the GPU hours already spent on the
            # others. The row is dropped, the reason is recorded, and the remaining
            # tiers still run.
            logger.error("Tier %s failed after loading (%s); its row is omitted and "
                         "the remaining tiers continue.", tier, e, exc_info=True)
        finally:
            del model
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        # Written after every tier, so an interruption keeps what has completed.
        write_csv(RESULTS_DIR / "probe_failure_summary.csv", rows, SUMMARY_COLUMNS)

    write_csv(RESULTS_DIR / "probe_failure_summary.csv", rows, SUMMARY_COLUMNS)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
