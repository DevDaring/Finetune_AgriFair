"""Base competence of the frozen subject models on the AgriFacts test set.

Greedy, temperature-0, deterministic JSON answer extraction (Instruction.md Section 9).
Writes a per-model summary and per-item predictions. Resumable and offline.

Run:  python GPU_Run/measure_base_competence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from GPU_Run.common import metrics as M
from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.inference import run_mcq_eval
from GPU_Run.common.logging_utils import append_jsonl, get_logger, write_csv
from GPU_Run.common.paths import RESULTS_DIR, TEST_INSTANCES_FROZEN
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("measure_base_competence")

SUMMARY_COLUMNS = [
    "tier",
    "hf_id",
    "attention_implementation_used",
    "model_revision_hash",
    "overall_accuracy",
    "accuracy_on_neq_condition",
    "accuracy_on_eq_condition",
    "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy",
    "json_parse_failure_rate_percent",
]


def _subset(records):
    n = int(os.environ.get("BASE_COMPETENCE_SUBSET_SIZE", "0") or 0)
    return records[:n] if n > 0 else records


def main(smoke: bool = False):
    set_global_determinism()
    test = _subset(read_jsonl(TEST_INSTANCES_FROZEN))
    if not test:
        raise SystemExit("Run build_template_instances.py first.")

    summary_rows = []
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        logger.info("Loading model tier=%s (smoke=%s)", tier, smoke)
        try:
            model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
        except Exception as e:
            logger.warning("Could not load %s (%s); skipping.", tier, e)
            continue
        preds = run_mcq_eval(model, tok, test, rationale_mode=False)
        pred_path = RESULTS_DIR / f"base_competence_predictions_{meta['tier']}.jsonl"
        if pred_path.exists():
            pred_path.unlink()
        for p in preds:
            append_jsonl(pred_path, p)
        row = {
            **{k: meta[k] for k in ("tier", "hf_id", "attention_implementation_used", "model_revision_hash")},
            "overall_accuracy": round(M.accuracy(preds), 4),
            "accuracy_on_neq_condition": round(M.condition_accuracy(preds, "neq"), 4),
            "accuracy_on_eq_condition": round(M.condition_accuracy(preds, "eq"), 4),
            "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy": round(M.contextual_fairness_score(preds), 4),
            "json_parse_failure_rate_percent": round(preds[0]["json_parse_failure_rate_percent"], 2) if preds else 0.0,
        }
        summary_rows.append(row)
        logger.info("tier=%s CFS=%.3f", meta["tier"], row["contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy"])
        del model

    write_csv(RESULTS_DIR / "base_competence_summary.csv", summary_rows, SUMMARY_COLUMNS)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
