"""Base competence of the frozen subject models on the AgriFacts test set.

Greedy, temperature-0, deterministic JSON answer extraction. Writes a per-model summary
and per-item predictions, including the two failure directions on the frozen base, which
are what the failure-polarity profile is built from. Resumable and offline.

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
from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.inference import run_mcq_eval
from GPU_Run.common.logging_utils import append_jsonl, get_logger, write_csv
from GPU_Run.common.paths import RESULTS_DIR, TEST_INSTANCES_FROZEN
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("measure_base_competence")

SUMMARY_COLUMNS = [
    "tier",
    "hf_id",
    "parameter_count_billions",
    "attention_implementation_used",
    "model_class_used",
    "model_revision_hash",
    "overall_accuracy",
    "accuracy_on_diff_condition",
    "accuracy_on_equal_condition",
    "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy",
    "gap_erasure_rate_on_diff_items",
    "gap_fabrication_rate_on_equal_items",
    "failure_polarity_index_positive_means_gap_erasure",
    "dominant_failure_polarity_label",
    "json_parse_failure_rate_percent",
    "answers_recovered_by_judge",
    "answers_left_unresolved",
]


def _balanced_subset(records, n):
    """First n records, but guaranteeing both conditions appear when n allows.

    The verification pass runs with a handful of items. Taking a plain prefix can yield items
    of a single condition, and then gap erasure (a "diff" item answered "roughly equal") or
    gap fabrication (an "equal" item answered with a group) is never computed, so the very
    code paths the verification exists to exercise stay untested while it reports success."""
    if n <= 0 or n >= len(records):
        return records
    picked, seen = [], set()
    for r in records:                      # one of each condition first
        c = r.get("condition")
        if c not in seen:
            seen.add(c)
            picked.append(r)
        if len(picked) >= n:
            break
    for r in records:                      # then fill in original order
        if len(picked) >= n:
            break
        if r not in picked:
            picked.append(r)
    order = {id(r): i for i, r in enumerate(records)}
    return sorted(picked[:n], key=lambda r: order[id(r)])


def _subset(records):
    n = int(os.environ.get("BASE_COMPETENCE_SUBSET_SIZE", "0") or 0)
    return _balanced_subset(records, n) if n > 0 else records


def _shown(v):
    """Render a metric for the log, keeping a real 0.0 distinct from an undefined value.

    `v or "undefined"` is wrong here: 0.0 is falsy, so a model that genuinely scored zero
    would be reported as if the metric could not be computed."""
    return "undefined" if v == "" else f"{float(v):.3f}"


def _num(v, nd=4):
    """Blank for an undefined value, so it is never mistaken for a real zero."""
    return round(float(v), nd) if v == v else ""


def main(smoke: bool = False):
    set_global_determinism()
    test = _subset(read_jsonl(TEST_INSTANCES_FROZEN))
    if not test:
        raise SystemExit("Run build_template_instances.py first.")

    judge = JudgeChain()
    if judge.available():
        logger.info("Answer-extraction judge chain: %s", " -> ".join(judge.route_names()))
    summary_rows = []
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        logger.info("Loading model tier=%s (smoke=%s)", tier, smoke)
        try:
            model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
        except Exception as e:
            logger.warning("Could not load %s (%s); skipping.", tier, e)
            continue
        try:
            preds = run_mcq_eval(model, tok, test, rationale_mode=False, judge=judge)
            pred_path = RESULTS_DIR / f"base_competence_predictions_{meta['tier']}.jsonl"
            if pred_path.exists():
                pred_path.unlink()
            for p in preds:
                append_jsonl(pred_path, p)
            polarity = M.failure_polarity_index(preds)
            row = {
                **{k: meta.get(k, "") for k in ("tier", "hf_id", "parameter_count_billions",
                                                "attention_implementation_used", "model_class_used",
                                                "model_revision_hash")},
                "overall_accuracy": round(M.accuracy(preds), 4),
                "accuracy_on_diff_condition": round(M.condition_accuracy(preds, "diff"), 4),
                "accuracy_on_equal_condition": round(M.condition_accuracy(preds, "equal"), 4),
                "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy": _num(M.balanced_awareness_score(preds)),
                "gap_erasure_rate_on_diff_items": round(M.gap_erasure_rate(preds), 4),
                "gap_fabrication_rate_on_equal_items": round(M.gap_fabrication_rate(preds), 4),
                "failure_polarity_index_positive_means_gap_erasure": round(polarity, 4) if polarity == polarity else "",
                "dominant_failure_polarity_label": M.polarity_label(polarity),
                "json_parse_failure_rate_percent": round(preds[0]["json_parse_failure_rate_percent"], 2) if preds else 0.0,
                "answers_recovered_by_judge": preds[0].get("answers_recovered_by_judge", 0) if preds else 0,
                "answers_left_unresolved": sum(1 for p in preds if p["pred_canonical"] == "unparsed"),
            }
            summary_rows.append(row)
            logger.info("tier=%s balanced awareness=%s erasure=%.3f fabrication=%.3f (%s)",
                        meta["tier"], _shown(row["balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy"]),
                        row["gap_erasure_rate_on_diff_items"], row["gap_fabrication_rate_on_equal_items"],
                        row["dominant_failure_polarity_label"])
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
        write_csv(RESULTS_DIR / "base_competence_summary.csv", summary_rows, SUMMARY_COLUMNS)

    write_csv(RESULTS_DIR / "base_competence_summary.csv", summary_rows, SUMMARY_COLUMNS)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
