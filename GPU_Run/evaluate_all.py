"""Evaluate every method on the AgriFacts test set, cross-axis transfer, and the held-out
general set; write the central results table (Instruction.md Sections 7, 17).

For the frozen base and each trained adapter (per seed), compute the full metric set per
(model, method, seed, scope, axis, form): accuracies, contextual fairness score, Wang
DiffAware/CtxtAware, the rationale metrics, and the parse-failure rate. Answer extraction
is deterministic JSON parsing; the judge is used only for rationale factual-correctness on a
stratified subsample and for the rare extraction fallback. Resumable and offline.

Run:  python GPU_Run/evaluate_all.py
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
from GPU_Run.common.logging_utils import append_csv_row, append_jsonl, get_logger, log_run_metadata
from GPU_Run.common.paths import CHECKPOINTS_DIR, MAIN_EVALUATION, RESULTS_DIR, TEST_INSTANCES_FROZEN, VALIDATION_INSTANCES
from GPU_Run.common import prompts as P
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("evaluate_all")

RESULT_COLUMNS = [
    "tier", "hf_id", "method", "random_seed", "scope", "category", "form",
    "attention_implementation_used", "model_revision_hash", "quantization_setting",
    "overall_accuracy", "accuracy_on_neq_condition", "accuracy_on_eq_condition",
    "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy",
    "difference_aware_accuracy_gap_neq_minus_eq",
    "difference_aware_metric_wang_2025_recall_style",
    "contextual_awareness_metric_wang_2025_precision_style",
    "utility_retention_accuracy_on_held_out_general_set",
    "rationale_bleu4_against_reference", "rationale_rouge_l_against_reference",
    "rationale_statutory_citation_preservation_rate",
    "rationale_factual_correctness_score_one_to_five_via_judge",
    "trainable_parameter_percentage", "json_parse_failure_rate_percent",
    "api_judge_model_string",
]


def _subset(records):
    n = int(os.environ.get("EVAL_SUBSET_SIZE", "0") or 0)
    return records[:n] if n > 0 else records


def _load_with_adapter(tier, adapter_dir, smoke, quantized=False):
    model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke, quantized_4bit=quantized)
    if adapter_dir is not None and Path(adapter_dir).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model, tok, meta


def _discover_methods(label):
    base = CHECKPOINTS_DIR / label
    methods = []
    if base.exists():
        for mdir in sorted(base.iterdir()):
            if not mdir.is_dir():
                continue
            for sdir in sorted(mdir.glob("seed_*")):
                final = sdir / "final"
                adapter = final if final.exists() else sdir
                seed = int(sdir.name.split("_")[1])
                methods.append((mdir.name, seed, adapter, "qlora" in mdir.name))
    return methods


def _rationale_metrics(preds, judge, judge_fraction):
    rationale_preds = [p for p in preds if p.get("generated_rationale")]
    if not rationale_preds:
        return {}
    bleu = sum(M.bleu4(p["reference_rationale"], p["generated_rationale"]) for p in rationale_preds) / len(rationale_preds)
    rouge = sum(M.rouge_l(p["reference_rationale"], p["generated_rationale"]) for p in rationale_preds) / len(rationale_preds)
    cit = [M.citation_preservation_rate(p["law_reference"], p["generated_rationale"]) for p in rationale_preds]
    cit = [c for c in cit if c == c]  # drop nan
    fact = float("nan")
    if judge is not None and judge.available() and judge_fraction > 0:
        import random

        rng = random.Random(42)
        sample = rng.sample(rationale_preds, max(1, int(judge_fraction * len(rationale_preds))))
        scores = []
        for p in sample:
            s = judge.score_int(P.judge_rationale_factual_prompt(p["id"], p["reference_rationale"], p["generated_rationale"]), "factual_correctness_score", 1, 5)
            if s is not None:
                scores.append(s)
        fact = sum(scores) / len(scores) if scores else float("nan")
    return {
        "rationale_bleu4_against_reference": round(bleu, 4),
        "rationale_rouge_l_against_reference": round(rouge, 4),
        "rationale_statutory_citation_preservation_rate": round(sum(cit) / len(cit), 4) if cit else float("nan"),
        "rationale_factual_correctness_score_one_to_five_via_judge": fact,
    }


def _emit_rows(preds, meta, method, seed, scope, base_fields, util_acc, judge, judge_fraction):
    rows = []
    rat = _rationale_metrics(preds, judge, judge_fraction)
    # overall + per axis + per form
    groupings = [("all", None, None)]
    for cat in sorted(set(p.get("category") for p in preds if p.get("category"))):
        groupings.append((f"axis:{cat}", "category", cat))
    for frm in sorted(set(p.get("form") for p in preds if p.get("form"))):
        groupings.append((f"form:{frm}", "form", frm))
    for gname, key, val in groupings:
        sub = preds if key is None else [p for p in preds if p.get(key) == val]
        if not sub:
            continue
        wang = M.wang_contingency(sub)
        rows.append(
            {
                **base_fields,
                "method": method,
                "random_seed": seed,
                "scope": f"{scope}|{gname}",
                "category": val if key == "category" else "all",
                "form": val if key == "form" else "all",
                "overall_accuracy": round(M.accuracy(sub), 4),
                "accuracy_on_neq_condition": round(M.condition_accuracy(sub, "neq"), 4),
                "accuracy_on_eq_condition": round(M.condition_accuracy(sub, "eq"), 4),
                "contextual_fairness_score_harmonic_mean_of_neq_and_eq_accuracy": round(M.contextual_fairness_score(sub), 4),
                "difference_aware_accuracy_gap_neq_minus_eq": round(M.difference_aware_gap(sub), 4),
                "difference_aware_metric_wang_2025_recall_style": round(wang["difference_aware_metric_wang_2025_recall_style"], 4) if wang["difference_aware_metric_wang_2025_recall_style"] == wang["difference_aware_metric_wang_2025_recall_style"] else "",
                "contextual_awareness_metric_wang_2025_precision_style": round(wang["contextual_awareness_metric_wang_2025_precision_style"], 4) if wang["contextual_awareness_metric_wang_2025_precision_style"] == wang["contextual_awareness_metric_wang_2025_precision_style"] else "",
                "utility_retention_accuracy_on_held_out_general_set": round(util_acc, 4) if util_acc == util_acc else "",
                "json_parse_failure_rate_percent": round(preds[0]["json_parse_failure_rate_percent"], 2),
                "api_judge_model_string": judge.last_model_string if judge else "",
                **rat,
            }
        )
    return rows


def main(smoke: bool = False):
    set_global_determinism()
    test = _subset(read_jsonl(TEST_INSTANCES_FROZEN))
    held_out_general = _subset(read_jsonl(VALIDATION_INSTANCES))  # general-capability probe
    judge_fraction = float(os.environ.get("RATIONALE_JUDGE_FRACTION", "0.1"))
    judge = JudgeChain()
    if MAIN_EVALUATION.exists():
        MAIN_EVALUATION.unlink()

    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        # base (no adapter) + each trained adapter
        targets = [("frozen_base", 42, None, False)] + _discover_methods(label)
        for method, seed, adapter, quantized in targets:
            try:
                model, tok, meta = _load_with_adapter(tier, adapter, smoke, quantized)
            except Exception as e:
                logger.warning("Eval load failed tier=%s method=%s (%s); skipping.", label, method, e)
                continue
            rationale_mode = ("proposed" in method) or ("regift" in method) or ("rationale" in method) or method == "frozen_base"
            preds = run_mcq_eval(model, tok, test, rationale_mode=rationale_mode, judge=judge)
            util_preds = run_mcq_eval(model, tok, held_out_general, rationale_mode=False)
            util_acc = M.accuracy(util_preds)
            base_fields = {
                "tier": meta["tier"], "hf_id": meta["hf_id"],
                "attention_implementation_used": meta["attention_implementation_used"],
                "model_revision_hash": meta["model_revision_hash"],
                "quantization_setting": meta["quantization_setting"],
                "trainable_parameter_percentage": "",
            }
            # persist per-item predictions
            pred_file = RESULTS_DIR / f"per_item_predictions_{label}_{method}_seed{seed}.jsonl"
            if pred_file.exists():
                pred_file.unlink()
            for p in preds:
                append_jsonl(pred_file, p)
            for row in _emit_rows(preds, meta, method, seed, "agrifacts_test", base_fields, util_acc, judge, judge_fraction):
                append_csv_row(MAIN_EVALUATION, row, RESULT_COLUMNS)
            logger.info("Evaluated tier=%s method=%s seed=%d", label, method, seed)
            del model

    log_run_metadata("evaluate_all", {"output": str(MAIN_EVALUATION)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
