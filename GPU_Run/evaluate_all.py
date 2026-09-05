"""Evaluate every method on the AgriFacts frozen test set and write the central results
table.

For the frozen base, every trained adapter (per seed), and the inference-time steering
baseline, this computes the full metric set per (model, method, seed, scope, axis, form):
accuracies, the balanced awareness score, Wang DiffAware/CtxtAware, the two failure
directions, identity-swap invariance and equivariance, option-rotation robustness,
external capability retention, the rationale metrics, and the parse-failure rate.

Three things this stage is careful about:
  * Capability retention is measured on the external probe built by Dataset_Prep (an MMLU
    subset), not on the in-domain validation split, so "preserve the rest" is not scored
    on the training distribution. The column records which probe was used.
  * Option-rotation robustness re-runs a subset under cyclic rotations of the displayed
    option order. A method whose canonical answer moves when the same three options are
    merely reordered is reading position, not content.
  * A leave-one-axis-out arm is evaluated only on its held-out axis, which is the slice it
    never saw in training.

Answer extraction is deterministic JSON parsing; the judge is used only for rationale
factual-correctness on a stratified subsample and for the rare extraction fallback.
Resumable: an existing per-item prediction file is reused unless FORCE_EVAL=1.

Run:  python GPU_Run/evaluate_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import time
from typing import Dict, List, Optional, Sequence

from GPU_Run.common import methods as METHODS
from GPU_Run.common import metrics as M
from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common import targets as TG
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.inference import run_capability_eval, run_mcq_eval
from GPU_Run.common.logging_utils import append_csv_row, append_jsonl, get_logger, log_run_metadata
from GPU_Run.common.paths import (
    COUNTERFACTUAL_PAIRS,
    EXTERNAL_CAPABILITY_PROBE,
    MAIN_EVALUATION,
    RESULTS_DIR,
    TEST_INSTANCES_FROZEN,
    VALIDATION_INSTANCES,
    split_loao_method,
)
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("evaluate_all")

RESULT_COLUMNS = [
    "tier", "hf_id", "parameter_count_billions", "method", "base_method", "held_out_axis",
    "random_seed", "scope", "category", "form",
    "attention_implementation_used", "model_class_used", "model_revision_hash", "quantization_setting",
    "number_of_items",
    "overall_accuracy", "accuracy_on_diff_condition", "accuracy_on_equal_condition",
    "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy",
    "diff_minus_equal_accuracy_gap",
    "difference_aware_metric_wang_2025_recall_style",
    "contextual_awareness_metric_wang_2025_precision_style",
    "capability_retention_accuracy_on_external_probe",
    "capability_probe_source_name",
    "gap_erasure_rate_on_diff_items",
    "gap_fabrication_rate_on_equal_items",
    "failure_polarity_index_positive_means_gap_erasure",
    "dominant_failure_polarity_label",
    "identity_swap_unlicensed_flip_rate",
    "identity_swap_invariance_rate_on_equal_items",
    "identity_swap_equivariance_rate_on_diff_items",
    "option_rotation_mean_accuracy",
    "option_rotation_canonical_consistency_rate",
    "option_rotation_position_following_rate",
    "rationale_bleu4_against_reference", "rationale_rouge_l_against_reference",
    "rationale_census_cell_citation_preservation_rate",
    "rationale_factual_correctness_score_one_to_five_via_judge",
    "trainable_parameter_percentage", "json_parse_failure_rate_percent",
    "answers_recovered_by_judge", "answers_left_unresolved",
    "api_judge_model_string",
]

SWAP_SUBSET_SIZE = int(os.environ.get("SWAP_SUBSET_SIZE", "200"))
# The identity-swap and option-rotation sweeps run on every seed by default, so their numbers
# carry the same variance estimate as the primary endpoint and a reviewer asking whether a
# flip rate or a rotation consistency is stable across seeds can be answered from the results.
# BEHAVIOURAL_SWEEP_SCOPE=first_seed narrows them to one seed per method, which is a declared
# reduction recorded in PREREGISTRATION.md, not a default.
BEHAVIOURAL_SWEEPS_ON_EVERY_SEED = os.environ.get("BEHAVIOURAL_SWEEP_SCOPE", "all") == "all"
ROTATION_SUBSET_SIZE = int(os.environ.get("ROTATION_SUBSET_SIZE", "150"))
CAPABILITY_SUBSET_SIZE = int(os.environ.get("CAPABILITY_SUBSET_SIZE", "0"))


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


def _subset(records, env_name="EVAL_SUBSET_SIZE"):
    n = int(os.environ.get(env_name, "0") or 0)
    return _balanced_subset(records, n) if n > 0 else records


def _force_eval() -> bool:
    return os.environ.get("FORCE_EVAL", "0") == "1"


def _capability_items():
    """(items, source_name). The external probe when present, else the in-domain
    validation split, which is recorded honestly as a weaker fallback."""
    if EXTERNAL_CAPABILITY_PROBE.exists():
        items = read_jsonl(EXTERNAL_CAPABILITY_PROBE)
        if items:
            if CAPABILITY_SUBSET_SIZE > 0:
                items = items[:CAPABILITY_SUBSET_SIZE]
            return items, items[0].get("source_name", "external_probe")
    return _subset(read_jsonl(VALIDATION_INSTANCES)), "in_domain_validation_split_fallback"


def _swap_records(test_records):
    test_ids = {r["id"] for r in test_records}
    pairs = read_jsonl(COUNTERFACTUAL_PAIRS)
    selected = [p for p in pairs if p.get("original", {}).get("id") in test_ids]
    if SWAP_SUBSET_SIZE > 0:
        selected = selected[:SWAP_SUBSET_SIZE]
    return [p["original"] for p in selected], [p["swapped"] for p in selected]


def _rationale_metrics(preds, judge, judge_fraction):
    rationale_preds = [p for p in preds if p.get("generated_rationale")]
    if not rationale_preds:
        return {}
    bleu = sum(M.bleu4(p["reference_rationale"], p["generated_rationale"]) for p in rationale_preds) / len(rationale_preds)
    rouge = sum(M.rouge_l(p["reference_rationale"], p["generated_rationale"]) for p in rationale_preds) / len(rationale_preds)
    cit = [M.citation_preservation_rate(p["law_reference"], p["generated_rationale"]) for p in rationale_preds]
    cit = [c for c in cit if c == c]
    fact = float("nan")
    if judge is not None and judge.available() and judge_fraction > 0:
        import random

        rng = random.Random(42)
        sample = rng.sample(rationale_preds, max(1, int(judge_fraction * len(rationale_preds))))
        scores = []
        for p in sample:
            s = judge.score_int(P.judge_rationale_factual_prompt(
                p.get("question") or p["id"], p["reference_rationale"], p["generated_rationale"]),
                "factual_correctness_score", 1, 5)
            if s is not None:
                scores.append(s)
        fact = sum(scores) / len(scores) if scores else float("nan")
    return {
        "rationale_bleu4_against_reference": round(bleu, 4),
        "rationale_rouge_l_against_reference": round(rouge, 4),
        "rationale_census_cell_citation_preservation_rate": round(sum(cit) / len(cit), 4) if cit else "",
        "rationale_factual_correctness_score_one_to_five_via_judge": round(fact, 3) if fact == fact else "",
    }


def _num(v, nd=4):
    return round(float(v), nd) if v == v else ""


def _emit_rows(preds, method, seed, base_fields, extras, rationale, per_axis=True) -> List[Dict]:
    rows = []
    groupings = [("all", None, None)]
    if per_axis:
        for cat in sorted({p.get("category") for p in preds if p.get("category")}):
            groupings.append((f"axis:{cat}", "category", cat))
        for frm in sorted({p.get("form") for p in preds if p.get("form")}):
            groupings.append((f"form:{frm}", "form", frm))
    base_method, held_out = split_loao_method(method)
    for gname, key, val in groupings:
        sub = preds if key is None else [p for p in preds if p.get(key) == val]
        if not sub:
            continue
        wang = M.wang_contingency(sub)
        polarity = M.failure_polarity_index(sub)
        overall = gname == "all"
        rows.append({
            **base_fields,
            "method": method,
            "base_method": base_method,
            "held_out_axis": held_out or "",
            "random_seed": seed,
            "scope": f"{extras['scope_prefix']}|{gname}",
            "category": val if key == "category" else "all",
            "form": val if key == "form" else "all",
            "number_of_items": len(sub),
            "overall_accuracy": _num(M.accuracy(sub)),
            "accuracy_on_diff_condition": _num(M.condition_accuracy(sub, "diff")),
            "accuracy_on_equal_condition": _num(M.condition_accuracy(sub, "equal")),
            "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy": _num(M.balanced_awareness_score(sub)),
            "diff_minus_equal_accuracy_gap": _num(M.diff_minus_equal_accuracy_gap(sub)),
            "difference_aware_metric_wang_2025_recall_style": _num(wang["difference_aware_metric_wang_2025_recall_style"]),
            "contextual_awareness_metric_wang_2025_precision_style": _num(wang["contextual_awareness_metric_wang_2025_precision_style"]),
            "capability_retention_accuracy_on_external_probe": _num(extras["capability_accuracy"]) if overall else "",
            "capability_probe_source_name": extras["capability_source"] if overall else "",
            "gap_erasure_rate_on_diff_items": _num(M.gap_erasure_rate(sub)),
            "gap_fabrication_rate_on_equal_items": _num(M.gap_fabrication_rate(sub)),
            "failure_polarity_index_positive_means_gap_erasure": _num(polarity),
            "dominant_failure_polarity_label": M.polarity_label(polarity),
            "identity_swap_unlicensed_flip_rate": _num(extras["swap_flip"]) if overall else "",
            "identity_swap_invariance_rate_on_equal_items": _num(extras["swap_invariance"]) if overall else "",
            "identity_swap_equivariance_rate_on_diff_items": _num(extras["swap_equivariance"]) if overall else "",
            "option_rotation_mean_accuracy": _num(extras["rotation"]["rotation_mean_accuracy"]) if overall else "",
            "option_rotation_canonical_consistency_rate": _num(extras["rotation"]["rotation_consistency_rate"]) if overall else "",
            "option_rotation_position_following_rate": _num(extras["rotation"]["rotation_position_following_rate"]) if overall else "",
            "json_parse_failure_rate_percent": round(sub[0].get("json_parse_failure_rate_percent", 0.0), 2),
            "answers_recovered_by_judge": sub[0].get("answers_recovered_by_judge", 0) if overall else "",
            "answers_left_unresolved": sum(1 for p in sub if p["pred_canonical"] == "unparsed"),
            "api_judge_model_string": extras.get("judge_model", ""),
            **rationale,
        })
    return rows


def _trainable_percentages() -> Dict:
    out = {}
    for fname in ("train_graft_runs.json", "train_baselines_runs.json"):
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        try:
            for r in json.loads(p.read_text(encoding="utf-8")):
                pct = r.get("trainable_parameter_percentage")
                if pct is not None:
                    out[(r.get("tier"), r.get("method"))] = float(pct)
        except Exception:
            continue
    return out


def _clear_stale_predictions(label: str, expected):
    from GPU_Run.common.paths import parse_per_item_prediction_path

    for path in sorted(RESULTS_DIR.glob(f"per_item_predictions_{label}_*.jsonl")):
        parsed = parse_per_item_prediction_path(path)
        if parsed is None:
            continue
        tier, method, seed = parsed
        if tier != label:
            continue
        if (method, seed) not in expected:
            # Archive rather than delete. These files are the output of paid GPU time, and an
            # arm counts as "retired" whenever the current run has a narrower scope than the
            # one that produced them, which a scope environment variable is enough to cause.
            retired = RESULTS_DIR / "retired_predictions"
            retired.mkdir(exist_ok=True)
            destination = retired / path.name
            if destination.exists():
                # Never overwrite: the point of archiving is to keep paid GPU output.
                stamp = time.strftime("%Y%m%dT%H%M%S")
                destination = retired / f"{path.stem}.{stamp}{path.suffix}"
            path.replace(destination)
            logger.info("Archived prediction file for arm %s not in this run's scope: %s -> %s/",
                        method, path.name, retired.name)


def _reusable_predictions(pred_file, test, target=None):
    """Cached predictions, but only if they cover exactly the items about to be evaluated.

    The filename records tier, method and seed, not which items were scored. A rehearsal run
    over a handful of items therefore produced a file that the real study would happily reuse,
    publishing a table computed from twelve predictions. Comparing the item ids closes that,
    and also rejects a file truncated by an interrupted run."""
    if not pred_file.exists():
        return None
    try:
        preds = read_jsonl(pred_file)
    except Exception as e:
        logger.warning("Cached predictions at %s are unreadable (%s); re-evaluating.",
                       pred_file.name, type(e).__name__)
        return None
    want = {r["id"] for r in test}
    got = {p.get("id") for p in preds}
    if got == want and len(preds) != len(want):
        # A set comparison alone accepts a file whose every item appears twice, which would
        # silently double every count the metrics are built from.
        logger.warning("Discarding cached predictions at %s: %d rows for %d items, so some "
                       "items are duplicated. Re-evaluating.", pred_file.name, len(preds), len(want))
        return None
    if got == want:
        # The ids can match exactly and still describe a different model: a verification pass
        # that evaluated a two-step rehearsal adapter over the full test set leaves such a
        # file. Reject anything older than the adapter that is about to be loaded, and
        # anything produced from a rehearsal.
        adapter_dir = getattr(target, "adapter_dir", None) if target is not None else None
        if adapter_dir is not None:
            weights = Path(adapter_dir) / "adapter_model.safetensors"
            try:
                if weights.exists() and weights.stat().st_mtime > pred_file.stat().st_mtime:
                    logger.warning("Discarding cached predictions at %s: the adapter was "
                                   "retrained after they were written. Re-evaluating.", pred_file.name)
                    return None
            except OSError:
                pass
            summary = Path(adapter_dir).parent / "train_summary.json"
            try:
                if summary.exists() and json.loads(
                        summary.read_text(encoding="utf-8")).get("step_capped_rehearsal"):
                    logger.warning("Discarding cached predictions at %s: they come from a "
                                   "step-capped rehearsal adapter. Re-evaluating.", pred_file.name)
                    return None
            except Exception:
                pass
        return preds
    logger.warning(
        "Discarding cached predictions at %s: they cover %d items but this evaluation needs "
        "%d (%d missing, %d unexpected). Re-evaluating.",
        pred_file.name, len(got), len(want), len(want - got), len(got - want))
    return None


def main(smoke: bool = False):
    set_global_determinism()
    test_all = _subset(read_jsonl(TEST_INSTANCES_FROZEN))
    capability_items, capability_source = _capability_items()
    swap_originals, swap_swapped = _swap_records(test_all)
    judge_fraction = float(os.environ.get("RATIONALE_JUDGE_FRACTION", "0.1"))
    judge = JudgeChain()
    trainable = _trainable_percentages()
    if MAIN_EVALUATION.exists():
        MAIN_EVALUATION.unlink()

    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        targets = TG.discover_targets(label)
        _clear_stale_predictions(label, {t.key for t in targets})
        for target in targets:
            method, seed = target.method, target.seed
            base_method, held_out = split_loao_method(method)
            test = [r for r in test_all if r.get("category") == held_out] if held_out else test_all
            if not test:
                logger.warning("No test items for %s (held-out axis %s); skipping.", method, held_out)
                continue
            pred_file = RESULTS_DIR / f"per_item_predictions_{label}_{method}_seed{seed}.jsonl"
            try:
                model, tok, meta = TG.load_target(tier, target, smoke=smoke)
            except Exception as e:
                logger.error("Eval load failed tier=%s method=%s (%s); skipping.", label, method, e)
                continue
            try:
                rationale_mode = METHODS.uses_rationale_arm(base_method)
                preds = _reusable_predictions(pred_file, test, target) if not _force_eval() else None
                if preds is not None:
                    logger.info("Reusing cached predictions for tier=%s method=%s seed=%d.", label, method, seed)
                else:
                    preds = run_mcq_eval(model, tok, test, rationale_mode=rationale_mode, judge=judge)
                    if pred_file.exists():
                        pred_file.unlink()
                    for p in preds:
                        append_jsonl(pred_file, p)

                cap = run_capability_eval(model, tok, capability_items, judge=judge) if capability_items and "external" in capability_source \
                    else run_mcq_eval(model, tok, capability_items, rationale_mode=False, judge=judge)
                capability_accuracy = (sum(c["correct"] for c in cap) / len(cap)) if cap and "correct" in cap[0] \
                    else M.accuracy(cap)

                first_seed_of_method = BEHAVIOURAL_SWEEPS_ON_EVERY_SEED or seed == min(
                    s for m, s in {t.key for t in targets} if m == method)
                swap_flip = swap_invariance = swap_equivariance = float("nan")
                if swap_originals and not held_out and first_seed_of_method:
                    try:
                        po = run_mcq_eval(model, tok, swap_originals, rationale_mode=False, judge=judge)
                        ps = run_mcq_eval(model, tok, swap_swapped, rationale_mode=False, judge=judge)
                        swap_flip = M.identity_swap_flip_rate(po, ps)
                        swap_invariance = M.identity_swap_invariance_rate(po, ps)
                        swap_equivariance = M.identity_swap_equivariance_rate(po, ps)
                    except Exception as e:
                        logger.warning("Identity-swap sweep failed (%s); left blank.", e)

                rotation = {"rotation_mean_accuracy": float("nan"), "rotation_consistency_rate": float("nan"),
                            "rotation_position_following_rate": float("nan")}
                if ROTATION_SUBSET_SIZE > 0 and not held_out and first_seed_of_method:
                    rot_items = test[:ROTATION_SUBSET_SIZE]
                    try:
                        # Rotations 1 and 2 are scored in answer mode, so rotation 0 must be
                        # too. Reusing the main predictions here would compare a rationale-mode
                        # baseline against answer-mode rotations for every rationale arm, and
                        # the "option-order consistency" number would mostly measure the change
                        # of prompt format instead of the change of option order.
                        if rationale_mode:
                            by_rot = {0: run_mcq_eval(model, tok, rot_items, rationale_mode=False,
                                                      rotation=0, judge=judge)}
                        else:
                            rot_ids = {r["id"] for r in rot_items}
                            by_rot = {0: [p for p in preds if p["id"] in rot_ids]}
                        for k in (1, 2):
                            by_rot[k] = run_mcq_eval(model, tok, rot_items, rationale_mode=False,
                                                     rotation=k, judge=judge)
                        rotation = M.option_rotation_consistency(by_rot)
                        for k in (1, 2):
                            rp = RESULTS_DIR / f"option_rotation_predictions_{label}_{method}_seed{seed}_rot{k}.jsonl"
                            if rp.exists():
                                rp.unlink()
                            for p in by_rot[k]:
                                append_jsonl(rp, p)
                    except Exception as e:
                        logger.warning("Option-rotation sweep failed (%s); left blank.", e)

                base_fields = {
                    "tier": meta["tier"], "hf_id": meta["hf_id"],
                    "parameter_count_billions": meta.get("parameter_count_billions", ""),
                    "attention_implementation_used": meta["attention_implementation_used"],
                    "model_class_used": meta.get("model_class_used", ""),
                    "model_revision_hash": meta["model_revision_hash"],
                    "quantization_setting": meta["quantization_setting"],
                    "trainable_parameter_percentage": round(trainable.get((label, method), float("nan")), 5)
                    if trainable.get((label, method)) is not None else "",
                }
                extras = {
                    "scope_prefix": f"held_out_axis:{held_out}" if held_out else "agrifacts_frozen_test",
                    "capability_accuracy": capability_accuracy,
                    "capability_source": capability_source,
                    "swap_flip": swap_flip, "swap_invariance": swap_invariance,
                    "swap_equivariance": swap_equivariance, "rotation": rotation,
                    "judge_model": judge.last_model_string if judge else "",
                }
                rationale = _rationale_metrics(preds, judge, judge_fraction)
                for row in _emit_rows(preds, method, seed, base_fields, extras, rationale, per_axis=not held_out):
                    append_csv_row(MAIN_EVALUATION, row, RESULT_COLUMNS)
                logger.info("Evaluated tier=%s method=%s seed=%d balanced_awareness=%.3f erasure=%.3f fabrication=%.3f",
                            label, method, seed, M.balanced_awareness_score(preds),
                            M.gap_erasure_rate(preds), M.gap_fabrication_rate(preds))
            except Exception as e:
                # One target failing must not abort the stage. With 33 arms per tier across
                # four tiers, an error on a single arm would otherwise throw away every
                # evaluation still queued behind it.
                logger.error("Evaluation failed for tier=%s method=%s seed=%d (%s); its rows "
                             "are omitted and the remaining targets continue.",
                             label, method, seed, e, exc_info=True)
            finally:
                TG.remove_steering(model)
                del model
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

    log_run_metadata("evaluate_all", {"output": str(MAIN_EVALUATION), "capability_probe": capability_source})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
