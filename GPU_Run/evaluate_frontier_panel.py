"""Evaluate the frozen frontier panel: hosted models, behaviour only.

These models cannot be repaired, so they are not baselines and they never appear in the main
comparison table. They answer the questions that need only behaviour, and those questions are
the ones that do not depend on how GRAFT ranks:

  * Which direction does a frontier model fail in? Gap erasure, or gap fabrication? Wang et
    al. and DART disagree about the skew, and every measurement so far in this study is on a
    3B to 12B open model.
  * How much of a frontier score does the state-blind prior explain? A model that never saw
    the training split, scoring near the surface-cue ceiling with errors that track the
    prior, is the sharpest available evidence that the benchmark measures structure rather
    than census knowledge. Scoring well above it on the structure-novel slice is evidence of
    real headroom.
  * Does agronomic advice move with the farmer's identity? An advisory system in India is far
    more likely to sit on a hosted model than on a local 3B one, so this is the panel's most
    deployment-relevant number.
  * Does the answer survive a rotation of the option order?

Per-item predictions are written in the same schema the local models use, under tier names
that carry no underscore, so `CPU_Run/audit_awareness_trade.py` picks the panel up and
extends the failure-polarity profile to hosted models with no further wiring. Because each
panel row is recorded as `frozen_base`, the awareness-trade audit correctly produces no
comparison rows for them: there is nothing to compare a frozen hosted model against.

Every call goes through GPU_Run/common/api_models.py, which enforces the fallback chain with
no retries. Results are cached per item, so an interrupted run resumes without paying twice.

Run:  RUN_FRONTIER_PANEL=1 python GPU_Run/evaluate_frontier_panel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np

from GPU_Run.common import metrics as M
from GPU_Run.common import prompts as P
from GPU_Run.common.api_models import (
    REGISTRY,
    ApiRouter,
    active_api_models,
    advice_token_budget,
    answer_token_budget,
)
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.logging_utils import append_csv_row, append_jsonl, get_logger, log_run_metadata
from GPU_Run.common.parsing import extract_answer_letter, extract_rationale, fallback_letter_from_freetext
from GPU_Run.common.paths import (
    AGRIADVICE_PAIRS,
    COUNTERFACTUAL_PAIRS,
    EVAL_CACHE_DIR,
    RESULTS_DIR,
    TEST_INSTANCES_FROZEN,
)
from GPU_Run.common.seeds import GLOBAL_SEED, set_global_determinism

logger = get_logger("evaluate_frontier_panel")

ANSWER_MAX_TOKENS = int(os.environ.get("FRONTIER_ANSWER_MAX_TOKENS", "160"))
ADVICE_MAX_TOKENS = int(os.environ.get("FRONTIER_ADVICE_MAX_TOKENS", "512"))
MAX_WORKERS = int(os.environ.get("FRONTIER_MAX_WORKERS", "4"))
ROTATION_SUBSET = int(os.environ.get("FRONTIER_ROTATION_SUBSET_SIZE", "150"))
SWAP_SUBSET = int(os.environ.get("FRONTIER_SWAP_SUBSET_SIZE", "150"))

PANEL_COLUMNS = [
    "model_tier", "model_display_name", "vendor", "test_slice_name", "number_of_items",
    "overall_accuracy", "accuracy_on_diff_condition", "accuracy_on_equal_condition",
    "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy",
    "difference_aware_metric_wang_2025_recall_style",
    "contextual_awareness_metric_wang_2025_precision_style",
    "gap_erasure_rate_on_diff_items", "gap_fabrication_rate_on_equal_items",
    "failure_polarity_index_positive_means_gap_erasure", "dominant_failure_polarity_label",
    "surface_cue_majority_ceiling_for_this_slice",
    "accuracy_above_surface_cue_ceiling",
    "identity_swap_unlicensed_flip_rate",
    "identity_swap_invariance_rate_on_equal_items",
    "identity_swap_equivariance_rate_on_diff_items",
    "option_rotation_mean_accuracy", "option_rotation_canonical_consistency_rate",
    "option_rotation_position_following_rate",
    "json_parse_failure_rate_percent", "unanswered_after_all_routes_failed",
    "responses_truncated_before_the_answer", "answers_recovered_by_judge",
]

DRIFT_COLUMNS = [
    "model_tier", "model_display_name", "toggle_axis", "pairs_evaluated",
    "advice_drift_embedding_cosine_distance_mean",
    "advice_drift_content_word_jaccard_distance_mean",
    "advice_drift_structured_feature_l1_mean",
    "advice_drift_flip_rate_above_preregistered_threshold",
    "advice_drift_threshold_used",
]

ROUTE_COLUMNS = [
    "model_tier", "model_display_name", "bedrock_model_id", "openrouter_model_id",
    "requests_total", "requests_served_by_route", "failures_by_route",
    "input_tokens_total", "output_tokens_total", "mean_latency_seconds",
]


def _subset(rows, env_name, default=0):
    n = int(os.environ.get(env_name, str(default)) or 0)
    return rows[:n] if n > 0 else rows


def _cache_path(tier: str, kind: str) -> Path:
    return EVAL_CACHE_DIR / f"frontier_{tier}_{kind}.jsonl"


def _load_cache(path: Path) -> Dict[str, Dict]:
    return {row["cache_key"]: row for row in read_jsonl(path) if "cache_key" in row}


def _ask(router: ApiRouter, tier: str, prompt: str, max_tokens: int) -> Dict:
    """One prompt through the fallback chain, flattened into a cacheable record."""
    response = router.complete(tier, prompt, max_tokens)
    if response is None:
        return {"text": "", "api_route": "all_routes_failed", "input_tokens": 0,
                "output_tokens": 0, "latency_seconds": 0.0, "failed_routes": ""}
    return {"text": response.text, "api_route": response.route,
            "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
            "latency_seconds": round(response.latency_seconds, 3),
            "failed_routes": response.failed_routes}


def _run_prompts(router: ApiRouter, tier: str, jobs: List[Dict], cache_path: Path,
                 max_tokens: int) -> Dict[str, Dict]:
    """Execute the prompts that are not already cached, appending each result as it lands."""
    cache = _load_cache(cache_path)
    todo = [j for j in jobs if j["cache_key"] not in cache]
    if not todo:
        logger.info("  %s: all %d prompts already cached.", tier, len(jobs))
        return cache
    logger.info("  %s: %d of %d prompts already cached, %d to fetch.",
                tier, len(jobs) - len(todo), len(jobs), len(todo))

    def work(job):
        return job, _ask(router, tier, job["prompt"], max_tokens)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as pool:
        for job, result in pool.map(work, todo):
            row = {"cache_key": job["cache_key"], **result}
            append_jsonl(cache_path, row)
            cache[job["cache_key"]] = row
            done += 1
            if done % 50 == 0:
                logger.info("  %s: %d/%d fetched.", tier, done, len(todo))
    return cache


def _mcq_predictions(router: ApiRouter, tier: str, records: Sequence[Dict], rotation: int,
                     judge: Optional[JudgeChain] = None) -> List[Dict]:
    built = [P.build_mcq_prompt(r, rationale_mode=False, rotation=rotation) for r in records]
    jobs = [{"cache_key": f"{r['id']}::rot{rotation}", "prompt": b["prompt"]}
            for r, b in zip(records, built)]
    budget = answer_token_budget(tier, ANSWER_MAX_TOKENS)
    cache = _run_prompts(router, tier, jobs, _cache_path(tier, f"mcq_rot{rotation}"), budget)

    out, parse_failures, unanswered, truncated, judge_recoveries = [], 0, 0, 0, 0
    for record, b, job in zip(records, built, jobs):
        entry = cache.get(job["cache_key"], {})
        raw = entry.get("text", "")
        route = entry.get("api_route", "")
        if route == "all_routes_failed":
            unanswered += 1
        letter, parse_ok = extract_answer_letter(raw)
        answer_source = "deterministic_json_parse"
        if not parse_ok:
            parse_failures += 1
            # Same judge chain the local models use, so a parse failure is resolved the same
            # way everywhere. The judge sees the output alone, never the question or the gold.
            if judge is not None and judge.available() and raw.strip():
                letter = judge.extract_answer_letter(raw)
                if letter is not None:
                    judge_recoveries += 1
                    answer_source = f"judge:{judge.last_route}"
            if letter is None:
                letter = fallback_letter_from_freetext(raw)
                answer_source = "free_text_heuristic" if letter else "unresolved"
            # An empty body that used the whole allowance is a truncation, not a model that
            # answered badly. It is counted separately so the two are never conflated.
            if not raw.strip() and entry.get("output_tokens", 0) >= budget:
                truncated += 1
        canonical = b["display_to_canonical"].get(letter) if letter else None
        out.append({
            "id": record["id"], "condition": record["condition"],
            "category": record.get("category"), "form": record.get("form"),
            "gold_canonical": record["correct_answer"],
            "pred_canonical": canonical if canonical else "unparsed",
            "pred_display_letter": letter or "", "rotation": rotation,
            "raw_output": raw, "generated_rationale": "",
            "law_reference": record.get("law_reference", ""),
            "reference_rationale": record.get("rationale", ""),
            "parse_ok": parse_ok, "answer_extraction_source": answer_source,
            "api_route": route, "api_failed_routes": entry.get("failed_routes", ""),
            "input_tokens": entry.get("input_tokens", 0), "output_tokens": entry.get("output_tokens", 0),
            "latency_seconds": entry.get("latency_seconds", 0.0),
        })
    rate = 100.0 * parse_failures / max(1, len(out))
    for row in out:
        row["json_parse_failure_rate_percent"] = rate
        row["unanswered_after_all_routes_failed"] = unanswered
        row["responses_truncated_before_the_answer"] = truncated
        row["answers_recovered_by_judge"] = judge_recoveries
    if truncated:
        logger.warning("  %s: %d of %d responses used the whole %d-token budget without emitting "
                       "an answer. Raise answer_max_tokens for this model.", tier, truncated, len(out), budget)
    return out


def _swap_predictions(router: ApiRouter, tier: str, test_ids, judge=None) -> Dict[str, List[Dict]]:
    pairs = [p for p in read_jsonl(COUNTERFACTUAL_PAIRS) if p.get("original", {}).get("id") in test_ids]
    if SWAP_SUBSET > 0:
        pairs = pairs[:SWAP_SUBSET]
    if not pairs:
        return {}
    originals = [p["original"] for p in pairs]
    swapped = [p["swapped"] for p in pairs]
    return {"original": _mcq_predictions(router, tier, originals, 0, judge),
            "swapped": _mcq_predictions(router, tier, swapped, 0, judge)}


def _advice_rows(router: ApiRouter, tier: str, pairs: Sequence[Dict]) -> List[Dict]:
    jobs = []
    for p in pairs:
        jobs.append({"cache_key": f"{p['pair_id']}::a", "prompt": P.build_advice_prompt(p["prompt_a"])})
        jobs.append({"cache_key": f"{p['pair_id']}::b", "prompt": P.build_advice_prompt(p["prompt_b"])})
    cache = _run_prompts(router, tier, jobs, _cache_path(tier, "advice"),
                         advice_token_budget(tier, ADVICE_MAX_TOKENS))
    answers_a = [cache.get(f"{p['pair_id']}::a", {}).get("text", "") for p in pairs]
    answers_b = [cache.get(f"{p['pair_id']}::b", {}).get("text", "") for p in pairs]
    embedding = M.advice_drift_embedding_distances(list(zip(answers_a, answers_b)))
    jaccard = [M.advice_content_word_jaccard_distance(a, b) for a, b in zip(answers_a, answers_b)]
    structured = [M.advice_structured_l1(a, b) for a, b in zip(answers_a, answers_b)]

    detail = RESULTS_DIR / f"frontier_advice_drift_{tier}.jsonl"
    if detail.exists():
        detail.unlink()
    for i, p in enumerate(pairs):
        append_jsonl(detail, {"pair_id": p["pair_id"], "toggle_axis": p["toggle_axis"],
                              "embedding_distance": embedding[i],
                              "content_word_jaccard_distance": jaccard[i],
                              "structured_feature_l1": structured[i],
                              "answer_a": answers_a[i], "answer_b": answers_b[i]})

    threshold = float(os.environ.get("ADVICE_DRIFT_THRESHOLD", "0.25"))
    groups = [("all", list(range(len(pairs))))]
    for axis in sorted({p["toggle_axis"] for p in pairs}):
        groups.append((axis, [i for i, p in enumerate(pairs) if p["toggle_axis"] == axis]))
    rows = []
    for axis_name, idx in groups:
        if not idx:
            continue
        e = [embedding[i] for i in idx]
        rows.append({
            "model_tier": tier, "model_display_name": REGISTRY[tier].display_name,
            "toggle_axis": axis_name, "pairs_evaluated": len(idx),
            "advice_drift_embedding_cosine_distance_mean": round(float(np.mean(e)), 4),
            "advice_drift_content_word_jaccard_distance_mean": round(float(np.mean([jaccard[i] for i in idx])), 4),
            "advice_drift_structured_feature_l1_mean": round(float(np.mean([structured[i] for i in idx])), 4),
            "advice_drift_flip_rate_above_preregistered_threshold": round(M.advice_flip_rate(e, threshold), 4),
            "advice_drift_threshold_used": threshold,
        })
    return rows


def _ceiling_for_slice(slice_name: str) -> float:
    """The surface-cue ceiling recorded for this slice, so a panel score is never read alone."""
    path = RESULTS_DIR / "surface_cue_ceiling_audit.csv"
    if not path.exists():
        return float("nan")
    import csv

    key = {"all": "test_all", "structure_familiar": "test_structure_familiar",
           "structure_novel": "test_structure_novel"}.get(slice_name, "test_all")
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("split_name") == key:
                try:
                    return float(row["surface_cue_majority_ceiling_for_condition_and_answer"])
                except (TypeError, ValueError):
                    return float("nan")
    return float("nan")


def _num(v, nd=4):
    return round(float(v), nd) if (v is not None and v == v) else ""


def main(smoke: bool = False):
    if os.environ.get("RUN_FRONTIER_PANEL", "0") != "1":
        logger.info("Frontier panel disabled. It spends money on hosted APIs, so it is opt-in: "
                    "set RUN_FRONTIER_PANEL=1 to run it.")
        return
    set_global_determinism()
    router = ApiRouter()
    judge = JudgeChain()
    if judge.available():
        logger.info("Judge chain for answer extraction: %s", " -> ".join(judge.route_names()))
    if not router.available():
        logger.error("No AWS or OpenRouter credentials in .env; the frontier panel cannot run.")
        return

    test = _subset(read_jsonl(TEST_INSTANCES_FROZEN), "FRONTIER_EVAL_SUBSET_SIZE")
    if not test:
        raise SystemExit("Run build_template_instances.py first.")
    slice_of = {r["id"]: r.get("test_slice", "") for r in test}
    advice_pairs = _subset(read_jsonl(AGRIADVICE_PAIRS), "FRONTIER_ADVICE_SUBSET_SIZE")

    panel_path = RESULTS_DIR / "frontier_panel_results.csv"
    drift_path = RESULTS_DIR / "frontier_panel_advice_drift.csv"
    route_path = RESULTS_DIR / "frontier_panel_route_log.csv"
    for path in (panel_path, drift_path, route_path):
        if path.exists():
            path.unlink()

    for tier in active_api_models():
        spec = REGISTRY[tier]
        logger.info("=== %s (%s) ===", spec.display_name, tier)
        before = dict(router.route_counts), dict(router.failure_counts)

        predictions = _mcq_predictions(router, tier, test, rotation=0, judge=judge)
        pred_file = RESULTS_DIR / f"per_item_predictions_{tier}_frozen_base_seed{GLOBAL_SEED}.jsonl"
        if pred_file.exists():
            pred_file.unlink()
        for row in predictions:
            append_jsonl(pred_file, row)

        rotation_stats = {"rotation_mean_accuracy": float("nan"),
                          "rotation_consistency_rate": float("nan"),
                          "rotation_position_following_rate": float("nan")}
        if ROTATION_SUBSET > 0:
            rotation_items = test[:ROTATION_SUBSET]
            wanted = {r["id"] for r in rotation_items}
            by_rotation = {0: [p for p in predictions if p["id"] in wanted]}
            for k in (1, 2):
                by_rotation[k] = _mcq_predictions(router, tier, rotation_items, rotation=k, judge=judge)
            rotation_stats = M.option_rotation_consistency(by_rotation)

        swap = _swap_predictions(router, tier, {r["id"] for r in test}, judge=judge)
        flip = invariance = equivariance = float("nan")
        if swap:
            flip = M.identity_swap_flip_rate(swap["original"], swap["swapped"])
            invariance = M.identity_swap_invariance_rate(swap["original"], swap["swapped"])
            equivariance = M.identity_swap_equivariance_rate(swap["original"], swap["swapped"])

        for slice_name in ("all", "structure_familiar", "structure_novel"):
            subset = predictions if slice_name == "all" else \
                [p for p in predictions if slice_of.get(p["id"]) == slice_name]
            if not subset:
                continue
            wang = M.wang_contingency(subset)
            polarity = M.failure_polarity_index(subset)
            accuracy = M.accuracy(subset)
            ceiling = _ceiling_for_slice(slice_name)
            overall = slice_name == "all"
            append_csv_row(panel_path, {
                "model_tier": tier, "model_display_name": spec.display_name, "vendor": spec.vendor,
                "test_slice_name": slice_name, "number_of_items": len(subset),
                "overall_accuracy": _num(accuracy),
                "accuracy_on_diff_condition": _num(M.condition_accuracy(subset, "diff")),
                "accuracy_on_equal_condition": _num(M.condition_accuracy(subset, "equal")),
                "balanced_awareness_score_harmonic_mean_of_diff_and_equal_accuracy":
                    _num(M.balanced_awareness_score(subset)),
                "difference_aware_metric_wang_2025_recall_style":
                    _num(wang["difference_aware_metric_wang_2025_recall_style"]),
                "contextual_awareness_metric_wang_2025_precision_style":
                    _num(wang["contextual_awareness_metric_wang_2025_precision_style"]),
                "gap_erasure_rate_on_diff_items": _num(M.gap_erasure_rate(subset)),
                "gap_fabrication_rate_on_equal_items": _num(M.gap_fabrication_rate(subset)),
                "failure_polarity_index_positive_means_gap_erasure": _num(polarity),
                "dominant_failure_polarity_label": M.polarity_label(polarity),
                "surface_cue_majority_ceiling_for_this_slice": _num(ceiling),
                "accuracy_above_surface_cue_ceiling": _num(accuracy - ceiling) if ceiling == ceiling else "",
                "identity_swap_unlicensed_flip_rate": _num(flip) if overall else "",
                "identity_swap_invariance_rate_on_equal_items": _num(invariance) if overall else "",
                "identity_swap_equivariance_rate_on_diff_items": _num(equivariance) if overall else "",
                "option_rotation_mean_accuracy": _num(rotation_stats["rotation_mean_accuracy"]) if overall else "",
                "option_rotation_canonical_consistency_rate": _num(rotation_stats["rotation_consistency_rate"]) if overall else "",
                "option_rotation_position_following_rate": _num(rotation_stats["rotation_position_following_rate"]) if overall else "",
                "json_parse_failure_rate_percent": round(subset[0]["json_parse_failure_rate_percent"], 2),
                "unanswered_after_all_routes_failed": subset[0]["unanswered_after_all_routes_failed"],
                "responses_truncated_before_the_answer": subset[0]["responses_truncated_before_the_answer"],
                "answers_recovered_by_judge": subset[0]["answers_recovered_by_judge"],
            }, PANEL_COLUMNS)

        if advice_pairs:
            for row in _advice_rows(router, tier, advice_pairs):
                append_csv_row(drift_path, row, DRIFT_COLUMNS)

        served = {k: router.route_counts.get(k, 0) - before[0].get(k, 0) for k in router.route_counts}
        failed = {k: router.failure_counts.get(k, 0) - before[1].get(k, 0) for k in router.failure_counts}
        served = {k: v for k, v in served.items() if v}
        failed = {k: v for k, v in failed.items() if v}
        latencies = [p.get("latency_seconds", 0.0) for p in predictions if p.get("latency_seconds")]
        append_csv_row(route_path, {
            "model_tier": tier, "model_display_name": spec.display_name,
            "bedrock_model_id": spec.bedrock_model_id or "", "openrouter_model_id": spec.openrouter_model_id or "",
            "requests_total": sum(served.values()),
            "requests_served_by_route": ";".join(f"{k}={v}" for k, v in sorted(served.items())),
            "failures_by_route": ";".join(f"{k}={v}" for k, v in sorted(failed.items())),
            "input_tokens_total": sum(p.get("input_tokens", 0) for p in predictions),
            "output_tokens_total": sum(p.get("output_tokens", 0) for p in predictions),
            "mean_latency_seconds": round(float(np.mean(latencies)), 3) if latencies else "",
        }, ROUTE_COLUMNS)
        logger.info("  %s served by %s", spec.display_name, served)

    log_run_metadata("evaluate_frontier_panel", {
        "models": active_api_models(), **router.usage_summary(), **judge.usage_summary(),
    })


if __name__ == "__main__":
    main()
