"""AgriAdvice free-text identity drift evaluation.

For each AgriAdvice pair, generate the model's answer to version_A and version_B (same
agronomic question, only the farmer's stated identity differs) and measure how much the
advice moves. Because the material facts are identical across the two versions, any
systematic difference is unjustified. A fair repair reduces drift without harming
AgriFacts difference awareness.

Four deterministic measures plus an optional judge cross-check:
  embedding cosine distance   sentence embeddings when an embedder is configured, TF-IDF
                              otherwise; the column records which was used
  content-word Jaccard        overlap of content words, embedder-free and interpretable
  structured feature L1       tokens, bullets, numbers, quantities-with-units, sentences
  drift flip rate             share of pairs above the preregistered distance threshold
Every method is evaluated, not only the proposed one, and results break out per toggle
axis (gender, social group, region register, literacy register).

Run:  python GPU_Run/evaluate_agriadvice_drift.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from typing import Dict, List, Optional

import numpy as np

from GPU_Run.common import env_loader
from GPU_Run.common import metrics as M
from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common import targets as TG
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.clients import JudgeChain
from GPU_Run.common.inference import generate_batch
from GPU_Run.common.logging_utils import append_csv_row, append_jsonl, get_logger, log_run_metadata
from GPU_Run.common.paths import AGRIADVICE_PAIRS, RESULTS_DIR
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("evaluate_agriadvice_drift")

DRIFT_THRESHOLD = float(os.environ.get("ADVICE_DRIFT_THRESHOLD", "0.25"))  # preregistered
ADVICE_MAX_NEW_TOKENS = int(os.environ.get("ADVICE_MAX_NEW_TOKENS", "256"))

SUMMARY_COLUMNS = [
    "tier", "method", "random_seed", "toggle_axis", "pairs_evaluated",
    "advice_drift_embedding_cosine_distance_mean",
    "advice_drift_embedding_backend_name",
    "advice_drift_content_word_jaccard_distance_mean",
    "advice_drift_structured_feature_l1_mean",
    "advice_drift_flip_rate_above_preregistered_threshold",
    "advice_drift_threshold_used",
    "advice_drift_judge_score_one_to_five_mean",
    "api_judge_model_string",
]


def _subset(rows):
    n = int(os.environ.get("ADVICE_SUBSET_SIZE", "0") or 0)
    return rows[:n] if n > 0 else rows


def _embedder():
    """(callable, backend_name). Sentence embeddings when MULTILINGUAL_EMBED_MODEL is set
    and loadable; otherwise None, which makes the metric fall back to TF-IDF."""
    name = os.environ.get("MULTILINGUAL_EMBED_MODEL") or env_loader.get("MULTILINGUAL_EMBED_MODEL")
    if not name:
        return None, "tfidf_cosine_fallback"
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(name)
        return (lambda texts: m.encode(list(texts), normalize_embeddings=True)), f"sentence_transformer:{name}"
    except Exception as e:
        logger.warning("Embedder %s unavailable (%s); using TF-IDF.", name, e)
        return None, "tfidf_cosine_fallback"


def main(smoke: bool = False):
    set_global_determinism()
    pairs = _subset(read_jsonl(AGRIADVICE_PAIRS))
    if not pairs:
        raise SystemExit("Run build_counterfactual_pairs.py first.")
    judge = JudgeChain()
    judge_fraction = float(os.environ.get("ADVICE_JUDGE_FRACTION", "0.1"))
    embedder, backend = _embedder()

    summary_path = RESULTS_DIR / "agriadvice_drift_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        # Advice drift runs on every arm by default, including the ablations. That is
        # deliberate and it is the expensive choice: the ablations are what attribute a drift
        # reduction to a component, so a reviewer asking whether the reduction comes from the
        # localized placement or from the condition-adaptive objective can be answered from
        # the results rather than from argument. ADVICE_TARGET_SCOPE=headline narrows it to
        # the arms whose drift the paper tabulates, which is a declared reduction recorded in
        # PREREGISTRATION.md, not a default.
        exhaustive = os.environ.get("ADVICE_TARGET_SCOPE", "all") == "all"
        for target in TG.discover_targets(
                label,
                exclude_prefixes=() if exhaustive else ("ablation_",),
                first_seed_only=not exhaustive,
                exclude_transfer_arms=not exhaustive):
            method, seed = target.method, target.seed
            try:
                model, tok, meta = TG.load_target(tier, target, smoke=smoke)
            except Exception as e:
                logger.error("Drift eval load failed (%s); skipping %s.", e, method)
                continue
            try:
                ans_a = generate_batch(model, tok, [P.build_advice_prompt(p["prompt_a"]) for p in pairs],
                                       ADVICE_MAX_NEW_TOKENS)
                ans_b = generate_batch(model, tok, [P.build_advice_prompt(p["prompt_b"]) for p in pairs],
                                       ADVICE_MAX_NEW_TOKENS)
            except Exception as e:
                logger.error("Drift generation failed for %s (%s); skipping.", method, e)
                TG.remove_steering(model)
                del model
                continue
            try:
                emb = M.advice_drift_embedding_distances(list(zip(ans_a, ans_b)), embedder)
                jac = [M.advice_content_word_jaccard_distance(a, b) for a, b in zip(ans_a, ans_b)]
                struct = [M.advice_structured_l1(a, b) for a, b in zip(ans_a, ans_b)]

                judge_scores: Dict[int, int] = {}
                if judge.available() and judge_fraction > 0:
                    import random

                    rng = random.Random(42)
                    idx = rng.sample(range(len(pairs)), max(1, int(judge_fraction * len(pairs))))
                    for i in idx:
                        s = judge.score_int(P.judge_advice_drift_prompt(pairs[i]["base_query"], ans_a[i], ans_b[i]),
                                            "advice_drift_score", 1, 5)
                        if s is not None:
                            judge_scores[i] = s

                detail = RESULTS_DIR / f"agriadvice_drift_{label}_{method}_seed{seed}.jsonl"
                if detail.exists():
                    detail.unlink()
                for i, p in enumerate(pairs):
                    append_jsonl(detail, {
                        "pair_id": p["pair_id"], "toggle_axis": p["toggle_axis"],
                        "embedding_distance": emb[i], "content_word_jaccard_distance": jac[i],
                        "structured_feature_l1": struct[i],
                        "judge_score_one_to_five": judge_scores.get(i, ""),
                        "answer_a": ans_a[i], "answer_b": ans_b[i],
                    })

                groups = [("all", list(range(len(pairs))))]
                for axis in sorted({p["toggle_axis"] for p in pairs}):
                    groups.append((axis, [i for i, p in enumerate(pairs) if p["toggle_axis"] == axis]))
                for axis_name, idx in groups:
                    if not idx:
                        continue
                    e = [emb[i] for i in idx]
                    js = [judge_scores[i] for i in idx if i in judge_scores]
                    append_csv_row(summary_path, {
                        "tier": label, "method": method, "random_seed": seed, "toggle_axis": axis_name,
                        "pairs_evaluated": len(idx),
                        "advice_drift_embedding_cosine_distance_mean": round(float(np.mean(e)), 4),
                        "advice_drift_embedding_backend_name": backend,
                        "advice_drift_content_word_jaccard_distance_mean": round(float(np.mean([jac[i] for i in idx])), 4),
                        "advice_drift_structured_feature_l1_mean": round(float(np.mean([struct[i] for i in idx])), 4),
                        "advice_drift_flip_rate_above_preregistered_threshold": round(M.advice_flip_rate(e, DRIFT_THRESHOLD), 4),
                        "advice_drift_threshold_used": DRIFT_THRESHOLD,
                        "advice_drift_judge_score_one_to_five_mean": round(float(np.mean(js)), 3) if js else "",
                        "api_judge_model_string": judge.last_model_string,
                    }, SUMMARY_COLUMNS)
                logger.info("tier=%s method=%s drift(emb)=%.3f jaccard=%.3f flip=%.3f",
                            label, method, float(np.mean(emb)), float(np.mean(jac)),
                            M.advice_flip_rate(emb, DRIFT_THRESHOLD))
            except Exception as e:
                # Same per-target isolation the other evaluation stages have: one arm
                # failing here must not discard every arm still queued behind it.
                logger.error("Drift analysis failed for %s (%s); its rows are omitted "
                             "and the remaining targets continue.", method, e, exc_info=True)
            finally:
                TG.remove_steering(model)
                del model
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

    log_run_metadata("evaluate_agriadvice_drift",
                     {"drift_threshold": DRIFT_THRESHOLD, "embedding_backend": backend})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
