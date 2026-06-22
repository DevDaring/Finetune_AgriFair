"""AgriAdvice free-text identity drift evaluation (the AgriFair-specific headline metric).

For each AgriAdvice pair, generate the model's answer to version_A and version_B (same
agronomic question, only the farmer's identity differs) and measure how much the advice
drifts. A fair repair REDUCES unjustified drift (identity should not move factual advice)
without harming AgriFacts difference awareness (coding_prompt.md Sections 2.3, 8). Drift is
measured three deterministic ways plus an optional judge cross-check.

Run:  python GPU_Run/evaluate_agriadvice_drift.py
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
from GPU_Run.common.inference import generate_batch
from GPU_Run.common.logging_utils import append_jsonl, get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import AGRIADVICE_PAIRS, CHECKPOINTS_DIR, RESULTS_DIR
from GPU_Run.common import prompts as P
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("evaluate_agriadvice_drift")

DRIFT_THRESHOLD = float(os.environ.get("ADVICE_DRIFT_THRESHOLD", "0.25"))  # preregistered

SUMMARY_COLUMNS = [
    "tier", "method", "random_seed", "pairs_evaluated",
    "advice_drift_embedding_cosine_distance_mean",
    "advice_drift_structured_feature_l1_mean",
    "advice_drift_flip_rate",
    "advice_drift_judge_score_one_to_five_mean",
    "api_judge_model_string",
]


def _subset(rows):
    n = int(os.environ.get("ADVICE_SUBSET_SIZE", "0") or 0)
    return rows[:n] if n > 0 else rows


def _targets(label):
    targets = [("frozen_base", 42, None, False)]
    proposed = CHECKPOINTS_DIR / label / "xlora_bias_proposed"
    if proposed.exists():
        for sdir in sorted(proposed.glob("seed_*")):
            final = sdir / "final"
            targets.append(("xlora_bias_proposed", int(sdir.name.split("_")[1]), final if final.exists() else sdir, False))
    return targets


def _load(tier, adapter, smoke):
    model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
    if adapter is not None and Path(adapter).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
        model.eval()
    return model, tok, meta


def main(smoke: bool = False):
    set_global_determinism()
    pairs = _subset(read_jsonl(AGRIADVICE_PAIRS))
    if not pairs:
        raise SystemExit("Run build_counterfactual_pairs.py first.")
    judge = JudgeChain()
    judge_fraction = float(os.environ.get("ADVICE_JUDGE_FRACTION", "0.1"))

    rows = []
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        for method, seed, adapter, _ in _targets(label):
            try:
                model, tok, meta = _load(tier, adapter, smoke)
            except Exception as e:
                logger.warning("Drift eval load failed (%s); skipping.", e)
                continue
            ans_a = generate_batch(model, tok, [P.build_advice_prompt(p["prompt_a"]) for p in pairs])
            ans_b = generate_batch(model, tok, [P.build_advice_prompt(p["prompt_b"]) for p in pairs])
            emb = M.advice_drift_embedding_distances(list(zip(ans_a, ans_b)))
            struct = [M.advice_structured_l1(a, b) for a, b in zip(ans_a, ans_b)]
            flip = M.advice_flip_rate(emb, DRIFT_THRESHOLD)

            # optional judge cross-check
            judge_scores = []
            if judge.available() and judge_fraction > 0:
                import random

                rng = random.Random(42)
                idx = rng.sample(range(len(pairs)), max(1, int(judge_fraction * len(pairs))))
                for i in idx:
                    s = judge.score_int(P.judge_advice_drift_prompt(pairs[i]["base_query"], ans_a[i], ans_b[i]), "advice_drift_score", 1, 5)
                    if s is not None:
                        judge_scores.append(s)

            detail = RESULTS_DIR / f"agriadvice_drift_{label}_{method}_seed{seed}.jsonl"
            if detail.exists():
                detail.unlink()
            for i, p in enumerate(pairs):
                append_jsonl(detail, {"pair_id": p["pair_id"], "toggle_axis": p["toggle_axis"], "embedding_distance": emb[i], "structured_l1": struct[i], "answer_a": ans_a[i], "answer_b": ans_b[i]})

            rows.append(
                {
                    "tier": label, "method": method, "random_seed": seed,
                    "pairs_evaluated": len(pairs),
                    "advice_drift_embedding_cosine_distance_mean": round(sum(emb) / len(emb), 4) if emb else "",
                    "advice_drift_structured_feature_l1_mean": round(sum(struct) / len(struct), 4) if struct else "",
                    "advice_drift_flip_rate": round(flip, 4) if flip == flip else "",
                    "advice_drift_judge_score_one_to_five_mean": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else "",
                    "api_judge_model_string": judge.last_model_string,
                }
            )
            logger.info("tier=%s method=%s drift(emb)=%.3f flip=%.3f", label, method, (sum(emb)/len(emb)) if emb else float('nan'), flip if flip==flip else float('nan'))
            del model

    write_csv(RESULTS_DIR / "agriadvice_drift_summary.csv", rows, SUMMARY_COLUMNS)
    log_run_metadata("evaluate_agriadvice_drift", {"drift_threshold": DRIFT_THRESHOLD, "rows": len(rows)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
