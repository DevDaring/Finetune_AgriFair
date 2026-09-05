"""Secondary-judge agreement and Cohen kappa on a 10 percent rationale subsample.

Re-runs a stratified 10 percent subsample of the rationale factual-correctness judgements
with the secondary judge and reports percent agreement and Cohen kappa against the primary
(Gemini-family) judge. If no judge keys are present, writes a
note and exits. CPU only (network for the judge call).

Run:  python CPU_Run/judge_robustness.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob
import random

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.clients import JudgeChain, primary_judge_chain, secondary_judge_chain
from GPU_Run.common import env_loader
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common import prompts as P
from GPU_Run.common.paths import RESULTS_DIR

logger = get_logger("judge_robustness")
COLUMNS = ["subsample_size", "percent_agreement", "cohen_kappa", "primary_judge", "secondary_judge"]


def _paired_scores(primary, secondary, prompt):
    """One rationale scored by the DeepSeek half of the chain and by the Mistral half.

    Splitting the chain is what makes this an agreement check rather than the same provider
    grading itself twice."""
    return (primary.score_int(prompt, "factual_correctness_score", 1, 5),
            secondary.score_int(prompt, "factual_correctness_score", 1, 5))


def main():
    primary = primary_judge_chain()
    secondary = secondary_judge_chain()
    if not (primary.available() and secondary.available()):
        write_csv(RESULTS_DIR / "judge_robustness_agreement.csv", [], COLUMNS)
        logger.info("No judge keys present; wrote an empty agreement table and exited.")
        return

    # gather rationale predictions across methods
    files = sorted(glob.glob(str(RESULTS_DIR / "per_item_predictions_*_*_seed*.jsonl")))
    pool = []
    for f in files:
        for r in read_jsonl(Path(f)):
            if r.get("generated_rationale"):
                pool.append(r)
    if not pool:
        logger.info("No rationale predictions found; nothing to re-judge.")
        return
    rng = random.Random(42)
    sample = rng.sample(pool, max(1, int(0.1 * len(pool))))

    primary_scores, secondary_scores = [], []
    for r in sample:
        # The question, not the item id: the judge cannot grade factual consistency against
        # an opaque identifier, and the inter-judge agreement measured here has to be the
        # agreement on the same task the main evaluation runs.
        prompt = P.judge_rationale_factual_prompt(
            r.get("question") or r["id"], r["reference_rationale"], r["generated_rationale"])
        a, b = _paired_scores(primary, secondary, prompt)
        if a is not None and b is not None:
            primary_scores.append(a)
            secondary_scores.append(b)

    if not primary_scores:
        logger.warning("Could not obtain paired judgements.")
        return
    agreement = sum(1 for a, b in zip(primary_scores, secondary_scores) if a == b) / len(primary_scores)
    kappa = _cohen_kappa(primary_scores, secondary_scores)
    write_csv(RESULTS_DIR / "judge_robustness_agreement.csv", [{
        "subsample_size": len(primary_scores),
        "percent_agreement": round(100 * agreement, 2),
        "cohen_kappa": round(kappa, 4),
        "primary_judge": primary.last_model_string or "deepseek",
        "secondary_judge": secondary.last_model_string or "mistral",
    }], COLUMNS)
    logger.info("Judge agreement %.1f%%, kappa %.3f over %d items.",
                100 * agreement, kappa, len(primary_scores))
    log_run_metadata("judge_robustness", {"agreement": agreement, "kappa": kappa})


def _cohen_kappa(a, b):
    import numpy as np

    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    conf = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    po = np.trace(conf) / n
    pe = sum((conf[i, :].sum() / n) * (conf[:, i].sum() / n) for i in range(len(labels)))
    return (po - pe) / (1 - pe) if (1 - pe) else 1.0


if __name__ == "__main__":
    main()
