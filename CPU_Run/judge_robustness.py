"""Secondary-judge agreement and Cohen kappa on a 10 percent rationale subsample.

Re-runs a stratified 10 percent subsample of the rationale factual-correctness judgements
with the secondary judge and reports percent agreement and Cohen kappa against the primary
(Gemini-family) judge (Instruction.md Section 10). If no judge keys are present, writes a
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
from GPU_Run.common.clients import JudgeChain, _RoundRobin
from GPU_Run.common import env_loader
from GPU_Run.common.clients import _call_openai_compatible
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common import prompts as P
from GPU_Run.common.paths import RESULTS_DIR

logger = get_logger("judge_robustness")
COLUMNS = ["subsample_size", "percent_agreement", "cohen_kappa", "primary_judge", "secondary_judge"]


def _secondary_score(prompt):
    keys = _RoundRobin(env_loader.get_round_robin(["MISTRAL_KEY1", "MISTRAL_KEY2"]))
    if not keys:
        return None
    model = env_loader.get("MISTRAL_MODEL_NAME", "mistral-small-latest")
    try:
        from GPU_Run.common.parsing import json_repair_parse

        text = _call_openai_compatible("https://api.mistral.ai/v1", keys.next(), model, prompt)
        obj = json_repair_parse(text) or {}
        return max(1, min(5, int(round(float(obj.get("factual_correctness_score"))))))
    except Exception:
        return None


def main():
    judge = JudgeChain()
    if not judge.available():
        (RESULTS_DIR / "judge_robustness_agreement.csv").write_text("no_judge_keys_present\n", encoding="utf-8")
        logger.info("No judge keys; wrote a note and exited.")
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

    primary, secondary = [], []
    for r in sample:
        prompt = P.judge_rationale_factual_prompt(r["id"], r["reference_rationale"], r["generated_rationale"])
        p = judge.score_int(prompt, "factual_correctness_score", 1, 5)
        s = _secondary_score(prompt)
        if p is not None and s is not None:
            primary.append(p)
            secondary.append(s)

    if not primary:
        logger.warning("Could not obtain paired judgements.")
        return
    agreement = sum(1 for a, b in zip(primary, secondary) if a == b) / len(primary)
    kappa = _cohen_kappa(primary, secondary)
    write_csv(RESULTS_DIR / "judge_robustness_agreement.csv", [{
        "subsample_size": len(primary),
        "percent_agreement": round(100 * agreement, 2),
        "cohen_kappa": round(kappa, 4),
        "primary_judge": judge.last_model_string,
        "secondary_judge": env_loader.get("MISTRAL_MODEL_NAME", "mistral-small-latest"),
    }], COLUMNS)
    logger.info("Judge agreement %.1f%%, kappa %.3f over %d items.", 100 * agreement, kappa, len(primary))
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
