"""Stage C: train the proposed XLoRA-Bias method, all ablations, the rank sweep, the
QLoRA arm (Instruction.md Section 2 Stage C; Section 6.4; coding_prompt.md Section 7).

The proposed method = attribution-guided placement + per-layer rank proportional to
attribution + condition-adaptive loss (upweight neq=diff) + rationale-aware multi-task
training. The proposed full-precision run sets the matched trainable-parameter budget;
baselines (train_baselines.py) must match it. Three seeds on primary models for the
proposed method; seed 42 elsewhere. Resumable per epoch.

Run:  python GPU_Run/train_xlora_bias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

from GPU_Run.common import model_registry
from GPU_Run.common import training as T
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import (
    CHECKPOINTS_DIR,
    GENERAL_REPLAY,
    RESULTS_DIR,
    TRAIN_INSTANCES,
)
from GPU_Run.common.seeds import GLOBAL_SEED, THREE_SEEDS, set_global_determinism

logger = get_logger("train_xlora_bias")


def _load_placements(label):
    path = RESULTS_DIR / f"lora_config_{label}.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}; run configure_layer_selective_lora.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _replay():
    return read_jsonl(GENERAL_REPLAY)


def _arms(configs, full_sweep: bool):
    """Define the proposed run plus ablations. Each arm: (method, placement_key, cfg_overrides)."""
    arms = [
        ("xlora_bias_proposed", "attribution_guided", {}),
    ]
    if full_sweep:
        arms += [
            ("ablation_placement_uniform", "uniform", {}),
            ("ablation_placement_random", "random", {}),
            ("ablation_no_rationale_loss", "attribution_guided", {"rationale_mode": False}),
            ("ablation_no_condition_adaptive", "attribution_guided", {"condition_adaptive": False}),
            ("ablation_rank_sweep_8", "attribution_guided", {"_force_rank": 8}),
            ("ablation_rank_sweep_16", "attribution_guided", {"_force_rank": 16}),
            ("ablation_rank_sweep_32", "attribution_guided", {"_force_rank": 32}),
            ("xlora_bias_proposed_qlora_nf4", "attribution_guided", {"_quantized": True}),
        ]
    return arms


def _apply_force_rank(placement, rank):
    p = json.loads(json.dumps(placement))
    p["layer_ranks"] = {k: rank for k in p["layer_ranks"]}
    p["default_rank"] = rank
    for k in list(p["rank_pattern"].keys()):
        p["rank_pattern"][k] = rank
    return p


def main(smoke: bool = False):
    set_global_determinism()
    train = read_jsonl(TRAIN_INSTANCES)
    replay = _replay()
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    full_sweep = not smoke and os.environ.get("XLORA_FULL_SWEEP", "1") == "1"

    runs = []
    reference_budget = None
    for tier in tiers:
        label = "smoke" if smoke else tier
        spec_role = "primary" if (not smoke and tier in model_registry.PRIMARY_TIERS) else "transfer"
        configs = _load_placements(label)
        for method, placement_key, overrides in _arms(configs, full_sweep):
            seeds = THREE_SEEDS if (method == "xlora_bias_proposed" and spec_role == "primary" and not smoke) else [GLOBAL_SEED]
            for seed in seeds:
                placement = configs[placement_key]
                if "_force_rank" in overrides:
                    placement = _apply_force_rank(placement, overrides["_force_rank"])
                cfg = T.TrainingConfig(seed=seed)
                if overrides.get("rationale_mode") is False:
                    cfg.rationale_mode = False
                if overrides.get("condition_adaptive") is False:
                    cfg.condition_adaptive = False
                quantized = bool(overrides.get("_quantized"))
                out_dir = CHECKPOINTS_DIR / label / method / f"seed_{seed}"
                logger.info("Training tier=%s method=%s seed=%d quantized=%s", label, method, seed, quantized)
                try:
                    model, tok, meta = model_registry.load_model_and_tokenizer(
                        tier, smoke=smoke, quantized_4bit=quantized, for_training=True
                    )
                    result = T.train_lora(
                        model, tok, train, placement, out_dir, cfg,
                        replay_records=replay,
                        reference_budget_pct=reference_budget if method != "xlora_bias_proposed" else None,
                    )
                    if method == "xlora_bias_proposed" and seed == seeds[0] and reference_budget is None and not quantized:
                        reference_budget = result["trainable_parameter_percentage"]
                        (RESULTS_DIR / "matched_budget_reference.json").write_text(
                            json.dumps({"tier": label, "reference_trainable_parameter_percentage": reference_budget}, indent=2),
                            encoding="utf-8",
                        )
                    runs.append({"tier": label, "method": method, "seed": seed, "quantized": quantized, **result})
                    del model
                except Exception as e:
                    logger.warning("Training arm failed (tier=%s method=%s seed=%d): %s", label, method, seed, e)
                    runs.append({"tier": label, "method": method, "seed": seed, "error": str(e)})

    (RESULTS_DIR / "train_xlora_bias_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    log_run_metadata("train_xlora_bias", {"num_runs": len(runs), "reference_budget": reference_budget})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
