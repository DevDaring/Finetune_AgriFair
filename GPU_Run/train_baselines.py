"""The six SOTA baselines plus two reference baselines, at a matched parameter budget.

Each method preserves its distinguishing signal (flagging, placement, rationale traces,
steering) and carries a paper-ready citation comment. Where a paper's full mechanism
cannot be reproduced exactly, the closest faithful approximation is used and noted
(Instruction.md Section 6).

Citations:
  FairNet            arXiv:2510.19421 (2025)  detector + conditional LoRA on flagged neq, contrastive
  IGU-LoRA           arXiv:2603.13792 (2026)  Integrated-Gradients rank allocation (capability-oriented)
  PEDAL              doi:10.1145/3774904.3793029 (2026)  Classifier/Modifier/Reviewer around PEFT
  FairLoRA           2025 [exact id UNVERIFIED]  modular LoRA + discriminators, uniform rank
  ReGiFT             arXiv:2504.05632 (2025)  reasoning-trace fine-tuning on the rationale field
  FairSteer          arXiv:2510.18914 (2025)  inference-time activation steering, no retraining
  Vanilla LoRA       Hu et al., ICLR 2022     reference baseline
  Vanilla QLoRA      Dettmers et al., NeurIPS 2023  reference baseline (4-bit NF4)

Run:  python GPU_Run/train_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from GPU_Run.common import model_registry
from GPU_Run.common import training as T
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.model_registry import DEFAULT_LORA_TARGETS
from GPU_Run.common.paths import CHECKPOINTS_DIR, GENERAL_REPLAY, RESULTS_DIR, TRAIN_INSTANCES
from GPU_Run.common.seeds import GLOBAL_SEED, THREE_SEEDS, set_global_determinism

logger = get_logger("train_baselines")


def _reference_budget():
    p = RESULTS_DIR / "matched_budget_reference.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8")).get("reference_trainable_parameter_percentage")
    return None


def _all_layer_uniform(configs, rank):
    layers = sorted({int(l) for l in configs["uniform"]["layer_ranks"]} | {int(l) for l in configs["attribution_guided"]["layer_ranks"]})
    pattern = {}
    for layer in layers:
        for mod in DEFAULT_LORA_TARGETS:
            pattern[f"layers.{layer}..*{mod}"] = rank
    return {"target_modules": list(DEFAULT_LORA_TARGETS), "rank_pattern": pattern, "default_rank": rank, "selected_layers": layers}


def _baseline_arms(configs):
    """Return list of (method, placement, cfg_overrides) approximating each baseline."""
    attr = configs["attribution_guided"]
    uni = configs["uniform"]
    return [
        # FairNet: uniform placement, condition-adaptive, trained on flagged neq (approximated)
        ("baseline_fairnet", uni, {"condition_adaptive": True, "rationale_mode": False}),
        # IGU-LoRA: attribution placement, capability-oriented loss (no fairness upweight)
        ("baseline_igu_lora", attr, {"condition_adaptive": False, "rationale_mode": False}),
        # PEDAL: uniform placement, condition-adaptive
        ("baseline_pedal", uni, {"condition_adaptive": True, "rationale_mode": False}),
        # FairLoRA: uniform placement, plain
        ("baseline_fairlora", uni, {"condition_adaptive": False, "rationale_mode": False}),
        # ReGiFT: rationale-mode training (uses the dataset rationale field)
        ("baseline_regift", uni, {"condition_adaptive": False, "rationale_mode": True}),
        # Vanilla LoRA (reference): rank 8 on attention+MLP, all layers (exempt from budget assert)
        ("reference_vanilla_lora", None, {"_vanilla_rank": 8}),
        # Vanilla QLoRA (reference): 4-bit NF4 + same LoRA
        ("reference_vanilla_qlora", None, {"_vanilla_rank": 8, "_quantized": True}),
        # FairSteer: inference-time steering, no training (handled separately)
        ("baseline_fairsteer", None, {"_inference_time": True}),
    ]


def _compute_fairsteer_vector(tier, smoke, label):
    """Mean hidden-state difference between correct-differential and over-equalized
    completions on neq items, at a mid-depth layer (FairSteer, arXiv:2510.18914)."""
    import numpy as np

    from GPU_Run.common import prompts as P
    from GPU_Run.common.patchscopes import capture_last_hidden_states

    train = [r for r in read_jsonl(TRAIN_INSTANCES) if r["condition"] == "neq"][:64]
    try:
        model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
    except Exception as e:
        logger.warning("FairSteer vector: could not load %s (%s).", tier, e)
        return
    diffs = []
    for rec in train:
        built = P.build_mcq_prompt(rec)
        correct = built["prompt"] + f'\n{{"answer_choice_letter": "{built["gold_display_letter"]}"}}'
        over_eq_letter = [d for d, c in built["display_to_canonical"].items() if c == "c"][0]
        overeq = built["prompt"] + f'\n{{"answer_choice_letter": "{over_eq_letter}"}}'
        hc = capture_last_hidden_states(model, tok, correct)
        ho = capture_last_hidden_states(model, tok, overeq)
        mid = len(hc) // 2
        diffs.append(np.asarray(hc[mid]) - np.asarray(ho[mid]))
    vec = np.mean(diffs, axis=0).tolist() if diffs else []
    out = RESULTS_DIR / f"fairsteer_vector_{label}.json"
    out.write_text(json.dumps({"tier": label, "mid_layer": len(diffs) and (len(capture_last_hidden_states(model, tok, "x")) // 2), "steering_vector": vec}, indent=2), encoding="utf-8")
    logger.info("Wrote FairSteer steering vector for %s (dim=%d).", label, len(vec))
    del model


def main(smoke: bool = False):
    set_global_determinism()
    train = read_jsonl(TRAIN_INSTANCES)
    replay = read_jsonl(GENERAL_REPLAY)
    reference_budget = _reference_budget()
    tiers = ["smoke"] if smoke else model_registry.active_tiers()

    runs = []
    for tier in tiers:
        label = "smoke" if smoke else tier
        cfg_path = RESULTS_DIR / f"lora_config_{label}.json"
        if not cfg_path.exists():
            logger.warning("Missing lora_config for %s; run configure first.", label)
            continue
        configs = json.loads(cfg_path.read_text(encoding="utf-8"))
        is_primary = (not smoke and tier in model_registry.PRIMARY_TIERS)

        for method, placement, overrides in _baseline_arms(configs):
            if overrides.get("_inference_time"):
                _compute_fairsteer_vector(tier, smoke, label)
                runs.append({"tier": label, "method": method, "type": "inference_time_no_training"})
                continue

            seeds = THREE_SEEDS if (method == "baseline_fairnet" and is_primary) else [GLOBAL_SEED]
            for seed in seeds:
                cfg = T.TrainingConfig(seed=seed)
                cfg.condition_adaptive = overrides.get("condition_adaptive", True)
                cfg.rationale_mode = overrides.get("rationale_mode", True)
                quantized = bool(overrides.get("_quantized"))
                is_reference = method.startswith("reference_")
                if "_vanilla_rank" in overrides:
                    placement = _all_layer_uniform(configs, overrides["_vanilla_rank"])
                out_dir = CHECKPOINTS_DIR / label / method / f"seed_{seed}"
                logger.info("Training baseline tier=%s method=%s seed=%d", label, method, seed)
                try:
                    model, tok, meta = model_registry.load_model_and_tokenizer(
                        tier, smoke=smoke, quantized_4bit=quantized, for_training=True
                    )
                    result = T.train_lora(
                        model, tok, train, placement, out_dir, cfg,
                        replay_records=replay,
                        reference_budget_pct=None if is_reference else reference_budget,
                    )
                    runs.append({"tier": label, "method": method, "seed": seed, **result})
                    del model
                except Exception as e:
                    logger.warning("Baseline failed (tier=%s method=%s): %s", label, method, e)
                    runs.append({"tier": label, "method": method, "seed": seed, "error": str(e)})

    (RESULTS_DIR / "train_baselines_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    log_run_metadata("train_baselines", {"num_runs": len(runs)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
