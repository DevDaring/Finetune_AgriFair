"""Stage C: train GRAFT (the proposed method), its ablations, the rank sweep, the QLoRA
arm, and the leave-one-axis-out transfer arms.

GRAFT = Gradient-Ranked Adapter Fairness Targeting: attribution-guided placement +
per-layer rank proportional to attribution + condition-adaptive loss (diff upweighted) +
rationale-aware multi-task training. The proposed full-precision run sets the matched
trainable-parameter budget that train_baselines.py must match; the reference is written
to results/matched_budget_reference.json so a resumed or separately-launched baseline run
reads the same number.

Three seeds on the primary models for the proposed method; seed 42 elsewhere. Resumable
per epoch, and an arm whose final adapter already exists is skipped unless FORCE_RETRAIN=1.

Run:  python GPU_Run/train_graft.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
from typing import Dict, List, Optional, Tuple

from GPU_Run.common import methods as METHODS
from GPU_Run.common import model_registry
from GPU_Run.common import training as T
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.paths import (
    CHECKPOINTS_DIR,
    GENERAL_REPLAY,
    LOAO_INFIX,
    MATCHED_BUDGET_REFERENCE,
    RESULTS_DIR,
    TRAIN_INSTANCES,
    lora_config_path,
)
from GPU_Run.common.seeds import GLOBAL_SEED, THREE_SEEDS, set_global_determinism

logger = get_logger("train_graft")


def load_placements(label: str, held_out_axis: Optional[str] = None) -> Dict:
    path = lora_config_path(label, held_out_axis)
    if not path.exists():
        raise SystemExit(f"Missing {path}; run configure_layer_selective_lora.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def runs_on_tier(env_name: str, tier: str) -> bool:
    """Whether an optional arm runs on this model.

    Unset means every tier, which is what the preregistration specifies. Setting one of these
    is a documented reduction of the study, never a silent one, and the reduction is recorded
    in PREREGISTRATION.md under the reduced-budget profile."""
    raw = os.environ.get(env_name)
    return True if not raw else tier in [t.strip() for t in raw.split(",") if t.strip()]


def transfer_axes() -> List[str]:
    raw = os.environ.get("LOAO_AXES", "social_group,landholding,gender")
    return [a.strip() for a in raw.split(",") if a.strip()]


def _budget_doc() -> dict:
    if MATCHED_BUDGET_REFERENCE.exists():
        try:
            return json.loads(MATCHED_BUDGET_REFERENCE.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def read_reference_budget(label: Optional[str] = None) -> Optional[float]:
    """The proposed method's trainable share FOR THIS TIER.

    The reference is per tier. A 3B and a 12B model do not have the same trainable-parameter
    percentage for the same placement, so carrying one tier's reference into another compared
    each baseline against the wrong number and made the matched-budget guarantee vacuous."""
    doc = _budget_doc()
    by_tier = doc.get("reference_trainable_parameter_percentage_by_tier") or {}
    if not by_tier and "reference_trainable_parameter_percentage" in doc:
        # A file written before the reference became per-tier. Its single value belongs to
        # whichever tier ran first, which is not knowable now, so it is discarded rather than
        # applied to the wrong backbone. Silently returning None here would skip every budget
        # assertion in the run, so say so loudly.
        logger.warning("matched_budget_reference.json is in the old single-value format and "
                       "cannot be attributed to a tier; it will be rebuilt. Budget assertions "
                       "are skipped until the proposed arm runs on each tier.")
        return None
    if label is not None:
        v = by_tier.get(label)
        return float(v) if v is not None else None
    # No tier asked for: only meaningful if exactly one is on record.
    if len(by_tier) == 1:
        return float(next(iter(by_tier.values())))
    return None


def write_reference_budget(label: str, pct: float) -> None:
    doc = _budget_doc()
    by_tier = doc.get("reference_trainable_parameter_percentage_by_tier") or {}
    by_tier[label] = pct
    MATCHED_BUDGET_REFERENCE.write_text(json.dumps(
        {"reference_trainable_parameter_percentage_by_tier": by_tier}, indent=2), encoding="utf-8")


def arms(full_sweep: bool) -> List[Tuple[str, str, Dict]]:
    """(method, placement_key, overrides). The proposed arm always runs first because it
    sets the matched-budget reference."""
    out = [(METHODS.PROPOSED, "attribution_guided", {})]
    if full_sweep:
        out += [
            ("ablation_placement_uniform", "uniform", {}),
            ("ablation_placement_random", "random", {}),
            ("ablation_placement_random_draw2", "random_draw2", {}),
            ("ablation_placement_random_draw3", "random_draw3", {}),
            ("ablation_no_rationale_loss", "attribution_guided", {"rationale_mode": False}),
            ("ablation_no_condition_adaptive", "attribution_guided", {"condition_adaptive": False}),
            ("ablation_rank_sweep_8", "attribution_guided", {"force_rank": 8}),
            ("ablation_rank_sweep_16", "attribution_guided", {"force_rank": 16}),
            ("ablation_rank_sweep_32", "attribution_guided", {"force_rank": 32}),
            (METHODS.PROPOSED_QLORA, "attribution_guided", {"quantized": True}),
        ]
    return out


def _placement(configs: Dict, key: str, overrides: Dict) -> Optional[Dict]:
    from GPU_Run.configure_layer_selective_lora import repack_with_rank

    if key not in configs:
        return None
    placement = configs[key]
    if "force_rank" in overrides:
        placement = repack_with_rank(placement, overrides["force_rank"])
    return placement


def _seeds_for(method: str, tier: str, smoke: bool) -> List[int]:
    """Three seeds where the preregistration asks for a variance estimate.

    The proposed method always gets three seeds on every primary model, whatever the budget.
    A study that cannot state the variance of its own method on a model has no basis for
    saying a difference is or is not separable there, and that sentence is the one the whole
    honest-reporting posture rests on.

    MULTI_SEED_TIERS narrows only the BASELINE variance arms. Losing a baseline's variance on
    two models costs breadth in the external comparison; losing the proposed method's would
    cost the argument itself, so it is not offered as an option."""
    if smoke:
        return [GLOBAL_SEED]
    if not model_registry.is_primary(tier):
        return [GLOBAL_SEED]
    if method == METHODS.PROPOSED:
        return list(THREE_SEEDS)
    if method in METHODS.MULTI_SEED_METHODS and runs_on_tier("MULTI_SEED_TIERS", tier):
        return list(THREE_SEEDS)
    return [GLOBAL_SEED]


def train_one(tier, label, method, placement, overrides, seed, train, replay, reference_budget, smoke, runs):
    out_dir = CHECKPOINTS_DIR / label / method / f"seed_{seed}"
    done = T.existing_final(out_dir)
    if done is not None:
        logger.info("tier=%s method=%s seed=%d already trained; skipping (FORCE_RETRAIN=1 to redo).",
                    label, method, seed)
        runs.append({"tier": label, "method": method, "seed": seed, "skipped_existing": True, **done})
        return done.get("trainable_parameter_percentage")
    cfg = T.TrainingConfig(seed=seed)
    if overrides.get("rationale_mode") is False:
        cfg.rationale_mode = False
    if overrides.get("condition_adaptive") is False:
        cfg.condition_adaptive = False
    quantized = bool(overrides.get("quantized"))
    logger.info("Training tier=%s method=%s seed=%d quantized=%s layers=%s",
                label, method, seed, quantized, placement.get("selected_layers"))
    try:
        model, tok, meta = model_registry.load_model_and_tokenizer(
            tier, smoke=smoke, quantized_4bit=quantized, for_training=True)
        result = T.train_lora(
            model, tok, train, placement, out_dir, cfg,
            replay_records=replay,
            reference_budget_pct=reference_budget if METHODS.is_budget_matched(method) else None,
            cost_record={"tier": label, "method": method, "random_seed": seed,
                         "quantization_setting": meta["quantization_setting"]},
        )
        runs.append({"tier": label, "method": method, "seed": seed, "quantized": quantized, **result})
        del model
        return result["trainable_parameter_percentage"]
    except Exception as e:
        logger.error("Training arm failed (tier=%s method=%s seed=%d): %s", label, method, seed, e)
        runs.append({"tier": label, "method": method, "seed": seed, "error": str(e)})
        return None


def main(smoke: bool = False):
    set_global_determinism()
    train = read_jsonl(TRAIN_INSTANCES)
    replay = read_jsonl(GENERAL_REPLAY)
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    full_sweep = os.environ.get("GRAFT_FULL_SWEEP", "0" if smoke else "1") == "1"
    run_transfer = os.environ.get("RUN_LEAVE_ONE_AXIS_OUT", "0" if smoke else "1") == "1"

    runs: List[Dict] = []
    for tier in tiers:
        label = "smoke" if smoke else tier
        # Per tier: the proposed method's trainable share differs between backbones, so each
        # tier's arms are matched against their own reference rather than another tier's.
        reference_budget = read_reference_budget(label)
        configs = load_placements(label)
        for method, key, overrides in arms(full_sweep):
            # RANK_SWEEP_TIERS narrows the rank sensitivity check, which is a sweep rather
            # than a hypothesis, so restricting it costs no registered claim.
            if method.startswith("ablation_rank_sweep") and not runs_on_tier("RANK_SWEEP_TIERS", tier):
                continue
            placement = _placement(configs, key, overrides)
            if placement is None:
                logger.warning("Placement %s missing for %s; arm %s skipped.", key, label, method)
                continue
            for seed in _seeds_for(method, tier, smoke):
                pct = train_one(tier, label, method, placement, overrides, seed, train, replay,
                                reference_budget, smoke, runs)
                if method == METHODS.PROPOSED and not overrides.get("quantized") and pct is not None \
                        and reference_budget is None:
                    reference_budget = pct
                    write_reference_budget(label, pct)
                    logger.info("Matched-budget reference set from %s on %s: %.4f percent trainable.",
                                method, label, pct)

        # Leave-one-axis-out transfer: train without one axis, evaluate on it later.
        if run_transfer and model_registry.is_primary(tier) and runs_on_tier("LOAO_TIERS", tier):
            for axis in transfer_axes():
                axis_configs_path = lora_config_path(label, axis)
                if not axis_configs_path.exists():
                    logger.warning("No leave-one-axis-out placement for %s/%s; skipping.", label, axis)
                    continue
                axis_configs = json.loads(axis_configs_path.read_text(encoding="utf-8"))
                axis_train = [r for r in train if r.get("category") != axis]
                if not axis_train:
                    continue
                for base_method in METHODS.TRANSFER_METHODS:
                    key = "attribution_guided" if base_method == METHODS.PROPOSED else None
                    if key is None:
                        from GPU_Run.train_baselines import all_layer_uniform_placement

                        placement = all_layer_uniform_placement(axis_configs, 8)
                    else:
                        placement = axis_configs[key]
                    method = f"{base_method}{LOAO_INFIX}{axis}"
                    train_one(tier, label, method, placement, {}, GLOBAL_SEED, axis_train, replay,
                              None, smoke, runs)

    (RESULTS_DIR / "train_graft_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    log_run_metadata("train_graft", {"num_runs": len(runs),
                                     "reference_budget_by_tier": _budget_doc().get(
                                         "reference_trainable_parameter_percentage_by_tier", {})})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
