"""The external baselines plus two reference baselines, at a matched parameter budget.

Each method keeps its distinguishing signal (gating, placement, rationale traces,
steering) and carries a paper-ready citation comment. Where a paper's full mechanism
cannot be reproduced exactly, the closest faithful approximation is used and noted, and
every divergence is listed in README under "Baseline fidelity".

  FairNet    arXiv:2510.19421, NeurIPS 2025      detector-gated conditional LoRA, contrastive
  IGU-LoRA   arXiv:2603.13792, ICLR 2026         Integrated-Gradients rank allocation (capability)
  PEDAL      doi:10.1145/3774904.3793029, WWW 2026  Classifier / Modifier / Reviewer around PEFT
  DART       arXiv:2604.16845, Findings of ACL 2026  distil -> audit -> severity-weighted repair
  ReGiFT     arXiv:2504.05632                    reasoning-trace fine-tuning on the rationale field
  LFTF       arXiv:2505.15475                    locate bias-relevant blocks first, fine-tune them
  FairSteer  arXiv:2504.14492, Findings of ACL 2025  inference-time activation steering, no retraining
  Vanilla LoRA   Hu et al., ICLR 2022, arXiv:2106.09685    reference baseline
  Vanilla QLoRA  Dettmers et al., NeurIPS 2023, arXiv:2305.14314  reference baseline (4-bit NF4)

LFTF is included because it is the closest published prior art to attribution-guided
placement: it also localizes first and fine-tunes only the located blocks, but it locates
with a bias-relevance score over hidden states rather than with a path-integrated
attribution of the specific failure, and its objective carries no notion of the item's
condition. It is therefore the control that isolates what the attribution signal and the
condition-adaptive objective each contribute.

Run:  python GPU_Run/train_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
from typing import Dict, List, Optional

import numpy as np

from GPU_Run.common import dart_baseline as dart
from GPU_Run.common import methods as METHODS
from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common import training as T
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.model_registry import DEFAULT_LORA_TARGETS
from GPU_Run.common.paths import (
    CHECKPOINTS_DIR,
    COUNTERFACTUAL_PAIRS,
    GENERAL_REPLAY,
    RESULTS_DIR,
    TRAIN_INSTANCES,
    VALIDATION_INSTANCES,
    lora_config_path,
)
from GPU_Run.common.seeds import GLOBAL_SEED, THREE_SEEDS, set_global_determinism

logger = get_logger("train_baselines")

LFTF_MAX_ITEMS = int(os.environ.get("LFTF_MAX_ITEMS", "64"))
FAIRSTEER_MAX_ITEMS = int(os.environ.get("FAIRSTEER_MAX_ITEMS", "64"))


def all_layer_uniform_placement(configs: Dict, rank: int) -> Dict:
    """Dense reference placement: every layer, one rank, suffix-matched target modules."""
    return {
        "placement": "all_layer_uniform",
        "target_modules": list(DEFAULT_LORA_TARGETS),
        "rank_pattern": {},
        "alpha_pattern": {},
        "default_rank": int(rank),
        "selected_layers": list(range(int(configs.get("number_of_layers", 0)))),
    }


def _pedal_classifier(record: Dict) -> bool:
    """PEDAL Classifier gate (doi:10.1145/3774904.3793029).

    PEDAL's Classifier decides whether a sample carries the shortcut pattern; the
    Modifier then breaks that shortcut on the flagged samples only. The AgriFacts
    shortcut is the "when two groups are named, answer Roughly equal" default, which is
    exactly what a diff item punishes. The gate therefore flags diff items; unflagged
    equal items train unchanged. Deterministic, no model in the loop."""
    return record.get("condition") == "diff"


# ------------------------------- LFTF placement ------------------------------

def compute_lftf_placement(tier: str, smoke: bool, configs: Dict, label: str) -> Optional[Dict]:
    """LFTF (arXiv:2505.15475): locate the blocks most relevant to the bias, then
    fine-tune only those.

    Approximation, recorded in README: the published block-level bias score is computed
    from a gender-bias probe set; here the same idea is applied to the AgriFair identity
    contrast, scoring each block by the mean L2 distance between its last-position hidden
    state on an item and on that item's identity-swapped counterfactual. Blocks whose
    normalized score clears the same 0.15 threshold are adapted at a uniform rank sized to
    the matched budget. The locating signal is hidden-state sensitivity to identity, not a
    path-integrated attribution of the failure, which is the difference this baseline is
    here to isolate."""
    from GPU_Run.common.patchscopes import capture_last_hidden_states
    from GPU_Run.configure_layer_selective_lora import pack, spread_budget

    pairs = read_jsonl(COUNTERFACTUAL_PAIRS)
    val_ids = {r["id"] for r in read_jsonl(VALIDATION_INSTANCES)}
    pairs = [c for c in pairs if c["original"]["id"] in val_ids][:LFTF_MAX_ITEMS]
    if not pairs:
        logger.warning("LFTF: no counterfactual pairs available; falling back to uniform placement.")
        return None
    try:
        model, tok, _ = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
    except Exception as e:
        logger.warning("LFTF: could not load %s (%s).", tier, e)
        return None
    add_special = P.prompt_add_special_tokens(tok)
    scores = None
    for c in pairs:
        ha = capture_last_hidden_states(model, tok, P.render_chat(tok, P.build_mcq_prompt(c["original"])["prompt"]), add_special)
        hb = capture_last_hidden_states(model, tok, P.render_chat(tok, P.build_mcq_prompt(c["swapped"])["prompt"]), add_special)
        d = np.array([float(np.linalg.norm(np.asarray(a) - np.asarray(b))) for a, b in zip(ha[1:], hb[1:])])
        scores = d if scores is None else scores + d
    del model
    scores = scores / max(1, len(pairs))
    mx = scores.max() if scores.size else 0.0
    norm = (scores / mx) if mx > 0 else scores
    threshold = float(os.environ.get("ATTRIBUTION_THRESHOLD", "0.15"))
    selected = [i for i, v in enumerate(norm) if v >= threshold] or [int(np.argmax(norm))]
    total_rank = sum(int(v) for v in configs["attribution_guided"]["layer_ranks"].values())
    spread = spread_budget(total_rank, selected)
    placement = pack("lftf_located", spread)
    (RESULTS_DIR / f"lftf_located_blocks_{label}.json").write_text(json.dumps({
        "tier": label, "block_bias_relevance_normalized": {str(i): float(v) for i, v in enumerate(norm)},
        "selected_blocks": selected, "threshold": threshold,
        "locating_signal": "mean_l2_hidden_state_distance_between_identity_swapped_prompts",
    }, indent=2), encoding="utf-8")
    logger.info("LFTF located %d/%d blocks for %s.", len(selected), len(norm), label)
    return placement


# ------------------------------ FairSteer vector -----------------------------

def compute_fairsteer_vector(tier: str, smoke: bool, label: str) -> None:
    """FairSteer (arXiv:2504.14492): a debiasing steering vector applied at inference.

    The vector is the mean hidden-state difference between a completion that names the
    correct group and one that answers "Roughly equal", over diff training items, at the
    mid-depth block. Adding it at inference pushes the model away from the gap-erasure
    completion without touching a weight. The layer index is recorded so evaluation can
    apply it at the same place it was measured."""
    from GPU_Run.common.patchscopes import capture_last_hidden_states, layer_output_state, number_of_layers

    train = [r for r in read_jsonl(TRAIN_INSTANCES) if r["condition"] == "diff"][:FAIRSTEER_MAX_ITEMS]
    try:
        model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
    except Exception as e:
        logger.warning("FairSteer vector: could not load %s (%s).", tier, e)
        return
    mid = max(0, number_of_layers(model) // 2)
    add_special = P.prompt_add_special_tokens(tok)
    diffs = []
    for rec in train:
        built = P.build_mcq_prompt(rec)
        rendered = P.render_chat(tok, built["prompt"]) + P.answer_prefix_text()
        correct = rendered + built["gold_display_letter"]
        erase_letter = [d for d, c in built["display_to_canonical"].items() if c == "c"][0]
        erased = rendered + erase_letter
        hc = layer_output_state(capture_last_hidden_states(model, tok, correct, add_special), mid)
        he = layer_output_state(capture_last_hidden_states(model, tok, erased, add_special), mid)
        diffs.append(np.asarray(hc) - np.asarray(he))
    vec = np.mean(diffs, axis=0) if diffs else np.array([])
    out = RESULTS_DIR / f"fairsteer_vector_{label}.json"
    out.write_text(json.dumps({
        "tier": label, "steering_layer_index": mid, "items_used": len(diffs),
        "steering_vector_dimension": int(vec.size), "steering_vector": vec.tolist(),
    }, indent=2), encoding="utf-8")
    logger.info("Wrote FairSteer steering vector for %s (layer %d, dim=%d).", label, mid, int(vec.size))
    del model


# --------------------------------- arms --------------------------------------

def baseline_arms(configs: Dict) -> List:
    """(method, placement_key_or_none, overrides) for every baseline."""
    return [
        # FairNet: uniform placement, condition-adaptive on every example; the contrastive
        # detector gate is approximated by the global diff upweight.
        ("baseline_fairnet", "uniform", {"condition_adaptive": True, "rationale_mode": False}),
        # IGU-LoRA: attribution placement, capability-oriented loss (no fairness upweight).
        ("baseline_igu_lora", "attribution_guided", {"condition_adaptive": False, "rationale_mode": False}),
        # PEDAL: uniform placement, no blanket condition weighting; the Classifier gate
        # selects the shortcut-carrying samples and the Modifier upweights only those.
        ("baseline_pedal", "uniform", {"condition_adaptive": False, "rationale_mode": False, "pedal": True}),
        # DART: rationale-mode distil phase, behavioural audit against the frozen base,
        # then severity-weighted repair.
        ("baseline_dart", "uniform", {"condition_adaptive": False, "rationale_mode": True, "dart": True}),
        # ReGiFT: rationale-mode training on the dataset's reference rationale.
        ("baseline_regift", "uniform", {"condition_adaptive": False, "rationale_mode": True}),
        # LFTF: locate bias-relevant blocks first, then fine-tune them with a plain loss.
        ("baseline_lftf", "lftf", {"condition_adaptive": False, "rationale_mode": False}),
        # Vanilla LoRA (reference): rank 8 on attention+MLP, all layers (budget-exempt).
        ("reference_vanilla_lora", None, {"vanilla_rank": 8, "rationale_mode": False}),
        # Vanilla QLoRA (reference): 4-bit NF4 + the same LoRA.
        ("reference_vanilla_qlora", None, {"vanilla_rank": 8, "quantized": True, "rationale_mode": False}),
    ]


def main(smoke: bool = False):
    set_global_determinism()
    train = read_jsonl(TRAIN_INSTANCES)
    replay = read_jsonl(GENERAL_REPLAY)
    from GPU_Run.train_graft import read_reference_budget

    tiers = ["smoke"] if smoke else model_registry.active_tiers()

    runs: List[Dict] = []
    for tier in tiers:
        label = "smoke" if smoke else tier
        # Read per tier: the proposed method's trainable share differs between a 3B and a 12B
        # backbone, so each tier's baselines must be matched against their own reference.
        reference_budget = read_reference_budget(label)
        if reference_budget is None:
            logger.info("No matched-budget reference for %s yet (train_graft.py sets it).", label)
        # Published baselines place adapters by their own algorithms, so their trainable share
        # is measured and reported per arm rather than forced to the proposed method's budget.
        # Only the placement ablations, which exist to isolate WHERE adapters go, are asserted
        # equal, and those are trained in train_graft.py.
        logger.info("Baseline budgets for %s are reported, not asserted; see "
                    "trainable_parameter_percentage in the results tables.", label)
        cfg_path = lora_config_path(label)
        if not cfg_path.exists():
            logger.warning("Missing lora_config for %s; run configure first.", label)
            continue
        configs = json.loads(cfg_path.read_text(encoding="utf-8"))
        lftf_placement = None

        for method, key, overrides in baseline_arms(configs):
            if key == "lftf":
                if lftf_placement is None:
                    lftf_placement = compute_lftf_placement(tier, smoke, configs, label)
                placement = lftf_placement or configs["uniform"]
            elif key is None:
                placement = all_layer_uniform_placement(configs, overrides["vanilla_rank"])
            else:
                placement = configs.get(key)
            if placement is None:
                logger.warning("Placement %s missing for %s; %s skipped.", key, label, method)
                continue

            from GPU_Run.train_graft import runs_on_tier

            multi_seed = (method in METHODS.MULTI_SEED_METHODS
                          and runs_on_tier("MULTI_SEED_TIERS", tier))
            seeds = THREE_SEEDS if (multi_seed and model_registry.is_primary(tier) and not smoke) else [GLOBAL_SEED]
            for seed in seeds:
                out_dir = CHECKPOINTS_DIR / label / method / f"seed_{seed}"
                done = T.existing_final(out_dir)
                if done is not None:
                    logger.info("tier=%s method=%s seed=%d already trained; skipping.", label, method, seed)
                    runs.append({"tier": label, "method": method, "seed": seed, "skipped_existing": True, **done})
                    continue
                cfg = T.TrainingConfig(seed=seed)
                cfg.condition_adaptive = overrides.get("condition_adaptive", True)
                cfg.rationale_mode = overrides.get("rationale_mode", True)
                if overrides.get("pedal"):
                    cfg.pedal_classifier_fn = _pedal_classifier
                quantized = bool(overrides.get("quantized"))
                logger.info("Training baseline tier=%s method=%s seed=%d", label, method, seed)
                model = None
                try:
                    model, tok, meta = model_registry.load_model_and_tokenizer(
                        tier, smoke=smoke, quantized_4bit=quantized, for_training=True)
                    audit_hook = None
                    if overrides.get("dart"):
                        # DART Stage II needs the frozen base predictions M0, captured
                        # before any adapter is attached to this model instance.
                        audit_records = dart.load_audit_records(train)
                        base_preds = dart.base_prediction_cache(model, tok, audit_records)
                        audit_hook = dart.make_audit_hook(audit_records, base_preds, label, seed)
                    result = T.train_lora(
                        model, tok, train, placement, out_dir, cfg,
                        replay_records=replay,
                        reference_budget_pct=reference_budget if METHODS.is_budget_matched(method) else None,
                        cost_record={"tier": label, "method": method, "random_seed": seed,
                                     "quantization_setting": meta["quantization_setting"]},
                        audit_hook=audit_hook,
                    )
                    runs.append({"tier": label, "method": method, "seed": seed, **result})
                except Exception as e:
                    logger.error("Baseline failed (tier=%s method=%s seed=%d): %s", label, method, seed, e)
                    runs.append({"tier": label, "method": method, "seed": seed, "error": str(e)})
                finally:
                    # See train_graft.train_one: releasing only on the success path let one
                    # failed arm cascade into every arm that followed it on the same tier.
                    T.release_model(model)

        # FairSteer changes no weight; it is prepared here and applied at evaluation.
        compute_fairsteer_vector(tier, smoke, label)
        runs.append({"tier": label, "method": "baseline_fairsteer", "type": "inference_time_no_training"})

    (RESULTS_DIR / "train_baselines_runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    log_run_metadata("train_baselines", {"num_runs": len(runs)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
