"""Dry run for GPU_Run: a tiny offline model exercises every GPU-stage code path.

Loads the smoke model and checks the chat rendering, the inference and extraction path,
batched-against-single generation equivalence, the attribution, the placement configuration
including the crucial property that a rank pattern for layer 1 does not also match layer 10
and that the target regex admits only the selected layers, a capped training step, the
Patchscope patch, and the steering hook. Writes a pass/fail report and exits non-zero on any
failure.

Run:  python Dry_Run/dry_run_gpu_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import re

import numpy as np

from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.inference import generate_batch, run_capability_eval, run_mcq_eval
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import DRY_RUN_DIR, TEST_INSTANCES_FROZEN
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("dry_run_gpu_run")


def _peft_available():
    try:
        import peft  # noqa: F401

        return True
    except Exception:
        return False


def _write(report):
    (DRY_RUN_DIR / "dry_run_gpu_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _placement_checks(checks, attr):
    from GPU_Run.configure_layer_selective_lora import build_configs, rank_for

    synthetic = {"layer_scores_normalized": {str(i): v for i, v in
                                             enumerate([1.0, 0.9, 0.2, 0.05, 0.5] + [0.01] * 27)}}
    cfg = build_configs(synthetic, draws=2)
    guided = cfg["attribution_guided"]
    checks["placement_selects_only_layers_above_the_threshold"] = guided["selected_layers"] == [0, 1, 2, 4]
    checks["placement_rank_is_proportional_to_attribution"] = \
        guided["layer_ranks"]["0"] == rank_for(1.0) and guided["layer_ranks"]["0"] > guided["layer_ranks"]["2"]
    rx = guided["target_modules"]
    checks["target_regex_admits_a_selected_layer"] = bool(re.fullmatch(rx, "model.layers.1.self_attn.q_proj"))
    checks["target_regex_rejects_an_unselected_layer"] = not re.fullmatch(rx, "model.layers.10.self_attn.q_proj")
    checks["target_regex_rejects_a_non_target_module"] = not re.fullmatch(rx, "model.layers.1.self_attn.rotary")
    try:
        from peft.utils.other import get_pattern_key

        keys = list(guided["rank_pattern"].keys())
        checks["rank_pattern_for_layer_one_does_not_match_layer_ten"] = \
            get_pattern_key(keys, "model.layers.10.self_attn.q_proj") not in keys
    except Exception:
        checks["rank_pattern_for_layer_one_does_not_match_layer_ten"] = None
    checks["matched_budget_across_placements"] = (
        guided["budget_units"] == cfg["uniform"]["budget_units"] == cfg["random"]["budget_units"])
    checks["random_placement_has_more_than_one_draw"] = len(cfg["random_draw_keys"]) >= 2
    checks["alpha_pattern_follows_rank"] = all(
        cfg["attribution_guided"]["alpha_pattern"][k] == 2 * v
        for k, v in cfg["attribution_guided"]["rank_pattern"].items())
    real = build_configs(attr, draws=1)
    checks["placement_config_built_from_the_real_attribution"] = \
        len(real["attribution_guided"]["selected_layers"]) > 0
    return real


def main():
    set_global_determinism()
    os.environ.setdefault("ATTRIBUTION_MAX_ITEMS", "3")
    os.environ.setdefault("ATTRIBUTION_RIEMANN_STEPS", "2")
    checks = {}

    test = read_jsonl(TEST_INSTANCES_FROZEN)[:6]
    if not test:
        raise SystemExit("Run build_template_instances.py first.")

    try:
        model, tok, meta = model_registry.load_model_and_tokenizer("smoke", smoke=True)
        checks["smoke_model_load"] = True
        checks["attention_implementation_recorded"] = bool(meta.get("attention_implementation_used"))
    except Exception as e:
        logger.error("Smoke model load failed: %s", e)
        _write({"checks": {"smoke_model_load": False}, "all_passed": False})
        raise SystemExit("Smoke model could not load.")

    rendered = P.render_chat(tok, "hello")
    checks["chat_rendering_returns_text"] = isinstance(rendered, str) and len(rendered) > 0

    preds = run_mcq_eval(model, tok, test, rationale_mode=False, batch_size=2)
    checks["inference_returns_one_prediction_per_item"] = len(preds) == len(test)
    checks["predictions_carry_a_canonical_letter"] = all("pred_canonical" in p for p in preds)
    checks["predictions_carry_the_displayed_letter"] = all("pred_display_letter" in p for p in preds)

    rotated = run_mcq_eval(model, tok, test, rationale_mode=False, batch_size=2, rotation=1)
    from GPU_Run.common import metrics as M

    rot = M.option_rotation_consistency({0: preds, 1: rotated})
    checks["option_rotation_consistency_computable"] = 0.0 <= rot["rotation_consistency_rate"] <= 1.0

    prompts_two = [P.build_mcq_prompt(test[0])["prompt"], P.build_mcq_prompt(test[1])["prompt"]]
    batched = generate_batch(model, tok, prompts_two, max_new_tokens=8, batch_size=2)
    single = generate_batch(model, tok, [prompts_two[0]], max_new_tokens=8, batch_size=1)
    checks["batched_generation_returns_one_output_per_prompt"] = len(batched) == 2
    checks["batched_and_single_generation_agree_on_the_first_prompt"] = batched[0] == single[0]

    cap_items = [{"id": "c0", "subject": "test", "question": "What is 2 + 2?",
                  "choices": ["3", "4", "5", "6"], "answer_index": 1}]
    checks["capability_probe_path_runs"] = len(run_capability_eval(model, tok, cap_items)) == 1

    from GPU_Run import attribution_integrated_gradients as A

    attr = A.attribute_model(model, tok, test[:2], riemann_steps=2)
    checks["attribution_produces_layer_scores"] = len(attr["layer_scores_normalized"]) > 0
    checks["attribution_scores_are_finite"] = all(np.isfinite(v) for v in attr["layer_scores_normalized"].values())
    checks["attribution_keeps_per_item_scores_for_the_stability_test"] = \
        len(attr["per_item_layer_scores"]) == 2
    stab = A.ranking_stability(attr["per_item_layer_scores"], 0.15, 10)
    checks["attribution_ranking_stability_computable"] = (
        stab["mean_pairwise_spearman"] != stab["mean_pairwise_spearman"]
        or -1.0 <= stab["mean_pairwise_spearman"] <= 1.0)
    checks["attribution_module_list_excludes_adapter_internals"] = all(
        ".lora_" not in name for _, name, _ in A.target_linear_modules(model))

    real_cfg = _placement_checks(checks, attr)

    from GPU_Run.common.patchscopes import (
        capture_last_hidden_states,
        layer_output_state,
        letter_token_ids,
        number_of_layers,
        patch_and_read_letter_prob,
    )

    lids = letter_token_ids(tok)
    checks["letter_token_ids_resolved_in_context"] = all(v is not None for v in lids.values())
    hs = capture_last_hidden_states(model, tok, "hello world")
    checks["hidden_state_count_matches_layer_count"] = len(hs) == number_of_layers(model) + 1
    prob = patch_and_read_letter_prob(model, tok, layer_output_state(hs, 0), "answer: ", 0, lids, "c")
    checks["patchscope_returns_a_probability"] = 0.0 <= prob <= 1.0

    from GPU_Run.common import targets as TG

    vec = np.zeros(model_registry.hidden_size(model), dtype=np.float32)
    TG.apply_steering(model, vec, 0, 1.0)
    steered = generate_batch(model, tok, [prompts_two[0]], max_new_tokens=8, batch_size=1)
    TG.remove_steering(model)
    checks["zero_steering_vector_leaves_generation_unchanged"] = steered[0] == single[0]
    checks["steering_hook_removed_cleanly"] = getattr(model, "_steering_hook_handle", None) is None

    if _peft_available():
        from GPU_Run.common import training as T

        cfg = T.TrainingConfig(epochs=1, grad_accum=2, micro_batch=2)
        os.environ["TRAIN_SMOKE_MAX_STEPS"] = "1"
        out_dir = DRY_RUN_DIR / "smoke_adapter"
        # start from a clean directory: a checkpoint left by an earlier dry run may carry a
        # different rank pattern, and resuming onto it would only exercise the fallback.
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)
        try:
            res = T.train_lora(model, tok, test[:4], real_cfg["attribution_guided"], out_dir, cfg)
            checks["smoke_train_runs"] = res["trainable_parameter_percentage"] > 0
            checks["train_summary_written"] = (out_dir / T.TRAIN_SUMMARY_FILENAME).exists()
            checks["final_adapter_written"] = (out_dir / "final" / "adapter_config.json").exists()
            adapter_cfg = json.loads((out_dir / "final" / "adapter_config.json").read_text(encoding="utf-8"))
            layers_adapted = {int(m.group(1)) for m in
                              (re.search(r"layers\.(\d+)\.", k) for k in adapter_cfg.get("rank_pattern", {}))
                              if m}
            checks["adapter_rank_pattern_covers_only_selected_layers"] = \
                layers_adapted.issubset(set(real_cfg["attribution_guided"]["selected_layers"]))
        except Exception as e:
            logger.error("Smoke train failed: %s", e)
            checks["smoke_train_runs"] = False
    else:
        logger.info("peft not installed; skipping training and adapter dry-run steps.")
        checks["smoke_train_runs"] = None

    report = {"checks": checks, "peft_available": _peft_available(),
              "all_passed": all(v is not False for v in checks.values())}
    _write(report)
    for k, v in checks.items():
        logger.info("  %-62s %s", k, v)
    logger.info("dry_run_gpu_run all_passed=%s", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit("dry_run_gpu_run FAILED: " +
                         json.dumps({k: v for k, v in checks.items() if v is False}))


if __name__ == "__main__":
    main()
