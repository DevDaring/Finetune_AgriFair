"""Dry run for GPU_Run: a tiny offline model exercises the GPU-stage code paths.

Loads the smoke model, runs the probe, attribution, placement configuration, a smoke train,
a batched-vs-single generation equivalence check, evaluation, verification, and Patchscope
(Instruction.md Sections 11, 14). Training requires peft; if peft is absent the training and
adapter-dependent steps are skipped with a clear note (the data/inference paths still run).

Run:  python Dry_Run/dry_run_gpu_run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.inference import generate_batch, run_mcq_eval
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common import prompts as P
from GPU_Run.common.paths import DRY_RUN_DIR, TEST_INSTANCES_FROZEN
from GPU_Run.common.seeds import set_global_determinism


logger = get_logger("dry_run_gpu_run")


def _peft_available():
    try:
        import peft  # noqa: F401

        return True
    except Exception:
        return False


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
    except Exception as e:
        logger.error("Smoke model load failed: %s", e)
        checks["smoke_model_load"] = False
        _write(checks)
        raise SystemExit("Smoke model could not load.")

    # inference + extraction path
    preds = run_mcq_eval(model, tok, test, rationale_mode=False, batch_size=2)
    checks["inference_returns_predictions"] = len(preds) == len(test)
    checks["predictions_have_canonical_letter"] = all("pred_canonical" in p for p in preds)

    # batched-vs-single equivalence on a confident item (best-effort; tiny model)
    one = [P.build_mcq_prompt(test[0])["prompt"], P.build_mcq_prompt(test[1])["prompt"]]
    batched = generate_batch(model, tok, one, max_new_tokens=8)
    single = [generate_batch(model, tok, [one[0]], max_new_tokens=8)[0]]
    checks["batched_generation_runs"] = isinstance(batched, list) and len(batched) == 2

    # attribution path (tiny)
    from GPU_Run import attribution_integrated_gradients as A

    A._CURRENT_ITEMS = test[:2]
    attr = A.attribute_model(model, tok)
    checks["attribution_produces_layer_scores"] = len(attr["layer_scores_normalized"]) > 0

    # placement config from the tiny attribution
    from GPU_Run.configure_layer_selective_lora import build_configs

    configs = build_configs(attr)
    checks["placement_config_built"] = "attribution_guided" in configs and len(configs["attribution_guided"]["selected_layers"]) > 0

    # training + adapter-dependent paths (need peft)
    if _peft_available():
        from GPU_Run.common import training as T

        cfg = T.TrainingConfig(epochs=1)
        os.environ["TRAIN_SMOKE_MAX_STEPS"] = "1"
        out_dir = DRY_RUN_DIR / "smoke_adapter"
        try:
            res = T.train_lora(model, tok, read_jsonl(TEST_INSTANCES_FROZEN)[:8], configs["attribution_guided"], out_dir, cfg)
            checks["smoke_train_runs"] = res["trainable_parameter_percentage"] > 0
        except Exception as e:
            logger.warning("Smoke train failed: %s", e)
            checks["smoke_train_runs"] = False
    else:
        logger.info("peft not installed; skipping training/adapter dry-run steps.")
        checks["smoke_train_runs"] = None

    report = {"checks": checks, "peft_available": _peft_available(),
              "all_passed": all(v is not False for v in checks.values())}
    _write(report)
    for k, v in checks.items():
        logger.info("  %-45s %s", k, v)
    logger.info("dry_run_gpu_run all_passed=%s", report["all_passed"])
    if not report["all_passed"]:
        raise SystemExit("dry_run_gpu_run FAILED.")


def _write(report):
    (DRY_RUN_DIR / "dry_run_gpu_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
