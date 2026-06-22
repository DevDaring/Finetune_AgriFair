"""Single entry point for the AgriFair XLoRA-Bias study (coding_prompt.md Section 11).

Runs the pipeline in dependency order, resumable. Stages:
  dataset_prep -> dry_run -> gpu_run -> cpu_run

Usage:
  python run_all.py                 # full pipeline, all stages, real models
  python run_all.py --stage dataset_prep
  python run_all.py --stage gpu_run
  python run_all.py --smoke         # tiny model + subsets + capped steps, offline end-to-end
  python run_all.py --smoke --stage gpu_run

After one download, every step is offline and resumable. Keys are read only from .env.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from GPU_Run.common.logging_utils import get_logger

logger = get_logger("run_all")

STAGES = ["dataset_prep", "dry_run", "gpu_run", "cpu_run"]


def _set_smoke_env():
    os.environ.setdefault("EVAL_SUBSET_SIZE", "16")
    os.environ.setdefault("BASE_COMPETENCE_SUBSET_SIZE", "16")
    os.environ.setdefault("ADVICE_SUBSET_SIZE", "8")
    os.environ.setdefault("ATTRIBUTION_MAX_ITEMS", "4")
    os.environ.setdefault("ATTRIBUTION_RIEMANN_STEPS", "3")
    os.environ.setdefault("PROBE_MAX_ITEMS", "12")
    os.environ.setdefault("PATCHSCOPE_MAX_ITEMS", "4")
    os.environ.setdefault("TRAIN_SMOKE_MAX_STEPS", "2")
    os.environ.setdefault("TRAIN_MICRO_BATCH_SIZE", "1")
    os.environ.setdefault("EVAL_BATCH_SIZE", "4")
    os.environ.setdefault("XLORA_FULL_SWEEP", "0")
    os.environ.setdefault("STRICT_BUDGET", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    # keep the smoke run fully offline: no live judge API calls
    os.environ.setdefault("DISABLE_JUDGE", "1")
    os.environ.setdefault("RATIONALE_JUDGE_FRACTION", "0")
    os.environ.setdefault("ADVICE_JUDGE_FRACTION", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _run(label, fn, *args, **kwargs):
    logger.info("=== %s ===", label)
    try:
        fn(*args, **kwargs)
        return True
    except SystemExit as e:
        logger.warning("%s exited: %s", label, e)
        return False
    except Exception:
        logger.error("%s FAILED:\n%s", label, traceback.format_exc())
        return False


def stage_dataset_prep(smoke):
    from Dataset_Prep import build_counterfactual_pairs, build_template_instances
    from Dataset_Prep import contamination_check, map_external_wang_datasets, prepare_prompts_deepseek

    _run("download_models_and_data (skipped unless missing)", _maybe_download)
    _run("build_template_instances", build_template_instances.main)
    _run("build_counterfactual_pairs", build_counterfactual_pairs.main)
    _run("prepare_prompts_deepseek", prepare_prompts_deepseek.main)
    _run("map_external_wang_datasets", map_external_wang_datasets.main)
    _run("contamination_check", contamination_check.main)


def _maybe_download():
    from GPU_Run.common.paths import AGRIADVICE_RAW, AGRIFACTS_RAW

    if AGRIFACTS_RAW.exists() and AGRIADVICE_RAW.exists():
        logger.info("AgriFair data already present; skipping download.")
        return
    from Dataset_Prep import download_models_and_data

    download_models_and_data.main()


def stage_dry_run(smoke):
    from Dry_Run import dry_run_analysis, dry_run_dataset_prep, dry_run_gpu_run

    _run("dry_run_dataset_prep", dry_run_dataset_prep.main)
    _run("dry_run_gpu_run", dry_run_gpu_run.main)
    _run("dry_run_analysis", dry_run_analysis.main)


def stage_gpu_run(smoke):
    from GPU_Run import (
        attribution_integrated_gradients,
        configure_layer_selective_lora,
        evaluate_agriadvice_drift,
        evaluate_all,
        measure_base_competence,
        patchscope_bias_verification,
        probe_subject_models,
        train_baselines,
        train_xlora_bias,
        verify_bias_subspace,
    )

    _run("measure_base_competence", measure_base_competence.main, smoke)
    _run("probe_subject_models", probe_subject_models.main, smoke)
    _run("attribution_integrated_gradients", attribution_integrated_gradients.main, smoke)
    _run("configure_layer_selective_lora", configure_layer_selective_lora.main, smoke)
    _run("train_xlora_bias", train_xlora_bias.main, smoke)
    _run("train_baselines", train_baselines.main, smoke)
    _run("evaluate_all", evaluate_all.main, smoke)
    _run("evaluate_agriadvice_drift", evaluate_agriadvice_drift.main, smoke)
    _run("verify_bias_subspace", verify_bias_subspace.main, smoke)
    _run("patchscope_bias_verification", patchscope_bias_verification.main, smoke)


def stage_cpu_run(smoke):
    from CPU_Run import (
        aggregate_results,
        figures_and_tables,
        judge_robustness,
        parameter_space_geometry,
        statistics_tests,
    )

    _run("aggregate_results", aggregate_results.main)
    _run("statistics_tests", statistics_tests.main)
    _run("figures_and_tables", figures_and_tables.main)
    _run("judge_robustness", judge_robustness.main)
    _run("parameter_space_geometry", parameter_space_geometry.main)


def main():
    ap = argparse.ArgumentParser(description="AgriFair XLoRA-Bias pipeline")
    ap.add_argument("--stage", choices=STAGES, help="run a single stage (default: all)")
    ap.add_argument("--smoke", action="store_true", help="tiny model + subsets + capped steps")
    args = ap.parse_args()
    if args.smoke:
        _set_smoke_env()
        logger.info("SMOKE mode: tiny model, subset data, capped steps.")

    stages = [args.stage] if args.stage else STAGES
    for st in stages:
        {"dataset_prep": stage_dataset_prep, "dry_run": stage_dry_run,
         "gpu_run": stage_gpu_run, "cpu_run": stage_cpu_run}[st](args.smoke)
    logger.info("run_all complete: stages=%s smoke=%s", stages, args.smoke)


if __name__ == "__main__":
    main()
