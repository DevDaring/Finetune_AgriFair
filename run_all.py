"""Single entry point for the AgriFair study.

Runs the pipeline in dependency order and is resumable at every stage. Stages:
  dataset_prep -> dry_run -> gpu_run -> cpu_run

Usage:
  python run_all.py                 # full pipeline, all stages, real models
  python run_all.py --stage gpu_run
  python run_all.py --smoke         # tiny model + subsets + capped steps, offline end to end
  python run_all.py --smoke --stage gpu_run
  python run_all.py --list          # print the stage and step order and exit

After one download every step is offline and resumable. Keys are read only from .env.
A step that fails is logged with its traceback and the run continues, so one broken arm
does not discard the rest; the exit status is non-zero if any step failed, and the failed
steps are named in the final summary.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from GPU_Run.common.logging_utils import get_logger, log_run_metadata

logger = get_logger("run_all")

STAGES = ["dataset_prep", "dry_run", "gpu_run", "frontier_panel", "cpu_run"]
_FAILURES = []


def _set_smoke_env():
    """Small, capped, offline settings for the end-to-end check."""
    defaults = {
        "EVAL_SUBSET_SIZE": "16",
        "BASE_COMPETENCE_SUBSET_SIZE": "16",
        "ADVICE_SUBSET_SIZE": "8",
        "ATTRIBUTION_MAX_ITEMS": "4",
        "ATTRIBUTION_RIEMANN_STEPS": "3",
        "ATTRIBUTION_BOOTSTRAP_RESAMPLES": "20",
        "PROBE_MAX_ITEMS": "12",
        "PROBE_SWAP_SUBSET_SIZE": "8",
        "PATCHSCOPE_MAX_ITEMS": "4",
        "SWAP_SUBSET_SIZE": "8",
        "ROTATION_SUBSET_SIZE": "8",
        "CAPABILITY_SUBSET_SIZE": "8",
        "DART_AUDIT_SUBSET_SIZE": "8",
        "LFTF_MAX_ITEMS": "4",
        "FAIRSTEER_MAX_ITEMS": "4",
        "GEOMETRY_MAX_MATRICES": "8",
        "GEOMETRY_BOOTSTRAP_RESAMPLES": "100",
        "STATISTICS_BOOTSTRAP_RESAMPLES": "100",
        "VERIFY_ATTRIBUTION_MAX_ITEMS": "2",
        "VERIFY_ATTRIBUTION_RIEMANN_STEPS": "2",
        "PATCHSCOPE_METHODS": "graft_proposed",
        "TRAIN_SMOKE_MAX_STEPS": "2",
        "TRAIN_MICRO_BATCH_SIZE": "2",
        "EVAL_BATCH_SIZE": "4",
        "GRAFT_FULL_SWEEP": "0",
        "RUN_LEAVE_ONE_AXIS_OUT": "0",
        "STRICT_BUDGET": "0",
        "HF_HUB_OFFLINE": "0",
        "DISABLE_JUDGE": "1",
        "RUN_FRONTIER_PANEL": "0",
        "RATIONALE_JUDGE_FRACTION": "0",
        "ADVICE_JUDGE_FRACTION": "0",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def _run(label, fn, *args, **kwargs):
    logger.info("=== %s ===", label)
    started = time.time()
    try:
        fn(*args, **kwargs)
        logger.info("--- %s done in %.1f s ---", label, time.time() - started)
        return True
    except SystemExit as e:
        logger.warning("%s exited: %s", label, e)
        _FAILURES.append(label)
        return False
    except Exception:
        logger.error("%s FAILED:\n%s", label, traceback.format_exc())
        _FAILURES.append(label)
        return False


def _maybe_download(smoke):
    """The only networked step. In smoke mode it is skipped entirely: the smoke run uses a
    tiny model and the AgriFair files already on disk, so nothing has to be fetched."""
    from GPU_Run.common.paths import AGRIADVICE_RAW, AGRIFACTS_RAW

    have_data = AGRIFACTS_RAW.exists() and AGRIADVICE_RAW.exists()
    if smoke:
        if not have_data:
            raise SystemExit("Smoke mode needs the AgriFair files in Dataset/; run the download stage first.")
        logger.info("Smoke mode: skipping every download; using the local AgriFair files and the tiny model.")
        return
    from Dataset_Prep import download_models_and_data

    download_models_and_data.main()


def stage_dataset_prep(smoke):
    from Dataset_Prep import (
        build_counterfactual_pairs,
        build_template_instances,
        contamination_check,
        map_external_wang_datasets,
        prepare_prompts_deepseek,
    )

    _run("download_models_and_data", _maybe_download, smoke)
    _run("build_template_instances", build_template_instances.main)
    _run("build_counterfactual_pairs", build_counterfactual_pairs.main)
    _run("prepare_prompts_deepseek", prepare_prompts_deepseek.main)
    _run("map_external_wang_datasets", map_external_wang_datasets.main)
    _run("contamination_check", contamination_check.main)


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
        train_graft,
        verify_bias_subspace,
    )

    _run("measure_base_competence", measure_base_competence.main, smoke)
    _run("probe_subject_models", probe_subject_models.main, smoke)
    _run("attribution_integrated_gradients", attribution_integrated_gradients.main, smoke)
    _run("configure_layer_selective_lora", configure_layer_selective_lora.main, smoke)
    _run("train_graft", train_graft.main, smoke)
    _run("train_baselines", train_baselines.main, smoke)
    _run("evaluate_all", evaluate_all.main, smoke)
    _run("evaluate_agriadvice_drift", evaluate_agriadvice_drift.main, smoke)
    _run("verify_bias_subspace", verify_bias_subspace.main, smoke)
    _run("patchscope_bias_verification", patchscope_bias_verification.main, smoke)


def stage_frontier_panel(smoke):
    """Hosted models, behaviour only. A no-op unless RUN_FRONTIER_PANEL=1, because it spends
    money on third-party APIs. It runs before cpu_run so the audit picks up its predictions."""
    from GPU_Run import evaluate_frontier_panel

    _run("evaluate_frontier_panel", evaluate_frontier_panel.main, smoke)


def stage_cpu_run(smoke):
    from CPU_Run import (
        aggregate_results,
        audit_awareness_trade,
        figures_and_tables,
        judge_robustness,
        parameter_space_geometry,
        repair_admissibility,
        statistics_tests,
    )

    _run("aggregate_results", aggregate_results.main)
    _run("statistics_tests", statistics_tests.main)
    _run("audit_awareness_trade", audit_awareness_trade.main)
    _run("judge_robustness", judge_robustness.main)
    # geometry must precede admissibility: the admissibility table reads the relative
    # Frobenius drift the geometry module writes and uses it as the invasiveness term.
    _run("parameter_space_geometry", parameter_space_geometry.main)
    _run("repair_admissibility", repair_admissibility.main)
    _run("figures_and_tables", figures_and_tables.main)


STAGE_FUNCTIONS = {
    "dataset_prep": stage_dataset_prep,
    "dry_run": stage_dry_run,
    "gpu_run": stage_gpu_run,
    "frontier_panel": stage_frontier_panel,
    "cpu_run": stage_cpu_run,
}

STAGE_STEPS = {
    "dataset_prep": ["download_models_and_data", "build_template_instances", "build_counterfactual_pairs",
                     "prepare_prompts_deepseek", "map_external_wang_datasets", "contamination_check"],
    "dry_run": ["dry_run_dataset_prep", "dry_run_gpu_run", "dry_run_analysis"],
    "gpu_run": ["measure_base_competence", "probe_subject_models", "attribution_integrated_gradients",
                "configure_layer_selective_lora", "train_graft", "train_baselines", "evaluate_all",
                "evaluate_agriadvice_drift", "verify_bias_subspace", "patchscope_bias_verification"],
    "frontier_panel": ["evaluate_frontier_panel"],
    "cpu_run": ["aggregate_results", "statistics_tests", "audit_awareness_trade", "judge_robustness",
                "parameter_space_geometry", "repair_admissibility", "figures_and_tables"],
}


def main():
    ap = argparse.ArgumentParser(description="AgriFair GRAFT pipeline")
    ap.add_argument("--stage", choices=STAGES, help="run a single stage (default: all)")
    ap.add_argument("--smoke", action="store_true", help="tiny model + subsets + capped steps")
    ap.add_argument("--list", action="store_true", help="print the stage and step order and exit")
    args = ap.parse_args()
    if args.list:
        for stage in STAGES:
            print(stage)
            for step in STAGE_STEPS[stage]:
                print(f"    {step}")
        return 0
    if args.smoke:
        _set_smoke_env()
        logger.info("SMOKE mode: tiny model, subset data, capped steps.")

    stages = [args.stage] if args.stage else STAGES
    started = time.time()
    for st in stages:
        STAGE_FUNCTIONS[st](args.smoke)
    minutes = (time.time() - started) / 60.0
    if _FAILURES:
        logger.error("run_all finished with %d failed step(s): %s", len(_FAILURES), ", ".join(_FAILURES))
    else:
        logger.info("run_all complete: stages=%s smoke=%s in %.1f minutes.", stages, args.smoke, minutes)
    log_run_metadata("run_all", {"stages": stages, "smoke": args.smoke,
                                 "wall_clock_minutes": round(minutes, 2), "failed_steps": _FAILURES})
    return 1 if _FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
