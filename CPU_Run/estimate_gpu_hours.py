"""GPU wall-clock and hardware estimate for the local half of the study.

The arm counts are read from the code that actually schedules them, not typed in here, so
this estimate moves when the study does. The timing model is physical rather than fitted:

  training     is compute-bound. tokens/second = GPU bf16 throughput x utilisation
               / (6 x parameters). The factor of six is forward plus backward; LoRA saves
               optimiser memory rather than compute, so it does not reduce it much.
  generation   is memory-bandwidth-bound, not compute-bound. Each decode step reads every
               weight once and serves the whole batch from that read, so
               tokens/second = bandwidth / (2 x parameters) x batch x efficiency.

That second point is the one that decides the hardware. Most of the wall clock in this study
is generation, not training, so memory bandwidth matters more than peak FLOPS, and an H100
beats an A6000 by roughly its bandwidth ratio rather than its FLOPS ratio.

Utilisation and efficiency are deliberately conservative: small micro-batches at sequence
length 1024 do not reach the numbers a throughput benchmark would quote.

Run:  python CPU_Run/estimate_gpu_hours.py
      python CPU_Run/estimate_gpu_hours.py --gpu "A100 80GB" --skip-advice-on-ablations
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

from GPU_Run.common import methods as METHODS
from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, write_csv
from GPU_Run.common.paths import AGRIADVICE_PAIRS, TEST_INSTANCES_FROZEN, TRAIN_INSTANCES
from GPU_Run.common.seeds import THREE_SEEDS

logger = get_logger("estimate_gpu_hours")

TRAINING_UTILISATION = 0.25     # fraction of peak bf16 achieved at these batch sizes
DECODE_EFFICIENCY = 0.45        # fraction of theoretical bandwidth-bound decode achieved
BYTES_PER_PARAMETER = 2         # bf16
LOAD_MINUTES_PER_RUN = 1.5      # checkpoint load, adapter attach, teardown


@dataclass(frozen=True)
class Gpu:
    name: str
    memory_gb: int
    bf16_tflops: float
    bandwidth_gb_s: float
    typical_usd_per_hour: float


# Prices are the 25th percentile of live Vast.ai single-GPU offers with at least 150 GB of
# disk and 200 Mbps down, read from their API on 2026-09-05. The cheapest listing is always
# lower than this and is usually the least reliable host, so the quartile is the number worth
# planning against. Re-read them before committing: this is a spot market.
GPUS = [
    Gpu("RTX 4090 24GB", 24, 165, 1008, 0.35),
    Gpu("L4 24GB", 24, 121, 300, 0.25),
    Gpu("A6000 48GB", 48, 155, 768, 0.404),
    Gpu("L40S 48GB", 48, 362, 864, 0.541),
    Gpu("A100 40GB", 40, 312, 1555, 0.668),
    Gpu("A100 80GB", 80, 312, 2039, 0.873),
    Gpu("H100 80GB", 80, 990, 3350, 2.002),
    Gpu("H200 141GB", 141, 990, 4800, 2.90),
]

COLUMNS = [
    "gpu_name", "memory_gigabytes", "fits_every_model_in_bf16",
    "training_hours", "generation_hours", "overhead_hours", "total_hours",
    "typical_usd_per_hour", "estimated_usd",
]


@dataclass
class Workload:
    training_runs_per_model: int
    eval_targets_per_model: int
    advice_targets_per_model: int
    train_tokens_per_run: float
    mcq_items: int
    capability_items: int
    swap_and_rotation_items: int
    advice_generations: int
    mcq_output_tokens: int
    advice_output_tokens: int
    capability_output_tokens: int


def count_arms(tier: str) -> Dict[str, int]:
    """Read the arm schedule for one model out of the modules that define it.

    This is tier-aware on purpose: MULTI_SEED_TIERS, RANK_SWEEP_TIERS and LOAO_TIERS change
    what runs on which model, so an estimate that ignored them would report the full study's
    cost for a reduced one."""
    from GPU_Run.train_baselines import baseline_arms
    from GPU_Run.train_graft import arms as graft_arms
    from GPU_Run.train_graft import runs_on_tier, transfer_axes

    def seeds_for(method: str) -> int:
        if method == METHODS.PROPOSED:
            return len(THREE_SEEDS)
        return len(THREE_SEEDS) if (method in METHODS.MULTI_SEED_METHODS
                                    and runs_on_tier("MULTI_SEED_TIERS", tier)) else 1

    graft = [(m, k, o) for m, k, o in graft_arms(full_sweep=True)
             if not (m.startswith("ablation_rank_sweep") and not runs_on_tier("RANK_SWEEP_TIERS", tier))]
    graft_runs = sum(seeds_for(m) for m, _, _ in graft)
    baseline_runs = sum(seeds_for(m) for m, _, _ in baseline_arms({"number_of_layers": 0})
                        if m != "baseline_fairsteer")
    loao_runs = (len(transfer_axes()) * len(METHODS.TRANSFER_METHODS)
                 if runs_on_tier("LOAO_TIERS", tier) else 0)
    training_runs = graft_runs + baseline_runs + loao_runs
    # every trained arm is evaluated, plus the frozen base and the steering baseline
    eval_targets = training_runs + 2
    # Advice drift covers every arm unless ADVICE_TARGET_SCOPE narrows it, so the estimate
    # must read the same switch the run does. Quoting the narrowed cost for a full run would
    # understate the bill by roughly half of all generation.
    ablation_methods = sum(1 for m, _, _ in graft if METHODS.is_ablation(m))
    if os.environ.get("ADVICE_TARGET_SCOPE", "all") == "all":
        advice_targets = eval_targets
    else:
        headline = (eval_targets
                    - sum(seeds_for(m) for m, _, _ in graft if METHODS.is_ablation(m))
                    - loao_runs)
        advice_targets = headline - sum(
            seeds_for(m) - 1 for m, _, _ in graft if not METHODS.is_ablation(m)) - sum(
            seeds_for(m) - 1 for m, _, _ in baseline_arms({"number_of_layers": 0})
            if m != "baseline_fairsteer")
    return {"training_runs": training_runs, "eval_targets": eval_targets,
            "advice_targets": max(1, advice_targets), "ablation_runs": ablation_methods}


def build_workload(tier: str) -> Workload:
    train = read_jsonl(TRAIN_INSTANCES)
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    advice = read_jsonl(AGRIADVICE_PAIRS)
    counts = count_arms(tier)
    # prompt plus rationale target, measured shape of the templated instances
    tokens_per_example = 460
    epochs = 3
    replay = 1.10
    return Workload(
        training_runs_per_model=counts["training_runs"],
        eval_targets_per_model=counts["eval_targets"],
        advice_targets_per_model=counts["advice_targets"],
        train_tokens_per_run=len(train) * epochs * tokens_per_example * replay,
        mcq_items=len(test),
        capability_items=int(os.environ.get("CAPABILITY_PROBE_SIZE", "500")),
        swap_and_rotation_items=600,
        advice_generations=2 * len(advice),
        mcq_output_tokens=180,       # answer arms are short, rationale arms longer
        advice_output_tokens=int(os.environ.get("ADVICE_MAX_NEW_TOKENS", "256")),
        capability_output_tokens=48,
    )


def memory_needed_gb(parameters_billion: float, micro_batch: int) -> float:
    """bf16 weights plus adapters, optimiser state and activations, with headroom."""
    weights = parameters_billion * BYTES_PER_PARAMETER
    activations = 0.9 * micro_batch * (parameters_billion / 4.0)
    return weights + activations + 3.0


def model_hours(gpu: Gpu, spec, work: Workload) -> Dict[str, float]:
    params = spec.parameter_count_billions
    micro = model_registry.safe_train_micro_batch_size(spec, 8)
    batch = model_registry.safe_eval_batch_size(spec, 16)

    train_tokens_per_second = (gpu.bf16_tflops * 1e12 * TRAINING_UTILISATION) / (6 * params * 1e9)
    decode_tokens_per_second = (gpu.bandwidth_gb_s * 1e9 / (2 * params * 1e9)) * batch * DECODE_EFFICIENCY

    training_seconds = work.training_runs_per_model * work.train_tokens_per_run / train_tokens_per_second
    generated = (
        work.eval_targets_per_model * work.mcq_items * work.mcq_output_tokens
        + work.eval_targets_per_model * work.capability_items * work.capability_output_tokens
        + work.eval_targets_per_model * work.swap_and_rotation_items * work.mcq_output_tokens
        + work.advice_targets_per_model * work.advice_generations * work.advice_output_tokens
    )
    generation_seconds = generated / decode_tokens_per_second
    overhead_seconds = (work.training_runs_per_model + work.eval_targets_per_model) * LOAD_MINUTES_PER_RUN * 60
    return {"training": training_seconds / 3600, "generation": generation_seconds / 3600,
            "overhead": overhead_seconds / 3600,
            "fits": memory_needed_gb(params, micro) <= gpu.memory_gb}


def main():
    ap = argparse.ArgumentParser(description="GPU hour and hardware estimate")
    ap.add_argument("--gpu", help="report only this GPU by name")
    ap.add_argument("--skip-advice-on-ablations", action="store_true", default=True)
    args = ap.parse_args()

    tiers = model_registry.active_tiers()
    specs = [model_registry.get_spec(t) for t in tiers]
    work_by_tier = {t: build_workload(t) for t in tiers}
    for t in tiers:
        w = work_by_tier[t]
        logger.info("%-20s %d training runs, %d evaluation targets, %d advice targets.",
                    t, w.training_runs_per_model, w.eval_targets_per_model, w.advice_targets_per_model)
    logger.info("Per training run: about %.2f million tokens.",
                work_by_tier[tiers[0]].train_tokens_per_run / 1e6)

    rows: List[Dict] = []
    for gpu in GPUS:
        if args.gpu and gpu.name != args.gpu:
            continue
        totals = {"training": 0.0, "generation": 0.0, "overhead": 0.0}
        fits_all = True
        for spec in specs:
            h = model_hours(gpu, spec, work_by_tier[spec.tier])
            for k in totals:
                totals[k] += h[k]
            fits_all = fits_all and h["fits"]
        total = sum(totals.values())
        rows.append({
            "gpu_name": gpu.name, "memory_gigabytes": gpu.memory_gb,
            "fits_every_model_in_bf16": "yes" if fits_all else "no",
            "training_hours": round(totals["training"], 1),
            "generation_hours": round(totals["generation"], 1),
            "overhead_hours": round(totals["overhead"], 1),
            "total_hours": round(total, 1),
            "typical_usd_per_hour": gpu.typical_usd_per_hour,
            "estimated_usd": round(total * gpu.typical_usd_per_hour, 0),
        })

    write_csv(Path(__file__).resolve().parents[1] / "results" / "gpu_hours_estimate.csv", rows, COLUMNS)
    total_runs = sum(w.training_runs_per_model for w in work_by_tier.values())
    print(f"\nAcross all four models: {total_runs} training runs, "
          f"{sum(w.eval_targets_per_model for w in work_by_tier.values())} evaluation targets")
    print(f"{'GPU':<16s}{'VRAM':>6s}{'fits all':>10s}{'train':>8s}{'generate':>10s}"
          f"{'total':>8s}{'$/h':>7s}{'est. $':>9s}")
    for r in rows:
        print(f"  {r['gpu_name']:<14s}{r['memory_gigabytes']:>5d}G{r['fits_every_model_in_bf16']:>10s}"
              f"{r['training_hours']:>8.1f}{r['generation_hours']:>10.1f}{r['total_hours']:>8.1f}"
              f"{r['typical_usd_per_hour']:>7.2f}{r['estimated_usd']:>9.0f}")

    per_model_note = []
    for spec in specs:
        need = memory_needed_gb(spec.parameter_count_billions,
                                model_registry.safe_train_micro_batch_size(spec, 8))
        per_model_note.append(f"{spec.tier} ({spec.parameter_count_billions:.1f}B) needs about {need:.0f} GB")
    print("\nMemory needed to train each model in bf16 at its configured micro-batch:")
    for note in per_model_note:
        print(f"  {note}")


if __name__ == "__main__":
    main()
