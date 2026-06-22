"""Stage B: layer-selective LoRA placement and rank configuration.

Selects layers whose normalized attribution is at or above a threshold (default 0.15 of
max), sets the per-layer LoRA rank proportional to normalized attribution clipped to
[4, 64], and places adapters on attention and MLP projection modules of the selected
layers. Also builds matched-budget UNIFORM and RANDOM placements for the placement
ablation (Instruction.md Section 2 Stage B; Section 6.4).

Run:  python GPU_Run/configure_layer_selective_lora.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import random

from GPU_Run.common import model_registry
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.model_registry import DEFAULT_LORA_TARGETS
from GPU_Run.common.paths import RESULTS_DIR
from GPU_Run.common.seeds import GLOBAL_SEED

logger = get_logger("configure_layer_selective_lora")

ATTRIBUTION_THRESHOLD = float(os.environ.get("ATTRIBUTION_THRESHOLD", "0.15"))
RANK_MIN = int(os.environ.get("LORA_RANK_MIN", "4"))
RANK_MAX = int(os.environ.get("LORA_RANK_MAX", "64"))


def _rank_for(norm_score: float) -> int:
    rank = RANK_MIN + norm_score * (RANK_MAX - RANK_MIN)
    return int(max(RANK_MIN, min(RANK_MAX, round(rank))))


def _rank_pattern_for_layers(layer_ranks: dict) -> dict:
    """Map a per-(layer, module) regex to a rank for peft's rank_pattern."""
    pattern = {}
    for layer, rank in layer_ranks.items():
        for mod in DEFAULT_LORA_TARGETS:
            pattern[f"layers.{layer}.\\w+.{mod}"] = rank
            pattern[f"layers.{layer}..*{mod}"] = rank
    return pattern


def _budget_units(layer_ranks: dict) -> int:
    """Crude budget proxy: sum of ranks across selected layers x #target modules."""
    return sum(layer_ranks.values()) * len(DEFAULT_LORA_TARGETS)


def build_configs(attribution: dict):
    norm = {int(k): v for k, v in attribution["layer_scores_normalized"].items()}
    if not norm:
        raise SystemExit("Attribution produced no layer scores; cannot configure placement.")
    all_layers = sorted(norm.keys())

    # attribution-guided placement
    selected = {l: norm[l] for l in all_layers if norm[l] >= ATTRIBUTION_THRESHOLD}
    if not selected:  # guard: keep the single most-attributed layer
        top = max(norm, key=norm.get)
        selected = {top: norm[top]}
    attr_ranks = {l: _rank_for(s) for l, s in selected.items()}
    budget = _budget_units(attr_ranks)

    # uniform placement at matched budget: spread the same total rank over all layers
    n_layers = len(all_layers)
    uniform_rank = max(RANK_MIN, min(RANK_MAX, round(sum(attr_ranks.values()) / max(1, len(attr_ranks)))))
    # adjust the layer count so uniform budget matches: use as many layers as needed
    target_total_rank = sum(attr_ranks.values())
    uniform_layers = all_layers[: max(1, round(target_total_rank / uniform_rank))]
    uniform_ranks = {l: uniform_rank for l in uniform_layers}

    # random placement at matched budget: random layers, same total rank
    rng = random.Random(GLOBAL_SEED)
    rand_layers = sorted(rng.sample(all_layers, k=min(len(uniform_ranks), n_layers)))
    rand_ranks = {l: uniform_rank for l in rand_layers}

    def pack(name, layer_ranks):
        return {
            "placement": name,
            "selected_layers": sorted(layer_ranks.keys()),
            "layer_ranks": {str(k): v for k, v in layer_ranks.items()},
            "target_modules": list(DEFAULT_LORA_TARGETS),
            "rank_pattern": _rank_pattern_for_layers(layer_ranks),
            "default_rank": RANK_MIN,
            "budget_units": _budget_units(layer_ranks),
        }

    return {
        "attribution_threshold": ATTRIBUTION_THRESHOLD,
        "rank_min": RANK_MIN,
        "rank_max": RANK_MAX,
        "attribution_guided": pack("attribution_guided", attr_ranks),
        "uniform": pack("uniform", uniform_ranks),
        "random": pack("random", rand_ranks),
    }


def main(smoke: bool = False):
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        resolved = model_registry.smoke_model_id() if smoke else tier
        label = "smoke" if smoke else tier
        attr_path = RESULTS_DIR / f"attribution_{label}.json"
        if not attr_path.exists():
            logger.warning("Attribution missing for %s (%s); run attribution first.", label, attr_path)
            continue
        attribution = json.loads(attr_path.read_text(encoding="utf-8"))
        configs = build_configs(attribution)
        out = RESULTS_DIR / f"lora_config_{label}.json"
        out.write_text(json.dumps(configs, indent=2), encoding="utf-8")
        logger.info(
            "tier=%s attribution layers=%s (budget %d), uniform budget %d, random budget %d",
            label,
            configs["attribution_guided"]["selected_layers"],
            configs["attribution_guided"]["budget_units"],
            configs["uniform"]["budget_units"],
            configs["random"]["budget_units"],
        )
        log_run_metadata("configure_layer_selective_lora", {"tier": label, "configs_written": str(out)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
