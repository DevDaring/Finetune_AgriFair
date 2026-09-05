"""Stage B: layer-selective LoRA placement and rank configuration.

Selects layers whose normalized attribution is at or above a threshold (default 0.15 of
max), sets the per-layer LoRA rank to clip(round(rank_max x normalized attribution),
rank_min, rank_max) (Algorithm 1 of the method), and restricts the adapters to the
attention and MLP projections OF THE SELECTED LAYERS ONLY, by handing PEFT a regular
expression over full module names. Also builds matched-budget UNIFORM (evenly spread over
the whole depth) and RANDOM (independently drawn layer sets, several draws) placements for
the placement ablation, and one placement per held-out axis for the leave-one-axis-out
arms.

Rank-pattern keys are anchored (`layers\\.3\\.` cannot match layer 31), and alpha follows
rank per layer so the effective scaling alpha/r is constant across layers.

Run:  python GPU_Run/configure_layer_selective_lora.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import random
from typing import Dict, List, Optional

import numpy as np

from GPU_Run.common import model_registry
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.model_registry import DEFAULT_LORA_TARGETS
from GPU_Run.common.paths import attribution_path, lora_config_path
from GPU_Run.common.seeds import GLOBAL_SEED

logger = get_logger("configure_layer_selective_lora")

ATTRIBUTION_THRESHOLD = float(os.environ.get("ATTRIBUTION_THRESHOLD", "0.15"))
RANK_MIN = int(os.environ.get("LORA_RANK_MIN", "4"))
RANK_MAX = int(os.environ.get("LORA_RANK_MAX", "64"))
ALPHA_MULTIPLIER = float(os.environ.get("LORA_ALPHA_MULTIPLIER", "2.0"))


def random_placement_draws() -> int:
    try:
        return max(1, int(os.environ.get("RANDOM_PLACEMENT_DRAWS", "3")))
    except ValueError:
        return 3


def loao_axes() -> List[str]:
    raw = os.environ.get("LOAO_AXES", "social_group,landholding,gender")
    return [a.strip() for a in raw.split(",") if a.strip()]


def rank_for(norm_score: float) -> int:
    """Algorithm 1: r = clip(round(rank_max x normalized attribution), rank_min, rank_max)."""
    return int(max(RANK_MIN, min(RANK_MAX, round(RANK_MAX * norm_score))))


# Multimodal checkpoints carry a vision encoder whose blocks are also called "layers", so a
# pattern keyed on layer index alone matches the vision tower as well as the language model.
# On Gemma 3 that put adapters on the SigLIP encoder: parameters trained on text-only data,
# counted in the trainable-parameter budget that every baseline is matched against, and mixed
# into the parameter-space geometry. The study is about language-model layers, so vision
# submodules are excluded by name.
NON_LANGUAGE_SUBMODULES = ("vision_tower", "vision_model", "visual", "image_encoder",
                           "multi_modal_projector", "audio_tower")
_NOT_VISION = rf"(?!.*(?:{'|'.join(NON_LANGUAGE_SUBMODULES)}))"


def target_regex(layers: List[int]) -> str:
    """Full-match regex over module names restricting LoRA to the given layers."""
    layer_alt = "|".join(str(l) for l in sorted(layers))
    mod_alt = "|".join(DEFAULT_LORA_TARGETS)
    return rf"{_NOT_VISION}.*\.(?:layers|h)\.({layer_alt})\.[^.]+\.({mod_alt})"


def rank_pattern_for_layers(layer_ranks: Dict[int, int]) -> Dict[str, int]:
    """Anchored per-(layer, module) regex -> rank for peft's rank_pattern."""
    pattern = {}
    for layer, rank in sorted(layer_ranks.items()):
        for mod in DEFAULT_LORA_TARGETS:
            pattern[rf"{_NOT_VISION}.*(?:layers|h)\.{layer}\.[^.]+\.{mod}"] = int(rank)
    return pattern


def alpha_pattern_for(rank_pattern: Dict[str, int]) -> Dict[str, int]:
    return {k: int(round(ALPHA_MULTIPLIER * v)) for k, v in rank_pattern.items()}


def budget_units(layer_ranks: Dict[int, int]) -> int:
    """Budget proxy: sum of ranks across selected layers x number of target modules. The
    real assertion is on trainable parameters after the adapter is built."""
    return sum(layer_ranks.values()) * len(DEFAULT_LORA_TARGETS)


def spread_budget(total_rank: int, all_layers: List[int], rng: Optional[random.Random] = None) -> Dict[int, int]:
    """Distribute `total_rank` over the depth with ranks in [RANK_MIN, RANK_MAX].

    This is the UNIFORM placement: evenly spaced layers, and every layer once the budget
    affords at least RANK_MIN each. The `rng` argument is retained only for callers that pass
    it positionally; random placement is built by random_placement() instead, because drawing
    a layer set here selects the whole depth and collapses back onto uniform."""
    n = len(all_layers)
    if total_rank >= RANK_MIN * n:
        k = n
    else:
        k = max(1, total_rank // RANK_MIN)
    base = max(RANK_MIN, min(RANK_MAX, total_rank // k))
    remainder = max(0, min(k, total_rank - base * k)) if base < RANK_MAX else 0
    ranks = [base + (1 if i < remainder else 0) for i in range(k)]
    if rng is None:
        idx = np.linspace(0, n - 1, k).round().astype(int).tolist() if k < n else list(range(n))
        layers = [all_layers[i] for i in sorted(set(idx))]
        while len(layers) < k:  # rounding collisions on very short stacks
            for l in all_layers:
                if l not in layers:
                    layers.append(l)
                    break
        layers = sorted(layers)
    else:
        layers = sorted(rng.sample(all_layers, k=k))
    return {l: r for l, r in zip(layers, ranks)}


def random_placement(attr_ranks: Dict[int, int], all_layers: List[int], rng: random.Random) -> Dict[int, int]:
    """The attribution-guided placement's layer COUNT and rank multiset, on a random layer set.

    This is the control that isolates layer identity: it holds constant everything the
    proposed method uses except WHICH layers were chosen, so a difference in outcome can only
    be attributed to the choice of layers.

    It must not go through spread_budget. That function distributes the budget over the whole
    depth, and when the budget affords at least RANK_MIN per layer it selects every layer, so
    the "random" draw returned all of them and sorted them back into the uniform placement.
    All three random draws were then byte-identical to `uniform`, and the randomisation
    control tested nothing at all."""
    k = min(len(attr_ranks), len(all_layers))
    if k >= len(all_layers):
        # Drawing every layer makes the "random" arm identical to uniform after sorting, which
        # is the exact degeneracy this function replaced. Say so rather than emit a control
        # that silently tests nothing.
        logger.warning("Attribution selected all %d layers, so a random layer set cannot "
                       "differ from it; the random placement ablation is degenerate for this "
                       "model and should be read as a duplicate of uniform.", len(all_layers))
    ranks = list(attr_ranks.values())
    rng.shuffle(ranks)                                  # rank is not tied to depth position
    layers = sorted(rng.sample(all_layers, k=k))
    return {l: r for l, r in zip(layers, ranks[:k])}


def pack(name: str, layer_ranks: Dict[int, int], draw_seed: Optional[int] = None) -> Dict:
    rank_pattern = rank_pattern_for_layers(layer_ranks)
    return {
        "placement": name,
        "selected_layers": sorted(layer_ranks.keys()),
        "layer_ranks": {str(k): v for k, v in sorted(layer_ranks.items())},
        "target_modules": target_regex(sorted(layer_ranks.keys())),
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern_for(rank_pattern),
        "default_rank": min(layer_ranks.values()) if layer_ranks else RANK_MIN,
        "budget_units": budget_units(layer_ranks),
        "draw_seed": draw_seed,
    }


def repack_with_rank(placement: Dict, rank: int) -> Dict:
    """Same layers, one forced rank everywhere (rank-sweep ablation)."""
    layer_ranks = {int(k): int(rank) for k in placement["layer_ranks"]}
    out = pack(placement["placement"] + f"_rank{rank}", layer_ranks, placement.get("draw_seed"))
    return out


def build_configs(attribution: dict, draws: Optional[int] = None) -> Dict:
    norm = {int(k): v for k, v in attribution["layer_scores_normalized"].items()}
    if not norm:
        raise SystemExit("Attribution produced no layer scores; cannot configure placement.")
    all_layers = sorted(norm.keys())
    draws = draws or random_placement_draws()

    # attribution-guided placement
    selected = {l: norm[l] for l in all_layers if norm[l] >= ATTRIBUTION_THRESHOLD}
    if not selected:  # guard: keep the single most-attributed layer
        top = max(norm, key=norm.get)
        selected = {top: norm[top]}
    attr_ranks = {l: rank_for(s) for l, s in selected.items()}
    total_rank = sum(attr_ranks.values())

    uniform_ranks = spread_budget(total_rank, all_layers)
    configs = {
        "attribution_threshold": ATTRIBUTION_THRESHOLD,
        "rank_min": RANK_MIN,
        "rank_max": RANK_MAX,
        "alpha_multiplier": ALPHA_MULTIPLIER,
        "number_of_layers": len(all_layers),
        "held_out_axis": attribution.get("held_out_axis", ""),
        "attribution_guided": pack("attribution_guided", attr_ranks),
        "uniform": pack("uniform", uniform_ranks),
    }
    for d in range(draws):
        rng = random.Random(GLOBAL_SEED + d)
        rand_ranks = random_placement(attr_ranks, all_layers, rng)
        key = "random" if d == 0 else f"random_draw{d + 1}"
        configs[key] = pack(key, rand_ranks, draw_seed=GLOBAL_SEED + d)
    configs["random_draw_keys"] = ["random"] + [f"random_draw{d + 1}" for d in range(1, draws)]
    return configs


def main(smoke: bool = False):
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        for held_out in [None] + loao_axes():
            attr_path = attribution_path(label, held_out)
            if not attr_path.exists():
                if held_out is None:
                    logger.warning("Attribution missing for %s (%s); run attribution first.", label, attr_path)
                continue
            attribution = json.loads(attr_path.read_text(encoding="utf-8"))
            configs = build_configs(attribution)
            out = lora_config_path(label, held_out)
            out.write_text(json.dumps(configs, indent=2), encoding="utf-8")
            logger.info(
                "tier=%s held_out=%s attribution layers=%s ranks=%s (budget %d), uniform layers=%d budget %d, random budget %d",
                label, held_out,
                configs["attribution_guided"]["selected_layers"],
                list(configs["attribution_guided"]["layer_ranks"].values()),
                configs["attribution_guided"]["budget_units"],
                len(configs["uniform"]["selected_layers"]),
                configs["uniform"]["budget_units"],
                configs["random"]["budget_units"],
            )
            log_run_metadata("configure_layer_selective_lora",
                             {"tier": label, "held_out_axis": held_out or "", "configs_written": str(out)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
