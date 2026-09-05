"""Discovery and loading of evaluation targets: the frozen base, every saved adapter,
and the inference-time steering baseline.

Every evaluation script (main evaluation, advice drift, identity probe, Patchscope)
uses this one discovery so the same set of methods appears in every table, and the
inference-time baseline (FairSteer) is not silently dropped because it has no adapter
directory.

# Li, Y., Fan, Z., Chen, R., et al. "FairSteer: Inference Time Debiasing for LLMs with
#   Dynamic Activation Steering." Findings of ACL 2025, arXiv:2510.18914.
#   [steering vector added to the residual stream at a chosen layer at inference time]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from GPU_Run.common import model_registry
from GPU_Run.common.logging_utils import get_logger
from GPU_Run.common.paths import CHECKPOINTS_DIR, LOAO_INFIX, RESULTS_DIR

logger = get_logger("targets")

FAIRSTEER_METHOD = "baseline_fairsteer"


@dataclass
class EvalTarget:
    method: str
    seed: int
    kind: str                      # frozen_base | adapter | steering
    adapter_dir: Optional[Path] = None
    quantized: bool = False
    steering_file: Optional[Path] = None

    @property
    def key(self):
        return (self.method, self.seed)


def fairsteer_vector_path(label: str) -> Path:
    return RESULTS_DIR / f"fairsteer_vector_{label}.json"


def _allow_rehearsals() -> bool:
    """True while a verification pass is running, when rehearsal adapters ARE the subject."""
    return bool(os.environ.get("TRAIN_SMOKE_MAX_STEPS"))


def _is_rehearsal(seed_dir) -> bool:
    summary = Path(seed_dir) / "train_summary.json"
    if not summary.exists():
        return False
    try:
        return bool(json.loads(summary.read_text(encoding="utf-8")).get("step_capped_rehearsal"))
    except Exception:
        return False


def discover_targets(
    label: str,
    include_frozen_base: bool = True,
    include_steering: bool = True,
    methods: Optional[Sequence[str]] = None,
    exclude_prefixes: Sequence[str] = (),
    first_seed_only: bool = False,
    exclude_transfer_arms: bool = False,
) -> List[EvalTarget]:
    """List every target for a model label.

    `methods` restricts to named methods; `exclude_prefixes` drops method-name prefixes such
    as "ablation_"; `first_seed_only` keeps one seed per method; `exclude_transfer_arms` drops
    the leave-one-axis-out adapters, which answer a question about AgriFacts transfer and have
    nothing to say about free-text advice drift or option-rotation robustness."""
    targets: List[EvalTarget] = []
    if include_frozen_base:
        targets.append(EvalTarget("frozen_base", 42, "frozen_base"))
    root = CHECKPOINTS_DIR / label
    if root.exists():
        for mdir in sorted(root.iterdir()):
            if not mdir.is_dir():
                continue
            name = mdir.name
            if methods is not None and name not in methods:
                continue
            if any(name.startswith(p) for p in exclude_prefixes):
                continue
            if exclude_transfer_arms and LOAO_INFIX in name:
                continue
            seed_dirs = sorted(mdir.glob("seed_*"))
            if first_seed_only:
                seed_dirs = seed_dirs[:1]
            for sdir in seed_dirs:
                final = sdir / "final"
                adapter = final if (final / "adapter_config.json").exists() else None
                if adapter is None:
                    epochs = sorted(sdir.glob("epoch_*"))
                    adapter = epochs[-1] if epochs else None
                if adapter is None:
                    logger.warning("No adapter found under %s; skipping.", sdir)
                    continue
                if _is_rehearsal(sdir) and not _allow_rehearsals():
                    # A step-capped rehearsal adapter has seen a couple of optimiser steps.
                    # Evaluating it would put a row in the results table that looks like every
                    # other row. The training stage discards these before retraining, but an
                    # arm that failed BEFORE reaching that point (an OOM at model load, a
                    # budget assertion) leaves one behind, and nothing downstream could tell.
                    logger.warning("Skipping %s/%s seed %s: its adapter is a step-capped "
                                   "rehearsal, not a trained arm.", label, name, sdir.name)
                    continue
                seed = int(sdir.name.split("_")[1])
                targets.append(EvalTarget(name, seed, "adapter", adapter_dir=adapter, quantized=("qlora" in name)))
    if include_steering and (methods is None or FAIRSTEER_METHOD in methods):
        sv = fairsteer_vector_path(label)
        if sv.exists() and not any(FAIRSTEER_METHOD.startswith(p) for p in exclude_prefixes):
            targets.append(EvalTarget(FAIRSTEER_METHOD, 42, "steering", steering_file=sv))
    return targets


# ------------------------------ steering hook --------------------------------

def steering_coefficient() -> float:
    try:
        return float(os.environ.get("FAIRSTEER_COEFFICIENT", "1.0"))
    except ValueError:
        return 1.0


def apply_steering(model, vector, layer_index: int, coefficient: float):
    """Add coefficient * vector to the residual stream output of one decoder layer at
    every position during the forward pass. Returns the hook handle."""
    import torch

    from GPU_Run.common.patchscopes import _decoder_layers

    layers = _decoder_layers(model)
    layer_index = max(0, min(layer_index, len(layers) - 1))
    vec = torch.tensor(vector, dtype=next(model.parameters()).dtype, device=model.device) * coefficient

    def hook(module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0] + vec
            return (hidden,) + tuple(out[1:])
        return out + vec

    handle = layers[layer_index].register_forward_hook(hook)
    model._steering_hook_handle = handle
    return handle


def remove_steering(model):
    handle = getattr(model, "_steering_hook_handle", None)
    if handle is not None:
        handle.remove()
        model._steering_hook_handle = None


def load_target(tier: str, target: EvalTarget, smoke: bool = False):
    """Load the base model for `tier` and apply the target's adapter or steering vector.
    Returns (model, tokenizer, meta) with meta["method"] set."""
    model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke, quantized_4bit=target.quantized)
    if target.kind == "adapter" and target.adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(target.adapter_dir))
        model.eval()
    elif target.kind == "steering" and target.steering_file is not None:
        data = json.loads(target.steering_file.read_text(encoding="utf-8"))
        vec = data.get("steering_vector") or []
        if vec:
            apply_steering(model, vec, int(data.get("steering_layer_index", 0)), steering_coefficient())
            meta["steering_layer_index"] = int(data.get("steering_layer_index", 0))
            meta["steering_coefficient"] = steering_coefficient()
        else:
            logger.warning("Steering file %s has no vector; evaluating the unsteered base.", target.steering_file)
    meta["method"] = target.method
    meta["random_seed"] = target.seed
    return model, tok, meta
