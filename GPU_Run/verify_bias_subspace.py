"""Stage D: attribution re-run and the linear identity probe.

Two readouts, computed for the frozen base and for every trained method so the panel is
demonstrably method-agnostic rather than asserted to be:

  attribution re-run   the same Integrated-Gradients attribution recomputed ON THE ADAPTED
                       MODEL and compared with the pre-repair attribution: the share of the
                       attribution mass still carried by the originally targeted layers, the
                       Spearman correlation between the pre- and post-repair layer rankings,
                       and the Jaccard overlap of their top-k layers. Recomputation is the
                       point; reading the stored pre-repair file back would compare a number
                       with itself.
  identity probe       cross-validated accuracy of a logistic probe decoding the swapped
                       identity from a mid-depth hidden state, on the AgriFacts MCQ swaps
                       and separately on the AgriAdvice persona pairs. Higher means more
                       residual identity signal.

Run:  python GPU_Run/verify_bias_subspace.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
from typing import Dict, Optional

import numpy as np

from GPU_Run.common import metrics as M
from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common import targets as TG
from GPU_Run.common import training as T
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import append_csv_row, get_logger, log_run_metadata
from GPU_Run.common.patchscopes import capture_last_hidden_states
from GPU_Run.common.paths import (
    AGRIADVICE_PAIRS,
    COUNTERFACTUAL_PAIRS,
    RESULTS_DIR,
    VALIDATION_INSTANCES,
    attribution_path,
    lora_config_path,
    split_loao_method,
)
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("verify_bias_subspace")

PROBE_MAX_ITEMS = int(os.environ.get("PROBE_MAX_ITEMS", "200"))
RERUN_ATTRIBUTION = os.environ.get("VERIFY_RERUN_ATTRIBUTION", "1") == "1"

COLUMNS = [
    "tier", "method", "random_seed",
    "attribution_mass_fraction_on_originally_targeted_layers",
    "attribution_rank_spearman_before_versus_after_repair",
    "attribution_top_k_layer_jaccard_before_versus_after_repair",
    "identity_probe_accuracy_on_agrifacts_swaps",
    "identity_probe_accuracy_on_agriadvice_personas",
    "residual_identity_signal_relative_to_frozen_base",
]


def _mid_hidden(model, tok, text, add_special):
    hs = capture_last_hidden_states(model, tok, text, add_special)
    return np.asarray(hs[len(hs) // 2])


def _probe_mcq(model, tok):
    cf = read_jsonl(COUNTERFACTUAL_PAIRS)
    val_ids = {r["id"] for r in read_jsonl(VALIDATION_INSTANCES)}
    cf = [c for c in cf if c["original"]["id"] in val_ids][:PROBE_MAX_ITEMS]
    add_special = P.prompt_add_special_tokens(tok)
    feats, labels = [], []
    for c in cf:
        feats.append(_mid_hidden(model, tok, P.render_chat(tok, P.build_mcq_prompt(c["original"])["prompt"]), add_special))
        labels.append(0)
        feats.append(_mid_hidden(model, tok, P.render_chat(tok, P.build_mcq_prompt(c["swapped"])["prompt"]), add_special))
        labels.append(1)
    if not feats:
        return float("nan")
    return M.linear_identity_probe(np.vstack(feats), np.asarray(labels))


def _probe_advice(model, tok):
    pairs = read_jsonl(AGRIADVICE_PAIRS)[:PROBE_MAX_ITEMS]
    add_special = P.prompt_add_special_tokens(tok)
    feats, labels = [], []
    for p in pairs:
        feats.append(_mid_hidden(model, tok, P.render_chat(tok, p["prompt_a"]), add_special))
        labels.append(0)
        feats.append(_mid_hidden(model, tok, P.render_chat(tok, p["prompt_b"]), add_special))
        labels.append(1)
    if not feats:
        return float("nan")
    return M.linear_identity_probe(np.vstack(feats), np.asarray(labels))


def _attribution_readout(label, method, model, tok) -> Dict[str, float]:
    """Recompute the attribution on this model and compare it with the pre-repair map."""
    from GPU_Run.attribution_integrated_gradients import (
        attribute_model,
        compare_layer_rankings,
        failure_items,
    )

    base_method, held_out = split_loao_method(method)
    attr_file = attribution_path(label, held_out)
    cfg_file = lora_config_path(label, held_out)
    if not (attr_file.exists() and cfg_file.exists()):
        return {}
    before = json.loads(attr_file.read_text(encoding="utf-8"))["layer_scores_normalized"]
    targeted = set(json.loads(cfg_file.read_text(encoding="utf-8"))["attribution_guided"]["selected_layers"])
    if not RERUN_ATTRIBUTION:
        return {}
    items = failure_items(label, exclude_axis=held_out,
                          max_items=int(os.environ.get("VERIFY_ATTRIBUTION_MAX_ITEMS", "16")))
    if not items:
        return {}
    steps = int(os.environ.get("VERIFY_ATTRIBUTION_RIEMANN_STEPS", "20"))
    after = attribute_model(model, tok, items, riemann_steps=steps)["layer_scores_normalized"]
    total = sum(after.values()) or 1.0
    on_target = sum(v for l, v in after.items() if int(l) in targeted)
    cmp = compare_layer_rankings(before, after, k=max(1, len(targeted)))
    return {
        "attribution_mass_fraction_on_originally_targeted_layers": on_target / total,
        "attribution_rank_spearman_before_versus_after_repair": cmp["spearman"],
        "attribution_top_k_layer_jaccard_before_versus_after_repair": cmp["top_k_jaccard"],
    }


def _num(v, nd=4):
    return round(float(v), nd) if (v is not None and v == v) else ""


def main(smoke: bool = False):
    set_global_determinism()
    out_path = RESULTS_DIR / "identity_probe_and_attribution_rerun.csv"
    if out_path.exists():
        out_path.unlink()
    n_rows = 0
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        base_probe: Optional[float] = None
        for target in TG.discover_targets(label, include_steering=True, exclude_prefixes=("ablation_",),
                                          first_seed_only=True):
            method, seed = target.method, target.seed
            model = None
            try:
                model, tok, meta = TG.load_target(tier, target, smoke=smoke)
            except Exception as e:
                # Release whatever was allocated before the failure. Leaving it resident is
                # what turned one out-of-memory error into a run of them, and a card under
                # pressure also produces confusing secondary errors such as a dispatched
                # model missing prepare_inputs_for_generation.
                logger.error("verify load failed for %s (%s); skipping.", method, e)
                T.release_model(model)
                continue
            try:
                probe_mcq = _probe_mcq(model, tok)
                probe_adv = _probe_advice(model, tok)
                attr = {}
                if target.kind == "adapter":
                    try:
                        attr = _attribution_readout(label, method, model, tok)
                    except Exception as e:
                        logger.warning("Attribution re-run failed for %s (%s); left blank.", method, e)
                if method == "frozen_base":
                    base_probe = probe_mcq
                relative = (probe_mcq / base_probe) if (base_probe and base_probe == base_probe and base_probe > 0
                                                       and probe_mcq == probe_mcq) else float("nan")
                append_csv_row(out_path, {
                    "tier": label, "method": method, "random_seed": seed,
                    "attribution_mass_fraction_on_originally_targeted_layers":
                        _num(attr.get("attribution_mass_fraction_on_originally_targeted_layers")),
                    "attribution_rank_spearman_before_versus_after_repair":
                        _num(attr.get("attribution_rank_spearman_before_versus_after_repair")),
                    "attribution_top_k_layer_jaccard_before_versus_after_repair":
                        _num(attr.get("attribution_top_k_layer_jaccard_before_versus_after_repair")),
                    "identity_probe_accuracy_on_agrifacts_swaps": _num(probe_mcq),
                    "identity_probe_accuracy_on_agriadvice_personas": _num(probe_adv),
                    "residual_identity_signal_relative_to_frozen_base": _num(relative),
                }, COLUMNS)
                n_rows += 1
                logger.info("tier=%s method=%s probe_mcq=%.3f probe_advice=%.3f",
                            label, method, probe_mcq if probe_mcq == probe_mcq else float("nan"),
                            probe_adv if probe_adv == probe_adv else float("nan"))
            finally:
                TG.remove_steering(model)
                T.release_model(model)

    log_run_metadata("verify_bias_subspace", {"rows": n_rows})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
