"""Training-free, layer-resolved Patchscope readout of the pull toward "Roughly equal".

Adapted from Patchscopes (Ghandeharioun et al., ICML 2024; arXiv:2401.06102). For the
frozen base and every trained method, at each attribution-targeted layer, the last-position
representation of an AgriFacts item is patched into a short letter-eliciting prompt in the
same model and the renormalized probability of the option letter standing for "Roughly
equal" is read. A successful localized repair LOWERS that probability on diff items while
KEEPING it high on equal items.

Two conventions make the number interpretable. The source prompt is built in canonical
option order, so the displayed letter c is always "Roughly equal" rather than whichever
option the shuffle happened to place third. And the letter token ids are resolved in the
JSON answer context, because tokenizers split a quoted letter differently from a bare one.

Run:  python GPU_Run/patchscope_bias_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
from typing import Dict, List

import numpy as np

from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common import targets as TG
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import append_csv_row, get_logger, log_run_metadata
from GPU_Run.common.patchscopes import (
    capture_last_hidden_states,
    layer_output_state,
    letter_token_ids,
    number_of_layers,
    patch_and_read_letter_prob,
)
from GPU_Run.common.paths import RESULTS_DIR, TEST_INSTANCES_FROZEN, lora_config_path, split_loao_method
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("patchscope_bias_verification")

PATCHSCOPE_MAX_ITEMS = int(os.environ.get("PATCHSCOPE_MAX_ITEMS", "48"))
TARGET_PROMPT_BODY = "Give the option letter for the correct answer."

PER_LAYER_COLUMNS = ["tier", "method", "random_seed", "condition", "layer_index",
                     "mean_probability_of_roughly_equal_option", "items_used"]
SUMMARY_COLUMNS = ["tier", "method", "random_seed", "condition",
                   "mean_probability_of_roughly_equal_option_over_targeted_layers",
                   "targeted_layers_used"]


def canonical_record(rec: Dict) -> Dict:
    """The item with its options in canonical order, so displayed c is "Roughly equal"."""
    return dict(rec, on_disk_choices=[rec["choice_a"], rec["choice_b"], rec["choice_c"]])


def targeted_layers(label: str, method: str, n_layers: int) -> List[int]:
    _, held_out = split_loao_method(method)
    cfg_path = lora_config_path(label, held_out)
    if cfg_path.exists():
        layers = json.loads(cfg_path.read_text(encoding="utf-8"))["attribution_guided"]["selected_layers"]
        layers = [l for l in layers if l < n_layers]
        if layers:
            return sorted(layers)
    stride = max(1, n_layers // 8)
    return list(range(0, n_layers, stride))


def main(smoke: bool = False):
    set_global_determinism()
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    diff_items = [canonical_record(r) for r in test if r["condition"] == "diff"][:PATCHSCOPE_MAX_ITEMS]
    equal_items = [canonical_record(r) for r in test if r["condition"] == "equal"][:PATCHSCOPE_MAX_ITEMS]

    per_layer_path = RESULTS_DIR / "patchscope_readout_per_layer.csv"
    summary_path = RESULTS_DIR / "patchscope_readout_summary.csv"
    for p in (per_layer_path, summary_path):
        if p.exists():
            p.unlink()

    wanted = os.environ.get("PATCHSCOPE_METHODS")
    method_filter = [m.strip() for m in wanted.split(",") if m.strip()] if wanted else None

    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        for target in TG.discover_targets(label, methods=method_filter, include_steering=False,
                                          exclude_prefixes=("ablation_",), first_seed_only=True):
            method, seed = target.method, target.seed
            try:
                model, tok, meta = TG.load_target(tier, target, smoke=smoke)
            except Exception as e:
                logger.error("patchscope load failed for %s (%s); skipping.", method, e)
                continue
            try:
                n_layers = number_of_layers(model)
                layers = targeted_layers(label, method, n_layers)
                lids = letter_token_ids(tok)
                add_special = P.prompt_add_special_tokens(tok)
                target_prompt = P.render_chat(tok, TARGET_PROMPT_BODY) + P.answer_prefix_text()

                for cond_name, items in (("diff", diff_items), ("equal", equal_items)):
                    if not items:
                        continue
                    sources = []
                    for rec in items:
                        text = P.render_chat(tok, P.build_mcq_prompt(rec)["prompt"]) + P.answer_prefix_text()
                        sources.append(capture_last_hidden_states(model, tok, text, add_special))
                    layer_means = {}
                    for layer in layers:
                        probs = []
                        for hs in sources:
                            pc = patch_and_read_letter_prob(
                                model, tok, layer_output_state(hs, layer), target_prompt, layer, lids,
                                read_letter="c", add_special_tokens=add_special)
                            if pc == pc:
                                probs.append(pc)
                        mean_pc = float(np.mean(probs)) if probs else float("nan")
                        layer_means[layer] = mean_pc
                        append_csv_row(per_layer_path, {
                            "tier": label, "method": method, "random_seed": seed, "condition": cond_name,
                            "layer_index": layer,
                            "mean_probability_of_roughly_equal_option": round(mean_pc, 4) if mean_pc == mean_pc else "",
                            "items_used": len(probs),
                        }, PER_LAYER_COLUMNS)
                    vals = [v for v in layer_means.values() if v == v]
                    overall = float(np.mean(vals)) if vals else float("nan")
                    append_csv_row(summary_path, {
                        "tier": label, "method": method, "random_seed": seed, "condition": cond_name,
                        "mean_probability_of_roughly_equal_option_over_targeted_layers":
                            round(overall, 4) if overall == overall else "",
                        "targeted_layers_used": ";".join(str(l) for l in layers),
                    }, SUMMARY_COLUMNS)
                    logger.info("tier=%s method=%s condition=%s mean P(Roughly equal)=%.3f over %d layers",
                                label, method, cond_name, overall if overall == overall else float("nan"), len(layers))
            finally:
                TG.remove_steering(model)
                del model

    log_run_metadata("patchscope_bias_verification", {"summary": str(summary_path)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
