"""Stage D+: training-free, layer-resolved Patchscope verification.

Adapted from Patchscopes (Ghandeharioun et al., ICML 2024; arXiv:2401.06102). For the frozen
base and the proposed adapter, at each attribution-targeted layer, patch that layer's last-
position representation into a letter-eliciting target prompt (same model, identity mapping,
single deterministic forward) and read the renormalized probability of option c ("Roughly
equal"). A successful localized repair LOWERS over-equalization on neq (diff) while KEEPING
it high on eq (equal) (Instruction.md Section 8.3).

Run:  python GPU_Run/patchscope_bias_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import append_csv_row, get_logger, log_run_metadata, write_csv
from GPU_Run.common.patchscopes import capture_last_hidden_states, letter_token_ids, patch_and_read_letter_prob
from GPU_Run.common import prompts as P
from GPU_Run.common.paths import CHECKPOINTS_DIR, RESULTS_DIR, TEST_INSTANCES_FROZEN
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("patchscope_bias_verification")

PATCHSCOPE_MAX_ITEMS = int(os.environ.get("PATCHSCOPE_MAX_ITEMS", "48"))
TARGET_PROMPT = (
    'Answer with one option letter. Respond with one JSON object only: '
    '{"answer_choice_letter": "'
)

PER_LAYER_COLUMNS = ["tier", "method", "condition", "layer_index", "mean_probability_of_option_c"]
SUMMARY_COLUMNS = ["tier", "method", "condition", "mean_probability_of_option_c_over_targeted_layers"]


def _targeted_layers(label, n_layers):
    cfg_path = RESULTS_DIR / f"lora_config_{label}.json"
    if cfg_path.exists():
        layers = json.loads(cfg_path.read_text(encoding="utf-8"))["attribution_guided"]["selected_layers"]
        if layers:
            return sorted(layers)
    return list(range(n_layers))  # all-layer fallback


def main(smoke: bool = False):
    set_global_determinism()
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    neq = [r for r in test if r["condition"] == "neq"][:PATCHSCOPE_MAX_ITEMS]
    eq = [r for r in test if r["condition"] == "eq"][:PATCHSCOPE_MAX_ITEMS]

    per_layer_path = RESULTS_DIR / "patchscope_bias_verification_per_layer.csv"
    if per_layer_path.exists():
        per_layer_path.unlink()
    summary_rows = []

    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        label = "smoke" if smoke else tier
        proposed = CHECKPOINTS_DIR / label / "xlora_bias_proposed"
        adapters = [("frozen_base", None)]
        if proposed.exists():
            sdir = sorted(proposed.glob("seed_*"))
            if sdir:
                final = sdir[0] / "final"
                adapters.append(("xlora_bias_proposed", final if final.exists() else sdir[0]))

        for method, adapter in adapters:
            try:
                model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke)
                if adapter is not None and Path(adapter).exists():
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(adapter))
                    model.eval()
            except Exception as e:
                logger.warning("patchscope load failed (%s); skipping.", e)
                continue
            n_layers = len(capture_last_hidden_states(model, tok, "x")) - 1
            layers = _targeted_layers(label, n_layers)
            lids = letter_token_ids(tok)

            for cond_name, items in (("neq", neq), ("eq", eq)):
                layer_means = {}
                for layer in layers:
                    probs = []
                    for rec in items:
                        src = capture_last_hidden_states(model, tok, P.build_mcq_prompt(rec)["prompt"])
                        pc = patch_and_read_letter_prob(model, tok, src[min(layer + 1, len(src) - 1)], TARGET_PROMPT, layer, lids, read_letter="c")
                        probs.append(pc)
                    mean_pc = sum(probs) / len(probs) if probs else float("nan")
                    layer_means[layer] = mean_pc
                    append_csv_row(per_layer_path, {"tier": label, "method": method, "condition": cond_name, "layer_index": layer, "mean_probability_of_option_c": round(mean_pc, 4) if mean_pc == mean_pc else ""}, PER_LAYER_COLUMNS)
                overall = sum(layer_means.values()) / len(layer_means) if layer_means else float("nan")
                summary_rows.append({"tier": label, "method": method, "condition": cond_name, "mean_probability_of_option_c_over_targeted_layers": round(overall, 4) if overall == overall else ""})
                logger.info("tier=%s method=%s cond=%s mean P(c)=%.3f", label, method, cond_name, overall if overall == overall else float("nan"))
            del model

    write_csv(RESULTS_DIR / "patchscope_bias_verification_summary.csv", summary_rows, SUMMARY_COLUMNS)
    log_run_metadata("patchscope_bias_verification", {"summary_rows": len(summary_rows)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
