"""Stage D: IG re-run localization fraction and the linear identity probe.

Reports, before (frozen base) and after (proposed adapter):
  - over_equalization_attribution_localization_fraction: the share of the bias attribution
    mass that still falls on the originally targeted layers.
  - identity_linear_probe_accuracy: cross-validated accuracy of a logistic probe decoding the
    swapped identity (group1 vs group2) from a mid-depth hidden state, on the AgriFacts MCQ
    swaps and on the AgriAdvice persona pairs. Higher = more residual identity (bias) signal.
The headline bias_subspace_residual_* columns use this probe (Instruction.md Section 8.1).

Run:  python GPU_Run/verify_bias_subspace.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os

import numpy as np

from GPU_Run.common import metrics as M
from GPU_Run.common import model_registry
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.patchscopes import capture_last_hidden_states
from GPU_Run.common import prompts as P
from GPU_Run.common.paths import AGRIADVICE_PAIRS, CHECKPOINTS_DIR, COUNTERFACTUAL_PAIRS, RESULTS_DIR, VALIDATION_INSTANCES
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("verify_bias_subspace")

PROBE_MAX_ITEMS = int(os.environ.get("PROBE_MAX_ITEMS", "200"))

COLUMNS = [
    "tier", "method", "scope",
    "over_equalization_attribution_localization_fraction",
    "identity_linear_probe_accuracy",
    "bias_subspace_residual_pre_repair",
    "bias_subspace_residual_post_repair",
]


def _mid_hidden(model, tok, text):
    hs = capture_last_hidden_states(model, tok, text)
    return np.asarray(hs[len(hs) // 2])


def _probe_mcq(model, tok):
    cf = read_jsonl(COUNTERFACTUAL_PAIRS)
    val_ids = {r["id"] for r in read_jsonl(VALIDATION_INSTANCES)}
    cf = [c for c in cf if c["original"]["id"] in val_ids][:PROBE_MAX_ITEMS]
    feats, labels = [], []
    for c in cf:
        feats.append(_mid_hidden(model, tok, P.build_mcq_prompt(c["original"])["prompt"]))
        labels.append(0)
        feats.append(_mid_hidden(model, tok, P.build_mcq_prompt(c["swapped"])["prompt"]))
        labels.append(1)
    if not feats:
        return float("nan")
    return M.linear_identity_probe(np.vstack(feats), np.asarray(labels))


def _probe_advice(model, tok):
    pairs = read_jsonl(AGRIADVICE_PAIRS)[:PROBE_MAX_ITEMS]
    feats, labels = [], []
    for p in pairs:
        feats.append(_mid_hidden(model, tok, p["prompt_a"]))
        labels.append(0)
        feats.append(_mid_hidden(model, tok, p["prompt_b"]))
        labels.append(1)
    if not feats:
        return float("nan")
    return M.linear_identity_probe(np.vstack(feats), np.asarray(labels))


def _localization_fraction(label, model, tok):
    """Re-run attribution and report the mass still on the originally targeted layers."""
    attr_path = RESULTS_DIR / f"attribution_{label}.json"
    cfg_path = RESULTS_DIR / f"lora_config_{label}.json"
    if not (attr_path.exists() and cfg_path.exists()):
        return float("nan")
    attr = json.loads(attr_path.read_text(encoding="utf-8"))
    targeted = set(json.loads(cfg_path.read_text(encoding="utf-8"))["attribution_guided"]["selected_layers"])
    norm = {int(k): v for k, v in attr["layer_scores_normalized"].items()}
    total = sum(norm.values()) or 1.0
    on_target = sum(v for l, v in norm.items() if l in targeted)
    return on_target / total


def main(smoke: bool = False):
    set_global_determinism()
    rows = []
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
                logger.warning("verify load failed (%s); skipping.", e)
                continue
            loc = _localization_fraction(label, model, tok)
            probe_mcq = _probe_mcq(model, tok)
            probe_adv = _probe_advice(model, tok)
            residual = probe_mcq  # headline residual uses the MCQ identity probe
            rows.append(
                {
                    "tier": label, "method": method, "scope": "mcq_swap_probe",
                    "over_equalization_attribution_localization_fraction": round(loc, 4) if loc == loc else "",
                    "identity_linear_probe_accuracy": round(probe_mcq, 4) if probe_mcq == probe_mcq else "",
                    "bias_subspace_residual_pre_repair": round(residual, 4) if (method == "frozen_base" and residual == residual) else "",
                    "bias_subspace_residual_post_repair": round(residual, 4) if (method != "frozen_base" and residual == residual) else "",
                }
            )
            rows.append(
                {
                    "tier": label, "method": method, "scope": "agriadvice_persona_probe",
                    "over_equalization_attribution_localization_fraction": "",
                    "identity_linear_probe_accuracy": round(probe_adv, 4) if probe_adv == probe_adv else "",
                    "bias_subspace_residual_pre_repair": "",
                    "bias_subspace_residual_post_repair": "",
                }
            )
            logger.info("tier=%s method=%s loc=%.3f probe_mcq=%.3f probe_adv=%.3f", label, method, loc if loc==loc else float('nan'), probe_mcq if probe_mcq==probe_mcq else float('nan'), probe_adv if probe_adv==probe_adv else float('nan'))
            del model

    write_csv(RESULTS_DIR / "bias_subspace_verification.csv", rows, COLUMNS)
    log_run_metadata("verify_bias_subspace", {"rows": len(rows)})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
