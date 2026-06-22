"""Stage A: Integrated Gradients attribution (localize the over-equalization failure).

Custom hook-based IG path conductance (Sundararajan et al., 2017, ICML; arXiv:1703.01365):
along the straight path from a zero-embedding baseline to the real input embeddings, with
50 Riemann steps, accumulate the absolute sum of (module output element-wise times its
gradient) of the cross-entropy loss for the correct answer letter. Attention and MLP
projection modules are scored; per-layer scores are the sum of their module scores; layer
scores are normalized by the max. Attribution is intentionally single-example
(Instruction.md Sections 2 Stage A, 14). Captum may be used to cross-check (optional).

Run:  python GPU_Run/attribution_integrated_gradients.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import re

from GPU_Run.common import model_registry
from GPU_Run.common import prompts as P
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata
from GPU_Run.common.model_registry import DEFAULT_LORA_TARGETS
from GPU_Run.common.paths import DATA_DIR, RESULTS_DIR, VALIDATION_INSTANCES
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("attribution_integrated_gradients")

RIEMANN_STEPS = int(os.environ.get("ATTRIBUTION_RIEMANN_STEPS", "50"))
MAX_ITEMS = int(os.environ.get("ATTRIBUTION_MAX_ITEMS", "64"))
_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.")
ANSWER_PREFIX = '\n{"answer_choice_letter": "'


def _is_linear_like(mod):
    """nn.Linear or transformers Conv1D (GPT-2 family) -- a leaf 2D-weight projection."""
    import torch.nn as nn

    if isinstance(mod, nn.Linear):
        return True
    try:
        from transformers.pytorch_utils import Conv1D

        return isinstance(mod, Conv1D)
    except Exception:
        return False


def _target_linear_modules(model):
    """Return list of (layer_index, module_name, module) for scored projection modules."""
    named = []
    for name, mod in model.named_modules():
        if not _is_linear_like(mod):
            continue
        m = _LAYER_RE.search(name)
        if not m:
            continue
        if any(name.endswith(suf) or f".{suf}" in name for suf in DEFAULT_LORA_TARGETS):
            named.append((int(m.group(1)), name, mod))
    if not named:  # fallback: any projection inside the decoder stack (discovered-suffix list)
        for name, mod in model.named_modules():
            if _is_linear_like(mod):
                m = _LAYER_RE.search(name)
                if m:
                    named.append((int(m.group(1)), name, mod))
    return named


def _failure_items(tier):
    fp = DATA_DIR / f"probe_failures_{tier}.jsonl"
    failures = read_jsonl(fp)
    fail_ids = {f.get("id") for f in failures}
    val = {r["id"]: r for r in read_jsonl(VALIDATION_INSTANCES)}
    # prioritise over-equalization failures, then any failure present in validation
    items = []
    for f in failures:
        rec = val.get(f.get("id"))
        if rec is not None:
            items.append(rec)
    if not items:  # fall back to all validation (e.g. perfect model or smoke)
        items = list(val.values())
    # dedup keep order
    seen, uniq = set(), []
    for r in items:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    return uniq[:MAX_ITEMS]


def attribute_model(model, tok):
    import torch
    import torch.nn.functional as F

    targets = _target_linear_modules(model)
    module_scores = {name: 0.0 for _, name, _ in targets}
    embed = model.get_input_embeddings()

    def gold_letter_token(built):
        ids = tok.encode(built["gold_display_letter"], add_special_tokens=False)
        return ids[0] if ids else tok.eos_token_id

    items = _CURRENT_ITEMS
    for rec in items:
        built = P.build_mcq_prompt(rec, rationale_mode=False)
        text = built["prompt"] + ANSWER_PREFIX
        enc = tok(text, return_tensors="pt").to(model.device)
        input_ids = enc["input_ids"]
        attn_mask = enc.get("attention_mask")
        with torch.no_grad():
            real_embeds = embed(input_ids)
        target_token = torch.tensor([gold_letter_token(built)], device=model.device)

        for step in range(1, RIEMANN_STEPS + 1):
            alpha = step / RIEMANN_STEPS
            scaled = (alpha * real_embeds).clone().detach().requires_grad_(True)
            captured = {}
            handles = []

            def make_hook(name):
                def hook(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    o.retain_grad()
                    captured[name] = o
                return hook

            for _, name, mod in targets:
                handles.append(mod.register_forward_hook(make_hook(name)))
            try:
                out = model(inputs_embeds=scaled, attention_mask=attn_mask)
                logits = out.logits[0, -1, :]
                loss = F.cross_entropy(logits.unsqueeze(0), target_token)
                model.zero_grad(set_to_none=True)
                loss.backward()
                for name, o in captured.items():
                    if o.grad is not None:
                        module_scores[name] += float((o * o.grad).abs().sum().detach().cpu())
            finally:
                for h in handles:
                    h.remove()

    # average over steps and items
    denom = max(1, RIEMANN_STEPS * len(items))
    module_scores = {k: v / denom for k, v in module_scores.items()}

    # per-layer = sum of its module scores; normalize by max
    layer_scores = {}
    name_to_layer = {name: li for li, name, _ in targets}
    for name, score in module_scores.items():
        li = name_to_layer[name]
        layer_scores[li] = layer_scores.get(li, 0.0) + score
    max_layer = max(layer_scores.values()) if layer_scores else 1.0
    normalized = {li: (s / max_layer if max_layer else 0.0) for li, s in layer_scores.items()}
    return {
        "module_scores": module_scores,
        "layer_scores_raw": {str(k): v for k, v in layer_scores.items()},
        "layer_scores_normalized": {str(k): v for k, v in normalized.items()},
        "riemann_steps": RIEMANN_STEPS,
        "items_used": len(items),
    }


_CURRENT_ITEMS = []


def main(smoke: bool = False):
    global _CURRENT_ITEMS
    set_global_determinism()
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        try:
            model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke, for_training=True)
        except Exception as e:
            logger.warning("Could not load %s (%s); skipping.", tier, e)
            continue
        _CURRENT_ITEMS = _failure_items(meta["tier"])
        logger.info("tier=%s attributing over %d items, %d Riemann steps.", meta["tier"], len(_CURRENT_ITEMS), RIEMANN_STEPS)
        result = attribute_model(model, tok)
        result["tier"] = meta["tier"]
        out = RESULTS_DIR / f"attribution_{meta['tier']}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        top = sorted(result["layer_scores_normalized"].items(), key=lambda x: -x[1])[:5]
        logger.info("tier=%s top layers (normalized): %s", meta["tier"], top)
        log_run_metadata("attribution_integrated_gradients", {"tier": meta["tier"], "top_layers": top})
        del model


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
