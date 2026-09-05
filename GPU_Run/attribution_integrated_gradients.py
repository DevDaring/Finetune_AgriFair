"""Stage A: Integrated Gradients attribution (localize the gap-erasure failure).

Custom hook-based IG path conductance (Sundararajan et al., 2017, ICML; arXiv:1703.01365):
along the straight path from a zero-embedding baseline to the real input embeddings, with
50 Riemann steps, accumulate the absolute sum of (module output element-wise times its
gradient) of the cross-entropy loss for the correct answer letter. Attention and MLP
projection modules are scored; per-layer scores are the sum of their module scores; layer
scores are normalized by the max.

Beyond the point estimate this script also stores the per-item layer scores, so the
stability of the layer ranking under resampling of the failure set can be measured
(mean pairwise Spearman over bootstrap resamples), and it writes one extra attribution
per held-out axis for the leave-one-axis-out arms, computed from failure items that
exclude that axis so the held-out axis never informs the placement.

Run:  python GPU_Run/attribution_integrated_gradients.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from GPU_Run.common import model_registry
from GPU_Run.common import training as T
from GPU_Run.common import prompts as P
from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import append_csv_row, get_logger, log_run_metadata
from GPU_Run.common.model_registry import DEFAULT_LORA_TARGETS
from GPU_Run.common.paths import DATA_DIR, RESULTS_DIR, VALIDATION_INSTANCES, attribution_path
from GPU_Run.common.seeds import set_global_determinism

logger = get_logger("attribution_integrated_gradients")

RIEMANN_STEPS = int(os.environ.get("ATTRIBUTION_RIEMANN_STEPS", "50"))
MAX_ITEMS = int(os.environ.get("ATTRIBUTION_MAX_ITEMS", "64"))
BOOTSTRAP_RESAMPLES = int(os.environ.get("ATTRIBUTION_BOOTSTRAP_RESAMPLES", "200"))
_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.")

STABILITY_COLUMNS = [
    "tier", "held_out_axis", "failure_items_used", "riemann_steps", "bootstrap_resamples",
    "mean_pairwise_spearman_of_layer_ranking_across_resamples",
    "selected_layer_set_agreement_rate_across_resamples",
]


def loao_axes() -> List[str]:
    raw = os.environ.get("LOAO_AXES", "social_group,landholding,gender")
    return [a.strip() for a in raw.split(",") if a.strip()]


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


def target_linear_modules(model):
    """Return list of (layer_index, module_name, module) for scored projection modules.

    Matches names that END with a target suffix, so a PEFT-wrapped projection is scored
    once through its wrapper (whose output includes the adapter's contribution) and its
    inner base_layer / lora_A / lora_B children are not double-counted."""
    named = []
    for name, mod in model.named_modules():
        if ".lora_" in name or name.endswith(".base_layer"):
            continue
        if not any(name.endswith(f".{suf}") for suf in DEFAULT_LORA_TARGETS):
            continue
        if not (_is_linear_like(mod) or hasattr(mod, "base_layer")):
            continue
        m = _LAYER_RE.search(name + ".")
        if not m:
            continue
        named.append((int(m.group(1)), name, mod))
    if not named:  # fallback: any projection inside the decoder stack
        for name, mod in model.named_modules():
            if _is_linear_like(mod) and ".lora_" not in name:
                m = _LAYER_RE.search(name + ".")
                if m:
                    named.append((int(m.group(1)), name, mod))
    return named


_PRIORITY = {"gap_erasure": 0, "identity_swap_flip": 1, "other_wrong": 2}


def failure_items(label: str, exclude_axis: Optional[str] = None, max_items: Optional[int] = None) -> List[Dict]:
    """Validation records that drive the attribution, gap-erasure failures first.

    exclude_axis drops every item of that axis (leave-one-axis-out placement)."""
    fp = DATA_DIR / f"probe_failures_{label}.jsonl"
    failures = read_jsonl(fp)
    val = {r["id"]: r for r in read_jsonl(VALIDATION_INSTANCES)}
    ranked = sorted(failures, key=lambda f: _PRIORITY.get(f.get("failure_type", "other_wrong"), 3))
    items = []
    for f in ranked:
        rec = val.get(f.get("id"))
        if rec is not None:
            items.append(rec)
    if not items:  # fall back to all validation (e.g. perfect model or smoke)
        items = list(val.values())
    if exclude_axis:
        items = [r for r in items if r.get("category") != exclude_axis]
    seen, uniq = set(), []
    for r in items:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    return uniq[: (max_items or MAX_ITEMS)]


def attribute_model(model, tok, items: Sequence[Dict], riemann_steps: Optional[int] = None) -> Dict:
    """Path-integrated gradient-times-activation on module outputs (Equation 5 of the
    method), summed over `items`. Returns module, layer, normalized layer, and per-item
    layer scores. All model parameters are frozen for the pass."""
    import torch
    import torch.nn.functional as F

    steps = riemann_steps or RIEMANN_STEPS
    for p in model.parameters():
        p.requires_grad_(False)
    targets = target_linear_modules(model)
    module_scores = {name: 0.0 for _, name, _ in targets}
    name_to_layer = {name: li for li, name, _ in targets}
    embed = model.get_input_embeddings()
    per_item_layer: List[Dict[str, float]] = []
    item_ids: List[str] = []

    def gold_letter_token(built):
        ids = tok.encode(built["gold_display_letter"], add_special_tokens=False)
        return ids[0] if ids else tok.eos_token_id

    for rec in items:
        built = P.build_mcq_prompt(rec, rationale_mode=False)
        text = P.render_chat(tok, built["prompt"]) + P.answer_prefix_text()
        enc = tok(text, return_tensors="pt", add_special_tokens=P.prompt_add_special_tokens(tok)).to(model.device)
        input_ids = enc["input_ids"]
        attn_mask = enc.get("attention_mask")
        with torch.no_grad():
            real_embeds = embed(input_ids)
        target_token = torch.tensor([gold_letter_token(built)], device=model.device)
        item_scores = {name: 0.0 for _, name, _ in targets}

        for step in range(1, steps + 1):
            alpha = step / steps
            scaled = (alpha * real_embeds).clone().detach().requires_grad_(True)
            captured = {}
            handles = []

            def make_hook(name):
                def hook(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    if o.requires_grad:
                        o.retain_grad()
                        captured[name] = o
                return hook

            for _, name, mod in targets:
                handles.append(mod.register_forward_hook(make_hook(name)))
            try:
                out = model(inputs_embeds=scaled, attention_mask=attn_mask)
                logits = out.logits[0, -1, :].float()
                loss = F.cross_entropy(logits.unsqueeze(0), target_token)
                loss.backward()
                for name, o in captured.items():
                    if o.grad is not None:
                        item_scores[name] += float((o.detach() * o.grad).abs().sum().cpu())
            finally:
                for h in handles:
                    h.remove()
        layer_item = {}
        for name, s in item_scores.items():
            s /= steps
            module_scores[name] += s
            li = name_to_layer[name]
            layer_item[str(li)] = layer_item.get(str(li), 0.0) + s
        per_item_layer.append(layer_item)
        item_ids.append(rec["id"])

    denom = max(1, len(items))
    module_scores = {k: v / denom for k, v in module_scores.items()}
    layer_scores: Dict[int, float] = {}
    for name, score in module_scores.items():
        li = name_to_layer[name]
        layer_scores[li] = layer_scores.get(li, 0.0) + score
    max_layer = max(layer_scores.values()) if layer_scores else 1.0
    normalized = {li: (s / max_layer if max_layer else 0.0) for li, s in layer_scores.items()}
    return {
        "module_scores": module_scores,
        "layer_scores_raw": {str(k): v for k, v in layer_scores.items()},
        "layer_scores_normalized": {str(k): v for k, v in normalized.items()},
        "per_item_layer_scores": per_item_layer,
        "item_ids": item_ids,
        "riemann_steps": steps,
        "items_used": len(items),
    }


# ------------------------------ stability -----------------------------------

def _layer_totals(per_item: Sequence[Dict[str, float]], idx: Sequence[int], layers: Sequence[str]) -> np.ndarray:
    tot = np.zeros(len(layers))
    for i in idx:
        row = per_item[i]
        for j, l in enumerate(layers):
            tot[j] += row.get(l, 0.0)
    return tot


def ranking_stability(per_item: Sequence[Dict[str, float]], threshold: float, resamples: int, seed: int = 42) -> Dict[str, float]:
    """Bootstrap the failure set, recompute the per-layer ranking each time, and report
    the mean pairwise Spearman correlation between resampled rankings plus the rate at
    which the thresholded layer set is identical across resample pairs."""
    from scipy.stats import spearmanr

    n = len(per_item)
    if n < 2 or resamples < 2:
        return {"mean_pairwise_spearman": float("nan"), "selected_set_agreement": float("nan")}
    layers = sorted({l for row in per_item for l in row}, key=int)
    rng = np.random.default_rng(seed)
    rankings, sets = [], []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        tot = _layer_totals(per_item, idx, layers)
        rankings.append(tot)
        mx = tot.max() if tot.size else 0.0
        sets.append(frozenset(l for j, l in enumerate(layers) if mx > 0 and tot[j] / mx >= threshold))
    pairs = [(i, j) for i in range(resamples) for j in range(i + 1, resamples)]
    if len(pairs) > 2000:
        pick = rng.choice(len(pairs), 2000, replace=False)
        pairs = [pairs[k] for k in pick]
    rhos, agree = [], []
    for i, j in pairs:
        if np.std(rankings[i]) == 0 or np.std(rankings[j]) == 0:
            continue
        rhos.append(spearmanr(rankings[i], rankings[j])[0])
        agree.append(float(sets[i] == sets[j]))
    return {
        "mean_pairwise_spearman": float(np.nanmean(rhos)) if rhos else float("nan"),
        "selected_set_agreement": float(np.mean(agree)) if agree else float("nan"),
    }


def compare_layer_rankings(before_norm: Dict[str, float], after_norm: Dict[str, float], k: int) -> Dict[str, float]:
    """Spearman correlation between two per-layer attribution vectors and the Jaccard
    overlap of their top-k layer sets."""
    from scipy.stats import spearmanr

    layers = sorted(set(before_norm) | set(after_norm), key=int)
    a = np.array([before_norm.get(l, 0.0) for l in layers])
    b = np.array([after_norm.get(l, 0.0) for l in layers])
    rho = float(spearmanr(a, b)[0]) if (np.std(a) > 0 and np.std(b) > 0) else float("nan")
    k = max(1, min(k, len(layers)))
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    jac = len(top_a & top_b) / max(1, len(top_a | top_b))
    return {"spearman": rho, "top_k_jaccard": jac, "k": k}


# ---------------------------------- main ------------------------------------

def _write(result: Dict, path: Path):
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def _drop_stability_rows(path, tier: str, held_out: str) -> None:
    """Remove earlier rows for this (tier, held-out axis) from the stability summary."""
    import csv

    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)
                    if not (r.get("tier") == tier and r.get("held_out_axis") == held_out)]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STABILITY_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as e:
        logger.warning("Could not de-duplicate %s (%s); rows may repeat.", path.name, type(e).__name__)


def main(smoke: bool = False):
    set_global_determinism()
    threshold = float(os.environ.get("ATTRIBUTION_THRESHOLD", "0.15"))
    stability_csv = RESULTS_DIR / "attribution_ranking_stability.csv"
    tiers = ["smoke"] if smoke else model_registry.active_tiers()
    for tier in tiers:
        try:
            model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=smoke, for_training=True)
        except Exception as e:
            logger.warning("Could not load %s (%s); skipping.", tier, e)
            continue
        label = meta["tier"]
        variants = [None] + loao_axes()
        for held_out in variants:
            items = failure_items(label, exclude_axis=held_out)
            if not items:
                logger.warning("tier=%s held_out=%s: no failure items; skipping.", label, held_out)
                continue
            logger.info("tier=%s held_out_axis=%s attributing over %d items, %d Riemann steps.",
                        label, held_out, len(items), RIEMANN_STEPS)
            try:
                result = attribute_model(model, tok, items)
            except Exception as e:
                logger.error("Attribution failed for tier=%s held_out=%s (%s); the remaining "
                             "variants and tiers continue.", label, held_out, e, exc_info=True)
                continue
            result["tier"] = label
            result["held_out_axis"] = held_out or ""
            stab = ranking_stability(result["per_item_layer_scores"], threshold, BOOTSTRAP_RESAMPLES)
            result["ranking_stability"] = stab
            _write(result, attribution_path(label, held_out))
            # Replace any row from an earlier run of this same (tier, held-out axis). The file
            # is append-only, so re-running the stage otherwise duplicates every row and any
            # downstream mean over it silently double-counts.
            _drop_stability_rows(stability_csv, label, held_out or "none")
            append_csv_row(stability_csv, {
                "tier": label, "held_out_axis": held_out or "none", "failure_items_used": len(items),
                "riemann_steps": RIEMANN_STEPS, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "mean_pairwise_spearman_of_layer_ranking_across_resamples": round(stab["mean_pairwise_spearman"], 4) if stab["mean_pairwise_spearman"] == stab["mean_pairwise_spearman"] else "",
                "selected_layer_set_agreement_rate_across_resamples": round(stab["selected_set_agreement"], 4) if stab["selected_set_agreement"] == stab["selected_set_agreement"] else "",
            }, STABILITY_COLUMNS)
            top = sorted(result["layer_scores_normalized"].items(), key=lambda x: -x[1])[:5]
            logger.info("tier=%s held_out=%s top layers (normalized): %s; ranking stability rho=%.3f",
                        label, held_out, top, stab["mean_pairwise_spearman"])
            log_run_metadata("attribution_integrated_gradients",
                             {"tier": label, "held_out_axis": held_out or "", "top_layers": top, **stab})
        T.release_model(model)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
