"""Parameter-space geometry analysis (where the weight update lands).

A measurement and explanation layer in parameter space, adapted from the geometry of
post-training updates (Shen et al. 2026 arXiv:2606.07082; Zhu et al. 2025 arXiv:2511.08567;
Liu et al. 2026 arXiv:2506.00772). For every weight-modifying method on both primary models
it computes the per-matrix update Delta_W (via PEFT get_delta_weight; zero for non-adapted
modules; inference_time_no_weight_update for FairSteer) and seven diagnostics, all on CPU
float64 SVD; assigns a regime label; and reports the Pearson correlation (with p-value) across
methods between normalized spectral shift and difference-awareness preservation, and between
principal-mask overlap and preservation (Instruction.md Section 8.2). CPU only; needs models/
base weights and checkpoints/ on this machine.

Run:  python CPU_Run/parameter_space_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

import numpy as np

from GPU_Run.common import model_registry
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import CHECKPOINTS_DIR, GEOMETRY_DIR, RESULTS_DIR

logger = get_logger("parameter_space_geometry")

ETA = 1e-3
ALPHA = 0.5
LAYER_STRIDE = int(os.environ.get("GEOMETRY_LAYER_STRIDE", "1"))

PER_METHOD_COLUMNS = [
    "tier", "method", "module_coverage_fraction", "mean_update_sparsity",
    "mean_principal_angle_rotation", "mean_normalized_spectral_shift",
    "mean_principal_mask_overlap", "mean_low_magnitude_mask_overlap",
    "mean_stable_rank", "mean_frobenius_norm", "mean_hill_tail_exponent",
    "regime_label",
]
CORR_COLUMNS = ["tier", "x_metric", "y_metric", "pearson_correlation", "p_value", "num_methods"]


def _svd(mat):
    return np.linalg.svd(mat.astype(np.float64), full_matrices=False)


def _principal_angle(U0, U1, k):
    a = U0[:, :k]
    b = U1[:, :k]
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    s = np.clip(s, -1, 1)
    return float(np.mean(np.arccos(s)))


def _hill_tail(sv, top=10):
    sv = np.sort(sv)[::-1]
    top = min(top, len(sv) - 1)
    if top < 2:
        return float("nan")
    xs = sv[:top]
    xmin = sv[top]
    if xmin <= 0:
        return float("nan")
    return float(top / np.sum(np.log(xs / xmin)))


def _diagnostics(W0, dW):
    out = {}
    out["sparsity"] = float(np.mean(np.abs(dW) < ETA))
    s0 = _svd(W0)[1]
    sU0, _, sVt0 = _svd(W0)
    sUd, sdv, sVtd = _svd(dW)
    sd = _svd(dW)[1]
    k = min(512, min(W0.shape))
    out["principal_angle"] = _principal_angle(sU0, sUd, k)
    denom = np.linalg.norm(s0) or 1.0
    n = min(len(s0), len(sd))
    out["spectral_shift"] = float(np.linalg.norm(s0[:n] - sd[:n]) / denom)
    # principal mask (top-alpha of rank-64 reconstruction) vs low-magnitude mask
    r = min(64, min(W0.shape))
    recon = (sU0[:, :r] * s0[:r]) @ sVt0[:r, :]
    flat_recon = np.abs(recon).ravel()
    flat_w0 = np.abs(W0).ravel()
    thr_hi = np.quantile(flat_recon, 1 - ALPHA)
    thr_lo = np.quantile(flat_w0, ALPHA)
    principal_mask = np.abs(recon) >= thr_hi
    low_mask = np.abs(W0) <= thr_lo
    update_mask = np.abs(dW) >= ETA
    upd_count = update_mask.sum() or 1
    out["principal_overlap"] = float((update_mask & principal_mask).sum() / upd_count)
    out["low_overlap"] = float((update_mask & low_mask).sum() / upd_count)
    fro = float(np.linalg.norm(dW))
    out["frobenius"] = fro
    out["stable_rank"] = float((np.sum(sd ** 2) / (sd[0] ** 2)) if sd.size and sd[0] > 0 else 0.0)
    out["hill"] = _hill_tail(sd)
    return out


def _iter_deltas(peft_model):
    """Yield (name, W0_numpy, dW_numpy) for adapted Linear modules."""
    import torch

    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
            try:
                adapter = list(mod.lora_A.keys())[0]
                dW = mod.get_delta_weight(adapter) if hasattr(mod, "get_delta_weight") else None
                W0 = mod.base_layer.weight.detach()
                if dW is None:
                    continue
                yield name, W0.float().cpu().numpy(), dW.detach().float().cpu().numpy()
            except Exception:
                continue


def _preservation_table():
    """post-repair CtxtAware / base CtxtAware per (tier, method)."""
    import pandas as pd

    path = RESULTS_DIR / "main_evaluation_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    col = "contextual_awareness_metric_wang_2025_precision_style"
    overall[col] = pd.to_numeric(overall[col], errors="coerce")
    base = overall[overall["method"] == "frozen_base"].groupby("tier")[col].mean()
    out = {}
    for (tier, method), g in overall.groupby(["tier", "method"]):
        b = base.get(tier, np.nan)
        if b and b == b:
            out[(tier, method)] = float(g[col].mean() / b)
    return out


def main():
    # analyse every tier that actually has checkpoints (includes "smoke" in a smoke run)
    tiers = [d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()] if CHECKPOINTS_DIR.exists() else []
    if not tiers:
        tiers = model_registry.PRIMARY_TIERS
    rows = []
    preservation = _preservation_table()
    per_tier_points = {}

    for tier in tiers:
        ckpt_root = CHECKPOINTS_DIR / tier
        if not ckpt_root.exists():
            logger.warning("No checkpoints for %s; skipping geometry.", tier)
            continue
        try:
            base_model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=(tier == "smoke"))
        except Exception as e:
            logger.warning("Geometry: could not load base %s (%s); skipping.", tier, e)
            continue

        for mdir in sorted(ckpt_root.iterdir()):
            if not mdir.is_dir():
                continue
            method = mdir.name
            if "fairsteer" in method:
                rows.append({"tier": tier, "method": method, "regime_label": "inference_time_no_weight_update",
                             **{c: "" for c in PER_METHOD_COLUMNS if c not in ("tier", "method", "regime_label")}})
                continue
            sdir = sorted(mdir.glob("seed_*"))
            if not sdir:
                continue
            final = (sdir[0] / "final")
            adapter = final if final.exists() else sdir[0]
            try:
                from peft import PeftModel

                pm = PeftModel.from_pretrained(base_model, str(adapter))
            except Exception as e:
                logger.warning("Could not load adapter %s (%s); skipping.", adapter, e)
                continue

            diags = []
            adapted = 0
            total_modules = 0
            for i, (name, W0, dW) in enumerate(_iter_deltas(pm)):
                total_modules += 1
                if i % LAYER_STRIDE != 0:
                    continue
                if np.any(np.abs(dW) >= ETA):
                    adapted += 1
                diags.append(_diagnostics(W0, dW))
            if not diags:
                continue
            agg = {k: float(np.nanmean([d[k] for d in diags])) for k in diags[0]}
            coverage = adapted / max(1, total_modules)
            shift = agg["spectral_shift"]
            overlap = agg["principal_overlap"]
            regime = _regime_label(overlap, shift)
            rows.append({
                "tier": tier, "method": method,
                "module_coverage_fraction": round(coverage, 4),
                "mean_update_sparsity": round(agg["sparsity"], 4),
                "mean_principal_angle_rotation": round(agg["principal_angle"], 4),
                "mean_normalized_spectral_shift": round(shift, 4),
                "mean_principal_mask_overlap": round(overlap, 4),
                "mean_low_magnitude_mask_overlap": round(agg["low_overlap"], 4),
                "mean_stable_rank": round(agg["stable_rank"], 4),
                "mean_frobenius_norm": round(agg["frobenius"], 4),
                "mean_hill_tail_exponent": round(agg["hill"], 4) if agg["hill"] == agg["hill"] else "",
                "regime_label": regime,
            })
            pres = preservation.get((tier, method))
            if pres is not None:
                per_tier_points.setdefault(tier, []).append((shift, overlap, pres))
            try:
                pm = pm.unload()
            except Exception:
                pass
        del base_model

    write_csv(GEOMETRY_DIR / "parameter_space_geometry_per_method.csv", rows, PER_METHOD_COLUMNS)

    corr_rows = []
    from scipy.stats import pearsonr

    for tier, pts in per_tier_points.items():
        if len(pts) >= 3:
            shifts = [p[0] for p in pts]
            overlaps = [p[1] for p in pts]
            pres = [p[2] for p in pts]
            for xname, xs in (("normalized_spectral_shift", shifts), ("principal_mask_overlap", overlaps)):
                r, p = pearsonr(xs, pres)
                corr_rows.append({"tier": tier, "x_metric": xname, "y_metric": "difference_awareness_preservation",
                                  "pearson_correlation": round(float(r), 4), "p_value": round(float(p), 4), "num_methods": len(pts)})
    write_csv(GEOMETRY_DIR / "geometry_vs_difference_awareness_correlation.csv", corr_rows, CORR_COLUMNS)
    log_run_metadata("parameter_space_geometry", {"methods_analyzed": len(rows), "correlations": len(corr_rows)})
    logger.info("Geometry: %d method rows, %d correlation rows.", len(rows), len(corr_rows))


def _regime_label(overlap, shift, anchor_shift=None):
    anchor = anchor_shift if anchor_shift else shift
    if overlap < 0.5 and shift <= 0.5 * (anchor if anchor else 1.0):
        return "off-principal-and-spectrum-preserving"
    if overlap >= 0.5:
        return "principal-aligned-and-distorting"
    return "relaxed-off-principal"


if __name__ == "__main__":
    main()
