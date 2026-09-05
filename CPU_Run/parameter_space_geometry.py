"""Parameter-space geometry analysis (where the weight update lands).

A measurement and explanation layer in parameter space. It is not a training method, and
the result is reported whichever way it lands, including if the geometry does not separate
the methods.

For every weight-modifying method on every analysed model it computes the per-matrix
update Delta_W (PEFT get_delta_weight; zero for non-adapted modules;
inference_time_no_weight_update for FairSteer), five core diagnostics plus two auxiliary
ones, all in CPU float64 (spectra and dominant subspaces come from a float64 Gram
eigendecomposition, and low-rank update spectra from an exact range projection, both verified
against the full SVD); assigns a regime label against a dense anchor; and reports
Pearson and Spearman correlations, bootstrap confidence intervals, and a partial
correlation controlling for the trainable-parameter percentage, between the geometry of
the update and how much difference awareness the method preserved.

Every diagnostic is defined on the pair (W0, W1 = W0 + Delta_W) where the definition calls
for the updated weight, and on Delta_W where it calls for the update itself. Getting that
distinction wrong silently changes what the numbers mean, so it is stated per diagnostic
in the code below.

# Parameter-space geometry diagnostics adapted from:
# Shen, Z., Li, Y., Yin, Q., Leong, C. T., Wang, Z., Chen, Y., Han, R., Lee, S.,
#   Fung, Y. R. "On the Geometry of On-Policy Distillation." arXiv:2606.07082, 2026.
#   [stable rank, principal-angle rotation, normalized spectral shift, update-mask
#    overlap, bf16-aware update sparsity; relaxed off-principal regime framing]
# Zhu, H., Zhang, Z., Huang, H., Su, D., Liu, Z., Zhao, J., Fedorov, I., Pirsiavash, H.,
#   Sha, Z., Lee, J., Pan, D. Z., Wang, Z., Tian, Y., Tai, K. S. "The Path Not Taken:
#   RLVR Provably Learns Off the Principals." arXiv:2511.08567, 2025.
#   [Three-Gate account; off-principal, geometry-preserving updates]
# Liu, Z., Pang, T., Balabanov, O., Yang, C., Huang, T., Yin, L., Yang, Y., Liu, S.
#   "Lift the Veil for the Truth: Principal Weights Emerge after Rank Reduction for
#   Reasoning-Focused Supervised Fine-Tuning." arXiv:2506.00772, 2026.
#   [principal weight mask as a high-curvature proxy]
# Mukherjee, S., Yuan, L., Hakkani-Tur, D., Peng, H. "Reinforcement Learning Finetunes
#   Small Subnetworks in Large Language Models." arXiv:2505.11711, 2025.
#   [localized, off-principal subnetwork updates]
# Shenfeld, I., Pari, J., Agrawal, P. "RL's Razor: Why Online Reinforcement Learning
#   Forgets Less." arXiv:2509.04259, 2025.
#   [link between off-principal updates and reduced forgetting / capability preservation]
# Hill, B. M. "A Simple General Approach to Inference About the Tail of a Distribution."
#   The Annals of Statistics, 3(5):1163-1174, 1975. [Hill tail exponent]
# Deb, K., Basu, A. "Language-Specific Bias Circuits in Multilingual Language Models."
#   ACM Transactions on Asian and Low-Resource Language Processing, 2026.
#   [companion verification: the minimal cut shows bias units are removed, while these
#    geometry diagnostics show the surrounding pretrained structure is left intact]

Run:  python CPU_Run/parameter_space_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import re

import numpy as np

from GPU_Run.common import model_registry
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import CHECKPOINTS_DIR, FIGURES_DIR, GEOMETRY_DIR, RESULTS_DIR

logger = get_logger("parameter_space_geometry")

# Documented thresholds (mirrored in README, "Implementation decisions").
ETA = 1e-3            # bf16-aware relative tolerance for "this entry changed"
ALPHA = 0.5           # mask fraction; the random-overlap baseline is therefore 0.5
PRINCIPAL_RANK = 64   # rank of the W0 reconstruction defining the principal mask
MAX_SUBSPACE_K = 512  # cap on the principal-angle subspace dimension
HILL_TOP = 10         # leading singular values used by the Hill estimator
LAYER_STRIDE = int(os.environ.get("GEOMETRY_LAYER_STRIDE", "1"))
# A full-precision SVD of every adapted matrix does not finish. Measured on this machine, one
# 3840x15360 MLP matrix takes ~141s and one 3840x3840 attention matrix ~38s, so a single dense
# 12B arm is about 7.7 hours and the stage as a whole runs into weeks. The diagnostics
# characterise WHERE in the network updates sit, which a stratified subsample estimates
# perfectly well, so the default caps the count and samples across depth and module type
# rather than taking whichever matrices come first. Set 0 to analyse every matrix.
MAX_MATRICES = int(os.environ.get("GEOMETRY_MAX_MATRICES", "48"))
BOOTSTRAP_RESAMPLES = int(os.environ.get("GEOMETRY_BOOTSTRAP_RESAMPLES", "1000"))
# Regime boundary: spectral shift below this multiple of the dense anchor's shift counts
# as spectrum-preserving.
SPECTRUM_PRESERVING_FRACTION_OF_ANCHOR = 0.5
# The dense, all-layer arm used as the principal-aligned anchor. Full fine-tuning is the
# anchor the source papers use; this study trains no full fine-tuning arm, so the densest
# available update takes its place and the substitution is recorded in the CSV.
DENSE_ANCHOR_METHODS = ("reference_vanilla_lora", "reference_vanilla_qlora")
# Arms excluded from the correlation because they change no weight.
NO_WEIGHT_UPDATE_METHODS = ("baseline_fairsteer",)

_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.")
_MODULE_RE = re.compile(r"\.([a-z_0-9]+)$")

PER_METHOD_COLUMNS = [
    "method_name", "base_model_name", "base_model_revision_hash", "weight_update_type",
    "lora_rank_budget", "fraction_of_analyzed_matrices_adapted", "module_coverage_fraction",
    "bf16_aware_update_sparsity_mean",
    "principal_angle_rotation_left_subspace_mean_degrees",
    "principal_angle_rotation_right_subspace_mean_degrees",
    "principal_angle_rotation_max_degrees",
    "normalized_spectral_shift_mean",
    "update_mask_overlap_with_principal_mask_mean",
    "update_mask_overlap_with_low_magnitude_mask_mean",
    "stable_rank_of_update_mean", "frobenius_norm_of_update_mean",
    "relative_frobenius_drift_mean", "hill_tail_exponent_mean",
    "parameter_space_regime_label", "regime_anchor_method_name",
    "contextual_awareness_metric_post_repair", "difference_aware_metric_post_repair",
    "contextual_awareness_preservation_relative_to_base",
    "trainable_parameter_percentage", "number_of_analyzed_matrices",
    "quantization_setting", "random_seed",
]

PER_LAYER_COLUMNS = [
    "method_name", "base_model_name", "random_seed", "layer_index", "module_type",
    "matrix_name", "bf16_aware_update_sparsity",
    "principal_angle_rotation_left_subspace_degrees",
    "principal_angle_rotation_right_subspace_degrees",
    "normalized_spectral_shift", "update_mask_overlap_with_principal_mask",
    "update_mask_overlap_with_low_magnitude_mask", "stable_rank_of_update",
    "frobenius_norm_of_update", "relative_frobenius_drift", "hill_tail_exponent",
]

CORR_COLUMNS = [
    "base_model_name", "geometry_metric_name", "preservation_metric_name",
    "pearson_correlation", "pearson_p_value",
    "spearman_correlation", "spearman_p_value",
    "bootstrap_confidence_interval_lower", "bootstrap_confidence_interval_upper",
    "partial_correlation_controlling_for_trainable_parameter_percentage",
    "number_of_configurations", "predicted_sign_preregistered",
]

RUN_LOG_COLUMNS = [
    "base_model_name", "method_name", "random_seed", "status_message",
    "number_of_analyzed_matrices",
]


# ----------------------------- diagnostics ----------------------------------

def _to_bf16_and_back(mat):
    """Round-trip through bfloat16 so a change is only counted when it survives the
    precision the model is actually trained and served in (Shen et al. 2026)."""
    try:
        import torch

        t = torch.from_numpy(np.ascontiguousarray(mat, dtype=np.float32))
        return t.to(torch.bfloat16).to(torch.float32).numpy()
    except Exception:
        # numpy-only fallback: truncate the float32 mantissa to bfloat16's 7 bits
        view = np.ascontiguousarray(mat, dtype=np.float32).view(np.uint32)
        return (view & np.uint32(0xFFFF0000)).view(np.float32)


def _changed_mask(W0, W1):
    """Core diagnostic 1 support: entries judged changed under the bf16-aware relative
    rule |W1 - W0| > eta * max(|W0|, |W1|). Relative, not absolute, so the rule does not
    silently declare every entry of a large-magnitude matrix changed."""
    a = _to_bf16_and_back(W0)
    b = _to_bf16_and_back(W1)
    scale = np.maximum(np.abs(a), np.abs(b))
    return np.abs(b - a) > (ETA * scale)


def _principal_angles_degrees(A, B, k):
    """Mean and max principal angle in degrees between the top-k column subspaces of A
    and B, as arccos of the singular values of the product of the two orthonormal bases."""
    a = A[:, :k]
    b = B[:, :k]
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.degrees(np.arccos(s))
    return float(np.mean(ang)), float(np.max(ang))


# A singular value below this fraction of the largest is numerically zero, not a measurement.
NUMERICAL_ZERO_FRACTION = 1e-12


def _hill_tail(singular_values, top=HILL_TOP):
    """Auxiliary diagnostic 7 (Hill 1975): inverse mean log-ratio of the leading singular
    values to the smallest selected one.

    Undefined when the reference value sv[top] is numerically zero. A LoRA update of rank r
    has exactly r non-zero singular values, so for r <= top the reference is a rounding
    artefact somewhere around 1e-14 of the leading value, and the "tail exponent" computed
    from it is a property of the floating-point arithmetic rather than of the update. It is
    reported as undefined instead of as a number, because a number here would be read as
    evidence about the update's spectrum."""
    sv = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    top = min(top, len(sv) - 1)
    if top < 2:
        return float("nan")
    xs = sv[:top]
    xmin = sv[top]
    if xmin <= 0 or np.any(xs <= 0) or (sv[0] > 0 and xmin < NUMERICAL_ZERO_FRACTION * sv[0]):
        return float("nan")
    return float(top / np.sum(np.log(xs / xmin)))


def _spectrum_and_top_vectors(W, k):
    """Full singular-value spectrum plus the top-k left and right singular vectors.

    Obtained from the eigendecomposition of the smaller Gram matrix rather than a full SVD.
    For a 3840x15360 weight this replaces a 23-45 second LAPACK call with about 2.5 seconds,
    which is what makes this stage finish at all: the exact formulation needs roughly 140
    CPU-hours across the study even after subsampling.

    Squaring the matrix squares the condition number, so the Gram MUST be formed in float64.
    In float32 the spectrum carries about 1.6e-05 relative error on an ill-conditioned
    attention matrix, and normalized_spectral_shift is a difference of two nearly identical
    spectra: for a small update that error is not small relative to the signal, it inflated
    the shift by up to 4.7x, and the bias is one-sided. hill_tail_exponent was inflated about
    4x for the same reason, because a rank-8 update's true zero singular values were replaced
    by a 2e-07 noise floor. In float64 the spectrum agrees with the exact SVD to ~4e-13 and
    the diagnostics are indistinguishable from it, at a fraction of a second's extra cost."""
    W = np.asarray(W, dtype=np.float64)
    transposed = W.shape[0] > W.shape[1]
    A = W.T if transposed else W                      # (m, n) with m <= n
    G = A @ A.T                                       # (m, m), BLAS-3, float64
    ev, U = np.linalg.eigh(G)
    ev = ev[::-1]
    U = U[:, ::-1]
    s = np.sqrt(np.clip(ev, 0.0, None))
    kk = int(min(k, U.shape[1]))
    Uk = U[:, :kk]
    sk = s[:kk]
    # V = A^T U / s, formed only for the columns actually needed.
    Vk = (A.T @ Uk) / np.where(sk > 0, sk, 1.0)
    if transposed:
        Uk, Vk = Vk, Uk
    return s, Uk, Vk


def _update_singular_values(dW, sketch=192):
    """Exact singular values of a low-rank update, without a full SVD.

    A LoRA delta has rank at most the adapter rank (<= RANK_MAX here). Projecting onto a
    random range of dimension `sketch` >= rank captures the whole row space, so the singular
    values of the small projected matrix are the singular values of dW, and the values beyond
    its rank come out as the true near-zeros rather than a squared-arithmetic noise floor."""
    dW = np.asarray(dW, dtype=np.float64)
    m, n = dW.shape
    k = int(min(sketch, m, n))
    if k >= min(m, n):
        return np.linalg.svd(dW, compute_uv=False)
    rs = np.random.default_rng(0)
    Q, _ = np.linalg.qr(dW @ rs.standard_normal((n, k)))
    sv = np.linalg.svd(Q.T @ dW, compute_uv=False)
    # Exact only while the sketch covers the update's rank. If the smallest returned value is
    # not negligible the range was not captured, and the spectrum is truncated: stable rank
    # and the tail exponent then come out low with nothing to show it. Fall back rather than
    # publish a quietly biased number.
    if sv.size and sv[0] > 0 and sv[-1] > 1e-8 * sv[0]:
        logger.warning("Update rank exceeds the %d-dimensional sketch; falling back to a full "
                       "SVD for this matrix so the spectrum is not truncated.", k)
        return np.linalg.svd(dW, compute_uv=False)
    return sv


def matrix_diagnostics(W0, dW):
    """All seven diagnostics for one weight matrix.

    W0 is the frozen base weight, dW the effective merged update, W1 = W0 + dW the updated
    weight. Diagnostics 2 and 3 compare W0 against W1 (the pretrained structure before and
    after the repair). Diagnostics 5, 6 and 7 describe dW itself."""
    W0 = np.asarray(W0, dtype=np.float64)
    dW = np.asarray(dW, dtype=np.float64)
    W1 = W0 + dW
    out = {}

    # 1. bf16-aware update sparsity: one minus the fraction of visibly changed entries.
    changed = _changed_mask(W0, W1)
    out["bf16_aware_update_sparsity"] = float(1.0 - changed.mean())

    # The top-k subspaces and the full spectra, without a full SVD. k must cover both the
    # principal-angle subspace and the rank-r reconstruction below.
    k = int(min(MAX_SUBSPACE_K, min(W0.shape)))
    r = int(min(PRINCIPAL_RANK, min(W0.shape)))
    s0, U0, V0 = _spectrum_and_top_vectors(W0, max(k, r))
    s1, U1, V1 = _spectrum_and_top_vectors(W1, max(k, r))

    # 2. principal-angle rotation of the dominant left and right subspaces of the weight.
    left_mean, left_max = _principal_angles_degrees(U0, U1, k)
    right_mean, right_max = _principal_angles_degrees(V0, V1, k)
    out["principal_angle_rotation_left_subspace_degrees"] = left_mean
    out["principal_angle_rotation_right_subspace_degrees"] = right_mean
    out["principal_angle_rotation_max_degrees"] = max(left_max, right_max)

    # 3. normalized spectral shift between the updated and base singular-value vectors.
    n = min(len(s0), len(s1))
    denom = float(np.linalg.norm(s0)) or 1.0
    out["normalized_spectral_shift"] = float(np.linalg.norm(s1[:n] - s0[:n]) / denom)

    # 4. overlap of the update mask with the principal and low-magnitude masks of W0.
    recon = (U0[:, :r] * s0[:r]) @ V0[:, :r].T
    principal_mask = np.abs(recon) >= np.quantile(np.abs(recon), 1.0 - ALPHA)
    low_mask = np.abs(W0) <= np.quantile(np.abs(W0), ALPHA)
    changed_count = int(changed.sum())
    if changed_count == 0:
        out["update_mask_overlap_with_principal_mask"] = float("nan")
        out["update_mask_overlap_with_low_magnitude_mask"] = float("nan")
    else:
        out["update_mask_overlap_with_principal_mask"] = float((changed & principal_mask).sum() / changed_count)
        out["update_mask_overlap_with_low_magnitude_mask"] = float((changed & low_mask).sum() / changed_count)

    # 5, 6, 7. properties of the update itself.
    sd = _update_singular_values(dW)
    fro = float(np.linalg.norm(dW))
    out["stable_rank_of_update"] = float((np.sum(sd ** 2) / (sd[0] ** 2)) if sd.size and sd[0] > 0 else 0.0)
    out["frobenius_norm_of_update"] = fro
    out["relative_frobenius_drift"] = float(fro / (np.linalg.norm(W0) or 1.0))
    out["hill_tail_exponent"] = _hill_tail(sd)
    return out


# --------------------------- delta extraction -------------------------------

def _iter_deltas(peft_model):
    """Yield (module_name, W0_numpy, dW_numpy) for every adapted Linear module.

    Merging through get_delta_weight makes the result correct regardless of the scaling
    convention (alpha/r or the rsLoRA alpha/sqrt(r))."""
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "base_layer"):
            try:
                adapter = list(mod.lora_A.keys())[0]
                if not hasattr(mod, "get_delta_weight"):
                    continue
                dW = mod.get_delta_weight(adapter)
                W0 = mod.base_layer.weight.detach()
                yield name, W0.float().cpu().numpy(), dW.detach().float().cpu().numpy()
            except Exception:
                continue


def _layer_and_module(name):
    m = _LAYER_RE.search(name)
    layer = int(m.group(1)) if m else -1
    mm = _MODULE_RE.search(name)
    return layer, (mm.group(1) if mm else "unknown")


def _rank_budget(adapter_dir):
    """Read the LoRA rank the adapter was actually saved with, so a low stable rank can be
    read against the capacity that was available."""
    import json

    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.exists():
        return ""
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return ""
    ranks = list((data.get("rank_pattern") or {}).values())
    default = data.get("r")
    if ranks:
        return f"{min(ranks)}-{max(ranks)}"
    return default if default is not None else ""


# ------------------------- joins to the evaluation --------------------------

def _evaluation_join():
    """Post-repair CtxtAware, DiffAware, and preservation relative to the frozen base.

    Read from the existing evaluation CSV, never recomputed here."""
    import pandas as pd

    path = RESULTS_DIR / "main_evaluation_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    overall = df[df["scope"].astype(str).str.endswith("|all")].copy()
    # the in-distribution frozen-test rows only; a held-out-axis row answers a different
    # question and must not be averaged into the preservation quantity
    in_dist = overall[overall["scope"].astype(str).str.startswith("agrifacts_frozen_test")]
    if not in_dist.empty:
        overall = in_dist.copy()
    ctxt = "contextual_awareness_metric_wang_2025_precision_style"
    diff = "difference_aware_metric_wang_2025_recall_style"
    for c in (ctxt, diff):
        overall[c] = pd.to_numeric(overall[c], errors="coerce")
    base = overall[overall["method"] == "frozen_base"].groupby("tier")[ctxt].mean()
    out = {}
    for (tier, method), g in overall.groupby(["tier", "method"]):
        b = base.get(tier, np.nan)
        post_ctxt = float(g[ctxt].mean())
        post_diff = float(g[diff].mean())
        pres = float(post_ctxt / b) if (b and b == b and b > 0) else float("nan")
        out[(tier, method)] = {
            "contextual_awareness_metric_post_repair": post_ctxt,
            "difference_aware_metric_post_repair": post_diff,
            "contextual_awareness_preservation_relative_to_base": pres,
        }
    return out


def _trainable_percentage_table():
    """Trainable-parameter percentage per (tier, method), for the partial correlation."""
    import json

    out = {}
    for fname in ("train_graft_runs.json", "train_baselines_runs.json"):
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        try:
            for r in json.loads(p.read_text(encoding="utf-8")):
                pct = r.get("trainable_parameter_percentage")
                if pct is not None:
                    out[(r.get("tier"), r.get("method"))] = float(pct)
        except Exception:
            continue
    return out


# ------------------------------ statistics ----------------------------------

def _bootstrap_pearson_ci(xs, ys, resamples=BOOTSTRAP_RESAMPLES, seed=42):
    """Percentile bootstrap 95 percent interval for the Pearson correlation."""
    from scipy.stats import pearsonr

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        if np.std(xs[idx]) == 0 or np.std(ys[idx]) == 0:
            continue
        stats.append(pearsonr(xs[idx], ys[idx])[0])
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def _partial_correlation(xs, ys, zs):
    """Pearson correlation of x and y with z partialled out.

    Reviewers will ask whether the geometry signal is only a proxy for how many
    parameters a method trained; z is the trainable-parameter percentage."""
    from scipy.stats import pearsonr

    xs, ys, zs = np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float)
    ok = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    if ok.sum() < 4 or np.std(zs[ok]) == 0:
        return float("nan")
    rxy = pearsonr(xs[ok], ys[ok])[0]
    rxz = pearsonr(xs[ok], zs[ok])[0]
    ryz = pearsonr(ys[ok], zs[ok])[0]
    denom = np.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return float((rxy - rxz * ryz) / denom)


def _regime_label(principal_overlap, spectral_shift, anchor_shift):
    """Regime placement against the dense anchor, with documented thresholds."""
    if principal_overlap != principal_overlap or spectral_shift != spectral_shift:
        return "undetermined_insufficient_update"
    if principal_overlap >= ALPHA:
        return "principal_aligned_and_distorting"
    if anchor_shift and anchor_shift == anchor_shift and anchor_shift > 0:
        if spectral_shift <= SPECTRUM_PRESERVING_FRACTION_OF_ANCHOR * anchor_shift:
            return "off_principal_and_spectrum_preserving"
        return "relaxed_off_principal"
    return "relaxed_off_principal"


# --------------------------------- figure -----------------------------------

def _regime_figure(rows, correlations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in rows if isinstance(r.get("normalized_spectral_shift_mean"), float)]
    if not pts:
        logger.warning("No weight-modifying rows; skipping the regime figure.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for r in pts:
        ax.scatter(r["normalized_spectral_shift_mean"], r["update_mask_overlap_with_principal_mask_mean"], s=42, color="#3a6ea5")
        ax.annotate(r["method_name"], (r["normalized_spectral_shift_mean"], r["update_mask_overlap_with_principal_mask_mean"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(ALPHA, linestyle="--", color="#888888", linewidth=1)
    ax.text(0.01, ALPHA + 0.01, "random overlap baseline", fontsize=7, color="#666666", transform=ax.get_yaxis_transform())
    ax.set_xlabel("normalized spectral shift (spectrum distortion)")
    ax.set_ylabel("update-mask overlap with principal mask")
    ax.set_title("Parameter-space regime placement")

    ax = axes[1]
    xs = [r["normalized_spectral_shift_mean"] for r in pts
          if isinstance(r.get("contextual_awareness_preservation_relative_to_base"), float)]
    ys = [r["contextual_awareness_preservation_relative_to_base"] for r in pts
          if isinstance(r.get("contextual_awareness_preservation_relative_to_base"), float)]
    if xs:
        ax.scatter(xs, ys, s=42, color="#a5533a")
        for r in pts:
            if isinstance(r.get("contextual_awareness_preservation_relative_to_base"), float):
                ax.annotate(r["method_name"], (r["normalized_spectral_shift_mean"], r["contextual_awareness_preservation_relative_to_base"]),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
        note = "; ".join(
            f"{c['base_model_name']} r={c['pearson_correlation']} p={c['pearson_p_value']}"
            for c in correlations if c["geometry_metric_name"] == "normalized_spectral_shift_mean"
        )
        if note:
            ax.set_title("Spectrum distortion vs preservation (" + note + ")", fontsize=9)
        else:
            ax.set_title("Spectrum distortion vs preservation")
    ax.set_xlabel("normalized spectral shift")
    ax.set_ylabel("difference-awareness preservation relative to base")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "parameter_space_regime.png", dpi=150)
    plt.close(fig)


# ---------------------------------- main ------------------------------------

def _stratified_matrix_sample(names, cap):
    """Choose at most `cap` matrices spread evenly across depth and module type.

    Taking the first `cap` matrices in module order means taking the earliest layers. For a
    study whose claim is about WHICH layers carry the fairness-relevant update, that would
    concentrate every geometric diagnostic at the bottom of the network and invite exactly the
    objection the diagnostics exist to answer. Sampling evenly across layers within each module
    type keeps depth coverage unbiased."""
    if not cap or len(names) <= cap:
        return list(names)
    by_type = {}
    for name in names:
        layer, module_type = _layer_and_module(name)
        by_type.setdefault(module_type, []).append((layer, name))
    chosen = []
    types = sorted(by_type)
    # Distribute the budget across module types, giving the remainder to the largest groups.
    per_type = {t: cap // len(types) for t in types}
    for t in sorted(types, key=lambda t: -len(by_type[t]))[: cap % len(types)]:
        per_type[t] += 1
    for j, t in enumerate(types):
        entries = sorted(by_type[t])
        k = min(per_type[t], len(entries))
        if k <= 0:
            continue
        if k == 1:
            # Stagger across module types; taking the middle for each put every type on the
            # same single layer whenever the cap was below one per type.
            idx = [min(len(entries) - 1, (len(entries) * (2 * j + 1)) // (2 * max(1, len(types))))]
        else:
            step = (len(entries) - 1) / (k - 1)
            # Offset each module type's grid by a fraction of one step. Without this every
            # type samples the same evenly spaced layers, and the whole sample collapses onto
            # about k distinct depths instead of spreading over the stack.
            shift = step * (j / max(1, len(types)))
            idx = sorted({min(len(entries) - 1, int(round(i * step + shift))) for i in range(k)})
        chosen.extend(entries[i][1] for i in idx)
    return chosen


def _analyse_adapter(base_model, adapter_dir):
    """Return (per_matrix_rows, adaptable_modules, adapted_modules) for one saved adapter.

    adaptable_modules is the number of Linear modules in the BASE model, not the number PEFT
    targeted. Using the targeted count made module_coverage_fraction identically 1.0 for every
    arm, since after training every targeted module has a non-zero update: the column carried
    no information, and the dense-anchor fallback that ranks arms by it tied on every row."""
    from peft import PeftModel

    pm = PeftModel.from_pretrained(base_model, str(adapter_dir))
    per_matrix = []
    total_modules = 0
    adapted_modules = 0
    try:
        # Enumerate first, so the subsample is chosen over the whole adapter rather than over
        # whatever happened to be visited before the cap was reached.
        all_names = [n for n, m in pm.named_modules()
                     if hasattr(m, "lora_A") and hasattr(m, "base_layer") and hasattr(m, "get_delta_weight")]
        # Denominator: every Linear the adapter COULD have targeted, so a sparse placement
        # scores low and a dense one scores high, which is what the column is read as meaning.
        import torch.nn as nn

        # PEFT replaces a targeted Linear with a wrapper whose ORIGINAL Linear moves to
        # `<name>.base_layer`. Excluding base_layer therefore excludes exactly the modules
        # that were adapted, and the denominator then shrank as placement grew denser. Count
        # every real Linear instead (base_layer plus untargeted), excluding only the injected
        # lora_A / lora_B factors, so the denominator is a property of the base model and is
        # identical across arms.
        adaptable_modules = sum(
            1 for name, mod in pm.named_modules()
            if isinstance(mod, nn.Linear) and ".lora_A" not in name and ".lora_B" not in name)
        total_modules = max(1, adaptable_modules)
        if LAYER_STRIDE > 1:
            all_names = [n for i, n in enumerate(all_names) if i % LAYER_STRIDE == 0]
        selected = set(_stratified_matrix_sample(all_names, MAX_MATRICES))
        if MAX_MATRICES and total_modules > len(selected):
            logger.info("Geometry: analysing a stratified %d of %d adapted matrices in %s.",
                        len(selected), total_modules, Path(adapter_dir).parent.name)
        for name, W0, dW in _iter_deltas(pm):
            # Counted over EVERY adapted module, before the sampling filter. Counting only
            # sampled matrices against the full module total inverted the published coverage
            # fractions: a dense 48-layer arm scored 48/336 = 0.14 while a sparse 10-layer one
            # scored 48/70 = 0.69, so the sparsest placement looked like the densest, and the
            # dense anchor that sets every regime label is chosen by exactly that column.
            if bool(np.any(dW != 0)):
                adapted_modules += 1
            if name not in selected:
                continue
            d = matrix_diagnostics(W0, dW)
            layer, module_type = _layer_and_module(name)
            per_matrix.append({"matrix_name": name, "layer_index": layer, "module_type": module_type, **d})
    finally:
        try:
            pm.unload()
        except Exception:
            logger.warning("Could not unload the adapter for %s; reloading the base model to "
                           "avoid carrying its weights into the next arm.", adapter_dir)
            raise
    return per_matrix, total_modules, adapted_modules


def main():
    tiers = [d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir()] if CHECKPOINTS_DIR.exists() else []
    if not tiers:
        tiers = list(model_registry.PRIMARY_TIERS)

    evaluation = _evaluation_join()
    trainable = _trainable_percentage_table()
    method_rows = []
    layer_rows = []
    run_log = []

    for tier in tiers:
        ckpt_root = CHECKPOINTS_DIR / tier
        if not ckpt_root.exists():
            run_log.append({"base_model_name": tier, "method_name": "", "random_seed": "",
                            "status_message": "no_checkpoint_directory", "number_of_analyzed_matrices": 0})
            continue
        try:
            base_model, tok, meta = model_registry.load_model_and_tokenizer(tier, smoke=(tier == "smoke"))
        except Exception as e:
            logger.warning("Geometry: could not load base %s (%s); skipping.", tier, e)
            run_log.append({"base_model_name": tier, "method_name": "", "random_seed": "",
                            "status_message": f"base_model_load_failed: {e}", "number_of_analyzed_matrices": 0})
            continue

        tier_rows = []
        for mdir in sorted(ckpt_root.iterdir()):
            if not mdir.is_dir():
                continue
            method = mdir.name
            common = {
                "method_name": method, "base_model_name": tier,
                "base_model_revision_hash": meta.get("model_revision_hash", ""),
                "quantization_setting": "nf4_4bit" if "qlora" in method else meta.get("quantization_setting", ""),
            }
            if any(m in method for m in NO_WEIGHT_UPDATE_METHODS):
                # Inference-time steering changes no weight. Record the row and leave the
                # weight diagnostics blank rather than fabricating them.
                method_rows.append({
                    **common, "weight_update_type": "inference_time_no_weight_update",
                    "parameter_space_regime_label": "inference_time_no_weight_update",
                    "random_seed": "", "number_of_analyzed_matrices": 0,
                })
                run_log.append({"base_model_name": tier, "method_name": method, "random_seed": "",
                                "status_message": "inference_time_no_weight_update", "number_of_analyzed_matrices": 0})
                continue

            for sdir in sorted(mdir.glob("seed_*")):
                seed = sdir.name.split("_")[-1]
                final = sdir / "final"
                adapter = final if final.exists() else sdir
                try:
                    per_matrix, total_modules, adapted_modules = _analyse_adapter(base_model, adapter)
                except Exception as e:
                    logger.warning("Could not analyse %s (%s); skipping.", adapter, e)
                    run_log.append({"base_model_name": tier, "method_name": method, "random_seed": seed,
                                    "status_message": f"adapter_load_failed: {e}", "number_of_analyzed_matrices": 0})
                    continue
                if not per_matrix:
                    run_log.append({"base_model_name": tier, "method_name": method, "random_seed": seed,
                                    "status_message": "no_adapted_matrices_found", "number_of_analyzed_matrices": 0})
                    continue

                for pm_row in per_matrix:
                    layer_rows.append({
                        "method_name": method, "base_model_name": tier, "random_seed": seed,
                        "layer_index": pm_row["layer_index"], "module_type": pm_row["module_type"],
                        "matrix_name": pm_row["matrix_name"],
                        **{k: (round(v, 6) if isinstance(v, float) and v == v else "")
                           for k, v in pm_row.items()
                           if k not in ("matrix_name", "layer_index", "module_type")},
                    })

                def m(key):
                    vals = [d[key] for d in per_matrix if d[key] == d[key]]
                    return float(np.mean(vals)) if vals else float("nan")

                ev = evaluation.get((tier, method), {})
                row = {
                    **common,
                    "weight_update_type": "merged_low_rank_adapter_update",
                    "lora_rank_budget": _rank_budget(adapter),
                    "fraction_of_analyzed_matrices_adapted": round(len(per_matrix) / max(1, total_modules), 4),
                    "module_coverage_fraction": round(adapted_modules / max(1, total_modules), 4),
                    "bf16_aware_update_sparsity_mean": round(m("bf16_aware_update_sparsity"), 6),
                    "principal_angle_rotation_left_subspace_mean_degrees": round(m("principal_angle_rotation_left_subspace_degrees"), 4),
                    "principal_angle_rotation_right_subspace_mean_degrees": round(m("principal_angle_rotation_right_subspace_degrees"), 4),
                    "principal_angle_rotation_max_degrees": round(m("principal_angle_rotation_max_degrees"), 4),
                    "normalized_spectral_shift_mean": round(m("normalized_spectral_shift"), 6),
                    "update_mask_overlap_with_principal_mask_mean": round(m("update_mask_overlap_with_principal_mask"), 4),
                    "update_mask_overlap_with_low_magnitude_mask_mean": round(m("update_mask_overlap_with_low_magnitude_mask"), 4),
                    "stable_rank_of_update_mean": round(m("stable_rank_of_update"), 4),
                    "frobenius_norm_of_update_mean": round(m("frobenius_norm_of_update"), 6),
                    "relative_frobenius_drift_mean": round(m("relative_frobenius_drift"), 6),
                    "hill_tail_exponent_mean": round(m("hill_tail_exponent"), 4) if m("hill_tail_exponent") == m("hill_tail_exponent") else "",
                    "trainable_parameter_percentage": trainable.get((tier, method), ""),
                    "number_of_analyzed_matrices": len(per_matrix),
                    "random_seed": seed,
                    "contextual_awareness_metric_post_repair": round(ev.get("contextual_awareness_metric_post_repair", float("nan")), 4) if ev.get("contextual_awareness_metric_post_repair", float("nan")) == ev.get("contextual_awareness_metric_post_repair", float("nan")) else "",
                    "difference_aware_metric_post_repair": round(ev.get("difference_aware_metric_post_repair", float("nan")), 4) if ev.get("difference_aware_metric_post_repair", float("nan")) == ev.get("difference_aware_metric_post_repair", float("nan")) else "",
                    "contextual_awareness_preservation_relative_to_base": ev.get("contextual_awareness_preservation_relative_to_base", float("nan")),
                }
                tier_rows.append(row)
                run_log.append({"base_model_name": tier, "method_name": method, "random_seed": seed,
                                "status_message": "analyzed", "number_of_analyzed_matrices": len(per_matrix)})

        # Regime labels need the dense anchor for this model, so they are assigned after
        # every method on the model has been measured.
        anchor_rows = [r for r in tier_rows if r["method_name"] in DENSE_ANCHOR_METHODS]
        if not anchor_rows and tier_rows:
            anchor_rows = [max(tier_rows, key=lambda r: r["module_coverage_fraction"])]
        anchor_shift = float(np.mean([r["normalized_spectral_shift_mean"] for r in anchor_rows])) if anchor_rows else float("nan")
        anchor_name = anchor_rows[0]["method_name"] if anchor_rows else ""
        for r in tier_rows:
            r["parameter_space_regime_label"] = _regime_label(
                r["update_mask_overlap_with_principal_mask_mean"], r["normalized_spectral_shift_mean"], anchor_shift)
            r["regime_anchor_method_name"] = anchor_name
        method_rows.extend(tier_rows)
        del base_model

    # correlations, per model and pooled across models
    corr_rows = []
    groups = {}
    for r in method_rows:
        if r.get("weight_update_type") != "merged_low_rank_adapter_update":
            continue
        groups.setdefault(r["base_model_name"], []).append(r)
    if len(groups) > 1:
        groups["pooled_across_models"] = [r for rows in groups.values() for r in rows]

    from scipy.stats import pearsonr, spearmanr

    for model_name, rows in groups.items():
        pres = [r["contextual_awareness_preservation_relative_to_base"] for r in rows]
        pcts = [float(r["trainable_parameter_percentage"]) if r["trainable_parameter_percentage"] != "" else float("nan") for r in rows]
        for metric in ("normalized_spectral_shift_mean", "update_mask_overlap_with_principal_mask_mean"):
            xs = [r[metric] for r in rows]
            ok = [i for i in range(len(rows)) if np.isfinite(xs[i]) and np.isfinite(pres[i])]
            if len(ok) < 3:
                continue
            x = [xs[i] for i in ok]
            y = [pres[i] for i in ok]
            z = [pcts[i] for i in ok]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            pr, pp = pearsonr(x, y)
            sr, sp = spearmanr(x, y)
            lo, hi = _bootstrap_pearson_ci(x, y)
            partial = _partial_correlation(x, y, z)
            corr_rows.append({
                "base_model_name": model_name,
                "geometry_metric_name": metric,
                "preservation_metric_name": "contextual_awareness_preservation_relative_to_base",
                "pearson_correlation": round(float(pr), 4), "pearson_p_value": round(float(pp), 4),
                "spearman_correlation": round(float(sr), 4), "spearman_p_value": round(float(sp), 4),
                "bootstrap_confidence_interval_lower": round(lo, 4) if lo == lo else "",
                "bootstrap_confidence_interval_upper": round(hi, 4) if hi == hi else "",
                "partial_correlation_controlling_for_trainable_parameter_percentage": round(partial, 4) if partial == partial else "",
                "number_of_configurations": len(ok),
                "predicted_sign_preregistered": "negative",
            })
            if len(ok) < 15:
                logger.warning(
                    "Geometry correlation on %s / %s uses %d configurations; the "
                    "preregistration asks for at least 15 before the law is claimed.",
                    model_name, metric, len(ok))

    # blank the non-finite preservation values only at write time, so the statistics above
    # still see the real numbers
    for r in method_rows:
        v = r.get("contextual_awareness_preservation_relative_to_base")
        r["contextual_awareness_preservation_relative_to_base"] = round(v, 4) if isinstance(v, float) and v == v else ""

    write_csv(GEOMETRY_DIR / "parameter_space_geometry_per_method.csv", method_rows, PER_METHOD_COLUMNS)
    write_csv(GEOMETRY_DIR / "per_layer_per_module_geometry.csv", layer_rows, PER_LAYER_COLUMNS)
    write_csv(GEOMETRY_DIR / "geometry_vs_difference_awareness_correlation.csv", corr_rows, CORR_COLUMNS)
    write_csv(GEOMETRY_DIR / "geometry_run_log.csv", run_log, RUN_LOG_COLUMNS)
    try:
        _regime_figure(method_rows, corr_rows)
    except Exception as e:
        logger.warning("Regime figure failed (%s).", e)

    log_run_metadata("parameter_space_geometry", {
        "method_rows": len(method_rows), "layer_rows": len(layer_rows), "correlations": len(corr_rows),
    })
    logger.info("Geometry: %d method rows, %d per-matrix rows, %d correlation rows.",
                len(method_rows), len(layer_rows), len(corr_rows))


if __name__ == "__main__":
    main()
