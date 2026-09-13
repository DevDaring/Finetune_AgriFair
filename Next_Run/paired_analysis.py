"""P2.2 / P2.3 - seed-explicit, dependence-aware paired analysis (plan sections 7.2, 7.3).

What this corrects relative to the legacy CPU_Run/statistics_tests.py:

  * one record per (model, method, seed, split, item) with explicit coverage, instead of
    "first available seed" silently standing in for a pooled estimate;
  * novel-test B as the primary endpoint, with every error type reported separately
    (erasure, wrong-group, fabrication, invalid) and an undefined B when a condition is absent;
  * paired SOURCE-CLUSTER bootstrap (10k draws) for intervals and a paired CLUSTER-SWAP
    permutation test for p-values, both on the same sampled clusters for all seeds and
    both methods; Monte Carlo p never exactly zero;
  * frozen comparison families: localisation (GRAFT vs uniform + 3 random placements,
    per model, up to 16 contrasts, Holm together, seed-42 paired because the controls have
    one seed) and a separate secondary family; unavailable contrasts stay visible;
  * a structural-key clustering sensitivity, with cluster counts reported;
  * the LOAO error taxonomy on cached transfer outputs, GRAFT vs vanilla LoRA.

CPU-efficient by construction: per-cluster, per-seed contingency counts are computed once
and resampled as arrays.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from Next_Run import common as C
from Next_Run import stats as S

SLICES = {"full": lambda r: True,
          "structure_novel": lambda r: r["test_slice"] == "structure_novel",
          "structure_familiar": lambda r: r["test_slice"] == "structure_familiar"}


def load_all_records(cfg: Dict, include_cpu_baselines: bool = True) -> Dict[C.Arm, List[Dict]]:
    items = C.test_items_by_id()
    recs: Dict[C.Arm, List[Dict]] = {}
    coverage = []
    for arm, path in C.limit_arms(C.prediction_files(), cfg).items():
        r, issues = C.normalise_predictions(arm, path, items)
        recs[arm] = r
        coverage.append({"tier": arm.tier, "method": arm.method, "seed": arm.seed, **issues})
    if include_cpu_baselines:
        for p in sorted((C.output_dir(cfg) / "cpu_baselines").glob("per_item_predictions_cpu-baseline_*_seed*.jsonl")):
            name = p.stem.replace("per_item_predictions_", "")
            tier, rest = name.split("_", 1)
            method, seed = rest.rsplit("_seed", 1)
            arm = C.Arm(tier, method, int(seed))
            r, issues = C.normalise_predictions(arm, p, items)
            recs[arm] = r
            coverage.append({"tier": tier, "method": method, "seed": int(seed), **issues})
    C.write_csv(C.output_dir(cfg) / "paired" / "coverage_by_arm.csv", coverage)
    return recs


def per_arm_metrics(recs: Dict[C.Arm, List[Dict]]) -> List[Dict]:
    rows = []
    for arm, r in sorted(recs.items(), key=lambda kv: kv[0].key):
        for sl, sel in SLICES.items():
            cnt = C.Counts()
            for x in r:
                if sel(x):
                    cnt.add(x["gold"], x["pred"] or None, x["parse_ok"])
            m = C.metrics_from_counts(cnt.as_array())
            rows.append({"tier": arm.tier, "method": arm.method, "seed": arm.seed, "slice": sl,
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
    return rows


def _seeds(recs, tier, method) -> List[int]:
    return sorted(a.seed for a in recs if a.tier == tier and a.method == method)


def _stack(recs, tier, method, seeds, sel, cluster_field, clusters) -> np.ndarray:
    mats = []
    for s in seeds:
        r = [x for x in recs[C.Arm(tier, method, s)] if sel(x)]
        m, _ = C.counts_by_cluster(r, cluster_field, clusters)
        mats.append(m)
    return np.stack(mats) if mats else np.zeros((0, len(clusters), 8))


def contrast(recs, tier, ref: str, cmp_: str, sl: str, cfg: Dict, cluster_field: str = "source_cell") -> Dict:
    """One paired contrast. Seed policy (plan 7.2): use the seeds both arms share; if the
    comparator has only seed 42, compare against seed-42 reference and say so."""
    sel = SLICES[sl]
    s_ref, s_cmp = _seeds(recs, tier, ref), _seeds(recs, tier, cmp_)
    shared = sorted(set(s_ref) & set(s_cmp))
    row = {"tier": tier, "reference": ref, "comparator": cmp_, "slice": sl, "cluster_field": cluster_field,
           "seeds_reference": s_ref, "seeds_comparator": s_cmp, "seeds_used": shared,
           "seed_policy": "matched_multi_seed" if len(shared) > 1 else ("seed_42_paired" if shared == [42] else "single_shared_seed" if shared else "unavailable")}
    if not shared:
        row.update({"status": "unavailable", "reason": "no shared seed"})
        return row
    # align items across the two arms per seed, then cluster
    aligned_ref, aligned_cmp, cov_total = [], [], {"common": 0, "only_in_a": 0, "only_in_b": 0}
    for s in shared:
        a, b, cov = C.align_pair([x for x in recs[C.Arm(tier, ref, s)] if sel(x)],
                                 [x for x in recs[C.Arm(tier, cmp_, s)] if sel(x)])
        aligned_ref.append(a); aligned_cmp.append(b)
        for k in cov_total: cov_total[k] += cov[k]
    clusters = sorted({x[cluster_field] for a in aligned_ref for x in a})
    A = np.stack([C.counts_by_cluster(a, cluster_field, clusters)[0] for a in aligned_ref])
    B = np.stack([C.counts_by_cluster(b, cluster_field, clusters)[0] for b in aligned_cmp])
    per_seed = [float(S.b_from_sums(A[i].sum(0)) - S.b_from_sums(B[i].sum(0))) for i in range(len(shared))]
    boot = S.paired_cluster_bootstrap(A, B, cfg["bootstrap_draws"], cfg["analysis_seed"])
    perm = S.cluster_swap_permutation(A, B, cfg["permutation_draws"], cfg["analysis_seed"] + 1)
    row.update({"status": "ok", "n_clusters": boot["n_clusters"], "items_common": cov_total["common"],
                "items_only_reference": cov_total["only_in_a"], "items_only_comparator": cov_total["only_in_b"],
                "b_reference_mean": round(float(np.nanmean([S.b_from_sums(A[i].sum(0)) for i in range(len(shared))])), 4),
                "b_comparator_mean": round(float(np.nanmean([S.b_from_sums(B[i].sum(0)) for i in range(len(shared))])), 4),
                "difference_reference_minus_comparator": round(S.observed_difference(A, B), 4),
                "per_seed_differences": [round(x, 4) for x in per_seed],
                "seed_spread_sd": round(float(np.std(per_seed)), 4) if len(per_seed) > 1 else "",
                "bootstrap_ci_lower": round(boot["lower"], 4), "bootstrap_ci_upper": round(boot["upper"], 4),
                "bootstrap_draws": boot["draws"],
                "permutation_p_two_sided": round(perm["p_value"], 5) if perm["p_value"] == perm["p_value"] else "",
                "permutation_draws": perm["draws"]})
    return row


def families(recs, cfg) -> List[Dict]:
    tiers = sorted({a.tier for a in recs if a.tier != "cpu-baseline"})
    ref = cfg["localisation_reference"]
    rows = []
    for sl in ("structure_novel", "full"):
        # localisation family: Holm across all its contrasts together
        fam = []
        for t in tiers:
            for ctl in cfg["localisation_controls"]:
                r = contrast(recs, t, ref, ctl, sl, cfg); r["family"] = "localisation"; fam.append(r)
        ps = [r.get("permutation_p_two_sided") if r.get("status") == "ok" else float("nan") for r in fam]
        ps = [float(p) if p not in ("", None) else float("nan") for p in ps]
        for r, adj in zip(fam, S.holm(ps)):
            r["holm_adjusted_p"] = round(adj, 5) if adj == adj else ""
            r["significant_after_holm_0_05"] = (adj == adj) and adj < 0.05
        rows += fam
        # secondary family, corrected separately and labelled secondary
        sec = []
        for t in tiers:
            for cmp_ in cfg["secondary_comparators"]:
                r = contrast(recs, t, ref, cmp_, sl, cfg); r["family"] = "secondary_baselines"; sec.append(r)
        ps = [float(r["permutation_p_two_sided"]) if r.get("status") == "ok" and r.get("permutation_p_two_sided") != "" else float("nan") for r in sec]
        for r, adj in zip(sec, S.holm(ps)):
            r["holm_adjusted_p"] = round(adj, 5) if adj == adj else ""
            r["significant_after_holm_0_05"] = (adj == adj) and adj < 0.05
        rows += sec
        # cpu baselines vs GRAFT, exploratory family
        cpu = []
        for t in tiers:
            for a in sorted(x for x in recs if x.tier == "cpu-baseline"):
                r = contrast_cross_tier(recs, t, ref, a, sl, cfg); r["family"] = "cpu_baselines_exploratory"; cpu.append(r)
        rows += cpu
    return rows


def contrast_cross_tier(recs, tier, ref, cpu_arm: C.Arm, sl, cfg) -> Dict:
    """GRAFT on `tier` (seed 42) versus a CPU baseline: same items, same clusters."""
    sel = SLICES[sl]
    if C.Arm(tier, ref, 42) not in recs:
        return {"tier": tier, "reference": ref, "comparator": cpu_arm.method, "slice": sl, "status": "unavailable",
                "reason": "reference seed 42 missing", "seed_policy": "seed_42_paired"}
    a, b, cov = C.align_pair([x for x in recs[C.Arm(tier, ref, 42)] if sel(x)], [x for x in recs[cpu_arm] if sel(x)])
    clusters = sorted({x["source_cell"] for x in a})
    A = C.counts_by_cluster(a, "source_cell", clusters)[0][None]
    B = C.counts_by_cluster(b, "source_cell", clusters)[0][None]
    boot = S.paired_cluster_bootstrap(A, B, cfg["bootstrap_draws"], cfg["analysis_seed"])
    perm = S.cluster_swap_permutation(A, B, cfg["permutation_draws"], cfg["analysis_seed"] + 1)
    return {"tier": tier, "reference": ref, "comparator": f"cpu-baseline/{cpu_arm.method}", "slice": sl,
            "cluster_field": "source_cell", "seeds_used": [42], "seed_policy": "seed_42_paired", "status": "ok",
            "n_clusters": boot["n_clusters"], "items_common": cov["common"],
            "b_reference_mean": round(float(S.b_from_sums(A[0].sum(0))), 4), "b_comparator_mean": round(float(S.b_from_sums(B[0].sum(0))), 4),
            "difference_reference_minus_comparator": round(S.observed_difference(A, B), 4),
            "bootstrap_ci_lower": round(boot["lower"], 4), "bootstrap_ci_upper": round(boot["upper"], 4),
            "permutation_p_two_sided": round(perm["p_value"], 5), "holm_adjusted_p": "", "significant_after_holm_0_05": ""}


def structural_key_sensitivity(recs, cfg) -> List[Dict]:
    """Same localisation contrasts clustered by the coarser state_blind_key. Few clusters make
    intervals unstable; the cluster count is reported so that is visible."""
    rows = []
    ref = cfg["localisation_reference"]
    for t in sorted({a.tier for a in recs if a.tier != "cpu-baseline"}):
        for ctl in cfg["localisation_controls"]:
            r = contrast(recs, t, ref, ctl, "structure_novel", cfg, cluster_field="state_blind_key")
            r["family"] = "localisation_sensitivity_state_blind_key"
            rows.append(r)
    return rows


def loao_error_taxonomy(recs) -> List[Dict]:
    """Which error type rises on each held-out axis: GRAFT LOAO vs vanilla-LoRA LOAO,
    restricted to the held-out axis's items, same seed. No new training."""
    from GPU_Run.common.paths import split_loao_method
    rows = []
    by = defaultdict(dict)
    for arm, r in recs.items():
        base, axis = split_loao_method(arm.method)
        if axis:
            by[(arm.tier, axis, arm.seed)][base] = r
    for (tier, axis, seed), d in sorted(by.items()):
        for base, r in sorted(d.items()):
            held = [x for x in r if x["axis"] == axis]
            cnt = C.Counts()
            for x in held:
                cnt.add(x["gold"], x["pred"] or None, x["parse_ok"])
            m = C.metrics_from_counts(cnt.as_array())
            rows.append({"tier": tier, "held_out_axis": axis, "seed": seed, "method": base,
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
    return rows


def main(cfg: Dict) -> Dict:
    out = C.output_dir(cfg) / "paired"
    out.mkdir(parents=True, exist_ok=True)
    recs = load_all_records(cfg)
    print(f"[paired_analysis] {len(recs)} arms loaded")
    metrics = per_arm_metrics(recs)
    C.write_csv(out / "per_arm_seed_explicit_metrics.csv", metrics)
    fam = families(recs, cfg)
    C.write_csv(out / "comparison_families.csv", fam)
    sens = structural_key_sensitivity(recs, cfg)
    C.write_csv(out / "sensitivity_state_blind_key_clustering.csv", sens)
    loao = loao_error_taxonomy(recs)
    C.write_csv(out / "loao_error_taxonomy.csv", loao)
    ok = sum(1 for r in fam if r.get("status") == "ok")
    unavail = sum(1 for r in fam if r.get("status") != "ok")
    sig = sum(1 for r in fam if r.get("significant_after_holm_0_05") is True)
    print(f"[paired_analysis] contrasts: {ok} computed, {unavail} unavailable (kept visible), {sig} significant after Holm")
    return {"arms": len(recs), "contrasts_ok": ok, "contrasts_unavailable": unavail, "significant": sig}


if __name__ == "__main__":
    main(C.load_config())
