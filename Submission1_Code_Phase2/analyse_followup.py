"""Analysis for R1 and R2, on the units the design actually has.

    python -m Submission1_Code_Phase2.analyse_followup [--stage main]

Every interval resamples **source comparisons**, never individual prompts: two wordings of one
comparison are repeated measurements of one fact, not two facts. Coarser parent-table and state
resamplings are reported beside the primary one with their real cluster counts, so a reader can
see when a sensitivity rests on two groups.

R1 primary endpoint: within-system accuracy difference between the original-family wording and
the new-family wording on the same comparisons.
R2 primary endpoint: the fraction of source bundles for which **all three** numerical relations
are answered correctly, reported per wording. A system that always answers the same letter gets
one third of individual items right and zero bundles all-correct; that baseline is printed.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from Next_Run import stats as S
from Submission1_Code_Phase2 import common as C


def load_predictions(cfg: Dict, stage: str) -> List[Dict]:
    """All generations for the stage, pilot included.

    run_inference excludes from the main run whatever the pilot already answered, so the units the
    pilot happened to draw exist ONLY in pilot_predictions.jsonl. Reading the main file alone
    analysed 30 of 34 R1 comparisons, 44 of 48 R2 bundles and 8 of 12 R2 diagnostics -- the data
    met the prespecified design, the analysis silently did not. The two files come from one code
    path with the same seed, ceiling and attention, so they pool directly. A generation is
    identified by (prompt_id, system): one prompt is answered by every system.
    """
    pred = C.CODES_ROOT / cfg["output_directory"] / "predictions"
    p = pred / f"{stage}_predictions.jsonl"
    if not p.exists():
        raise SystemExit(f"no predictions at {p}; run inference first")
    rows, seen = [], set()
    for src in (p, pred / "pilot_predictions.jsonl") if stage == "main" else (p,):
        if not src.exists():
            continue
        for r in C.read_jsonl(src):
            k = (r.get("prompt_id"), r.get("system"))
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def _paired_cluster_ci(values_a: Dict[str, float], values_b: Dict[str, float], draws: int, seed: int) -> Dict:
    """Bootstrap the mean paired difference over shared cluster keys."""
    keys = sorted(set(values_a) & set(values_b))
    if len(keys) < 2:
        return {"n_clusters": len(keys), "difference": float("nan"), "lower": float("nan"), "upper": float("nan")}
    d = np.array([values_a[k] - values_b[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    boots = d[idx].mean(axis=1)
    return {"n_clusters": len(keys), "difference": round(float(d.mean()), 4),
            "lower": round(float(np.percentile(boots, 2.5)), 4), "upper": round(float(np.percentile(boots, 97.5)), 4)}


# ------------------------------------------------------------------ R1

def r1_analysis(cfg: Dict, rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    r1 = [r for r in rows if r["study"] == "r1_fresh"]
    if not r1:
        return [], []
    per_system = []
    for sysid in sorted({r["system"] for r in r1}):
        sub = [r for r in r1 if r["system"] == sysid]
        acc = {}
        for wording in sorted({r["wording"] for r in sub}):
            w = [r for r in sub if r["wording"] == wording]
            c = C.Counts()
            for r in w:
                gold = "c" if r["condition"] == "equal" else "a"          # per-item correctness is what matters here
                c.add(gold, gold if r["correct"] else "b", r["parse_ok"])
            acc[wording] = {"n": len(w), "accuracy": round(sum(r["correct"] for r in w) / max(1, len(w)), 4),
                            "invalid": round(sum(not r["parse_ok"] for r in w) / max(1, len(w)), 4)}
        by_cmp = collections.defaultdict(dict)
        for r in sub:
            by_cmp[r["comparison_id"]][r["wording"]] = float(r["correct"])
        a = {k: v["original_family"] for k, v in by_cmp.items() if "original_family" in v}
        b = {k: v["new_family"] for k, v in by_cmp.items() if "new_family" in v}
        ci = _paired_cluster_ci(a, b, cfg["bootstrap_draws"], cfg["analysis_seed"])
        row = {"system": sysid, **{f"{w}_{k}": v for w, d in acc.items() for k, v in d.items()},
               "original_minus_new_wording": ci["difference"], "ci_lower": ci["lower"], "ci_upper": ci["upper"],
               "n_comparisons": ci["n_clusters"]}
        for unit in ("parent_table", "state"):
            g_a = collections.defaultdict(list); g_b = collections.defaultdict(list)
            for r in sub:
                (g_a if r["wording"] == "original_family" else g_b)[r[unit]].append(float(r["correct"]))
            s_ci = _paired_cluster_ci({k: float(np.mean(v)) for k, v in g_a.items()},
                                      {k: float(np.mean(v)) for k, v in g_b.items()},
                                      cfg["bootstrap_draws"], cfg["analysis_seed"])
            row[f"sensitivity_{unit}_difference"] = s_ci["difference"]
            row[f"sensitivity_{unit}_n_groups"] = s_ci["n_clusters"]
        per_system.append(row)
    by_axis = []
    for sysid in sorted({r["system"] for r in r1}):
        for axis in sorted({r["axis"] for r in r1}):
            sub = [r for r in r1 if r["system"] == sysid and r["axis"] == axis]
            if sub:
                by_axis.append({"system": sysid, "axis": axis, "n": len(sub),
                                "accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4)})
    return per_system, by_axis


# ------------------------------------------------------------------ R2

def r2_analysis(cfg: Dict, rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    main = [r for r in rows if r["study"] == "r2_main"]
    diag = [r for r in rows if r["study"] == "r2_diagnostic"]
    per_system: List[Dict] = []
    if main:
        for sysid in sorted({r["system"] for r in main}):
            sub = [r for r in main if r["system"] == sysid]
            row = {"system": sysid, "n_outputs": len(sub),
                   "overall_accuracy": round(sum(r["correct"] for r in sub) / len(sub), 4),
                   "invalid_rate": round(sum(not r["parse_ok"] for r in sub) / len(sub), 4)}
            for rel in sorted({r["relation"] for r in sub}):
                s = [r for r in sub if r["relation"] == rel]
                row[f"accuracy_{rel}"] = round(sum(r["correct"] for r in s) / len(s), 4)
            picked = collections.Counter(r.get("picked_choice") for r in sub)
            top = picked.most_common(1)[0] if picked else ("", 0)
            row["most_frequent_answer_share"] = round(top[1] / len(sub), 4)
            for wording in sorted({r["wording"] for r in sub}):
                w = [r for r in sub if r["wording"] == wording]
                by_bundle = collections.defaultdict(list)
                for r in w:
                    by_bundle[r["bundle_id"]].append(bool(r["correct"]))
                complete = {k: v for k, v in by_bundle.items() if len(v) == 3}
                row[f"all_three_relations_correct_{wording}"] = round(
                    sum(all(v) for v in complete.values()) / max(1, len(complete)), 4)
                row[f"bundles_{wording}"] = len(complete)
            per_system.append(row)
        # GRAFT minus frozen, paired on source bundles, within each wording
        contrasts = []
        for tier in cfg["systems"]["tiers"]:
            g = f"{tier}|graft_proposed|seed{cfg['systems']['seed']}"
            f_ = f"{tier}|frozen_base|seed{cfg['systems']['seed']}"
            for wording in sorted({r["wording"] for r in main}):
                def bundle_scores(sysid):
                    d = collections.defaultdict(list)
                    for r in main:
                        if r["system"] == sysid and r["wording"] == wording:
                            d[r["bundle_id"]].append(bool(r["correct"]))
                    return {k: float(all(v)) for k, v in d.items() if len(v) == 3}
                ci = _paired_cluster_ci(bundle_scores(g), bundle_scores(f_), cfg["bootstrap_draws"], cfg["analysis_seed"])
                contrasts.append({"tier": tier, "wording": wording, "endpoint": "all_three_relations_correct",
                                  "graft_minus_frozen": ci["difference"], "ci_lower": ci["lower"],
                                  "ci_upper": ci["upper"], "n_bundles": ci["n_clusters"]})
    else:
        contrasts = []
    diag_rows: List[Dict] = []
    for sysid in sorted({r["system"] for r in diag}):
        for variant in sorted({r["variant"] for r in diag}):
            s = [r for r in diag if r["system"] == sysid and r["variant"] == variant]
            if s:
                diag_rows.append({"system": sysid, "variant": variant, "n": len(s),
                                  "accuracy": round(sum(r["correct"] for r in s) / len(s), 4),
                                  "chose_insufficient_evidence": round(
                                      sum(1 for r in s if (r.get("picked_choice") or "").startswith("The table does not")) / len(s), 4),
                                  "invalid_rate": round(sum(not r["parse_ok"] for r in s) / len(s), 4)})
    return per_system, contrasts, diag_rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="main"); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "tables")
    rows = load_predictions(cfg, a.stage)
    r1_sys, r1_axis = r1_analysis(cfg, rows)
    if r1_sys:
        C.write_csv(out / "r1_template_transfer_results.csv", r1_sys)
        C.write_csv(out / "r1_by_axis.csv", r1_axis)
    r2_sys, r2_con, r2_diag = r2_analysis(cfg, rows)
    if r2_sys:
        C.write_csv(out / "r2_evidence_results.csv", r2_sys)
        C.write_csv(out / "r2_graft_minus_frozen.csv", r2_con)
    if r2_diag:
        C.write_csv(out / "r2_diagnostics.csv", r2_diag)
    C.write_json(out / "followup_summary.json", {
        "stage": a.stage, "predictions": len(rows),
        "studies": dict(collections.Counter(r["study"] for r in rows)),
        "constant_label_baseline": {"r2_item_accuracy": round(1 / 3, 4), "r2_all_three_correct": 0.0},
        "cluster_units": {"primary": "source comparison / source bundle", "sensitivities": ["parent_table", "state"]},
        "interpretation_note": ("A wording drop in R1 indicates template sensitivity. R2 separates response to numbers "
                                "from response to framing, because every table is hypothetical and only the numbers move. "
                                "Neither endpoint establishes comprehension, and neither requires GRAFT to win.")})
    print(f"[analyse_followup] {len(rows)} predictions -> {out}")
    for r in r1_sys:
        print(f"  R1 {r['system']}: original {r.get('original_family_accuracy')} vs new {r.get('new_family_accuracy')} "
              f"(diff {r['original_minus_new_wording']} [{r['ci_lower']}, {r['ci_upper']}], {r['n_comparisons']} comparisons)")
    for r in r2_sys:
        print(f"  R2 {r['system']}: overall {r['overall_accuracy']}, all-three "
              f"{r.get('all_three_relations_correct_context_rich')} / {r.get('all_three_relations_correct_plain_new')}")


if __name__ == "__main__":
    main()
