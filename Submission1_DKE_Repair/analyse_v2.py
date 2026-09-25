"""Analyse the version-2 generations: E1 (repaired wording), E3 (clean control), E4 (neutral).

    python -m Submission1_DKE_Repair.analyse_v2

Every contrast is prespecified in the plan and reported for all four systems and all retained
items, whichever direction the result runs. Items are never excluded on the basis of a model
mistake -- only on source or rendering validity, and version 2 has no such exclusions.
"""
from __future__ import annotations

import argparse
import collections
import json
from typing import Dict, List

import numpy as np

from Submission1_Code_Phase2 import common as C

OUT_DIR = "results_submission1_dke_repair_v2"


def load() -> List[Dict]:
    return list(C.read_jsonl(C.CODES_ROOT / OUT_DIR / "v2_predictions.jsonl"))


def _paired_cluster_ci(a: Dict[str, float], b: Dict[str, float], draws: int, seed: int) -> Dict:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return {"n_clusters": len(keys), "difference": None, "lower": None, "upper": None}
    d = np.array([a[k] - b[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    stats = d[idx].mean(axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"n_clusters": len(keys), "difference": round(float(d.mean()), 4),
            "lower": round(float(lo), 4), "upper": round(float(hi), 4)}


# ------------------------------------------------------------------ E1
def e1(rows: List[Dict], draws: int, seed: int) -> Dict:
    sel = [r for r in rows if r["study"] == "r1_corrected"]
    out, by_cond = [], []
    for sysname in sorted({r["system"] for r in sel}):
        s = [r for r in sel if r["system"] == sysname]
        acc = {}
        for w in ("wording_a", "wording_b"):
            v = [r for r in s if r["wording"] == w]
            acc[w] = sum(bool(r["correct"]) for r in v) / len(v) if v else None
        a = {r["comparison_id"]: float(bool(r["correct"])) for r in s if r["wording"] == "wording_a"}
        b = {r["comparison_id"]: float(bool(r["correct"])) for r in s if r["wording"] == "wording_b"}
        ci = _paired_cluster_ci(a, b, draws, seed)
        out.append({"system": sysname, "wording_a_accuracy": round(acc["wording_a"], 4),
                    "wording_b_accuracy": round(acc["wording_b"], 4),
                    "a_minus_b": ci["difference"], "ci_lower": ci["lower"], "ci_upper": ci["upper"],
                    "n_comparisons": ci["n_clusters"],
                    "invalid_rate": round(sum(1 for r in s if not r.get("parse_ok", True)) / len(s), 4)})
        for cond in ("diff", "equal"):
            v = [r for r in s if r["condition"] == cond]
            by_cond.append({"system": sysname, "condition": cond, "n_responses": len(v),
                            "accuracy": round(sum(bool(r["correct"]) for r in v) / len(v), 4) if v else None})
    return {"per_system": out, "by_condition": by_cond}


# ------------------------------------------------------------------ E3
def e3(rows: List[Dict]) -> List[Dict]:
    """Clean control against each perturbation, on the same 12 bundles."""
    clean = {(r["system"], r["bundle_id"]): bool(r["correct"])
             for r in rows if r["study"] == "r2_diagnostic_clean"}
    old = list(C.read_jsonl(C.CODES_ROOT / "results_submission1_phase2" / "predictions" /
                            "main_predictions.jsonl"))
    old += list(C.read_jsonl(C.CODES_ROOT / "results_submission1_phase2" / "predictions" /
                             "pilot_predictions.jsonl"))
    seen, diag = set(), []
    for r in old:
        if r.get("study") != "r2_diagnostic":
            continue
        k = (r.get("prompt_id"), r.get("system"))
        if k in seen:
            continue
        seen.add(k)
        diag.append(r)
    out = []
    for sysname in sorted({r["system"] for r in rows if r["study"] == "r2_diagnostic_clean"}):
        base = [v for (s, _), v in clean.items() if s == sysname]
        base_acc = sum(base) / len(base) if base else None
        for variant in sorted({d["variant"] for d in diag}):
            v = [d for d in diag if d["system"] == sysname and d["variant"] == variant]
            paired = [(clean.get((sysname, d["bundle_id"])), bool(d["correct"])) for d in v
                      if (sysname, d["bundle_id"]) in clean]
            if not paired:
                continue
            acc = sum(b for _, b in paired) / len(paired)
            cl = sum(bool(a) for a, _ in paired) / len(paired)
            out.append({"system": sysname, "variant": variant, "n_bundles": len(paired),
                        "clean_accuracy": round(cl, 4), "variant_accuracy": round(acc, 4),
                        "variant_minus_clean": round(acc - cl, 4),
                        "clean_accuracy_all_12": round(base_acc, 4) if base_acc is not None else None})
    return out


# ------------------------------------------------------------------ E4
def e4(rows: List[Dict]) -> List[Dict]:
    """Neutral entities against the matched agricultural items, same bundles and values."""
    neutral = [r for r in rows if r["study"] == "r2_neutral"]
    old = list(C.read_jsonl(C.CODES_ROOT / "results_submission1_phase2" / "predictions" /
                            "main_predictions.jsonl"))
    old += list(C.read_jsonl(C.CODES_ROOT / "results_submission1_phase2" / "predictions" /
                             "pilot_predictions.jsonl"))
    seen, agri = set(), {}
    for r in old:
        if r.get("study") != "r2_main":
            continue
        k = (r.get("prompt_id"), r.get("system"))
        if k in seen:
            continue
        seen.add(k)
        agri[(r["system"], r["prompt_id"])] = bool(r["correct"])
    out = []
    for sysname in sorted({r["system"] for r in neutral}):
        pairs = []
        for r in neutral:
            if r["system"] != sysname:
                continue
            orig_id = r["prompt_id"].replace("v2neutral-", "")
            if (sysname, orig_id) in agri:
                pairs.append((agri[(sysname, orig_id)], bool(r["correct"])))
        if not pairs:
            continue
        a = sum(x for x, _ in pairs) / len(pairs)
        n = sum(y for _, y in pairs) / len(pairs)
        out.append({"system": sysname, "n_matched_items": len(pairs),
                    "agricultural_accuracy": round(a, 4), "neutral_accuracy": round(n, 4),
                    "neutral_minus_agricultural": round(n - a, 4)})
    return out


def main(argv=None) -> None:
    argparse.ArgumentParser().parse_args(argv)
    cfg = C.load_config()
    out = C.CODES_ROOT / OUT_DIR
    rows = load()
    draws = int(cfg.get("bootstrap_draws", 10000))
    seed = int(cfg.get("analysis_seed", 20260925))

    r_e1, r_e3, r_e4 = e1(rows, draws, seed), e3(rows), e4(rows)
    C.write_csv(out / "r1_corrected_results.csv", r_e1["per_system"])
    C.write_csv(out / "r1_corrected_by_condition.csv", r_e1["by_condition"])
    C.write_csv(out / "r2_diagnostic_clean_control_results.csv", r_e3)
    C.write_csv(out / "r2_neutral_transfer_results.csv", r_e4)
    C.write_json(out / "analysis_v2.json",
                 {"n_predictions": len(rows),
                  "by_study": dict(collections.Counter(r["study"] for r in rows)),
                  "E1": r_e1, "E3": r_e3, "E4": r_e4,
                  "note": ("Wording A versus wording B sensitivity. Neither wording reproduces an "
                           "original template family under the paper's own identifier, so this is "
                           "not a test of familiarity with a seen training template.")})
    print(json.dumps({"n": len(rows), "E1": r_e1["per_system"], "E4": r_e4}, indent=1))


if __name__ == "__main__":
    main()
