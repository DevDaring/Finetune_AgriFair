"""Compose the per-pillar tables and the composite "wise" profile from the analysis CSVs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from Submission2_Run import common as C

DISPLAY = {"frontier-grok-4-3": "Grok 4.3", "frontier-gpt-5-6-luna": "GPT-5.6 Luna",
           "frontier-nova-2-lite": "Nova 2 Lite", "frontier-kimi-k2-5": "Kimi K2.5", "frontier-glm-4-7": "GLM-4.7",
           "frontier-qwen3-next-80b": "Qwen3-Next 80B", "frontier-nemotron-nano-3-30b": "Nemotron Nano 3 30B"}


def _csv(p: Path) -> List[Dict]:
    return list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []


def _mean_over_repeats(rows: List[Dict], keys: List[str], value: str) -> Dict[tuple, float]:
    acc = defaultdict(list)
    for r in rows:
        try:
            acc[tuple(r[k] for k in keys)].append(float(r[value]))
        except (KeyError, ValueError):
            pass
    return {k: float(np.nanmean(v)) for k, v in acc.items()}


def composite(an: Path, cfg: Dict) -> List[Dict]:
    """One row per model, method=plain, English: the three pillars side by side. No single score."""
    e1 = _mean_over_repeats(_csv(an / "E1_facts_summary.csv"), ["tier", "method", "language"], "harmonic_b")
    e2 = _mean_over_repeats(_csv(an / "E2_evidence_summary.csv"), ["tier", "language"], "joint_evidence_following")
    e3 = _mean_over_repeats(_csv(an / "E3_advice_summary.csv"), ["tier", "method", "language"], "exact_duplicate_rate")
    cost = {(r["tier"], r["method"], r["language"], r["set"]): r for r in _csv(an / "E5_E6_cost_energy.csv")}
    stab = {(r["tier"], r["kind"]): float(r["answer_change_rate"]) for r in _csv(an / "E7_stability.csv")}
    hs = _csv(an.parent / "human_study" / "human_rubric_outcomes.csv")
    hsm = {(r["tier"], r["language"]): r for r in hs}
    rows = []
    for tier in cfg["models"]:
        adv = cost.get((tier, "plain", "en", "advice"), {})
        rows.append({"model": DISPLAY.get(tier, tier),
                     "equity_balanced_awareness_en": e1.get((tier, "plain", "en")),
                     "equity_balanced_awareness_hi": e1.get((tier, "plain", "hi")),
                     "equity_balanced_awareness_bn": e1.get((tier, "plain", "bn")),
                     "evidence_following_en": e2.get((tier, "en")),
                     "advice_duplicate_rate_en": e3.get((tier, "plain", "en")),
                     "human_unsupported_change_en": hsm.get((tier, "en"), {}).get("unsupported_change_rate"),
                     "human_supported_rate_en": hsm.get((tier, "en"), {}).get("supported_rate"),
                     "economic_inr_per_1000_advice": adv.get("inr_per_1000"),
                     "economic_latency_p50_s": adv.get("latency_p50_s"),
                     "hindi_token_ratio": cost.get((tier, "plain", "hi", "advice"), {}).get("token_ratio_to_english"),
                     "environment_wh_per_advice_low": adv.get("wh_per_prompt_low"),
                     "environment_wh_per_advice_high": adv.get("wh_per_prompt_high"),
                     "stability_change_across_repeats": stab.get((tier, "across_repeats")),
                     "stability_change_across_routes": stab.get((tier, "across_routes"))})
    return rows


def method_table(an: Path) -> List[Dict]:
    """E1 by method: does any published prompting recipe beat plain on balanced awareness?"""
    e1 = _mean_over_repeats(_csv(an / "E1_facts_summary.csv"), ["tier", "method", "language"], "harmonic_b")
    con = {(r["tier"], r["method"]): r for r in _csv(an / "E1_method_contrasts.csv")}
    rows = []
    for (tier, method, lang), b in sorted(e1.items()):
        if lang != "en":
            continue
        c = con.get((tier, method), {})
        rows.append({"model": DISPLAY.get(tier, tier), "method": method, "balanced_awareness": round(b, 4),
                     "difference_vs_plain": c.get("difference_B"), "ci_lower": c.get("ci_lower"), "ci_upper": c.get("ci_upper"),
                     "p_holm": c.get("p_holm")})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.CODES_ROOT / (cfg["output_directory"] + ("_SMOKE" if a.smoke else ""))
    an = out / "analysis"; rp = out / "report"; rp.mkdir(exist_ok=True)
    C.write_csv(rp / "table_composite_profile.csv", composite(an, cfg))
    C.write_csv(rp / "table_methods_vs_plain.csv", method_table(an))
    print(f"report tables -> {rp}")


if __name__ == "__main__":
    main()
