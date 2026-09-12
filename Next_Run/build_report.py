"""P5 - tables, figures and a machine-readable claims-to-evidence report.

Reads only the final-audit output directory. Produces the four main-paper tables the plan
names (dataset/source audit; main benchmark results with CPU controls; transfer error
taxonomy; human advice outcomes), the optional evidence panel table, three figures, and a
claims_to_evidence.json that says for each intended claim which artifact supports it and
whether that artifact exists yet. A claim whose artifact is missing is marked
"unsupported" - the paper must not make it.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from Next_Run import common as C


def _csv(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _md_table(rows: List[Dict], cols: List[str]) -> str:
    if not rows:
        return "_no rows_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def table_main_results(root: Path) -> List[Dict]:
    """Novel-test B per (tier, method) with seed count, plus the CPU controls, in one table."""
    m = _csv(root / "paired" / "per_arm_seed_explicit_metrics.csv")
    by = defaultdict(list)
    for r in m:
        if r["slice"] == "structure_novel":
            by[(r["tier"], r["method"])].append(r)
    rows = []
    for (tier, method), rs in sorted(by.items()):
        bs = [float(r["harmonic_b"]) for r in rs if r["harmonic_b"] not in ("", None)]
        rows.append({"tier": tier, "method": method, "n_seeds": len(rs), "seeds": ",".join(r["seed"] for r in rs),
                     "novel_B_mean": round(sum(bs) / len(bs), 4) if bs else "undefined",
                     "novel_B_by_seed": ";".join(f"{b:.4f}" for b in bs),
                     "erasure_on_diff": rs[0]["erasure_rate_on_diff"], "wrong_group_on_diff": rs[0]["wrong_group_rate_on_diff"],
                     "fabrication_on_equal": rs[0]["fabrication_rate_on_equal"], "invalid": rs[0]["invalid_rate"]})
    return rows


def figures(root: Path, out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []
    fam = [r for r in _csv(root / "paired" / "comparison_families.csv") if r.get("status") == "ok" and r.get("slice") == "structure_novel"]
    if fam:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(fam))))
        labels = [f"{r['tier']} · {r['comparator']} ({r['family'][:4]})" for r in fam]
        d = [float(r["difference_reference_minus_comparator"]) for r in fam]
        lo = [float(r["bootstrap_ci_lower"]) for r in fam]; hi = [float(r["bootstrap_ci_upper"]) for r in fam]
        y = range(len(fam))
        ax.errorbar(d, y, xerr=[[a - b for a, b in zip(d, lo)], [b - a for a, b in zip(d, hi)]], fmt="o", capsize=3)
        ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("GRAFT − comparator, novel-test B (paired source-cluster bootstrap 95% CI)")
        ax.set_title("Paired contrasts with dependence-aware uncertainty"); fig.tight_layout()
        p = out / "fig_paired_contrasts.png"; fig.savefig(p, dpi=160); plt.close(fig); made.append(p.name)
    adv = _json(root / "advice" / "advice_paired_summary.json") or []
    adv = [r for r in adv if r["subset"] == "complete" and r["toggle_axis"] == "all" and r["comparator"] == "graft_proposed"]
    if adv:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        tiers = [r["tier"] for r in adv]
        ax[0].bar(tiers, [r["frozen_words_mean"] for r in adv], width=0.4, label="frozen", align="edge")
        ax[0].bar(tiers, [r["comparator_words_mean"] for r in adv], width=-0.4, label="GRAFT", align="edge")
        ax[0].set_ylabel("mean words per answer"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
        ax[1].bar(tiers, [r["frozen_tfidf_distance"].get("mean", 0) for r in adv], width=0.4, label="frozen", align="edge")
        ax[1].bar(tiers, [r["comparator_tfidf_distance"].get("mean", 0) for r in adv], width=-0.4, label="GRAFT", align="edge")
        ax[1].set_ylabel("TF-IDF cosine distance (not a semantic embedding)"); ax[1].legend(); ax[1].tick_params(axis="x", rotation=30)
        fig.suptitle("Advice consistency alongside output length — association only, not causation"); fig.tight_layout()
        p = out / "fig_advice_length_vs_distance.png"; fig.savefig(p, dpi=160); plt.close(fig); made.append(p.name)
    loao = _csv(root / "paired" / "loao_error_taxonomy.csv")
    if loao:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        keys = sorted({(r["tier"], r["held_out_axis"], r["method"]) for r in loao})
        x = range(len(keys)); bottoms = [0.0] * len(keys)
        for et, col in (("erasure_rate_on_diff", "#d95f02"), ("wrong_group_rate_on_diff", "#7570b3"), ("fabrication_rate_on_equal", "#1b9e77"), ("invalid_rate", "#666666")):
            vals = []
            for k in keys:
                r = next(r for r in loao if (r["tier"], r["held_out_axis"], r["method"]) == k)
                vals.append(float(r[et]) if r[et] not in ("", None) else 0.0)
            ax.bar(list(x), vals, bottom=bottoms, label=et.replace("_rate", "").replace("_", " "), color=col)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(list(x)); ax.set_xticklabels([f"{t[:5]}·{a[:6]}·{m[:10]}" for t, a, m in keys], rotation=75, fontsize=6)
        ax.set_ylabel("rate"); ax.set_title("Leave-one-axis-out transfer: which error type rises"); ax.legend(fontsize=7); fig.tight_layout()
        p = out / "fig_loao_error_taxonomy.png"; fig.savefig(p, dpi=160); plt.close(fig); made.append(p.name)
    return made


def claims_to_evidence(root: Path) -> List[Dict]:
    def exists(*parts): return (root.joinpath(*parts)).exists()
    src = _json(root / "sources" / "verify_sources_summary.json") or {}
    hs = _json(root / "advice" / "human_study" / "ratings_analysis.json")
    ep = _json(root / "evidence_panel" / "evidence_panel_run_manifest.json")
    claims = [
        {"claim": "Frozen test set is the preregistered one", "artifact": "inventory/manifest.json:frozen_test_hash_matches_config",
         "status": "supported" if (_json(root / "inventory" / "manifest.json") or {}).get("frozen_test_hash_matches_config") else "unsupported"},
        {"claim": "Census-derived labels were independently verified", "artifact": "sources/discrepancy_report.csv",
         "status": "supported" if src.get("recompute", {}).get("status") == "ok" and src["recompute"].get("disagree", 1) == 0 else
                   ("partially_supported" if src.get("recompute", {}).get("status") == "ok" else "unsupported")},
        {"claim": "Simple train-only baselines reach a stated fraction of neural novel-test B", "artifact": "cpu_baselines/cpu_baseline_summary.csv",
         "status": "supported" if exists("cpu_baselines", "cpu_baseline_summary.csv") else "unsupported"},
        {"claim": "GRAFT vs placement controls, with dependence-aware uncertainty (localisation family)", "artifact": "paired/comparison_families.csv",
         "status": "supported" if exists("paired", "comparison_families.csv") else "unsupported"},
        {"claim": "Transfer failures are characterised by error type, not only accuracy", "artifact": "paired/loao_error_taxonomy.csv",
         "status": "supported" if exists("paired", "loao_error_taxonomy.csv") else "unsupported"},
        {"claim": "Advice drift reduction coincides with shorter, more duplicated answers (association)", "artifact": "advice/advice_paired_summary.json",
         "status": "supported" if exists("advice", "advice_paired_summary.json") else "unsupported"},
        {"claim": "Blinded expert assessment links automated consistency to correctness/usefulness", "artifact": "advice/human_study/ratings_analysis.json",
         "status": "supported" if hs and hs.get("paired_endpoints") else "unsupported (study not completed - narrow the claim)"},
        {"claim": "Evidence-sensitivity panel results", "artifact": "evidence_panel/evidence_panel_scores.csv",
         "status": "supported" if ep and exists("evidence_panel", "evidence_panel_scores.csv") else "unsupported (P4 not run)"},
        {"claim": "GRAFT is superior to ordinary adapters / localisation is necessary", "artifact": "—",
         "status": "do_not_claim (preregistered hypothesis unsupported; report as such)"},
    ]
    return claims


def main(cfg: Dict) -> Dict:
    root = C.output_dir(cfg)
    out = root / "report"; out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    t_main = table_main_results(root)
    C.write_csv(out / "table_main_results_novel_B.csv", t_main)
    t_src = _csv(root / "sources" / "audit_sample_frozen.csv")
    t_loao = _csv(root / "paired" / "loao_error_taxonomy.csv")
    t_cpu = _csv(root / "cpu_baselines" / "cpu_baseline_summary.csv")
    figs = figures(root, out / "figures")
    claims = claims_to_evidence(root)
    C.write_json(out / "claims_to_evidence.json", claims)
    md = ["# Final audit report (auto-generated; numbers are from artifacts, not typed in)\n",
          "## Table 1 - source audit sample (frozen before inspecting model errors)\n", _md_table(t_src[:12], ["source_cell", "axis", "frozen_condition", "audit_stage"]),
          f"\n_{len(t_src)} comparisons in the frozen sample; see sources/discrepancy_report.csv once the ledger is filled._\n",
          "\n## Table 2 - main results, novel-test B, seed-explicit, with CPU controls\n", _md_table(t_main, ["tier", "method", "n_seeds", "novel_B_mean", "erasure_on_diff", "wrong_group_on_diff", "fabrication_on_equal", "invalid"]),
          "\n### CPU baselines\n", _md_table([r for r in t_cpu if r["slice"] == "structure_novel"], ["baseline", "slice", "harmonic_b", "overall_accuracy"]),
          "\n## Table 3 - transfer error taxonomy (LOAO)\n", _md_table(t_loao, ["tier", "held_out_axis", "method", "harmonic_b", "erasure_rate_on_diff", "wrong_group_rate_on_diff", "fabrication_rate_on_equal", "invalid_rate"]),
          "\n## Table 4 - human advice outcomes\n", "_populated from advice/human_study/ratings_analysis.json when the study is complete_\n",
          "\n## Figures\n" + "".join(f"- {f}\n" for f in figs),
          "\n## Claims to evidence\n", _md_table(claims, ["claim", "artifact", "status"])]
    (out / "REPORT.md").write_text("".join(md), encoding="utf-8")
    print(f"[build_report] {len(t_main)} main rows, {len(figs)} figures, {sum(1 for c in claims if c['status'].startswith('supported'))}/{len(claims)} claims supported -> {out}")
    return {"figures": figs, "claims": claims}


if __name__ == "__main__":
    main(C.load_config())
