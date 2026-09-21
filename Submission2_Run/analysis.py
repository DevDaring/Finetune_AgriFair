"""Analysis for E1-E7. Reads only the prediction files; writes tables under results/analysis/.

Equity:      balanced awareness B + error taxonomy per model x method x language (Next_Run.common)
Evidence:    joint verified+hypothetical correctness, anonymised consistency
Advice:      TF-IDF cosine, Jaccard, exact-duplicate rate, words (per model x method x language)
Economic:    cost per query and per 1,000 queries (USD, INR), latency p50/p95, Indic/English token ratio
Environment: Wh per query, low/high proxy
Stability:   answer-change rate across repeats and across forced routes
Contrasts:   paired source-cluster bootstrap + permutation between methods, Holm within family
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from Next_Run import stats as S
from Submission2_Run import common as C


def _load(out: Path, prefix: str) -> List[Dict]:
    rows = []
    for p in sorted(out.glob(f"{prefix}_r*.jsonl")):
        rows += C.read_jsonl(p)
    return rows


def _pct(v, q): return float(np.percentile(v, q)) if len(v) else float("nan")


# ------------------------------------------------------------------ equity (E1)
def facts_summary(rows: List[Dict]) -> List[Dict]:
    out = []
    groups = defaultdict(list)
    for r in rows:
        if r["condition"] == "no_evidence":
            groups[(r["tier"], r["method"], r["language"], r["repeat"])].append(r)
    for (tier, method, lang, rep), grp in sorted(groups.items()):
        c = C.Counts()
        for r in grp:
            c.add(r["gold"], r["pred"], r["parse_ok"])
        out.append({"tier": tier, "method": method, "language": lang, "repeat": rep, **C.metrics_from_counts(c.as_array())})
    return out


# ------------------------------------------------------------------ evidence (E2)
def evidence_summary(rows: List[Dict]) -> List[Dict]:
    by = defaultdict(dict)
    for r in rows:
        if r["method"] == "evidence" or r["condition"] == "no_evidence":
            by[(r["tier"], r["language"], r["repeat"], r["item_id"])][r["condition"]] = r
    acc = defaultdict(lambda: defaultdict(int))
    for (tier, lang, rep, item), conds in by.items():
        k = (tier, lang, rep); acc[k]["items"] += 1
        ok = {c: (conds.get(c, {}).get("pred") == conds.get(c, {}).get("gold")) for c in conds}
        for c, v in ok.items():
            acc[k][f"correct_{c}"] += v
        acc[k]["joint"] += ok.get("verified_evidence", False) and ok.get("hypothetical_evidence", False)
        acc[k]["anon_consistent"] += ok.get("anonymised_evidence", False) and ok.get("verified_evidence", False)
        acc[k]["hypothetical_equal_answer"] += conds.get("hypothetical_evidence", {}).get("pred") == "c"
    out = []
    for (tier, lang, rep), d in sorted(acc.items()):
        n = d["items"]
        out.append({"tier": tier, "language": lang, "repeat": rep, "items": n,
                    "no_evidence_acc": d["correct_no_evidence"] / n, "verified_acc": d["correct_verified_evidence"] / n,
                    "hypothetical_acc": d["correct_hypothetical_evidence"] / n, "anonymised_acc": d["correct_anonymised_evidence"] / n,
                    "joint_evidence_following": d["joint"] / n, "anonymised_consistent_and_correct": d["anon_consistent"] / n,
                    "hypothetical_equal_answer_rate": d["hypothetical_equal_answer"] / n})
    return out


# ------------------------------------------------------------------ advice (E3)
def _tfidf_cos(a: str, b: str) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not a.strip() or not b.strip():
        return float("nan")
    X = TfidfVectorizer().fit_transform([a, b]); sim = (X[0] @ X[1].T).toarray()[0, 0]
    return float(1 - sim)


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return 1 - len(wa & wb) / len(wa | wb) if (wa | wb) else float("nan")


def advice_pairs(rows: List[Dict]) -> List[Dict]:
    by = defaultdict(dict)
    for r in rows:
        by[(r["tier"], r["method"], r["repeat"], r["pair_id"])][r["side"]] = r
    out = []
    for (tier, method, rep, pid), sides in by.items():
        if "A" not in sides or "B" not in sides:
            continue
        a, b = sides["A"]["raw_text"], sides["B"]["raw_text"]
        out.append({"tier": tier, "method": method, "repeat": rep, "pair_id": pid, "query_id": sides["A"]["query_id"],
                    "language": sides["A"]["language"], "toggle_axis": sides["A"]["toggle_axis"],
                    "tfidf_cosine_distance": _tfidf_cos(a, b), "jaccard_distance": _jaccard(a, b),
                    "exact_duplicate": a.strip() == b.strip(), "words_A": len(a.split()), "words_B": len(b.split()),
                    "empty_or_refusal": any(not t.strip() or re.search(r"\b(i can(?:no|')t|i cannot|as an ai)\b", t.lower()) for t in (a, b))})
    return out


def advice_summary(pairs: List[Dict]) -> List[Dict]:
    groups = defaultdict(list)
    for p in pairs:
        groups[(p["tier"], p["method"], p["language"], p["repeat"])].append(p)
    out = []
    for (tier, method, lang, rep), g in sorted(groups.items()):
        d = np.array([p["tfidf_cosine_distance"] for p in g], dtype=float)
        out.append({"tier": tier, "method": method, "language": lang, "repeat": rep, "pairs": len(g),
                    "tfidf_distance_mean": float(np.nanmean(d)), "jaccard_mean": float(np.nanmean([p["jaccard_distance"] for p in g])),
                    "exact_duplicate_rate": float(np.mean([p["exact_duplicate"] for p in g])),
                    "words_mean": float(np.mean([(p["words_A"] + p["words_B"]) / 2 for p in g])),
                    "empty_or_refusal_rate": float(np.mean([p["empty_or_refusal"] for p in g]))})
    return out


# ------------------------------------------------------------------ economic + environmental (E5, E6)
def cost_summary(cfg: Dict, rows: List[Dict], label: str) -> List[Dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tier"], r["method"], r.get("language", "en"))].append(r)
    out = []
    for (tier, method, lang), g in sorted(groups.items()):
        toks = [r["input_tokens"] + r["output_tokens"] for r in g]; usd = [r["cost_usd"] for r in g]; lat = [r["latency_seconds"] for r in g]
        e = C.energy_wh(cfg, tier, int(np.mean(toks))) if toks else {"low": float("nan"), "high": float("nan")}
        out.append({"set": label, "tier": tier, "method": method, "language": lang, "prompts": len(g),
                    "tokens_per_prompt": float(np.mean(toks)), "usd_per_prompt": float(np.mean(usd)),
                    "inr_per_1000": C.cost_inr(cfg, float(np.mean(usd)) * 1000), "latency_p50_s": _pct(lat, 50), "latency_p95_s": _pct(lat, 95),
                    "wh_per_prompt_low": e["low"], "wh_per_prompt_high": e["high"]})
    en = {(r["tier"], r["method"]): r["tokens_per_prompt"] for r in out if r["language"] == "en"}
    for r in out:
        base = en.get((r["tier"], r["method"]))
        r["token_ratio_to_english"] = r["tokens_per_prompt"] / base if base else float("nan")
    return out


# ------------------------------------------------------------------ stability (E7)
def stability_summary(facts_rows: List[Dict], route_rows: List[Dict]) -> List[Dict]:
    out = []
    by = defaultdict(dict)
    for r in facts_rows:
        if r["condition"] == "no_evidence" and r["method"] == "plain":
            by[(r["tier"], r["language"], r["item_id"])][r["repeat"]] = r["pred"]
    per_tier = defaultdict(lambda: [0, 0])
    for (tier, lang, item), reps in by.items():
        if len(reps) > 1:
            per_tier[tier][1] += 1; per_tier[tier][0] += len(set(reps.values())) > 1
    for tier, (chg, n) in sorted(per_tier.items()):
        out.append({"tier": tier, "kind": "across_repeats", "items": n, "answer_change_rate": chg / n if n else float("nan")})
    byr = defaultdict(dict)
    for r in route_rows:
        byr[(r["tier"], r["item_id"], r["repeat"])][r["forced_route"]] = r["pred"]
    per = defaultdict(lambda: [0, 0])
    for (tier, item, rep), routes in byr.items():
        if len(routes) > 1:
            per[tier][1] += 1; per[tier][0] += len(set(routes.values())) > 1
    for tier, (chg, n) in sorted(per.items()):
        out.append({"tier": tier, "kind": "across_routes", "items": n, "answer_change_rate": chg / n if n else float("nan")})
    return out


# ------------------------------------------------------------------ contrasts between methods (E1 primary endpoint)
def method_contrasts(cfg: Dict, rows: List[Dict], reference: str = "plain") -> List[Dict]:
    recs = defaultdict(list)
    for r in rows:
        if r["condition"] == "no_evidence" and r["language"] == "en":
            recs[(r["tier"], r["method"])].append({**r, "item_id": r["item_id"]})
    out = []
    for tier in sorted({t for t, _ in recs}):
        fam = []
        for method in sorted({m for t, m in recs if t == tier and m != reference}):
            a, b = recs[(tier, method)], recs[(tier, reference)]
            clusters = sorted({r["source_cell"] for r in a} | {r["source_cell"] for r in b})
            reps = sorted({r["repeat"] for r in a})
            ca = np.stack([C.counts_by_cluster([r for r in a if r["repeat"] == k], "source_cell", clusters)[0] for k in reps])
            cb = np.stack([C.counts_by_cluster([r for r in b if r["repeat"] == k], "source_cell", clusters)[0] for k in reps])
            bs = S.paired_cluster_bootstrap(ca, cb, cfg["bootstrap_draws"], cfg["analysis_seed"])
            pm = S.cluster_swap_permutation(ca, cb, cfg["permutation_draws"], cfg["analysis_seed"])
            fam.append({"tier": tier, "method": method, "reference": reference, "difference_B": S.observed_difference(ca, cb),
                        "ci_lower": bs["lower"], "ci_upper": bs["upper"], "n_clusters": bs["n_clusters"], "p_value": pm["p_value"]})
        for r, p in zip(fam, S.holm([r["p_value"] for r in fam])):
            r["p_holm"] = p; r["significant_after_holm_0_05"] = p < 0.05
        out += fam
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.CODES_ROOT / (cfg["output_directory"] + ("_SMOKE" if a.smoke else "")); an = out / "analysis"; an.mkdir(exist_ok=True)
    facts, advice, ref, routes = (_load(out, p) for p in ("facts_predictions", "advice_predictions", "reference_answers", "route_panel"))
    C.write_csv(an / "E1_facts_summary.csv", facts_summary(facts))
    C.write_csv(an / "E2_evidence_summary.csv", evidence_summary(facts))
    pairs = advice_pairs(advice); C.write_csv(an / "E3_advice_pairs.csv", pairs); C.write_csv(an / "E3_advice_summary.csv", advice_summary(pairs))
    C.write_csv(an / "E5_E6_cost_energy.csv", cost_summary(cfg, facts, "facts") + cost_summary(cfg, advice, "advice") + cost_summary(cfg, ref, "reference"))
    C.write_csv(an / "E7_stability.csv", stability_summary(facts, routes))
    if facts:
        C.write_csv(an / "E1_method_contrasts.csv", method_contrasts(cfg, facts))
    print(f"analysis written -> {an}")


if __name__ == "__main__":
    main()
