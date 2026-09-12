"""P0 - inventory: prove what we have before interpreting any of it.

Validates hashes, schemas, id uniqueness, seed coverage, checkpoint availability and
cached-output coverage, and writes a manifest plus a deviations-log scaffold. It reads
the legacy directories and never writes into them.

The frozen test hash is compared against the value recorded in config; a mismatch is a
finding to investigate, not something to overwrite.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from Next_Run import common as C

REQUIRED_ITEM_FIELDS = ["id", "category", "form", "condition", "question", "choice_a", "choice_b",
                        "choice_c", "correct_answer", "source_cell", "state_blind_key", "split"]
REQUIRED_PRED_FIELDS = ["id", "gold_canonical", "pred_canonical", "parse_ok"]


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "sklearn", "torch", "transformers", "peft"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "not installed"
    return out


def _git_snapshot() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.CODES_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not a git repository"


def check_splits(findings: List[Dict]) -> Dict[str, Dict]:
    """Schema, id uniqueness, and cross-split leakage of ids and source cells."""
    out = {}
    ids_by_split, cells_by_split = {}, {}
    for split in ("train", "validation", "test"):
        rows = C.load_split(split)
        missing = Counter()
        for r in rows:
            for f in REQUIRED_ITEM_FIELDS:
                if f not in r:
                    missing[f] += 1
            if r.get("correct_answer") not in C.CANONICAL:
                missing["correct_answer_not_canonical"] += 1
            if r.get("condition") not in ("equal", "diff"):
                missing["condition_not_equal_or_diff"] += 1
            if (r.get("condition") == "equal") != (r.get("correct_answer") == C.EQUAL_LETTER):
                missing["condition_gold_inconsistent"] += 1
        ids = [r["id"] for r in rows]
        dup = len(ids) - len(set(ids))
        ids_by_split[split] = set(ids)
        cells_by_split[split] = {r["source_cell"] for r in rows}
        bad_cells = sum(1 for r in rows if not C.parse_source_cell(r["source_cell"])["parse_ok"])
        bad_keys = sum(1 for r in rows if not C.parse_state_blind_key(r["state_blind_key"])["parse_ok"])
        out[split] = {"items": len(rows), "duplicate_ids": dup, "distinct_source_cells": len(cells_by_split[split]),
                      "distinct_state_blind_keys": len({r["state_blind_key"] for r in rows}),
                      "condition": dict(Counter(r["condition"] for r in rows)),
                      "axis": dict(Counter(r["category"] for r in rows)),
                      "test_slice": dict(Counter(r.get("test_slice", "") for r in rows)),
                      "unparseable_source_cells": bad_cells, "unparseable_state_blind_keys": bad_keys,
                      "schema_issues": dict(missing)}
        if dup:
            findings.append({"severity": "blocker", "where": split, "finding": f"{dup} duplicate item ids"})
        if missing:
            findings.append({"severity": "blocker", "where": split, "finding": f"schema issues {dict(missing)}"})
    for a, b in (("train", "test"), ("validation", "test"), ("train", "validation")):
        id_leak = len(ids_by_split[a] & ids_by_split[b])
        cell_leak = len(cells_by_split[a] & cells_by_split[b])
        out[f"{a}_vs_{b}"] = {"shared_ids": id_leak, "shared_source_cells": cell_leak}
        if id_leak:
            findings.append({"severity": "blocker", "where": f"{a}/{b}", "finding": f"{id_leak} item ids shared"})
        if cell_leak:
            findings.append({"severity": "major", "where": f"{a}/{b}", "finding": f"{cell_leak} source cells shared across splits"})
    return out


def check_predictions(findings: List[Dict]) -> Dict:
    """Coverage matrix over (tier, method, seed): rows, parse_ok, missing/unknown/duplicate ids."""
    items = C.test_items_by_id()
    files = C.prediction_files()
    cov = []
    by_tier_method = defaultdict(set)
    for arm, path in sorted(files.items(), key=lambda kv: kv[0].key):
        raw = C.read_jsonl(path)
        field_missing = Counter(f for r in raw[:5] for f in REQUIRED_PRED_FIELDS if f not in r)
        recs, issues = C.normalise_predictions(arm, path, items)
        parse_ok = sum(1 for r in recs if r["parse_ok"])
        by_tier_method[(arm.tier, arm.method)].add(arm.seed)
        cov.append({"tier": arm.tier, "method": arm.method, "seed": arm.seed, "file": path.name,
                    "raw_rows": len(raw), "normalised_records": issues["records"], "parse_ok": parse_ok,
                    "missing_ids": issues["missing_ids"], "unknown_ids": issues["unknown_ids"],
                    "duplicate_ids": issues["duplicate_ids"], "expected_scope": issues["expected_scope"],
                    "schema_issues": dict(field_missing)})
        if issues["missing_ids"] or issues["duplicate_ids"] or issues["unknown_ids"]:
            findings.append({"severity": "major", "where": path.name,
                             "finding": f"coverage issues: {issues}"})
    seeds = {f"{t}/{m}": sorted(s) for (t, m), s in by_tier_method.items()}
    return {"files": len(files), "coverage": cov, "seeds_per_arm": seeds}


def check_advice(findings: List[Dict]) -> Dict:
    pairs = C.read_jsonl(C.legacy_data_dir() / "agriadvice_pairs.jsonl")
    pair_ids = {p["pair_id"] for p in pairs}
    out = {"pairs": len(pairs), "axes": dict(Counter(p["toggle_axis"] for p in pairs)),
           "distinct_base_query": len({p["base_query"] for p in pairs}), "cached_arms": []}
    for p in sorted(C.legacy_results_dir().glob("agriadvice_drift_*_seed*.jsonl")):
        rows = C.read_jsonl(p)
        ids = {r["pair_id"] for r in rows}
        judged = sum(1 for r in rows if str(r.get("judge_score_one_to_five", "")).strip())
        out["cached_arms"].append({"file": p.name, "pairs": len(rows), "missing_pairs": len(pair_ids - ids),
                                   "judge_scored": judged, "has_token_counts": any("generated_tokens" in r or "finish_reason" in r for r in rows[:3])})
    return out


def check_checkpoints() -> Dict:
    """Adapter configs alone do not establish usable checkpoints (plan section 4)."""
    root = C.CODES_ROOT / "checkpoints"
    out = {"root_exists": root.exists(), "arms_with_config": 0, "arms_with_weights": 0, "missing_weights": []}
    if not root.exists():
        return out
    for cfg in root.glob("*/*/seed_*/**/adapter_config.json"):
        out["arms_with_config"] += 1
        if (cfg.parent / "adapter_model.safetensors").exists():
            out["arms_with_weights"] += 1
        else:
            out["missing_weights"].append(str(cfg.parent.relative_to(root)))
    return out


def main(cfg: Dict) -> Dict:
    out = C.output_dir(cfg) / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    findings: List[Dict] = []

    test_path = C.legacy_data_dir() / "test_instances_frozen.jsonl"
    test_hash = C.sha256_file(test_path)
    hash_ok = test_hash == cfg["frozen_test_sha256"]
    if not hash_ok:
        findings.append({"severity": "blocker", "where": str(test_path),
                         "finding": f"frozen test hash {test_hash} != configured {cfg['frozen_test_sha256']}; investigate before any analysis"})

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_snapshot": _git_snapshot(),
        "versions": _versions(),
        "config_sha256": C.sha256_obj(cfg),
        "analysis_seed": cfg["analysis_seed"],
        "inputs": {p.name: C.sha256_file(p) for p in sorted(C.legacy_data_dir().glob("*.jsonl"))},
        "frozen_test_hash_matches_config": hash_ok,
        "splits": check_splits(findings),
        "predictions": check_predictions(findings),
        "advice": check_advice(findings),
        "checkpoints": check_checkpoints(),
        "findings": findings,
    }
    C.write_json(out / "manifest.json", manifest)
    C.write_csv(out / "prediction_coverage.csv", manifest["predictions"]["coverage"])
    dev = out / "deviations_log.md"
    if not dev.exists():
        dev.write_text(
            "# Deviations and amendments log\n\n"
            "Dated entries only. Four sections, kept apart (plan section 5.1):\n\n"
            "## A. Original preregistered hypotheses (unchanged; see PREREGISTRATION.md)\n\n"
            "## B. Implementation deviations from the preregistration\n\n"
            "## C. Post-hoc diagnostics already inspected before this round\n"
            f"- {time.strftime('%Y-%m-%d')}: train-only metadata and TF-IDF classifiers (plan section 2.2); "
            "advice length/duplication diagnostic (section 2.3). Post-hoc, not confirmatory.\n\n"
            "## D. New analyses frozen before their outputs were examined\n"
            f"- {time.strftime('%Y-%m-%d')}: Next_Run protocol frozen: source-cluster bootstrap and cluster-swap "
            "permutation, seed-explicit families, advice audit, optional evidence panel.\n",
            encoding="utf-8")
    print(f"[inventory] test hash ok={hash_ok}  prediction files={manifest['predictions']['files']}  "
          f"findings={len(findings)}  -> {out}")
    for f in findings:
        print(f"   [{f['severity']}] {f['where']}: {C.scrub_secrets(f['finding'])}")
    return manifest


if __name__ == "__main__":
    main(C.load_config())
