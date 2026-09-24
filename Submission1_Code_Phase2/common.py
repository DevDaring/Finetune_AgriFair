"""Shared helpers for the Phase 2 repair package.

Metric definitions, the error taxonomy and the cluster statistics are imported from Next_Run
rather than reimplemented, so the new studies are scored exactly like the originals. Nothing
here writes into results_final_audit_20260912; Phase 2 outputs live in their own directory.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

from Next_Run.common import (CANONICAL, EQUAL_LETTER, Counts, error_type, harmonic_b,  # noqa: F401
                             counts_by_cluster, metrics_from_counts, read_jsonl, scrub_secrets,
                             sha256_file, sha256_obj, write_csv, write_json, write_jsonl)

CODES_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"
AXES = ("gender", "landholding", "social_group")

# Existing assets this package reads (never writes).
ORIGINAL_AUDIT = CODES_ROOT / "results_final_audit_20260912"
CONSTRUCTION = CODES_ROOT / "Source_Records" / "extracted" / "Dataset"
FACTS_CSV = CONSTRUCTION / "data" / "interim" / "agrifacts_facts.csv"
DATASET_FACTS = CODES_ROOT / "Dataset" / "agrifacts.jsonl"
DATASET_ADVICE = CODES_ROOT / "Dataset" / "agriadvice.jsonl"
LEDGER = ORIGINAL_AUDIT / "sources" / "evidence_ledger_filled.csv"


def load_config(path: Path = CONFIG_PATH) -> Dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def out_dir(cfg: Dict, *parts: str) -> Path:
    p = CODES_ROOT / cfg["output_directory"]
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def systems(cfg: Dict) -> List[Dict]:
    s = cfg["systems"]
    return [{"tier": t, "method": m, "seed": s["seed"]} for t in s["tiers"] for m in s["methods"]]


def system_id(sys: Mapping) -> str:
    return f"{sys['tier']}|{sys['method']}|seed{sys['seed']}"


# ------------------------------------------------------------------ source-cell handling

_CELL = re.compile(r"^(?P<edition>\S+)\s+(?P<table>\S+)\s+(?P<state>.+?)/(?P<size_class>[^/]+)/(?P<metric>[^/]+)/(?P<comparison>.+)$")


def parse_cell(cell: str) -> Dict[str, str]:
    """'AgCensus2015-16 T2-4 Manipur/Marginal/number/SCvsOthers' -> its parts."""
    m = _CELL.match(cell.strip())
    if not m:
        return {"edition": "", "table": "", "state": "", "size_class": "", "metric": "", "comparison": cell}
    return m.groupdict()


def parent_table(cell: str) -> str:
    p = parse_cell(cell)
    return f"{p['edition']} {p['table']}"


def state_of(cell: str) -> str:
    return parse_cell(cell)["state"]


# ------------------------------------------------------------------ template families

_STATES_CACHE: Optional[set] = None
GROUP_WORDS = ["Scheduled Castes", "Scheduled Tribes", "Other social groups", "All social groups",
               "marginal holdings", "small holdings", "semi-medium holdings", "medium holdings", "large holdings",
               "marginal operated area", "small operated area", "semi-medium operated area", "medium operated area",
               "large operated area", "female operational holders", "male operational holders", "women", "men"]
METRIC_WORDS = ["number of holdings", "agricultural holdings", "operated area", "agricultural operated area",
                "holdings", "operated"]


def known_states() -> set:
    """States seen in the ledger; used only to blank surface tokens, never labels."""
    global _STATES_CACHE
    if _STATES_CACHE is None:
        import csv
        _STATES_CACHE = {parse_cell(r["source_cell"])["state"] for r in csv.DictReader(LEDGER.open(encoding="utf-8"))}
    return _STATES_CACHE


def canonical_template(question: str, groups: Sequence[str] = (), state: str = "") -> str:
    """Blank every content slot, leaving the sentence skeleton = the template family key.

    Content slots are entity names, the state, metric words and digits. This is a purely
    surface canonicalization: it never reads the gold label or any model output.
    """
    t = " " + (question or "").strip() + " "
    for g in sorted(set(list(groups) + GROUP_WORDS), key=len, reverse=True):
        if g:
            t = re.sub(re.escape(g), " <G> ", t, flags=re.I)
    for s in sorted(known_states() | ({state} if state else set()), key=len, reverse=True):
        if s:
            t = re.sub(re.escape(s), " <S> ", t, flags=re.I)
    for m in sorted(METRIC_WORDS, key=len, reverse=True):
        t = re.sub(re.escape(m), " <M> ", t, flags=re.I)
    t = re.sub(r"\d[\d,.\-]*", " <N> ", t)
    t = re.sub(r"[^\w<>]+", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def template_id(question: str, groups: Sequence[str] = (), state: str = "") -> str:
    return hashlib.sha256(canonical_template(question, groups, state).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ answers

def canonical_letter(choice_text: str, group1: str, group2: str) -> str:
    """a = group1, b = group2, c = roughly equal, matching the original scoring convention."""
    c = (choice_text or "").strip().lower()
    if c.startswith("roughly equal"):
        return "c"
    if c == (group1 or "").strip().lower():
        return "a"
    if c == (group2 or "").strip().lower():
        return "b"
    return ""


def freeze(obj) -> str:
    """Stable digest of a design object, written into every manifest."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def manifest(cfg: Dict, stage: str, payload: Dict) -> Dict:
    return {"stage": stage, "analysis_seed": cfg["analysis_seed"], "comparison_rule": cfg["comparison_rule"],
            "design_sha256": freeze(payload), **payload}
