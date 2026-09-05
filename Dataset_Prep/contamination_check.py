"""Contamination check for a templated benchmark, with hard stops.

AgriFacts questions are heavily templated, so raw n-gram overlap between any two splits is
high by construction and is not a contamination signal on its own. The legal-prose
threshold of 0.5 percent raw n-gram overlap would reject every valid split here. This
script therefore gates on what is meaningful and reports the rest:

  gating   exact fact-level source_cell disjointness across the three splits, which is
           stricter than any n-gram heuristic because source_cell is what determines the
           answer; and exact-text duplicate detection between test and train.
  gating   structural disjointness of the structure-novel test slice: no item in that slice
           may share a state_blind_key with training, which is the property the slice
           exists to guarantee.
  reported de-boilerplated 8-gram overlap, raw 8-gram overlap, maximum TF-IDF cosine, and
           the share of each test slice whose state-blind structure is also in training.
           These are expected to be high for a templated benchmark and are diagnostics.

Run:  python Dataset_Prep/contamination_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collections
import os

from GPU_Run.common.checkpointing import read_jsonl
from GPU_Run.common.logging_utils import get_logger, log_run_metadata, write_csv
from GPU_Run.common.paths import (
    CONTAMINATION_REPORT,
    TEST_INSTANCES_FROZEN,
    TRAIN_INSTANCES,
    VALIDATION_INSTANCES,
)

logger = get_logger("contamination_check")

TEMPLATE_DF_FRACTION = 0.20      # 8-grams in >=20% of items are treated as boilerplate
DIAGNOSTIC_NOTE = "diagnostic only; AgriFacts is templated"
REPORT_COLUMNS = ["metric_name", "metric_value", "threshold_value", "is_gating", "check_passed"]


def _eight_grams(text: str):
    toks = text.lower().split()
    return set(tuple(toks[i : i + 8]) for i in range(len(toks) - 7))


def _text_of(rec):
    return " ".join([
        rec.get("question", ""),
        " ".join(str(c) for c in rec.get("on_disk_choices", [])),
        rec.get("rationale", ""),
    ])


def _template_grams(all_rows):
    df = collections.Counter()
    for r in all_rows:
        for g in _eight_grams(_text_of(r)):
            df[g] += 1
    cutoff = TEMPLATE_DF_FRACTION * len(all_rows)
    return {g for g, c in df.items() if c >= cutoff}


def _overlap_fraction(test, train_grams, template=frozenset()):
    total = overlap = 0
    for r in test:
        g = _eight_grams(_text_of(r)) - template
        total += len(g)
        overlap += len(g & train_grams)
    return overlap / max(1, total)


def _max_cosine(train, test, embed_model):
    import numpy as np

    train_texts = [_text_of(r) for r in train]
    test_texts = [_text_of(r) for r in test]
    if embed_model:
        try:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(embed_model)
            tr = m.encode(train_texts, normalize_embeddings=True)
            te = m.encode(test_texts, normalize_embeddings=True)
            return float((te @ tr.T).max())
        except Exception as e:
            logger.warning("Embed model failed (%s); TF-IDF fallback.", e)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer().fit(train_texts + test_texts)
    tr = normalize(vec.transform(train_texts))
    te = normalize(vec.transform(test_texts))
    max_c = 0.0
    for i in range(0, te.shape[0], 256):
        block = (te[i : i + 256] @ tr.T).toarray()
        max_c = max(max_c, float(block.max()) if block.size else 0.0)
    return max_c


def main():
    train = read_jsonl(TRAIN_INSTANCES)
    val = read_jsonl(VALIDATION_INSTANCES)
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    if not train or not test:
        raise SystemExit("Run build_template_instances.py first.")

    tr_cells = {r["source_cell"] for r in train}
    va_cells = {r["source_cell"] for r in val}
    te_cells = {r["source_cell"] for r in test}
    fact_overlap_test = len(tr_cells & te_cells)
    fact_overlap_val = len(tr_cells & va_cells)

    def _sig(r):
        return r.get("question", "") + " || " + " | ".join(str(c) for c in r.get("on_disk_choices", []))

    train_sigs = {_sig(r) for r in train}
    exact_dupes = sum(1 for r in test if _sig(r) in train_sigs)

    train_keys = {r["state_blind_key"] for r in train}
    novel = [r for r in test if r.get("test_slice") == "structure_novel"]
    familiar = [r for r in test if r.get("test_slice") == "structure_familiar"]
    novel_leak = sum(1 for r in novel if r["state_blind_key"] in train_keys)
    familiar_share = (sum(1 for r in familiar if r["state_blind_key"] in train_keys) / len(familiar)) if familiar else float("nan")

    template = _template_grams(train + val + test)
    train_content = set()
    train_all = set()
    for r in train:
        g = _eight_grams(_text_of(r))
        train_all |= g
        train_content |= g - template
    deboiler = _overlap_fraction(test, train_content, template)
    raw = _overlap_fraction(test, train_all)
    max_cosine = _max_cosine(train, test, os.environ.get("MULTILINGUAL_EMBED_MODEL"))

    report = [
        {"metric_name": "fact_level_source_cell_overlap_train_test", "metric_value": fact_overlap_test,
         "threshold_value": 0, "is_gating": True, "check_passed": fact_overlap_test == 0},
        {"metric_name": "fact_level_source_cell_overlap_train_validation", "metric_value": fact_overlap_val,
         "threshold_value": 0, "is_gating": True, "check_passed": fact_overlap_val == 0},
        {"metric_name": "exact_text_duplicate_test_in_train", "metric_value": exact_dupes,
         "threshold_value": 0, "is_gating": True, "check_passed": exact_dupes == 0},
        {"metric_name": "structure_novel_slice_items_sharing_a_state_blind_key_with_train",
         "metric_value": novel_leak, "threshold_value": 0, "is_gating": True, "check_passed": novel_leak == 0},
        {"metric_name": "structure_familiar_slice_share_sharing_a_state_blind_key_with_train",
         "metric_value": round(familiar_share, 6) if familiar_share == familiar_share else "",
         "threshold_value": "expected_high_by_design", "is_gating": False, "check_passed": True},
        {"metric_name": "deboilerplated_eight_gram_overlap_fraction_diagnostic",
         "metric_value": round(deboiler, 6), "threshold_value": DIAGNOSTIC_NOTE, "is_gating": False, "check_passed": True},
        {"metric_name": "raw_eight_gram_overlap_fraction_diagnostic", "metric_value": round(raw, 6),
         "threshold_value": DIAGNOSTIC_NOTE, "is_gating": False, "check_passed": True},
        {"metric_name": "max_test_train_cosine_similarity_diagnostic", "metric_value": round(max_cosine, 6),
         "threshold_value": DIAGNOSTIC_NOTE, "is_gating": False, "check_passed": True},
    ]
    write_csv(CONTAMINATION_REPORT, report, REPORT_COLUMNS)
    log_run_metadata("contamination_check", {r["metric_name"]: r["metric_value"] for r in report})

    logger.info("GATING fact-level source_cell overlap train/test=%d, train/validation=%d (must be 0).",
                fact_overlap_test, fact_overlap_val)
    logger.info("GATING exact-text duplicates test-in-train=%d (must be 0).", exact_dupes)
    logger.info("GATING structure-novel items sharing a state-blind key with train=%d (must be 0).", novel_leak)
    logger.info("DIAGNOSTIC de-boilerplated 8-gram %.2f%%, raw 8-gram %.2f%%, max cosine %.4f (templated; not gating).",
                100 * deboiler, 100 * raw, max_cosine)

    failed = [r["metric_name"] for r in report if r["is_gating"] and not r["check_passed"]]
    if failed:
        raise SystemExit("HARD STOP: gating contamination checks failed: " + ", ".join(failed))
    logger.info("Contamination check passed.")


if __name__ == "__main__":
    main()
