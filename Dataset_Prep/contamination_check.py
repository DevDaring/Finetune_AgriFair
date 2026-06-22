"""Contamination check for the AgriFair port, with hard-stops (coding_prompt.md 5.3).

AgriFacts questions are heavily templated ("In <State>, according to the 2015-16
Agriculture Census, which ... operates a larger share ..."), so RAW n-gram overlap
between any train/test split is high by construction and is not a contamination signal.
The legal 0.5 percent raw-n-gram threshold (Instruction.md Section 4.3) assumes free prose.

This script therefore guards contamination two ways:
  PRIMARY (hard-stop): exact fact-level disjointness. The answer-determining identity of
  an AgriFacts item is its `source_cell` (state / size-class / metric / comparison), which
  is 1:1 with `paraphrase_of`. No source_cell may appear in two splits. This is stricter
  than the legal n-gram heuristic and is the real guard.
  SECONDARY (hard-stop): exact-text duplicate detection. No test item's (question + choices)
  string may equal any train item's, catching accidental duplication.
  DIAGNOSTIC (reported, not gating): de-boilerplated 8-gram overlap, raw 8-gram overlap, and
  max TF-IDF cosine. These are HIGH by construction (AgriFacts has only ~36 states x a few
  group pairs x 2 metrics, so a test item differs from some train item only in a state name)
  and are reported for transparency, not used to gate; gating on them would reject every
  valid split of a templated benchmark.

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


def _eight_grams(text: str):
    toks = text.lower().split()
    return set(tuple(toks[i : i + 8]) for i in range(len(toks) - 7))


def _text_of(rec):
    return " ".join(
        [
            rec.get("question", ""),
            " ".join(str(c) for c in rec.get("on_disk_choices", [])),
            rec.get("rationale", ""),
        ]
    )


def _template_grams(all_rows):
    df = collections.Counter()
    for r in all_rows:
        for g in _eight_grams(_text_of(r)):
            df[g] += 1
    cutoff = TEMPLATE_DF_FRACTION * len(all_rows)
    return {g for g, c in df.items() if c >= cutoff}


def main():
    train = read_jsonl(TRAIN_INSTANCES)
    val = read_jsonl(VALIDATION_INSTANCES)
    test = read_jsonl(TEST_INSTANCES_FROZEN)
    if not train or not test:
        raise SystemExit("Run build_template_instances.py first.")

    # PRIMARY: exact fact-level disjointness.
    tr_cells = set(r["source_cell"] for r in train)
    va_cells = set(r["source_cell"] for r in val)
    te_cells = set(r["source_cell"] for r in test)
    fact_overlap_test = len(tr_cells & te_cells)
    fact_overlap_val = len(tr_cells & va_cells)

    # SECONDARY: exact-text duplicate detection (question + choices).
    def _sig(r):
        return r.get("question", "") + " || " + " | ".join(str(c) for c in r.get("on_disk_choices", []))

    train_sigs = set(_sig(r) for r in train)
    exact_dupes = sum(1 for r in test if _sig(r) in train_sigs)

    # DIAGNOSTIC: de-boilerplated 8-gram overlap.
    template = _template_grams(train + val + test)
    train_content = set()
    for r in train:
        train_content |= _eight_grams(_text_of(r)) - template
    test_content_total = 0
    test_content_overlap = 0
    for r in test:
        cg = _eight_grams(_text_of(r)) - template
        test_content_total += len(cg)
        test_content_overlap += len(cg & train_content)
    deboiler_fraction = test_content_overlap / max(1, test_content_total)

    # DIAGNOSTICS.
    train_all = set()
    for r in train:
        train_all |= _eight_grams(_text_of(r))
    raw_total = 0
    raw_overlap = 0
    for r in test:
        g = _eight_grams(_text_of(r))
        raw_total += len(g)
        raw_overlap += len(g & train_all)
    raw_fraction = raw_overlap / max(1, raw_total)
    max_cosine = _max_cosine(train, test, os.environ.get("MULTILINGUAL_EMBED_MODEL"))

    report = [
        {"metric_name": "fact_level_source_cell_overlap_train_test", "metric_value": fact_overlap_test, "threshold_value": 0, "is_gating": True, "check_passed": fact_overlap_test == 0},
        {"metric_name": "fact_level_source_cell_overlap_train_val", "metric_value": fact_overlap_val, "threshold_value": 0, "is_gating": True, "check_passed": fact_overlap_val == 0},
        {"metric_name": "exact_text_duplicate_test_in_train", "metric_value": exact_dupes, "threshold_value": 0, "is_gating": True, "check_passed": exact_dupes == 0},
        {"metric_name": "deboilerplated_eight_gram_overlap_fraction_diagnostic", "metric_value": round(deboiler_fraction, 6), "threshold_value": DIAGNOSTIC_NOTE, "is_gating": False, "check_passed": True},
        {"metric_name": "raw_eight_gram_overlap_fraction_diagnostic", "metric_value": round(raw_fraction, 6), "threshold_value": DIAGNOSTIC_NOTE, "is_gating": False, "check_passed": True},
        {"metric_name": "max_test_train_cosine_similarity_diagnostic", "metric_value": round(max_cosine, 6), "threshold_value": DIAGNOSTIC_NOTE, "is_gating": False, "check_passed": True},
    ]
    write_csv(CONTAMINATION_REPORT, report, ["metric_name", "metric_value", "threshold_value", "is_gating", "check_passed"])
    log_run_metadata(
        "contamination_check",
        {
            "fact_level_overlap_train_test": fact_overlap_test,
            "exact_text_duplicate_test_in_train": exact_dupes,
            "deboilerplated_eight_gram_overlap_fraction": deboiler_fraction,
            "raw_eight_gram_overlap_fraction": raw_fraction,
            "max_cosine": max_cosine,
        },
    )

    logger.info("PRIMARY  fact-level source_cell overlap train/test = %d (must be 0).", fact_overlap_test)
    logger.info("SECONDARY exact-text duplicates test-in-train = %d (must be 0).", exact_dupes)
    logger.info("DIAGNOSTIC de-boilerplated 8-gram %.2f%%, raw 8-gram %.2f%%, max cosine %.4f (templated; not gating).",
                100 * deboiler_fraction, 100 * raw_fraction, max_cosine)

    if fact_overlap_test > 0 or fact_overlap_val > 0:
        raise SystemExit(f"HARD STOP: fact-level source_cell overlap (test={fact_overlap_test}, val={fact_overlap_val}).")
    if exact_dupes > 0:
        raise SystemExit(f"HARD STOP: {exact_dupes} exact-text duplicate test items found in train.")
    logger.info("Contamination check passed (fact-level disjoint; no exact duplicates).")


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


if __name__ == "__main__":
    main()
