"""P2.1 - exactly three train-only CPU baselines, fixed in advance (plan sections 2.2, 7.1).

  1. training-majority: predicts the training set's majority canonical letter; the
     always-equal predictor is reported alongside as a descriptive diagnostic.
  2. metadata logistic regression: one-hot of the four fields in state_blind_key
     (axis, metric, size class, comparison token). No condition, gold, rationale or
     raw source identifier.
  3. question-only TF-IDF logistic regression: word 1-2 grams, min_df=2,
     max_features=20000, C=1, max_iter=1000, on the canonical question text only.

All three are fitted on the training partition alone; preprocessing is fitted on training
text only. Nothing is tuned on the test set. Per-item predictions and probabilities are
saved so that paired_analysis can treat these exactly like model arms.

Optional sensitivity: schema-aware geography masking. It is only reported if masking can
be verified on the source-table schema; examples are written out so a reader can check
that what was masked was a state and not a category token.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from Next_Run import common as C

TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=20000)
LOGREG = dict(C=1.0, max_iter=1000)


def _records(split: str) -> List[Dict]:
    return C.load_split(split)


def _meta_features(rows: List[Dict]) -> List[Dict[str, str]]:
    return [{f"axis={p['axis']}": 1, f"metric={p['metric']}": 1, f"size_class={p['size_class']}": 1,
             f"comparison={p['comparison']}": 1}
            for p in (C.parse_state_blind_key(r["state_blind_key"]) for r in rows)]


def _state_tokens(rows: List[Dict]) -> Tuple[set, List[Dict]]:
    """States, taken from the parsed source_cell, not from a fixed token position. Returns the
    set and a few examples so the masking can be eyeballed."""
    states = set()
    examples = []
    for r in rows:
        p = C.parse_source_cell(r["source_cell"])
        if p["parse_ok"]:
            states.add(p["state"])
            if len(examples) < 6:
                examples.append({"source_cell": r["source_cell"], "state_parsed": p["state"]})
    return states, examples


def _mask_states(text: str, states: set) -> str:
    for s in sorted(states, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(s) + r"\b", "STATE", text)
    return text


def _fit_predict(train_rows, test_rows, kind: str, states=None):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    y_train = [r["correct_answer"] for r in train_rows]
    if kind == "majority":
        maj = Counter(y_train).most_common(1)[0][0]
        preds = [maj] * len(test_rows)
        proba = [{c: float(c == maj) for c in C.CANONICAL} for _ in test_rows]
        return preds, proba, {"majority_letter": maj, "train_distribution": dict(Counter(y_train))}
    if kind == "always_equal":
        return [C.EQUAL_LETTER] * len(test_rows), [{c: float(c == C.EQUAL_LETTER) for c in C.CANONICAL} for _ in test_rows], {}
    if kind == "metadata":
        pipe = make_pipeline(DictVectorizer(sparse=True), LogisticRegression(**LOGREG))
        Xtr, Xte = _meta_features(train_rows), _meta_features(test_rows)
    elif kind in ("tfidf", "tfidf_state_blind"):
        tx = (lambda t: _mask_states(t, states)) if kind == "tfidf_state_blind" else (lambda t: t)
        pipe = make_pipeline(TfidfVectorizer(**TFIDF), LogisticRegression(**LOGREG))
        Xtr = [tx(r["question"]) for r in train_rows]
        Xte = [tx(r["question"]) for r in test_rows]
    else:
        raise ValueError(kind)
    pipe.fit(Xtr, y_train)
    preds = list(pipe.predict(Xte))
    P = pipe.predict_proba(Xte)
    classes = list(pipe.classes_)
    proba = [{c: float(P[i, classes.index(c)]) if c in classes else 0.0 for c in C.CANONICAL} for i in range(len(test_rows))]
    info = {"classes": classes}
    # descriptive feature inspection: strongest coefficients per class
    try:
        vec, clf = pipe.steps[0][1], pipe.steps[-1][1]
        names = np.asarray(vec.get_feature_names_out())
        top = {}
        for ci, cls in enumerate(classes):
            coef = clf.coef_[ci] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if ci else -1)
            top[cls] = [(str(names[j]), round(float(coef[j]), 3)) for j in np.argsort(-coef)[:12]]
        info["top_features_by_class"] = top
        info["n_features"] = int(len(names))
    except Exception:
        pass
    return preds, proba, info


def evaluate(test_rows: List[Dict], preds: List[str]) -> Dict[str, Dict]:
    out = {}
    for name, sel in (("full", lambda r: True), ("structure_novel", lambda r: r.get("test_slice") == "structure_novel"),
                      ("structure_familiar", lambda r: r.get("test_slice") == "structure_familiar")):
        cnt = C.Counts()
        for r, p in zip(test_rows, preds):
            if sel(r):
                cnt.add(r["correct_answer"], p, True)
        out[name] = C.metrics_from_counts(cnt.as_array())
    return out


def main(cfg: Dict) -> Dict:
    out = C.output_dir(cfg) / "cpu_baselines"
    out.mkdir(parents=True, exist_ok=True)
    train, test = _records("train"), _records("test")
    states, examples = _state_tokens(train + test)
    # the schema check the plan asks for: are any "states" actually category tokens?
    suspicious = sorted(s for s in states if s in ("ST", "SC", "All", "Others", "Total") or len(s) <= 3)
    state_blind_ok = len(states) > 0 and not suspicious

    summary_rows, feature_info = [], {}
    kinds = ["majority", "always_equal", "metadata", "tfidf"] + (["tfidf_state_blind"] if state_blind_ok else [])
    for kind in kinds:
        preds, proba, info = _fit_predict(train, test, kind, states)
        feature_info[kind] = info
        # per-item predictions in the same shape as a model arm, so paired_analysis can use them
        recs = [{"id": r["id"], "gold_canonical": r["correct_answer"], "pred_canonical": p, "parse_ok": True,
                 "proba_a": pr["a"], "proba_b": pr["b"], "proba_c": pr["c"]}
                for r, p, pr in zip(test, preds, proba)]
        C.write_jsonl(out / f"per_item_predictions_cpu-baseline_{kind}_seed{cfg['analysis_seed']}.jsonl", recs)
        ev = evaluate(test, preds)
        for slice_name, m in ev.items():
            summary_rows.append({"baseline": kind, "slice": slice_name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
        print(f"[cpu_baselines] {kind:18s} full B={ev['full']['harmonic_b']:.4f}  novel B={ev['structure_novel']['harmonic_b']:.4f}")
    C.write_csv(out / "cpu_baseline_summary.csv", summary_rows)
    C.write_json(out / "feature_definitions.json", {
        "tfidf": TFIDF, "logistic_regression": LOGREG,
        "metadata_fields": ["axis", "metric", "size_class", "comparison"], "metadata_source": "state_blind_key",
        "text_field": "question", "fitted_on": "train_instances.jsonl only",
        "state_blind_sensitivity_reported": state_blind_ok,
        "state_tokens_found": sorted(states), "suspicious_state_tokens": suspicious, "masking_examples": examples,
        "feature_inspection": feature_info,
        "note": "post-hoc diagnostic reproduced in a saved pipeline (plan section 2.2); lexical associations are descriptive, not causal",
    })
    return {"kinds": kinds, "state_blind_reported": state_blind_ok}


if __name__ == "__main__":
    main(C.load_config())
