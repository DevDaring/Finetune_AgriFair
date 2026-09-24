"""R1-A: CPU audit of the original resource under template and source controls.

    python -m Submission1_Code_Phase2.r1_template_audit

Four things the original novel-slice split could not show:

1. **Template families.** Every question is canonicalized to a sentence skeleton with entities,
   state, metric words and digits blanked (common.canonical_template). Identical skeletons form
   one family. The key is computed from surface text only, never from the gold label.
2. **Baselines reproduced under a grouped holdout.** The training-majority, metadata and question
   TF-IDF baselines are refitted inside each fold of a GroupKFold over template families, and
   again over parent-table/state groups. A vectorizer fitted on the whole corpus would leak
   across folds, so it is fitted inside the fold.
3. **Overlap audit.** How much the original train/test split shares template families and source
   comparisons; an option-letter audit that checks canonical answers against displayed letters,
   so shuffling cannot create a bookkeeping shortcut.
4. **Text-only sensitivities.** Masking state names, and masking group names. Masking destroys
   part of the task, so a drop is a diagnostic of what the text carries, not proof of bias; the
   output says so in its own column.

Outputs (results_submission1_phase2/templates/): template_manifest.csv, overlap_audit.json,
shallow_grouped_cv.csv, text_sensitivities.csv, option_letter_audit.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Submission1_Code_Phase2 import common as C


def load_items() -> List[Dict]:
    items = C.read_jsonl(C.DATASET_FACTS)
    split = {r["id"]: r.get("split", "") for r in C.read_jsonl(C.CODES_ROOT / "data" / "test_instances_frozen.jsonl")} if (C.CODES_ROOT / "data" / "test_instances_frozen.jsonl").exists() else {}
    for it in items:
        p = C.parse_cell(it["source_cell"])
        groups = [c for c in it["choices"] if not c.lower().startswith("roughly equal")]
        it["state"] = p["state"]; it["size_class"] = p["size_class"]; it["parent_table"] = C.parent_table(it["source_cell"])
        it["template_id"] = C.template_id(it["question"], groups, p["state"])
        it["template_text"] = C.canonical_template(it["question"], groups, p["state"])
        it["gold_letter"] = "c" if it["answer"].lower().startswith("roughly equal") else ("a" if it["answer"] == groups[0] else "b")
        it["in_frozen_test"] = it["id"] in split
    return items


# ------------------------------------------------------------------ feature builders

def metadata_features(items: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Structural fields only: axis, metric, size class, comparison token. No question text."""
    rows = []
    for it in items:
        p = C.parse_cell(it["source_cell"])
        rows.append({"axis": it["axis"], "metric": it["metric"], "size_class": p["size_class"], "comparison": p["comparison"]})
    return rows, ["axis", "metric", "size_class", "comparison"]


def mask_states(text: str) -> str:
    out = text
    for s in sorted(C.known_states(), key=len, reverse=True):
        out = re.sub(re.escape(s), "[STATE]", out, flags=re.I)
    return out


def mask_groups(text: str, groups: List[str]) -> str:
    out = text
    for g in sorted(set(groups + C.GROUP_WORDS), key=len, reverse=True):
        if g:
            out = re.sub(re.escape(g), "[GROUP]", out, flags=re.I)
    return out


# ------------------------------------------------------------------ grouped cross-validation

def grouped_cv(items: List[Dict], group_key: str, cfg: Dict, text_fn=None) -> List[Dict]:
    """Refit each shallow baseline inside every fold of a GroupKFold over `group_key`."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import OneHotEncoder

    y = np.array([it["gold_letter"] for it in items])
    groups = np.array([str(it[group_key]) for it in items])
    n_groups = len(set(groups))
    folds = min(int(cfg["r1"]["cv_folds"]), n_groups)
    if folds < 2:
        return [{"group_key": group_key, "baseline": b, "status": f"not evaluable: {n_groups} group(s)"} for b in ("majority", "metadata_lr", "question_tfidf_lr")]
    texts = [(text_fn(it) if text_fn else it["question"]) for it in items]
    meta_rows, meta_cols = metadata_features(items)
    meta_mat = [[r[c] for c in meta_cols] for r in meta_rows]
    gkf = GroupKFold(n_splits=folds)
    acc = {b: [] for b in ("majority", "metadata_lr", "question_tfidf_lr")}
    per_fold = []
    for k, (tr, te) in enumerate(gkf.split(texts, y, groups)):
        if len(tr) < int(cfg["r1"]["min_train_rows_per_fold"]):
            continue
        maj = collections.Counter(y[tr]).most_common(1)[0][0]
        a_maj = float((y[te] == maj).mean())
        enc = OneHotEncoder(handle_unknown="ignore").fit([meta_mat[i] for i in tr])
        lr_m = LogisticRegression(max_iter=2000, C=1.0).fit(enc.transform([meta_mat[i] for i in tr]), y[tr])
        a_meta = float(lr_m.score(enc.transform([meta_mat[i] for i in te]), y[te]))
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000).fit([texts[i] for i in tr])
        lr_t = LogisticRegression(max_iter=2000, C=1.0).fit(vec.transform([texts[i] for i in tr]), y[tr])
        a_tfidf = float(lr_t.score(vec.transform([texts[i] for i in te]), y[te]))
        for b, a in (("majority", a_maj), ("metadata_lr", a_meta), ("question_tfidf_lr", a_tfidf)):
            acc[b].append(a)
        per_fold.append({"fold": k, "n_train": len(tr), "n_test": len(te), "test_groups": len(set(groups[te])),
                         "majority": round(a_maj, 4), "metadata_lr": round(a_meta, 4), "question_tfidf_lr": round(a_tfidf, 4)})
    out = []
    for b, vals in acc.items():
        out.append({"group_key": group_key, "baseline": b, "folds": len(vals), "n_groups": n_groups,
                    "accuracy_mean": round(float(np.mean(vals)), 4) if vals else float("nan"),
                    "accuracy_min": round(float(np.min(vals)), 4) if vals else float("nan"),
                    "accuracy_max": round(float(np.max(vals)), 4) if vals else float("nan"),
                    "status": "ok" if vals else "no fold met the minimum training size"})
    return out, per_fold


# ------------------------------------------------------------------ audits

def overlap_audit(items: List[Dict]) -> Dict:
    test = [it for it in items if it["in_frozen_test"]]
    train = [it for it in items if not it["in_frozen_test"]]
    t_tpl, r_tpl = {i["template_id"] for i in test}, {i["template_id"] for i in train}
    t_cell, r_cell = {i["source_cell"] for i in test}, {i["source_cell"] for i in train}
    fam_sizes = collections.Counter(i["template_id"] for i in items)
    return {"n_items": len(items), "n_train_like": len(train), "n_frozen_test": len(test),
            "template_families_total": len(fam_sizes),
            "template_families_train": len(r_tpl), "template_families_test": len(t_tpl),
            "test_families_also_in_train": len(t_tpl & r_tpl),
            "test_items_whose_family_is_in_train": sum(1 for i in test if i["template_id"] in r_tpl),
            "share_of_test_items_with_seen_template": round(sum(1 for i in test if i["template_id"] in r_tpl) / max(1, len(test)), 4),
            "source_cells_test_also_in_train": len(t_cell & r_cell),
            "largest_families": fam_sizes.most_common(5),
            "note": ("The original split withheld structural keys, not linguistic templates. A test item whose "
                     "template family also occurs in training can be answered from wording alone.")}


def option_letter_audit(items: List[Dict]) -> Dict:
    """Canonical answer vs displayed option position: shuffling must not encode the label."""
    pos = collections.Counter()
    mismatched = []
    for it in items:
        groups = [c for c in it["choices"] if not c.lower().startswith("roughly equal")]
        idx = it["choices"].index(it["answer"])
        pos[("abc"[idx], it["gold_letter"])] += 1
        letter = C.canonical_letter(it["answer"], groups[0], groups[1])
        if letter != it["gold_letter"]:
            mismatched.append(it["id"])
    by_display = collections.Counter(k[0] for k in pos.elements())
    n = sum(by_display.values())
    return {"displayed_position_of_correct_answer": dict(by_display),
            "max_position_share": round(max(by_display.values()) / max(1, n), 4),
            "canonical_letter_mismatches": len(mismatched), "mismatch_ids": mismatched[:20],
            "verdict": ("no positional shortcut: the correct answer is spread across displayed positions"
                        if max(by_display.values()) / max(1, n) < 0.45 else
                        "positional imbalance: check the shuffling seed before interpreting accuracy")}


def dominant_features(items: List[Dict], top: int = 15) -> List[Dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    y = [it["gold_letter"] for it in items]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
    X = vec.fit_transform([it["question"] for it in items])
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    rows = []
    for ci, cls in enumerate(lr.classes_):
        coef = lr.coef_[ci] if lr.coef_.shape[0] > 1 else lr.coef_[0] * (1 if ci else -1)
        for j in np.argsort(coef)[-top:][::-1]:
            rows.append({"class": cls, "feature": names[j], "weight": round(float(coef[j]), 4)})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(argv)
    cfg = C.load_config(); out = C.out_dir(cfg, "templates")
    items = load_items()
    if a.smoke:
        items = items[: max(50, cfg["smoke"]["items"] * 10)]

    C.write_csv(out / "template_manifest.csv",
                [{"item_id": it["id"], "template_id": it["template_id"], "axis": it["axis"], "metric": it["metric"],
                  "state": it["state"], "size_class": it["size_class"], "parent_table": it["parent_table"],
                  "source_cell": it["source_cell"], "in_frozen_test": it["in_frozen_test"],
                  "template_text": it["template_text"][:180]} for it in items])

    ov = overlap_audit(items); C.write_json(out / "overlap_audit.json", ov)
    C.write_json(out / "option_letter_audit.json", option_letter_audit(items))

    rows, folds = [], []
    for key in ("template_id", "parent_table", "state", "source_cell"):
        r, f = grouped_cv(items, key, cfg)
        rows += r; folds += [{**x, "group_key": key} for x in f]
    C.write_csv(out / "shallow_grouped_cv.csv", rows)
    C.write_csv(out / "shallow_grouped_cv_folds.csv", folds)

    sens = []
    for name, fn, note in (("unmasked", None, "full question text"),
                           ("state_masked", lambda it: mask_states(it["question"]), "state names blanked; task partly destroyed, diagnostic only"),
                           ("group_masked", lambda it: mask_groups(it["question"], [c for c in it["choices"] if not c.lower().startswith("roughly equal")]), "group names blanked; the question can no longer be answered faithfully, diagnostic only")):
        r, _ = grouped_cv(items, "template_id", cfg, text_fn=fn)
        for x in r:
            if x["baseline"] == "question_tfidf_lr":
                sens.append({"masking": name, "note": note, **{k: v for k, v in x.items() if k != "baseline"}})
    C.write_csv(out / "text_sensitivities.csv", sens)
    C.write_csv(out / "dominant_features.csv", dominant_features(items))

    C.write_json(out / "manifest.json", C.manifest(cfg, "r1_template_audit", {
        "n_items": len(items), "template_families": ov["template_families_total"],
        "share_of_test_items_with_seen_template": ov["share_of_test_items_with_seen_template"],
        "grouped_cv_keys": ["template_id", "parent_table", "state", "source_cell"]}))
    print(f"[r1_template_audit] {len(items)} items, {ov['template_families_total']} template families; "
          f"{ov['share_of_test_items_with_seen_template']:.1%} of frozen-test items reuse a training template -> {out}")


if __name__ == "__main__":
    main()
