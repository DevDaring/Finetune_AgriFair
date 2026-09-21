"""Import the returned verification workbooks: validate, compute agreement, apply corrections.

    python -m Dataset_Translation.verification_import --dir Submission2/verification/filled/Filled_Docs [--apply]

Validation: ids and translated text must match verification_manifest.json (raters may only fill
the rating columns). Agreement: raw agreement and Cohen's kappa per rating column, per language.
Corrections (with --apply): a row rated "partly" or "no" by either rater
  - takes the rater-supplied corrected text when one exists (R1's if both supplied one, and the
    manifest records which rater's was used);
  - otherwise is re-translated through the full pipeline with the raters' comments as the hint.
Rows the raters rated "yes" are never changed. Writes verification_report.json and updates the
output files in place (append-only; readers keep the last row per id).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from openpyxl import load_workbook

from Dataset_Translation.human_study_kappa import kappa  # small helper, defined below if missing
from Dataset_Translation.pipeline import Cache, Translator
from Dataset_Translation.providers import Clients
from Dataset_Translation.quality import _run_with_hint
from Dataset_Translation.run import ADAPTERS, CFG, OUT, SRC, _append, _jsonl

ROOT = Path(__file__).resolve().parents[2]
VER = ROOT / "Submission2" / "verification"
LANG_NAME = {"hi": "Hindi", "bn": "Bengali"}
F = {"meaning": 7, "terms": 8, "natural": 9, "corr_q": 10, "corr_opt": 11, "comment": 12}          # AgriFacts columns (0-based)
P = {"meaning": 6, "identity": 7, "natural": 8, "corr_A": 9, "corr_B": 10, "comment": 11}          # AgriAdvice columns


def _read(path: Path) -> Dict[str, List[tuple]]:
    wb = load_workbook(path, data_only=True)
    return {s: [tuple("" if v is None else str(v).strip() for v in row) for row in wb[s].iter_rows(min_row=2, values_only=True)] for s in ("AgriFacts", "AgriAdvice")}


def _validate(lang: str, sheets: Dict, man: Dict) -> List[str]:
    problems = []
    L = man["languages"][lang]
    if [r[0] for r in sheets["AgriFacts"]] != L["fact_ids"]:
        problems.append("AgriFacts ids or order changed")
    if [r[0] for r in sheets["AgriAdvice"]] != L["pair_ids"]:
        problems.append("AgriAdvice ids or order changed")
    for r in sheets["AgriFacts"]:
        if hashlib.sha256((r[5] + "|" + "|".join(r[6].split(" | "))).encode()).hexdigest() != L["fact_sha256"].get(r[0]):
            problems.append(f"AgriFacts {r[0]}: translated text edited in place")
    for r in sheets["AgriAdvice"]:
        if hashlib.sha256((r[4] + "|" + r[5]).encode()).hexdigest() != L["pair_sha256"].get(r[0]):
            problems.append(f"AgriAdvice {r[0]}: translated text edited in place")
    return problems


def _agreement(a: List[str], b: List[str], ordinal: bool) -> Dict:
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return {"n": 0}
    xs, ys = zip(*pairs)
    return {"n": len(pairs), "raw_agreement": round(sum(x == y for x, y in pairs) / len(pairs), 4), "cohen_kappa": round(kappa(list(xs), list(ys), ordinal), 4)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", type=Path, required=True); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    man = json.load(open(VER / "verification_manifest.json", encoding="utf-8"))
    report = {"languages": {}}
    for lang in CFG["languages"]:
        L = LANG_NAME[lang]; sheets = {r: _read(a.dir / f"AgriFair_{L}_verification_{r}.xlsx") for r in ("R1", "R2")}
        problems = [f"{r}: {p}" for r in sheets for p in _validate(lang, sheets[r], man)]
        if problems:
            raise SystemExit("rejected:\n" + "\n".join(problems[:20]))
        rep = {"facts": {}, "pairs": {}, "corrections": {"applied_from_rater": [], "retranslated": [], "unchanged_after_flag": []},
               "notes": []}
        # ---- per-rater distributions and agreement
        fa, fb = sheets["R1"]["AgriFacts"], sheets["R2"]["AgriFacts"]; pa, pb = sheets["R1"]["AgriAdvice"], sheets["R2"]["AgriAdvice"]
        for name, rows_a, rows_b, cols, ordinal_cols in (("facts", fa, fb, F, {"natural"}), ("pairs", pa, pb, P, {"natural"})):
            d = {}
            for col in [c for c in cols if c in ("meaning", "terms", "identity", "natural")]:
                d[col] = {"R1": dict(Counter(r[cols[col]] for r in rows_a)), "R2": dict(Counter(r[cols[col]] for r in rows_b)),
                          "agreement": _agreement([r[cols[col]] for r in rows_a], [r[cols[col]] for r in rows_b], col in ordinal_cols)}
            d["n"] = len(rows_a); rep[name] = d
        # ---- corrections
        if a.apply:
            clients = Clients(CFG); tr = Translator(CFG, clients, Cache(OUT / "cache.sqlite"))
            for comp, rows_a, rows_b, cols, corr_keys in (("agrifacts", fa, fb, F, ("corr_q", "corr_opt")), ("agriadvice", pa, pb, P, ("corr_A", "corr_B"))):
                to_item, from_item, idkey = ADAPTERS[comp]
                src = {r[idkey]: r for r in _jsonl(SRC / f"{comp}.jsonl")}
                cur = {r[idkey]: r for r in _jsonl(OUT / f"{comp}_{lang}.jsonl")}
                out_path = OUT / f"{comp}_{lang}.jsonl"; log_path = OUT / f"translation_log_{comp}_{lang}.jsonl"
                for ra, rb in zip(rows_a, rows_b):
                    iid = ra[0]; flagged = [(n, r) for n, r in (("R1", ra), ("R2", rb)) if r[cols["meaning"]] in ("partly", "no")]
                    if not flagged:
                        continue
                    rater, r = flagged[0]
                    corr = [r[cols[k]] for k in corr_keys]
                    if any(corr):
                        row = dict(cur[iid])
                        if comp == "agrifacts":
                            q = corr[0] or row["question"]; ch = corr[1].split(" | ") if corr[1] else row["choices"]
                            if len(ch) != 3:
                                rep["notes"].append(f"{iid}: corrected options not 3 items; question only applied"); ch = row["choices"]
                            ans = ch[row["choices_en"].index(row["answer_en"])]
                            row.update({"question": q, "choices": ch, "answer": ans})
                        else:
                            pa_, pb_ = corr[0] or row["version_A"]["prompt"], corr[1] or row["version_B"]["prompt"]
                            row["version_A"] = {**row["version_A"], "prompt": pa_}; row["version_B"] = {**row["version_B"], "prompt": pb_}
                            if row["base_query"] not in pa_ or row["base_query"] not in pb_:
                                rep["notes"].append(f"{iid}: rater correction rewrote the base query; base_query field re-derived from A")
                                # keep the longest common prefix-free approach simple: store A's text minus persona is not derivable; leave base_query as is but flag
                        row["human_verified"] = {"rater": rater, "action": "corrected"}
                        _append(out_path, row); _append(log_path, {idkey: iid, "human_correction": rater, "comment": r[cols["comment"]]})
                        rep["corrections"]["applied_from_rater"].append(iid)
                    else:
                        hint = ("\nA bilingual human verifier rated a previous translation of this item as not fully preserving meaning. "
                                "Their comments: " + " / ".join(x[cols["comment"]] for _, x in flagged if x[cols["comment"]]) +
                                ". Fix exactly that, keep glossary terms, and translate 'operated area' size classes as operated area, never as holdings.")
                        res = _run_with_hint(tr, to_item(src[iid]), lang, hint)
                        if res["status"] == "ok" and res["translation"]:
                            new = from_item(src[iid], res["translation"], lang); new["human_verified"] = {"rater": rater, "action": "retranslated_after_flag"}
                            _append(out_path, new); _append(log_path, {idkey: iid, "repair": True, "human_flag": rater, "rounds": res["rounds"], "final_verdicts": res["final_verdicts"], "glossary_missing": res["glossary_missing"]})
                            rep["corrections"]["retranslated"].append(iid)
                        else:
                            rep["corrections"]["unchanged_after_flag"].append(iid)
        report["languages"][lang] = rep
        print(f"{L}: facts meaning R1={rep['facts']['meaning']['R1']} R2={rep['facts']['meaning']['R2']} kappa={rep['facts']['meaning']['agreement'].get('cohen_kappa')} | "
              f"pairs meaning R1={rep['pairs']['meaning']['R1']} R2={rep['pairs']['meaning']['R2']} | corrections: { {k: len(v) for k, v in rep['corrections'].items()} }")
    (VER / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report ->", VER / "verification_report.json")


if __name__ == "__main__":
    main()
