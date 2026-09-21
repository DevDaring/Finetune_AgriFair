"""Update the dataset card: language tags, configs for the translated files, and a section that
describes how the translations were produced and how to read the log."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict


def _log_stats(out: Path, comp: str, lang: str) -> Dict:
    p = out / f"translation_log_{comp}_{lang}.jsonl"
    rows = [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []
    idkey = "id" if comp == "agrifacts" else "pair_id"
    last = {}                                   # a repaired item has several entries; the last completed one describes the released row
    for r in rows:
        if r.get("final_verdicts"):
            last[r.get(idkey)] = r
    ok = list(last.values())
    all_ok = sum(1 for r in ok if all(v == "OK" for v in r["final_verdicts"].values()))
    majority_ok = sum(1 for r in ok if sum(v == "OK" for v in r["final_verdicts"].values()) >= 2)
    rounds = [max([x.get("round", 1) for x in r["rounds"]] or [1]) for r in ok]
    repaired = sum(1 for r in ok if r.get("repair"))
    return {"n": len(ok), "all_ok": all_ok, "majority_ok": majority_ok, "repaired": repaired,
            "mean_rounds": round(sum(rounds) / len(rounds), 2) if rounds else 0}


def _spot(out: Path) -> str:
    p = out / "quality_log.jsonl"
    scores = []
    for l in (p.open(encoding="utf-8") if p.exists() else []):
        scores += [x["score"] for x in json.loads(l).get("spot", []) if x.get("score", 0) > 0]
    return f"mean {sum(scores)/len(scores):.2f}/5 over {len(scores)} sampled items, none rated 3 or below" if scores and min(scores) > 3 else (f"mean {sum(scores)/len(scores):.2f}/5 over {len(scores)} sampled items" if scores else "not run")


def updated_readme(readme: str, cfg: Dict, out: Path) -> str:
    # --- front matter: languages and configs
    fm_end = readme.index("\n---", 4)
    fm, body = readme[: fm_end + 4], readme[fm_end + 4:]
    fm = re.sub(r"language:\n(- \w+\n)+", "language:\n- en\n- hi\n- bn\n", fm)
    cfg_block = "configs:\n- config_name: agrifacts\n  data_files: agrifacts.jsonl\n- config_name: agriadvice\n  data_files: agriadvice.jsonl\n"
    for comp in ("agrifacts", "agriadvice"):
        for lang in cfg["languages"]:
            cfg_block += f"- config_name: {comp}_{lang}\n  data_files: {comp}_{lang}.jsonl\n"
    fm = re.sub(r"configs:\n(- config_name: .*\n  data_files: .*\n)+", cfg_block, fm)
    if "multilingual" not in fm:
        fm = fm.replace("- difference-awareness\n", "- difference-awareness\n- multilingual\n- hindi\n- bengali\n")

    # --- body section (replace if present)
    body = re.sub(r"\n## Hindi and Bengali translations.*?(?=\n## |\Z)", "", body, flags=re.S)
    m = cfg["models"]
    rev = ", ".join(f"{r['provider']}/{r['model']}" for r in m["reviewers"])
    enh = " and ".join(f"{e['provider']}/{e['model']}" for e in m["enhance"])
    lines = ["\n## Hindi and Bengali translations\n",
             "Every AgriFacts question with its three options, and every AgriAdvice pair, is also released in Hindi "
             "(`*_hi.jsonl`) and Bengali (`*_bn.jsonl`). Rows keep the English schema and add `language` plus the English "
             "originals (`question_en`, `choices_en`, `answer_en`; `base_query_en`, `version_A_en`, `version_B_en`).\n",
             "**How they were made.** Machine translation with a fixed glossary for census categories (e.g. Scheduled Castes -> "
             "अनुसूचित जाति / তফসিলি জাতি; Roughly equal -> लगभग बराबर / প্রায় সমান), followed by up to "
             f"{cfg['max_rounds']} rounds of independent review and revision:\n",
             f"- translate: {m['translate']['provider']}/{m['translate']['model']}\n",
             f"- review (each round, in order): {rev}\n",
             f"- revise after a REVISE verdict: {enh}, alternating\n",
             "- stop when every reviewer returns OK, or after the last round.\n",
             "For AgriAdvice the base question and the two persona wrappers were translated separately and recomposed, so "
             "version A and version B still differ only in the identity span. Option order is unchanged and the answer is the "
             "translated option at the same index.\n",
             "**What was not done.** No human post-editing has been applied to the released files. Treat the translations as "
             "high-quality machine output with model review, not as expert-verified text; for a human-rated study, verify a "
             "sample first. Numbers are written in ASCII digits in all three languages. A separate automated quality check "
             f"(script coverage, numbers, glossary, structure, and a GPT-4o fidelity rating on random samples: {_spot(out)}) "
             "re-translated the rows it flagged. The per-item log (`translation_log_<component>_<lang>.jsonl`, in the code repository) records every "
             "reviewer verdict and the model that served each step.\n",
             "| File | Items | All 3 reviewers OK | At least 2 of 3 OK | Re-translated after quality check | Mean review rounds |\n|---|---|---|---|---|---|\n"]
    for comp in ("agrifacts", "agriadvice"):
        for lang in cfg["languages"]:
            s = _log_stats(out, comp, lang)
            lines.append(f"| `{comp}_{lang}.jsonl` | {s['n']} | {s['all_ok']} | {s['majority_ok']} | {s['repaired']} | {s['mean_rounds']} |\n")
    lines += _verification_section(out)
    lines.append("\nGlossary and pipeline code: `Codes/Dataset_Translation/` in the linked repository.\n")
    return fm + body.rstrip() + "\n" + "".join(lines)


def _verification_section(out: Path) -> list:
    rep_p = out.parents[1] / "Submission2" / "verification" / "verification_report.json"
    if not rep_p.exists():
        return []
    rep = json.load(open(rep_p, encoding="utf-8"))
    L = {"hi": "Hindi", "bn": "Bengali"}
    lines = ["\n**Human verification of a sample.** Two bilingual agricultural professionals independently rated a stratified sample "
             "(100 AgriFacts questions and 50 AgriAdvice pairs per language, drawn by seed across every axis and condition) for "
             "meaning preservation, glossary use, identity-only difference and naturalness. Rows either rater marked as not fully "
             "preserving meaning were corrected from the rater's text or re-translated; the `human_verified` field marks them. "
             "A systematic defect the raters found (operated-area size classes rendered as holdings) was then fixed across all "
             "affected rows. No other rows were altered.\n",
             "| Language | Set | Meaning preserved R1 / R2 | Raw agreement | Cohen's kappa | Corrected |\n|---|---|---|---|---|---|\n"]
    for lang, r in rep["languages"].items():
        for name, key, n in (("facts", "facts", 100), ("pairs", "pairs", 50)):
            m = r[key]["meaning"]; c = r["corrections"]
            y1, y2 = m["R1"].get("yes", 0), m["R2"].get("yes", 0)
            corr = sum(1 for i in c["applied_from_rater"] + c["retranslated"] if (i.startswith("agrifacts") == (key == "facts")))
            lines.append(f"| {L[lang]} | {name} | {y1}/{r[key]['n']} / {y2}/{r[key]['n']} | {m['agreement']['raw_agreement']} | {m['agreement']['cohen_kappa']} | {corr} |\n")
    lines.append("\nKappa on \"meaning preserved\" is low where one rater flagged items the other accepted (near-unanimous marginals); raw agreement "
                 "is the more informative figure here. Naturalness (1-3) agreement was low in both languages, as expected for a stylistic judgement.\n")
    return lines
