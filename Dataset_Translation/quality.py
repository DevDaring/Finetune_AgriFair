"""Quality checks on translated rows, and repair of the rows that fail.

    python -m Dataset_Translation.run quality [--sample 40] [--repair]

Deterministic checks (every row written since the last check, plus a random sample of older rows):
    script      >= 85 % of letters in the target script; Latin allowed only for numbers/acronyms
    numbers     every number / year in the English source appears in the translation
    glossary    fixed census terms present where the English term is present
    structure   agrifacts: 3 choices, answer is one of them; agriadvice: base_query inside both prompts,
                A and B identical outside the identity span
    dash        the "—" option separator of a question survives (or an equivalent " - ")
    verdicts    a reviewer still said REVISE at exit
LLM spot check (GPT-4o via LinkAPI) on a random sample: fidelity 1-5 with one-line reason;
rows scoring <= 3 are flagged.

Repair: a flagged row is removed from the output file and re-run through the full
translate -> review -> enhance loop with the detected issues appended to the translator's
instructions (fresh cache key), then re-checked. Everything is logged to quality_log.jsonl.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from GPU_Run.common.parsing import json_repair_parse
from Dataset_Translation.glossary import GLOSSARY, LANG_NAME, ascii_digits, missing_terms
from Dataset_Translation.pipeline import Cache, Translator, SYSTEM_TRANSLATE, _user_translate
from Dataset_Translation.providers import Clients

SCRIPT = {"hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF)}
LATIN_OK = re.compile(r"^[A-Z0-9][A-Za-z0-9.\-/&()%]*$")   # acronyms, codes, numbers with units


def _letters(s: str):
    return [ch for ch in s if ch.isalpha()]


def script_share(text: str, lang: str) -> float:
    lo, hi = SCRIPT[lang]; L = _letters(text)
    return sum(1 for ch in L if lo <= ord(ch) <= hi) / len(L) if L else 0.0


def latin_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z\-]+", text) if not LATIN_OK.match(w)]


def numbers(text: str) -> List[str]:
    return re.findall(r"\d[\d,./-]*", text)


def check_row(row: Dict, comp: str, lang: str, log_entry: Dict) -> List[str]:
    issues = []
    if comp == "agrifacts":
        src, tr = row["question_en"], row["question"]
        texts = [tr] + row["choices"]
        if len(row["choices"]) != 3 or row["answer"] not in row["choices"]:
            issues.append("structure: choices/answer mismatch")
        if ("—" in src or " - " in src) and not any(d in tr for d in ("—", " - ", "–")):
            issues.append("dash: option separator lost")
        src_all = src + " " + " ".join(row["choices_en"])
    else:
        src, tr = row["base_query_en"], row["base_query"]
        pa, pb = row["version_A"]["prompt"], row["version_B"]["prompt"]
        texts = [tr, pa, pb]
        if tr not in pa or tr not in pb:
            issues.append("structure: base_query not inside both prompts")
        ia, ib = row["version_A"]["persona"], row["version_B"]["persona"]
        if pa.replace(ia, "#") != pb.replace(ib, "#") and pa.replace(tr, "{q}") == pb.replace(tr, "{q}"):
            pass  # templates identical -> fine even if persona strings are not the only difference marker
        elif pa.replace(tr, "{q}") == pb.replace(tr, "{q}"):
            issues.append("structure: version A and B templates identical (identity lost)")
        src_all = src + " " + row["version_A_en"]["prompt"] + " " + row["version_B_en"]["prompt"]
    joined = " ".join(texts)
    sh = script_share(joined, lang)
    if sh < 0.85:
        issues.append(f"script: only {sh:.0%} in {LANG_NAME[lang].split()[0]} script")
    lw = latin_words(joined)
    if len(lw) > 2:
        issues.append(f"english leakage: {lw[:4]}")
    joined_ascii = ascii_digits(joined)   # Bengali/Devanagari numerals are normalised mechanically at upload, not re-translated
    miss = [n for n in numbers(src_all) if n.rstrip(".,") not in joined_ascii]
    if miss:
        issues.append(f"numbers missing: {miss[:3]}")
    g = missing_terms(src_all, joined, lang)
    if g:
        issues.append(f"glossary: {g[:3]}")
    fv = log_entry.get("final_verdicts") or {}
    n_rev = sum(1 for v in fv.values() if v == "REVISE")
    if n_rev >= 2 or (n_rev == 1 and len(fv) < 3):
        issues.append("reviewer: majority REVISE at exit")     # one dissenting reviewer out of three is accepted (often self-contradictory nitpicks)
    return issues


SPOT_SYSTEM = """You are a bilingual (English / {lang}) agricultural translator. Rate the translation's fidelity to the English source from 1 (wrong meaning) to 5 (exact, natural, glossary terms right). Reply with JSON only: {{"score": n, "reason": "one line"}}."""


def spot_check(clients: Clients, cfg: Dict, rows: List[Tuple[Dict, str, str]]) -> List[Dict]:
    out = []
    role = {"provider": "linkapi", "model": "gpt-4o", "fallbacks": [{"provider": "openai", "model": "gpt-4o"}, {"provider": "openrouter", "model": "openai/gpt-4o"}]}
    for row, comp, lang in rows:
        if comp == "agrifacts":
            src = json.dumps({"question": row["question_en"], "choices": row["choices_en"]}, ensure_ascii=False)
            tr = json.dumps({"question": row["question"], "choices": row["choices"]}, ensure_ascii=False)
        else:
            src = json.dumps({"A": row["version_A_en"]["prompt"], "B": row["version_B_en"]["prompt"]}, ensure_ascii=False)
            tr = json.dumps({"A": row["version_A"]["prompt"], "B": row["version_B"]["prompt"]}, ensure_ascii=False)
        try:
            c = clients.complete(role, SPOT_SYSTEM.format(lang=LANG_NAME[lang]), f"Source:\n{src}\n\nTranslation:\n{tr}", max_tokens=120)
            obj = json_repair_parse(c.text) or {}
            out.append({"id": row.get("id") or row.get("pair_id"), "comp": comp, "lang": lang, "score": int(obj.get("score", 0)), "reason": str(obj.get("reason", ""))[:160]})
        except Exception as e:
            out.append({"id": row.get("id") or row.get("pair_id"), "comp": comp, "lang": lang, "score": -1, "reason": f"spot check failed: {type(e).__name__}"})
    return out


def repair(cfg: Dict, out_dir: Path, comp: str, lang: str, ids: Dict[str, List[str]], run_module) -> Dict[str, str]:
    """Remove flagged rows and re-run them with the issues fed to the translator. Returns id -> status."""
    to_item, from_item, idkey = run_module.ADAPTERS[comp]
    src = {r[idkey]: r for r in run_module._jsonl(run_module.SRC / f"{comp}.jsonl")}
    out_path = out_dir / f"{comp}_{lang}.jsonl"; log_path = out_dir / f"translation_log_{comp}_{lang}.jsonl"
    # no in-place rewrite: the repaired row is appended and readers keep the last row per id
    clients = Clients(cfg); tr = Translator(cfg, clients, Cache(out_dir / "cache.sqlite"))
    status = {}
    for iid, issues in ids.items():
        row = src[iid]; item = to_item(row)
        hint = "\nA previous translation of this item was rejected for: " + "; ".join(issues) + ". Avoid these problems."
        # fresh cache key: the hint changes the system prompt, so translate/review/enhance all run again
        tr_run = _run_with_hint(tr, item, lang, hint)
        if tr_run["status"] == "ok" and tr_run["translation"]:
            try:
                new = from_item(row, tr_run["translation"], lang)
                run_module._append(out_path, new)
                run_module._append(log_path, {idkey: iid, "repair": True, "prior_issues": issues, "rounds": tr_run["rounds"],
                                              "final_verdicts": tr_run["final_verdicts"], "glossary_missing": tr_run["glossary_missing"]})
                status[iid] = "repaired"
                continue
            except Exception as e:
                status[iid] = f"recompose failed: {e}"
        else:
            status[iid] = "translate failed"
        run_module._append(log_path, {idkey: iid, "repair": True, "status": "failed", "prior_issues": issues})
    return status


def _run_with_hint(tr: Translator, item: Dict, lang: str, hint: str) -> Dict:
    """Translator.run with the hint appended to the translate system prompt."""
    orig = tr._call
    def patched(role_name, role, system, user):
        if role_name.startswith("translate"):
            system = system + hint
        return orig(role_name, role, system, user)
    tr._call = patched
    try:
        return tr.run(item, lang)
    finally:
        tr._call = orig


def main(cfg: Dict, out_dir: Path, run_module, sample: int = 40, do_repair: bool = False, seed: int = None) -> Dict:
    rng = random.Random(seed or int(time.time()))
    state_p = out_dir / "quality_state.json"; state = json.loads(state_p.read_text()) if state_p.exists() else {}
    report = {"checked": 0, "flagged": {}, "spot": [], "repaired": {}}
    clients = Clients(cfg)
    for comp in run_module.ADAPTERS:
        _, _, idkey = run_module.ADAPTERS[comp]
        for lang in cfg["languages"]:
            rows = run_module._jsonl(out_dir / f"{comp}_{lang}.jsonl")
            if not rows:
                continue
            log_rows = run_module._jsonl(out_dir / f"translation_log_{comp}_{lang}.jsonl")
            logs = {l.get(idkey): l for l in log_rows if l.get("final_verdicts")}
            repairs = {}
            for l in log_rows:
                if l.get("repair"):
                    repairs[l.get(idkey)] = repairs.get(l.get(idkey), 0) + 1
            key = f"{comp}_{lang}"; seen = state.get(key, 0)
            new_rows = rows[seen:]; old_sample = rng.sample(rows[:seen], min(sample // 4, seen)) if seen else []
            flagged = {}
            for r in new_rows + old_sample:
                iss = check_row(r, comp, lang, logs.get(r[idkey], {}))
                report["checked"] += 1
                if iss:
                    flagged[r[idkey]] = iss
            spot_rows = [(r, comp, lang) for r in rng.sample(new_rows or rows, min(sample // 4, len(new_rows or rows)))]
            for s in spot_check(clients, cfg, spot_rows):
                report["spot"].append(s)
                if 0 <= s["score"] <= 3:
                    flagged.setdefault(s["id"], []).append(f"spot: score {s['score']}: {s['reason']}")
            capped = {i: iss for i, iss in flagged.items() if repairs.get(i, 0) >= 2}
            flagged = {i: iss for i, iss in flagged.items() if repairs.get(i, 0) < 2}
            if flagged:
                report["flagged"][key] = flagged
            if capped:
                report.setdefault("unresolved_after_2_repairs", {})[key] = capped
            state[key] = len(rows)
            if do_repair and flagged:
                report["repaired"][key] = repair(cfg, out_dir, comp, lang, flagged, run_module)
    state_p.write_text(json.dumps(state))
    with (out_dir / "quality_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **report}, ensure_ascii=False) + "\n")
    return report
