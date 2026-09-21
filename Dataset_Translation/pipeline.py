"""Translate -> review -> enhance loop for one item, with a persistent cache and a full log.

An item is a dict of named English strings, e.g. for AgriFacts
    {"question": ..., "choice_0": ..., "choice_1": ..., "choice_2": ...}
and for AgriAdvice
    {"base_query": ..., "template_A": "... {q} ...", "template_B": "... {q} ..."}
The AgriAdvice templates keep the literal placeholder {q}; the two prompts are recomposed
after translation, which guarantees that version A and version B still differ only in the
identity span (the property the dataset is built on).

Every model reply must be a JSON object with exactly the item's keys. The three roles:
    translate  -> JSON translation
    review     -> {"verdict": "OK" | "REVISE", "issues": [...]}
    enhance    -> JSON translation (revised)
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from GPU_Run.common.parsing import json_repair_parse
from Dataset_Translation.glossary import LANG_NAME, glossary_block, missing_terms
from Dataset_Translation.providers import Clients, Completion, ProviderError

# ------------------------------------------------------------------ prompts

SYSTEM_TRANSLATE = """You are a professional translator of Indian agricultural statistics and farm-extension text into {lang}.
Rules:
1. Translate meaning exactly. Add nothing, omit nothing, do not explain.
2. Use these fixed terms verbatim wherever the English term appears:
{glossary}
3. Keep numbers, years (e.g. 2015-16), percentages and the placeholder {{q}} exactly as they are.
4. Write Indian state and district names in their standard {lang} spelling.
5. Keep the sentence form: a question stays a question; an option label stays a short label; a first-person farmer's sentence stays first person, natural for a farmer to say.
6. Output one JSON object with exactly the same keys as the input and nothing else."""

SYSTEM_REVIEW = """You are a bilingual reviewer (English and {lang}) for an agricultural dataset.
Compare the English source with the {lang} translation, key by key. Check:
- meaning preserved, nothing added or omitted, numbers and years unchanged;
- the fixed glossary terms below are used verbatim where the English term appears;
- option labels are short labels, questions are questions, farmer sentences sound natural in {lang};
- the placeholder {{q}}, if present, is kept exactly;
- script is {lang} throughout (no untranslated English words except proper nouns, acronyms and numbers).
Glossary:
{glossary}
Reply with one JSON object only: {{"verdict": "OK" or "REVISE", "issues": ["key: short description", ...]}}. Use "OK" with an empty list when nothing needs to change."""

SYSTEM_ENHANCE = """You are a senior {lang} translator improving a draft translation of Indian agricultural text.
You receive the English source, the current {lang} draft and a reviewer's issues. Fix exactly what the issues describe and nothing else. Keep the fixed glossary terms verbatim:
{glossary}
Keep numbers, years and the placeholder {{q}} unchanged. Output one JSON object with exactly the same keys as the draft and nothing else."""


def _user_translate(item: Dict) -> str:
    return "Translate this JSON object:\n" + json.dumps(item, ensure_ascii=False, indent=1)


def _user_review(item: Dict, draft: Dict) -> str:
    return ("English source:\n" + json.dumps(item, ensure_ascii=False, indent=1) +
            "\n\nTranslation to check:\n" + json.dumps(draft, ensure_ascii=False, indent=1))


def _user_enhance(item: Dict, draft: Dict, issues: List[str]) -> str:
    return ("English source:\n" + json.dumps(item, ensure_ascii=False, indent=1) +
            "\n\nCurrent draft:\n" + json.dumps(draft, ensure_ascii=False, indent=1) +
            "\n\nReviewer issues:\n" + "\n".join(f"- {i}" for i in issues))


# ------------------------------------------------------------------ cache

class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False); self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, provider TEXT, model TEXT, text TEXT, "
                          "input_tokens INTEGER, output_tokens INTEGER, created_utc TEXT)")
        self.conn.commit()

    @staticmethod
    def key(role: str, model: str, system: str, user: str) -> str:
        return hashlib.sha256(f"{role}|{model}|{system}|{user}".encode()).hexdigest()

    def get(self, k: str) -> Optional[Completion]:
        with self.lock:
            row = self.conn.execute("SELECT provider, model, text, input_tokens, output_tokens FROM calls WHERE key=?", (k,)).fetchone()
        return Completion(row[2], row[0], row[1], row[3], row[4]) if row else None

    def put(self, k: str, c: Completion) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?)",
                              (k, c.provider, c.model, c.text, c.input_tokens, c.output_tokens, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
            self.conn.commit()


# ------------------------------------------------------------------ the loop

class Translator:
    def __init__(self, cfg: Dict, clients: Clients, cache: Cache):
        self.cfg, self.clients, self.cache = cfg, clients, cache
        self._enh_lock = threading.Lock(); self._enh_i = 0

    def _call(self, role_name: str, role: Dict, system: str, user: str) -> Completion:
        k = Cache.key(role_name, role["model"], system, user)
        hit = self.cache.get(k)
        if hit:
            return hit
        c = self.clients.complete(role, system, user)
        self.cache.put(k, c); return c

    def _next_enhancer(self) -> Tuple[str, Dict]:
        with self._enh_lock:
            roles = self.cfg["models"]["enhance"]; role = roles[self._enh_i % len(roles)]; self._enh_i += 1
        return f"enhance_{role['provider']}", role

    @staticmethod
    def _parse_obj(text: str, keys: List[str]) -> Optional[Dict]:
        obj = json_repair_parse(text)
        if not isinstance(obj, dict) or set(obj) != set(keys):
            return None
        return {k: str(obj[k]).strip() for k in keys}

    @staticmethod
    def _parse_verdict(text: str) -> Tuple[str, List[str]]:
        obj = json_repair_parse(text) or {}
        v = str(obj.get("verdict", "")).strip().upper()
        issues = [str(i) for i in obj.get("issues", []) if str(i).strip()]
        if v not in ("OK", "REVISE"):
            v = "REVISE" if issues else "OK"
        return v, issues

    def run(self, item: Dict, lang: str) -> Dict:
        """Returns {"translation": {...}, "rounds": [...], "final_verdicts": {...}, "glossary_missing": [...]}."""
        L = LANG_NAME[lang]; g = glossary_block(lang); keys = list(item)
        log: List[Dict] = []
        sys_t, sys_r, sys_e = (SYSTEM_TRANSLATE.format(lang=L, glossary=g), SYSTEM_REVIEW.format(lang=L, glossary=g),
                               SYSTEM_ENHANCE.format(lang=L, glossary=g))
        c = self._call("translate", self.cfg["models"]["translate"], sys_t, _user_translate(item))
        draft = self._parse_obj(c.text, keys)
        if draft is None:  # one retry with a stricter nudge, then give up on this item
            c = self._call("translate_retry", self.cfg["models"]["translate"], sys_t + "\nReturn ONLY the JSON object.", _user_translate(item))
            draft = self._parse_obj(c.text, keys)
        log.append({"stage": "translate", "provider": c.provider, "model": c.model, "ok": draft is not None})
        if draft is None:
            return {"translation": None, "rounds": log, "final_verdicts": {}, "glossary_missing": [], "status": "translate_failed"}

        verdicts = {}
        for rnd in range(1, int(self.cfg["max_rounds"]) + 1):
            changed = False
            for rev in self.cfg["models"]["reviewers"]:
                rev_name = rev["name"]
                rc = self._call(rev_name, rev, sys_r, _user_review(item, draft))
                verdict, issues = self._parse_verdict(rc.text)
                verdicts[rev_name] = verdict
                log.append({"stage": rev_name, "round": rnd, "provider": rc.provider, "model": rc.model, "verdict": verdict, "issues": issues})
                if verdict == "REVISE" and issues:
                    en_name, en_role = self._next_enhancer()
                    ec = self._call(en_name, en_role, sys_e, _user_enhance(item, draft, issues))
                    new = self._parse_obj(ec.text, keys)
                    log.append({"stage": en_name, "round": rnd, "provider": ec.provider, "model": ec.model, "ok": new is not None,
                                "changed": bool(new and new != draft)})
                    if new and new != draft:
                        draft, changed = new, True
            if not changed and all(v == "OK" for v in verdicts.values()):
                break
            if not changed:            # reviewers still object but enhancers produce nothing new: stop, record it
                break
        src_text = " ".join(item.values()); tr_text = " ".join(draft.values())
        return {"translation": draft, "rounds": log, "final_verdicts": verdicts,
                "glossary_missing": missing_terms(src_text, tr_text, lang), "status": "ok"}
