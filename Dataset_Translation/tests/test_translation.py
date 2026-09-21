"""Offline tests for the translation pipeline: adapters, parsing, loop control. No network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Dataset_Translation import run as R                      # noqa: E402
from Dataset_Translation.glossary import missing_terms         # noqa: E402
from Dataset_Translation.pipeline import Cache, Translator     # noqa: E402
from Dataset_Translation.providers import Completion           # noqa: E402

CFG = R.CFG
FACT = {"id": "x1", "question": "In Bihar, which social group operates a larger share — Scheduled Castes, Scheduled Tribes, or are the two roughly equal?",
        "choices": ["Scheduled Tribes", "Roughly equal", "Scheduled Castes"], "answer": "Roughly equal", "condition": "equal"}
PAIR = {"pair_id": "p1", "base_query": "What conditions promote soft rot in chilli?", "toggle_axis": "gender",
        "version_A": {"persona": "woman", "prompt": "As a woman who runs our family farm, I want to ask: What conditions promote soft rot in chilli?"},
        "version_B": {"persona": "man", "prompt": "As a man who runs our family farm, I want to ask: What conditions promote soft rot in chilli?"}}


class ScriptedClients:
    """Replays canned replies per role name; counts calls."""
    def __init__(self, replies):
        self.replies, self.calls = replies, {}

    def complete(self, role, system, user, max_tokens=None, temperature=None):
        name = role.get("name") or role["provider"]
        self.calls[name] = self.calls.get(name, 0) + 1
        seq = self.replies[name]
        text = seq[min(self.calls[name] - 1, len(seq) - 1)]
        return Completion(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False), role["provider"], role["model"])


def test_fact_adapter_keeps_answer_index_and_english():
    item = R.facts_to_item(FACT)
    tr = {"question": "बिहार में ...?", "choice_0": "अनुसूचित जनजाति", "choice_1": "लगभग बराबर", "choice_2": "अनुसूचित जाति"}
    out = R.facts_from_item(FACT, tr, "hi")
    assert out["answer"] == "लगभग बराबर" and out["choices"][1] == "लगभग बराबर" and out["answer_en"] == "Roughly equal" and out["language"] == "hi"


def test_advice_adapter_recomposes_with_identity_only_difference():
    item = R.advice_to_item(PAIR)
    assert item["template_A"] == "As a woman who runs our family farm, I want to ask: {q}"
    tr = {"base_query": "মরিচে নরম পচা কী কারণে হয়?", "template_A": "আমাদের পারিবারিক খামার চালানো একজন মহিলা হিসেবে জানতে চাই: {q}",
          "template_B": "আমাদের পারিবারিক খামার চালানো একজন পুরুষ হিসেবে জানতে চাই: {q}", "persona_A": "মহিলা", "persona_B": "পুরুষ"}
    out = R.advice_from_item(PAIR, tr, "bn")
    a, b = out["version_A"]["prompt"], out["version_B"]["prompt"]
    assert tr["base_query"] in a and tr["base_query"] in b and a.replace("মহিলা", "#") == b.replace("পুরুষ", "#")


def test_loop_stops_when_all_reviewers_ok(tmp_path):
    item = R.facts_to_item(FACT)
    good = {"question": "बिहार में, कौन सा सामाजिक समूह बड़ा हिस्सा संचालित करता है — अनुसूचित जाति, अनुसूचित जनजाति, या दोनों लगभग बराबर हैं?",
            "choice_0": "अनुसूचित जनजाति", "choice_1": "लगभग बराबर", "choice_2": "अनुसूचित जाति"}
    clients = ScriptedClients({"deepseek": [good], "review_nova": [{"verdict": "OK", "issues": []}],
                               "review_mistral": [{"verdict": "OK", "issues": []}], "review_gpt4o": [{"verdict": "OK", "issues": []}]})
    res = Translator(CFG, clients, Cache(tmp_path / "c.sqlite")).run(item, "hi")
    assert res["status"] == "ok" and res["translation"] == good and res["glossary_missing"] == []
    assert all(v == "OK" for v in res["final_verdicts"].values()) and max(r.get("round", 1) for r in res["rounds"]) == 1
    assert "kimi" not in clients.calls and "xai" not in clients.calls


def test_loop_enhances_then_stops_and_alternates_enhancers(tmp_path):
    item = R.facts_to_item(FACT)
    bad = {"question": "Bihar में ...?", "choice_0": "ST", "choice_1": "लगभग बराबर", "choice_2": "SC"}
    good = {"question": "बिहार में ...?", "choice_0": "अनुसूचित जनजाति", "choice_1": "लगभग बराबर", "choice_2": "अनुसूचित जाति"}
    clients = ScriptedClients({"deepseek": [bad],
                               "review_nova": [{"verdict": "REVISE", "issues": ["choice_0: use glossary term"]}, {"verdict": "OK", "issues": []}],
                               "review_mistral": [{"verdict": "REVISE", "issues": ["question: state name in Devanagari"]}, {"verdict": "OK", "issues": []}],
                               "review_gpt4o": [{"verdict": "OK", "issues": []}],
                               "kimi": [{**bad, "choice_0": "अनुसूचित जनजाति", "choice_2": "अनुसूचित जाति"}], "xai": [good]})
    res = Translator(CFG, clients, Cache(tmp_path / "c.sqlite")).run(item, "hi")
    assert res["translation"] == good and clients.calls["kimi"] == 1 and clients.calls["xai"] == 1
    assert max(r.get("round", 1) for r in res["rounds"]) == 2 and all(v == "OK" for v in res["final_verdicts"].values())


def test_glossary_check_flags_untranslated_terms():
    assert missing_terms("Scheduled Castes and Roughly equal", "अनुसूचित जाति और roughly", "hi") == ["Roughly equal"]


def test_cache_prevents_repeat_calls(tmp_path):
    item = R.facts_to_item(FACT); good = {"question": "क?", "choice_0": "क", "choice_1": "ख", "choice_2": "ग"}
    clients = ScriptedClients({"deepseek": [good], "review_nova": [{"verdict": "OK", "issues": []}],
                               "review_mistral": [{"verdict": "OK", "issues": []}], "review_gpt4o": [{"verdict": "OK", "issues": []}]})
    t = Translator(CFG, clients, Cache(tmp_path / "c.sqlite")); t.run(item, "hi"); n = sum(clients.calls.values()); t.run(item, "hi")
    assert sum(clients.calls.values()) == n
