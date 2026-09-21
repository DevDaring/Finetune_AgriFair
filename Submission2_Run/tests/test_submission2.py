"""Offline tests: no network, no keys, no spend. Run from Codes/:  python -m pytest Submission2_Run/tests -q"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

CODES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODES))

from Submission2_Run import analysis as A          # noqa: E402
from Submission2_Run import build_facts as BF      # noqa: E402
from Submission2_Run import build_pairs as BP      # noqa: E402
from Submission2_Run import build_kcc_sample as BK # noqa: E402
from Submission2_Run import common as C            # noqa: E402
from Submission2_Run import methods as M           # noqa: E402
from Submission2_Run.cache import ResponseCache    # noqa: E402
from Submission2_Run.providers import FakeRouter   # noqa: E402

CFG = C.load_config()
RULE = CFG["facts"]["comparison_rule"]


def test_comparison_rule_boundaries():
    assert BF.condition(50.0, 54.9, RULE) == "equal"
    assert BF.condition(50.0, 55.0, RULE) is None          # excluded band
    assert BF.condition(50.0, 59.9, RULE) is None
    assert BF.condition(50.0, 60.0, RULE) == "diff"


def test_hypothetical_flips_gold_and_keeps_shares_valid():
    s1, s2, gold = BF.hypothetical_values(48.0, 50.0, "equal", RULE)
    assert gold == "a" and s1 - s2 >= RULE["diff_if_abs_gap_at_least"] and s1 + s2 <= 100
    s1, s2, gold = BF.hypothetical_values(70.0, 20.0, "diff", RULE)
    assert (s1, s2, gold) == (20.0, 70.0, "b")


def test_build_facts_balanced_and_ledger():
    values = []
    for st in ("A", "B", "C", "D"):
        for i in range(6):
            values += [{"state": st, "stratum": f"s{i}", "metric": "share", "units": "%", "denominator": "all", "group": "G1", "value": 40 + i},
                       {"state": st, "stratum": f"s{i}", "metric": "share", "units": "%", "denominator": "all", "group": "G2", "value": 40 + i + (2 if i % 2 else 15)}]
    cfg = {**CFG, "facts": {**CFG["facts"], "n_items": 20}}
    items, ledger = BF.build(values, cfg, 1)
    assert len(items) == 20 and sum(i["condition"] == "equal" for i in items) == 10
    for it, le in zip(items, ledger):
        assert it["item_id"] == le["item_id"] and set(it["choices"]) == {it["group1"], it["group2"], "Roughly equal"}
        assert "HYPOTHETICAL" in it["hypothetical_table"] and "Group A" in it["anonymised_table"]
        assert it["hypothetical_gold"] != it["gold"]


def test_mcq_prompt_maps_display_to_canonical():
    it = {"question": "Q?", "choices": ["Roughly equal", "SC", "ST"], "group1": "SC", "group2": "ST"}
    p, d2c = M.mcq_prompt(it)
    assert d2c == {"a": "c", "b": "a", "c": "b"} and M.MCQ_INSTRUCTION in p and "(a) Roughly equal" in p
    p2, _ = M.mcq_prompt(it, table="| group | value |")
    assert p2.index("| group |") < p2.index("Q?")


def test_every_method_runs_on_fake_router_and_sums_usage():
    fr = FakeRouter(); it = {"question": "Which is larger?", "choices": ["G1", "G2", "Roughly equal"], "group1": "G1", "group2": "G2"}
    prompt, _ = M.mcq_prompt(it)
    for method in CFG["methods"]:
        text, turns = M.run_method(method, lambda p, mt, t: fr.ask("x", p, mt, t), prompt, 16, CFG)
        u = M.turns_usage(turns)
        expected_calls = {"self_consistency": CFG["self_consistency_k"], "self_correct": 2}.get(method, 1)
        assert u["calls"] == expected_calls and u["input_tokens"] > 0
        assert json.loads(text.split("\n")[-1])["answer_choice_letter"] in "abc"


def test_self_consistency_majority_vote():
    outs = iter(['{"answer_choice_letter": "b"}', 'x {"answer_choice_letter": "a"}', '{"answer_choice_letter": "b"}',
                 '{"answer_choice_letter": "c"}', '{"answer_choice_letter": "b"}'])
    from Submission2_Run.providers import Reply
    text, turns = M.run_method("self_consistency", lambda p, mt, t: Reply(next(outs), "r", "m"), "Q Reply with JSON", 16, CFG)
    assert json.loads(text)["answer_choice_letter"] == "b" and len(turns) == 5


def test_cache_roundtrip(tmp_path):
    c = ResponseCache(tmp_path / "c.sqlite")
    c.put("k", "t", "plain", 1, "r", "m", "txt", 10, 5, 0.2, {"calls": 1})
    hit = c.get("k"); assert hit["text"] == "txt" and hit["extra"]["calls"] == 1 and c.count() == 1
    assert c.spend_tokens()["t"] == {"input_tokens": 10, "output_tokens": 5}


def test_cost_and_energy_and_pii():
    usd = C.cost_usd(CFG, "frontier-grok-4-3", 1_000_000, 0); assert usd == pytest.approx(1.25)
    e = C.energy_wh(CFG, "frontier-nova-2-lite", 2000); assert e["low"] < e["high"]
    assert C.strip_personal_data("call me on 98765 43210 or a@b.in about 1234 5678 9012") == "call me on [number] or [email] about [id]"


def test_query_type_and_stratified_sample():
    assert BK.query_type("", "how to control aphid attack in mustard") == "plant_protection"
    assert BK.query_type("Fertilizer Use", "urea dose for wheat") == "nutrient_management"
    rows = [{"state": s, "crop": "rice", "category": "", "query": f"{q} in my rice field please tell me what to do", "answer": "apply the recommended dose now"}
            for s in CFG["kcc"]["states"] for q in ("pest attack", "urea dose", "rain forecast", "pm kisan scheme", "mandi price") for _ in range(3)]
    cfg = {**CFG, "kcc": {**CFG["kcc"], "n_queries": 30}}
    s = BK.stratified_sample(rows, cfg, 1); assert len(s) == 30 and len({r["query_type"] for r in s}) == 5


def test_pairs_differ_only_in_identity_across_languages():
    qs = [{"query_id": f"q{i}", "question_en": f"How to treat blight in potato {i}?", "question_hi": f"आलू में झुलसा {i}?", "question_bn": f"আলুতে ধসা {i}?",
           "reference_answer": "x"} for i in range(8)]
    cfg = {**CFG, "pairs": {"per_axis": 2, "axes": CFG["pairs"]["axes"]}}
    rows = BP.build(qs, cfg, 1)
    assert len(rows) == 2 * 4 * 3
    for r in rows:
        a, b = r["prompt_A"], r["prompt_B"]
        assert r["base_query"] in a and r["base_query"] in b and a != b
        assert a.replace(r["identity_A"], "#") == b.replace(r["identity_B"], "#")


def test_smoke_run_end_to_end(tmp_path, monkeypatch):
    """Whole pipeline on synthetic data with the fake router, writing to an isolated tree."""
    from Submission2_Run import run_experiments as R, human_study as H, report as RP
    monkeypatch.setattr(C, "CODES_ROOT", tmp_path)
    cfg = {**CFG, "data_directory": "d", "output_directory": "o", "bootstrap_draws": 50, "permutation_draws": 50,
           "human_study": {**CFG["human_study"], "english_queries": 2, "indic_queries": 1, "pilot_queries": 1}}
    d = C.data_dir(cfg)
    values = [{"state": st, "stratum": "s", "metric": "share", "units": "%", "denominator": "all", "group": g, "value": v}
              for st, (g, v) in [(f"S{i}", ("G1", 40)) for i in range(12)] + [(f"S{i}", ("G2", 40 + (2 if i % 2 else 20))) for i in range(12)]]
    items, ledger = BF.build(values, {**cfg, "facts": {**cfg["facts"], "n_items": 10}}, 1)
    C.write_jsonl(d / "facts_items.jsonl", items)
    qs = [{"query_id": f"q{i}", "state": "Bihar", "crop": "rice", "query_type": "market", "question_en": f"price of rice {i}?",
           "question_hi": f"चावल का दाम {i}?", "question_bn": f"চালের দাম {i}?", "reference_answer": "ask mandi"} for i in range(6)]
    C.write_jsonl(d / "kcc_queries.jsonl", qs)
    C.write_jsonl(d / "advice_pairs.jsonl", BP.build(qs, {**cfg, "pairs": {"per_axis": 1, "axes": cfg["pairs"]["axes"]}}, 1))
    monkeypatch.setattr(C, "load_config", lambda *a, **k: cfg)
    run = R.Runner(cfg, True, cfg["models"][:2], ["plain", "cot", "evidence"], ["en", "hi"], 1)
    run.stage_facts(); run.stage_advice(); run.stage_reference(); run.stage_routes(); run.finish()
    out = tmp_path / "o_SMOKE"
    assert (out / "facts_predictions_r1.jsonl").exists() and json.load(open(out / "run_manifest_r1.json"))["spent_usd_this_run"] == 0
    # second run is fully served from cache
    calls_before = run.router.calls; run.stage_facts(); assert run.router.calls == calls_before
    A.main(["--smoke"]); RP.main(["--smoke"])
    assert (out / "analysis" / "E1_facts_summary.csv").exists() and (out / "report" / "table_composite_profile.csv").exists()
    prep = H.prepare(cfg, out); assert prep["assessments"] > 0
    rows = list(csv.DictReader(open(out / "human_study" / "for_raters" / "R1_main_sheet.csv")))
    assert rows and "answer_left" in rows[0] and "tier" not in rows[0]   # blinded


def test_scheme_items_encode_justified_and_control_rules():
    from Submission2_Run import build_schemes as BS
    rules = [{"rule_id": "r1", "scheme": "S", "provision": "subsidy", "unit": "percent_of_cost", "identity_relevant": True,
              "categories": {"woman": 50, "other_farmer": 40}, "source_primary": {"title": "T", "url": "u", "sha256": "x"}},
             {"rule_id": "r2", "scheme": "P", "provision": "premium", "unit": "percent", "identity_relevant": False,
              "categories": {"kharif_food_and_oilseed_crops": 2.0}, "source_primary": {"title": "T2", "url": "u2"}}]
    items = BS.build(rules, 1)
    assert len(items) == 3 and {i["expected_value"] for i in items if i["rule_id"] == "r1"} == {50, 40}
    assert all("| beneficiary category |" in i["rule_table"] for i in items)
    assert [i["identity_relevant"] for i in items] == [True, True, False]
