"""Offline tests: no GPU, no network, no checkpoint. Run from Codes/:

    python -m pytest Submission1_Code_Phase2/tests -q
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

CODES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODES))

from Submission1_Code_Phase2 import common as C            # noqa: E402
from Submission1_Code_Phase2 import r1_fresh_panel as R1    # noqa: E402
from Submission1_Code_Phase2 import r2_evidence as R2       # noqa: E402
from Submission1_Code_Phase2 import run_inference as RI     # noqa: E402

CFG = C.load_config()


# ---------------------------------------------------------------- template canonicalization

def test_template_key_ignores_content_but_keeps_structure():
    a = C.canonical_template("In Bihar, which social group operates a larger share of operated area — Scheduled Castes, Scheduled Tribes, or are the two roughly equal?",
                             ["Scheduled Castes", "Scheduled Tribes"], "Bihar")
    b = C.canonical_template("In Punjab, which social group operates a larger share of operated area — Other social groups, Scheduled Tribes, or are the two roughly equal?",
                             ["Other social groups", "Scheduled Tribes"], "Punjab")
    assert a == b, "same skeleton, different content -> one template family"
    c = C.canonical_template("Rank these two groups by their share of operated area in Bihar: Scheduled Castes or Scheduled Tribes.",
                             ["Scheduled Castes", "Scheduled Tribes"], "Bihar")
    assert a != c, "different sentence structure -> different family"


def test_template_key_never_reads_the_label():
    q = "In Bihar, which group is larger — A, B, or roughly equal?"
    assert C.template_id(q, ["A", "B"]) == C.template_id(q, ["A", "B"])   # deterministic, label-free


# ---------------------------------------------------------------- R1

def test_r1_inventory_reports_real_availability():
    inv = R1.inventory(CFG)
    assert inv["validated_comparisons"] > inv["used_by_benchmark"] > 0
    assert inv["unused_available"] == inv["validated_comparisons"] - inv["used_by_benchmark"]
    assert inv["feasible_total"] <= inv["plan_target_total"]
    if inv["feasible_total"] < inv["plan_target_total"]:
        assert inv["shortfalls"], "a shortfall must be named, not silently absorbed"


def test_r1_panel_uses_only_unused_cells_and_passes_invariants():
    panel, _ = R1.select(CFG)
    assert panel, "panel is empty"
    used = R1.used_cells()
    assert not ({p["source_cell"] for p in panel} & used), "a fresh comparison reused a benchmark cell"
    assert R1.validate_panel(panel, CFG) == []
    for p in panel:
        assert p["template_id_original"] != p["template_id_new"]
        assert p["gold_choice_text"] in p["choices"]


def test_r1_checker_worksheet_hides_answers(tmp_path):
    panel, _ = R1.select(CFG)
    R1.checker_sheets(panel[:5], tmp_path)
    text = (tmp_path / "fresh_panel_checker_worksheet.csv").read_text(encoding="utf-8")
    for p in panel[:5]:
        assert str(p["share1_pct"]) not in text and p["gold_choice_text"] not in text.split("\n")[0]
    assert "share_1_percent_recomputed" in text


# ---------------------------------------------------------------- R2

def test_r2_value_generation_respects_the_rule_and_bounds():
    rng = random.Random(1)
    for relation in R2.RELATIONS:
        for _ in range(60):
            v1, v2 = R2.make_values(relation, rng, CFG)
            gap = abs(v1 - v2)
            assert v1 + v2 <= CFG["r2"]["max_share_sum"]
            assert 0 < min(v1, v2) and max(v1, v2) <= 99.0
            if relation == "approximately_equal":
                assert 0 < gap < CFG["comparison_rule"]["equal_if_abs_gap_below"]
                assert v1 != v2, "equality band must use nonzero unequal values"
            else:
                assert gap >= CFG["comparison_rule"]["diff_if_abs_gap_at_least"]
                higher_is_first = v1 > v2
                assert higher_is_first == (relation == "first_higher")


def test_r2_every_table_is_hypothetical_and_only_numbers_move():
    prompts, diag = R2.build(CFG)
    assert R2.validate(prompts, diag, CFG) == []
    for p in prompts:
        assert R2.WARNING in p["prompt"], "framing must be identical across conditions"
        assert R2.RULE_TEXT in p["prompt"]
    # same bundle+relation under two wordings shows identical numbers
    pairs = {}
    for p in prompts:
        pairs.setdefault((p["bundle_id"], p["relation"]), set()).add((p["value1"], p["value2"]))
    assert all(len(v) == 1 for v in pairs.values())
    # each bundle+wording carries all three relations -> answers balanced by construction
    per = {}
    for p in prompts:
        per.setdefault((p["bundle_id"], p["wording"]), set()).add(p["gold_canonical"])
    assert all(v == {"a", "b", "c"} for v in per.values())


def test_r2_diagnostics_only_allow_insufficient_when_values_are_absent():
    _, diag = R2.build(CFG)
    for d in diag:
        assert "insufficient" in d["prompt"].lower()
        if d["variant"] == "values_removed":
            assert d["gold_choice_text"] == R2.INSUFFICIENT and "not reported" in d["prompt"]
        else:
            assert d["gold_choice_text"] != R2.INSUFFICIENT and "not reported" not in d["prompt"]


def test_r2_row_order_variant_keeps_the_same_answer():
    _, diag = R2.build(CFG)
    by = {}
    for d in diag:
        by.setdefault(d["bundle_id"], {})[d["variant"]] = d
    for b, variants in by.items():
        if "row_order_reversed" in variants and "irrelevant_column" in variants:
            assert variants["row_order_reversed"]["gold_choice_text"] == variants["irrelevant_column"]["gold_choice_text"]


# ---------------------------------------------------------------- scoring and the runner

def test_scoring_marks_the_right_option():
    p = {"choices": ["Roughly equal", "B", "A"], "gold_choice_text": "A"}
    assert RI.score_row(p, '{"answer_choice_letter": "c"}')["correct"] is True
    assert RI.score_row(p, '{"answer_choice_letter": "a"}')["correct"] is False
    bad = RI.score_row(p, "I cannot answer that")
    assert bad["parse_ok"] is False and bad["correct"] is False, "a parse failure is a failure, never a retry"


def test_insufficient_evidence_scoring():
    p = {"choices": ["A", "B", "Roughly equal", "The table does not contain the values needed"],
         "gold_choice_text": "insufficient_evidence"}
    assert RI.score_row(p, '{"answer_choice_letter": "d"}')["correct"] is True
    assert RI.score_row(p, '{"answer_choice_letter": "a"}')["correct"] is False


def test_smoke_run_writes_scored_rows_and_never_loads_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CODES_ROOT", tmp_path)
    cfg = dict(CFG); cfg["output_directory"] = "out"; cfg["gpu_enabled"] = False
    d = C.out_dir(cfg, "evidence")
    prompts, diag = R2.build(CFG)
    C.write_jsonl(d / "r2_main_prompts.jsonl", prompts[:8])
    C.write_jsonl(d / "r2_diagnostic_prompts.jsonl", diag[:4])
    man = RI.run(cfg, "pilot", smoke=True)
    assert man["outputs_written"] > 0
    rows = C.read_jsonl(tmp_path / "out" / "predictions" / "pilot_predictions.jsonl")
    assert all("correct" in r and "prompt_sha256" in r and "prompt" not in r for r in rows)
    assert len({r["system"] for r in rows}) == len(C.systems(cfg))


def test_gpu_gate_blocks_a_real_run(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CODES_ROOT", tmp_path)
    cfg = dict(CFG); cfg["output_directory"] = "out"; cfg["gpu_enabled"] = False
    d = C.out_dir(cfg, "evidence")
    prompts, _ = R2.build(CFG)
    C.write_jsonl(d / "r2_main_prompts.jsonl", prompts[:4])
    with pytest.raises(SystemExit, match="gpu_enabled"):
        RI.run(cfg, "pilot", smoke=False)
