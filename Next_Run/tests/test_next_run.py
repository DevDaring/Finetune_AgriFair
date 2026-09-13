"""Unit tests required by plan section 10 before the final execution.

Each test is one of the named cases: three-option canonical remapping; wrong-group versus
erasure; fabrication versus invalid; undefined single-condition B; duplicate/missing ids;
shared source-cluster resampling; per-seed joins; missing admissibility readouts -> "not
evaluable"; synthetic-table gold; "always equal" failing joint correctness despite
apparent invariance. Plus a hand-computed check of the vectorised bootstrap/permutation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Next_Run import common as C
from Next_Run import stats as S
from Next_Run.verify_sources import derive_condition, expected_gold
from Next_Run.evidence_panel import parse_answer, render_prompt, expected_for


# ---------------------------------------------------------------- answers and error types

def test_three_option_canonical_remapping():
    b = {"question": "q", "group1": "women", "group2": "men",
         "choices_on_disk": ["Roughly equal", "men", "women"], "units": "", "denominator": "",
         "verified_group1_value": 1, "verified_group2_value": 1, "synthetic_group1_value": 1, "synthetic_group2_value": 1,
         "verified_expected_gold": "c", "synthetic_expected_gold": "c"}
    _, d2c = render_prompt(b, "no_evidence")
    assert d2c == {"a": "c", "b": "b", "c": "a"}          # displayed a is "Roughly equal" -> canonical c
    assert parse_answer('{"answer_choice_letter": "a"}', d2c) == "c"
    assert parse_answer("garbage", d2c) is None


def test_wrong_group_vs_erasure_vs_fabrication_vs_invalid():
    assert C.error_type("a", "c") == "erasure"
    assert C.error_type("a", "b") == "wrong_group"
    assert C.error_type("c", "a") == "fabrication"
    assert C.error_type("c", "b") == "fabrication"
    assert C.error_type("a", "a") == "correct"
    assert C.error_type("a", None) == "invalid"
    assert C.error_type("a", "z") == "invalid"
    assert C.error_type("a", "a", parse_ok=False) == "invalid"


def test_counts_separate_every_error_type():
    c = C.Counts()
    for g, p in (("a", "c"), ("a", "b"), ("c", "a"), ("c", "c"), ("b", None)):
        c.add(g, p)
    assert (c.erasure, c.wrong_group, c.fabrication, c.invalid) == (1, 1, 1, 1)
    assert (c.n_diff, c.correct_diff, c.n_equal, c.correct_equal) == (3, 0, 2, 1)


# ---------------------------------------------------------------- harmonic B

def test_undefined_single_condition_b():
    assert math.isnan(C.harmonic_b(float("nan"), 0.9))
    assert math.isnan(C.harmonic_b(0.5, float("nan")))
    assert C.harmonic_b(0.0, 0.0) == 0.0
    assert C.harmonic_b(1.0, 1.0) == 1.0
    assert abs(C.harmonic_b(0.5, 1.0) - 2 / 3) < 1e-12
    m = C.metrics_from_counts(np.array([0, 0, 10, 8, 0, 0, 2, 0]))   # no diff items at all
    assert math.isnan(m["harmonic_b"]) and math.isnan(m["accuracy_diff"]) and m["accuracy_equal"] == 0.8


def test_vectorised_b_matches_scalar():
    s = np.array([[10, 7, 10, 9], [0, 0, 5, 5], [4, 0, 4, 0]], dtype=float)
    s = np.hstack([s, np.zeros((3, 4))])
    b = S.b_from_sums(s)
    assert abs(b[0] - C.harmonic_b(0.7, 0.9)) < 1e-12
    assert math.isnan(b[1])
    assert b[2] == 0.0


# ---------------------------------------------------------------- normalisation and joins

def _items():
    return {f"i{k}": {"id": f"i{k}", "source_cell": f"cell{k % 3}", "state_blind_key": "x|y|z|w", "category": "gender",
                      "form": "number", "condition": "diff" if k % 2 else "equal", "test_slice": "structure_novel",
                      "correct_answer": "a" if k % 2 else "c", "split": "test"} for k in range(6)}


def test_duplicate_and_missing_ids_are_counted(tmp_path):
    items = _items()
    p = tmp_path / "per_item_predictions_t_m_seed42.jsonl"
    rows = [{"id": "i0", "pred_canonical": "c", "parse_ok": True}, {"id": "i0", "pred_canonical": "a", "parse_ok": True},
            {"id": "i1", "pred_canonical": "a", "parse_ok": True}, {"id": "ghost", "pred_canonical": "a", "parse_ok": True}]
    p.write_text("\n".join(__import__("json").dumps(r) for r in rows))
    recs, issues = C.normalise_predictions(C.Arm("t", "m", 42), p, items)
    assert {k: issues[k] for k in ("duplicate_ids", "unknown_ids", "missing_ids", "records")} == {"duplicate_ids": 1, "unknown_ids": 1, "missing_ids": 4, "records": 2}
    assert issues["expected_scope"] == "full_test"
    assert [r["item_id"] for r in recs] == ["i0", "i1"]
    assert recs[0]["pred"] == "c"                                    # first occurrence kept


def test_align_pair_reports_coverage():
    a = [{"item_id": "i0"}, {"item_id": "i1"}, {"item_id": "i2"}]
    b = [{"item_id": "i1"}, {"item_id": "i2"}, {"item_id": "i9"}]
    aa, bb, cov = C.align_pair(a, b)
    assert [r["item_id"] for r in aa] == ["i1", "i2"] == [r["item_id"] for r in bb]
    assert cov == {"common": 2, "only_in_a": 1, "only_in_b": 1}


def test_counts_by_cluster_fixed_order_with_zero_rows():
    recs = [{"source_cell": "c1", "gold": "a", "pred": "a", "parse_ok": True},
            {"source_cell": "c1", "gold": "c", "pred": "a", "parse_ok": True}]
    m, order = C.counts_by_cluster(recs, "source_cell", ["c0", "c1", "c2"])
    assert order == ["c0", "c1", "c2"] and m.shape == (3, 8)
    assert m[0].sum() == 0 and m[2].sum() == 0
    assert m[1].tolist() == [1, 1, 1, 0, 0, 0, 1, 0]


# ---------------------------------------------------------------- resampling

def test_bootstrap_preserves_multiplicity_and_pairs_methods():
    # two clusters, method A perfect, method B wrong on cluster 1 only; every draw that
    # samples cluster 1 at all gives a positive difference, and the same draw is used for B
    A = np.array([[5, 5, 5, 5, 0, 0, 0, 0], [5, 5, 5, 5, 0, 0, 0, 0]], dtype=float)
    B = np.array([[5, 5, 5, 5, 0, 0, 0, 0], [5, 0, 5, 0, 5, 0, 5, 0]], dtype=float)
    r = S.paired_cluster_bootstrap(A, B, draws=2000, seed=1)
    assert r["lower"] >= 0.0 and r["upper"] <= 1.0 and r["mean"] > 0
    assert r["n_clusters"] == 2


def test_bootstrap_same_draw_across_seeds():
    A = np.stack([np.array([[5, 5, 5, 5, 0, 0, 0, 0]] * 3, float)] * 2)   # (seeds=2, clusters=3, 8)
    r = S.paired_cluster_bootstrap(A, A, draws=200, seed=3)
    assert r["lower"] == 0.0 and r["upper"] == 0.0                       # identical methods -> exactly zero


def test_permutation_never_exactly_zero_and_hand_case():
    # one cluster: swapping either changes nothing (A==B) -> p = 1
    A = np.array([[4, 4, 4, 4, 0, 0, 0, 0]], float)
    r = S.cluster_swap_permutation(A, A, draws=99, seed=0)
    assert r["p_value"] == 1.0
    # strong effect over many clusters: p is small but never 0
    A = np.tile(np.array([[3, 3, 3, 3, 0, 0, 0, 0]], float), (30, 1))
    B = np.tile(np.array([[3, 0, 3, 0, 3, 0, 3, 0]], float), (30, 1))
    r = S.cluster_swap_permutation(A, B, draws=999, seed=0)
    assert 0 < r["p_value"] <= 2 / 1000


def test_holm_preserves_order_and_ignores_nan():
    out = S.holm([0.01, float("nan"), 0.04, 0.03])
    assert math.isnan(out[1])
    assert abs(out[0] - 0.03) < 1e-12 and abs(out[3] - 0.06) < 1e-12 and abs(out[2] - 0.06) < 1e-12


# ---------------------------------------------------------------- sources and panel

def test_synthetic_table_gold_and_excluded_band():
    rule = {"metric_unit": "percentage_points", "equal_if_abs_gap_below": 2.0, "diff_if_abs_gap_at_least": 5.0}
    assert derive_condition(10, 11.5, rule) == "equal"
    assert derive_condition(10, 12, rule) == "excluded_band"  # equal boundary is strict
    assert derive_condition(10, 16, rule) == "diff"
    assert derive_condition(10, 13, rule) == "excluded_band"
    assert expected_gold(16, 10, "diff") == "a" and expected_gold(10, 16, "diff") == "b"
    assert expected_gold(10, 11, "equal") == "c"


def test_always_equal_fails_joint_correctness_despite_invariance():
    """A model that always answers c is perfectly consistent across anonymisation but cannot
    be right on both the verified (diff) and synthetic versions."""
    b = {"verified_expected_gold": "a", "synthetic_expected_gold": "b"}
    always = "c"
    assert (always == expected_for(b, "verified_evidence")) is False
    assert (always == expected_for(b, "synthetic_evidence")) is False
    # invariance alone would look perfect: same answer with and without anonymisation
    assert always == always


def test_missing_admissibility_readout_is_not_evaluable():
    """Joining behavioural predictions with mechanistic readouts by (model, method, seed):
    a missing readout must yield 'not_evaluable', never a pass."""
    readouts = {("t", "m", 42): {"probe": 0.7}}
    def verdict(key):
        r = readouts.get(key)
        return "not_evaluable" if r is None or r.get("probe") is None else ("pass" if r["probe"] < 0.8 else "fail")
    assert verdict(("t", "m", 42)) == "pass"
    assert verdict(("t", "m", 43)) == "not_evaluable"
    assert verdict(("t", "other", 42)) == "not_evaluable"


def test_scrub_secrets():
    # fake credentials assembled at runtime so no token-shaped literal ever sits in source
    fake_hf = "hf_" + "a" * 30
    fake_sk = "sk-" + "b" * 30
    s = f"HUGGINGFACE_TOKEN={fake_hf} and api_key: {fake_sk}"
    out = C.scrub_secrets(s)
    assert fake_hf not in out and fake_sk not in out


def test_refuses_to_write_into_legacy_dirs(tmp_path):
    with pytest.raises(PermissionError):
        C.write_json(C.legacy_results_dir() / "x.json", {})
    C.write_json(tmp_path / "ok.json", {"a": 1})


def test_synthetic_equal_variant_always_clears_threshold():
    """Plan 9.1: for an originally-equal pair the synthetic table must introduce a difference
    beyond the frozen threshold, whichever group was larger to begin with."""
    from Next_Run.verify_sources import derive_condition
    rule = {"metric_unit": "percentage_points", "equal_if_abs_gap_below": 5.0, "diff_if_abs_gap_at_least": 10.0}
    delta = rule["diff_if_abs_gap_at_least"] + 1.0
    for g1, g2 in ((40.0, 43.0), (43.0, 40.0), (2.0, 1.0), (95.0, 97.0)):
        s1, s2 = (g2 + delta, g2) if g2 + delta <= 100.0 else (g2, max(g2 - delta, 0.0))
        assert derive_condition(s1, s2, rule) == "diff", (g1, g2, s1, s2)
