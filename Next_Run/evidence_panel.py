"""P4 - optional, gated evidence-use stress test (plan section 9).

The only new foundation-model experiment the plan allows, and it stays off unless every
gate passes. Gates, checked in order and all required:

  1. config: gpu_enabled true, allow_paid_api false (there is no API path in this file)
  2. the source ledger has been filled and recomputed with zero unresolved discrepancies
     for every source cell the panel will use (a bundle built on an unverified value is
     an invalid experiment, not a model failure)
  3. the 48 balanced main bundles plus 8 additional pilot bundles, including their
     synthetic variants, carry a human "manually_checked"
     flag in the bundle file
  4. base weights and the six adapter checkpoints exist on disk (config files alone do not
     count)
  5. the pilot's measured cost, times 1.5, plus remaining loads fits inside the remaining
     minutes of the 120-minute cap; if the balanced 48-bundle panel does not fit, P4 is
     skipped

The scorer is deterministic: expected answers come from the supplied numbers and the frozen
rule, never from a model. The synthetic condition is labelled hypothetical in the prompt and
in every output row, and altered numbers are never written anywhere as census facts.
"""
from __future__ import annotations

import csv
import argparse
import gc
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Next_Run import common as C
from Next_Run.verify_sources import derive_condition, expected_gold

CONDITIONS = ("no_evidence", "verified_evidence", "synthetic_evidence", "anonymised_verified_evidence")
ANSWER_RE = re.compile(r'"answer_choice_letter"\s*:\s*"([abc])"|\b([ABC])\b')
BUNDLE_SPEC_VERSION = "p4-balanced-48-plus-8-pilot-v3"


# ----------------------------------------------------------------------------- bundles

def _ledger_values(cfg: Dict) -> Dict[str, Dict]:
    p = C.output_dir(cfg) / "sources" / "discrepancy_report.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def admissible(r):
        agrees = r.get("agrees_with_frozen") in ("True", "true", "1")
        status = r.get("verification_status", "")
        if C.smoke(cfg):
            return agrees and status == "SYNTHETIC_SMOKE_VALUE" and r.get("denominator_or_stratum")
        return (agrees and status not in ("", "unverified") and r.get("page_or_sheet")
                and r.get("source_url_or_doc_hash") and r.get("denominator_or_stratum"))
    return {r["source_cell"]: r for r in rows if admissible(r)}


def _candidate_strata(items: List[Dict], verified: Dict[str, Dict], excluded_cells=None):
    """Return one deterministic candidate item per verified source cell and stratum."""
    excluded_cells = set(excluded_cells or ())
    by = defaultdict(list)
    seen_cells = set(excluded_cells)
    for it in sorted(items, key=lambda r: r["id"]):
        cell = it["source_cell"]
        if cell in verified and cell not in seen_cells:
            by[(it["category"], it["condition"])].append(it)
            seen_cells.add(cell)
    return by


def _bundle_from_item(it: Dict, verified: Dict[str, Dict], cfg: Dict, bundle_id: str,
                      panel_role: str, source_split: str) -> Dict:
    rule = cfg["comparison_rule"]
    v = verified[it["source_cell"]]
    g1, g2 = float(v["group1_value"]), float(v["group2_value"])
    cond = it["condition"]
    # Reverse a real difference, or open an unambiguous gap for an equal item.
    if cond == "diff":
        s1, s2 = g2, g1
    else:
        delta = float(rule["diff_if_abs_gap_at_least"]) + 1.0
        # Use a feasible two-share construction: each value is within 0..100 and their
        # sum never exceeds 100. This also works for comparisons with separate strata.
        s2 = min(g2, (100.0 - delta) / 2.0)
        s1 = s2 + delta
    verified_condition = derive_condition(g1, g2, rule)
    synthetic_condition = derive_condition(s1, s2, rule)
    assert verified_condition == cond
    assert synthetic_condition == "diff"
    verified_gold = expected_gold(g1, g2, verified_condition)
    synthetic_gold = expected_gold(s1, s2, synthetic_condition)
    assert synthetic_gold != verified_gold, "synthetic evidence must change the expected answer"
    return {
        "bundle_spec_version": BUNDLE_SPEC_VERSION,
        "bundle_id": bundle_id,
        "panel_role": panel_role,
        "source_split": source_split,
        "item_id": it["id"],
        "source_cell": it["source_cell"],
        "axis": it["category"],
        "original_condition": cond,
        "nested_subset": panel_role == "main",
        "question": it["question"],
        "group1": it["group1"],
        "group2": it["group2"],
        "choices_on_disk": it["on_disk_choices"],
        "units": v.get("units", ""),
        "denominator": v.get("denominator_or_stratum", ""),
        "source_location": v.get("page_or_sheet", ""),
        "source_reference": v.get("source_url_or_doc_hash", ""),
        "verified_group1_value": g1,
        "verified_group2_value": g2,
        "verified_expected_gold": verified_gold,
        "synthetic_group1_value": s1,
        "synthetic_group2_value": s2,
        "synthetic_condition": synthetic_condition,
        "synthetic_expected_gold": synthetic_gold,
        "automated_checks_passed": False,
        "manually_checked": bool(C.smoke(cfg).get("auto_mark_bundles_checked")),
        "manual_check_notes": "SMOKE: auto-marked, no human check" if C.smoke(cfg) else "",
    }


def build_bundles(cfg: Dict) -> Tuple[List[Dict], Dict]:
    """Build 48 balanced main bundles and 8 additional validation-split pilot bundles.

    Every bundle uses a distinct verified source comparison. The main panel has exactly
    eight equal and eight different cells per axis. Pilot cells are excluded from scoring.
    """
    ep = cfg["evidence_panel"]
    verified = _ledger_values(cfg)
    if not verified:
        return [], {"status": "blocked", "reason": "no verified source values (run verify_sources with a filled ledger first)"}
    rng = random.Random(cfg["analysis_seed"])
    by = _candidate_strata(C.load_split("test"), verified)
    bundles, selected_cells = [], set()
    per = ep["bundles_per_axis"] // 2
    for axis in C.AXES:
        for cond in ("equal", "diff"):
            cand = list(by[(axis, cond)])
            rng.shuffle(cand)
            if len(cand) < per:
                return [], {"status": "blocked", "reason": f"balanced main stratum {axis}/{cond} needs {per}, has {len(cand)}"}
            for it in cand[:per]:
                selected_cells.add(it["source_cell"])
                bundles.append(_bundle_from_item(
                    it, verified, cfg, f"B{len(bundles) + 1:03d}", "main", "test"
                ))

    # Pilot bundles are genuinely additional and come from the validation split. The smoke
    # fixture has a synthetic test-only ledger, so its non-scientific rehearsal uses leftover
    # test cells while retaining distinct pilot/main source comparisons.
    pilot_split = "test" if C.smoke(cfg) else "validation"
    pilot_by = _candidate_strata(C.load_split(pilot_split), verified, selected_cells)
    pilot_items = []
    for axis in C.AXES:
        for cond in ("equal", "diff"):
            cand = list(pilot_by[(axis, cond)])
            rng.shuffle(cand)
            if cand:
                pilot_items.append(cand.pop())
                pilot_by[(axis, cond)] = cand
    remainder = [it for cand in pilot_by.values() for it in cand]
    rng.shuffle(remainder)
    pilot_items.extend(remainder[:max(0, ep["pilot_bundles"] - len(pilot_items))])
    pilot_items = pilot_items[:ep["pilot_bundles"]]
    if len(pilot_items) != ep["pilot_bundles"]:
        return [], {"status": "blocked", "reason": f"pilot needs {ep['pilot_bundles']} independent validation cells, has {len(pilot_items)}"}
    for i, it in enumerate(pilot_items, 1):
        if it["source_cell"] in selected_cells:
            raise AssertionError("pilot and main source cells overlap")
        selected_cells.add(it["source_cell"])
        bundles.append(_bundle_from_item(it, verified, cfg, f"P{i:03d}", "pilot", pilot_split))

    audit = automated_bundle_audit(bundles, cfg)
    if audit["issues"]:
        return [], {"status": "blocked", "reason": "automated bundle audit failed", "issues": audit["issues"]}
    for b in bundles:
        b["automated_checks_passed"] = True
    return bundles, {
        "status": "built",
        "bundle_spec_version": BUNDLE_SPEC_VERSION,
        "main_bundles": sum(b["panel_role"] == "main" for b in bundles),
        "pilot_bundles": sum(b["panel_role"] == "pilot" for b in bundles),
        "total_bundles": len(bundles),
        "main_strata": audit["main_strata"],
    }


def _fmt_value(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _anonymise(text: str, group1: str, group2: str) -> str:
    """Replace possibly overlapping entity names (for example men/women) simultaneously."""
    placeholders = {group1: "__P4_GROUP_A__", group2: "__P4_GROUP_B__"}
    out = text
    for entity in sorted(placeholders, key=len, reverse=True):
        out = out.replace(entity, placeholders[entity])
    return out.replace("__P4_GROUP_A__", "Group A").replace("__P4_GROUP_B__", "Group B")


def _table(g1_name, g2_name, v1, v2, units, denom, hypothetical: bool, anon: bool,
           citation: str = "") -> str:
    n1, n2 = ("Group A", "Group B") if anon else (g1_name, g2_name)
    head = ("HYPOTHETICAL TABLE (not real census data). Answer only about this table." if hypothetical
            else "Table (2015-16 Agriculture Census).")
    source = "" if hypothetical else f"\n(source location: {citation or 'recorded in the evidence ledger'})"
    anon_note = "\n(Group A and Group B are anonymous labels used consistently.)" if anon else ""
    display_units = (units or "percentage share (0-100)").split(";", 1)[0].strip()
    return (
        f"{head}\n| group | value ({display_units}) |\n|---|---|\n"
        f"| {n1} | {_fmt_value(v1)} |\n| {n2} | {_fmt_value(v2)} |\n"
        f"(denominator: {denom or 'as stated'}){source}{anon_note}"
    )


def render_prompt(b: Dict, condition: str) -> Tuple[str, Dict[str, str]]:
    """Fixed three-choice short-answer prompt. Choice order is the on-disk order for every
    condition so counterbalancing is identical across compared systems."""
    q = b["question"]; choices = list(b["choices_on_disk"])
    g1, g2 = b["group1"], b["group2"]
    if condition == "anonymised_verified_evidence":
        q = _anonymise(q, g1, g2)
        choices = [_anonymise(c, g1, g2) for c in choices]
    elif condition == "synthetic_evidence":
        # The original stem often asserts that values come from the real census. Replacing it
        # avoids contradicting the explicitly hypothetical table.
        q = (
            f"Using only the hypothetical table, which has the larger share—{g1}, {g2}, "
            "or are the two roughly equal?"
        )
    display_to_canonical = {}
    for letter, c in zip("abc", choices):
        canon = "c" if c.lower().startswith("roughly equal") else ("a" if c in (g1, "Group A") or c.replace("Group A", g1) == g1 else "b")
        display_to_canonical[letter] = canon
    table = ""
    if condition == "verified_evidence":
        table = _table(g1, g2, b["verified_group1_value"], b["verified_group2_value"], b["units"], b["denominator"], False, False, b.get("source_location", ""))
    elif condition == "synthetic_evidence":
        table = _table(g1, g2, b["synthetic_group1_value"], b["synthetic_group2_value"], b["units"],
                       "hypothetical percentage scale; values are not census observations", True, False)
    elif condition == "anonymised_verified_evidence":
        table = _table(g1, g2, b["verified_group1_value"], b["verified_group2_value"], b["units"], b["denominator"], False, True, b.get("source_location", ""))
    opts = "\n".join(f"{l}) {c}" for l, c in zip("abc", choices))
    prompt = (f"{table}\n\n" if table else "") + f"{q}\n{opts}\n" + 'Reply with JSON only: {"answer_choice_letter": "<a|b|c>"}'
    return prompt, display_to_canonical


def parse_answer(text: str, display_to_canonical: Dict[str, str]) -> Optional[str]:
    m = ANSWER_RE.search(text or "")
    if not m:
        return None
    letter = (m.group(1) or m.group(2) or "").lower()
    return display_to_canonical.get(letter)


def expected_for(b: Dict, condition: str) -> str:
    return b["synthetic_expected_gold"] if condition == "synthetic_evidence" else b["verified_expected_gold"]


def automated_bundle_audit(bundles: List[Dict], cfg: Dict) -> Dict:
    """Exhaustive deterministic checks. This supplements but never impersonates human review."""
    issues = []
    seen_ids, seen_cells = set(), set()
    main_strata = defaultdict(int)
    for b in bundles:
        bid = b.get("bundle_id", "")
        if bid in seen_ids:
            issues.append(f"duplicate bundle_id: {bid}")
        if b.get("source_cell") in seen_cells:
            issues.append(f"reused source cell: {b.get('source_cell')}")
        seen_ids.add(bid); seen_cells.add(b.get("source_cell"))
        if b.get("panel_role") == "main":
            main_strata[f"{b['axis']}/{b['original_condition']}"] += 1
            if b.get("source_split") != "test":
                issues.append(f"{bid}: main bundle is not from test")
        elif b.get("panel_role") == "pilot":
            required_pilot_split = "test" if C.smoke(cfg) else "validation"
            if b.get("source_split") != required_pilot_split:
                issues.append(f"{bid}: pilot bundle is not from validation")
        else:
            issues.append(f"{bid}: invalid panel_role")
        if not (0 <= float(b["verified_group1_value"]) <= 100 and 0 <= float(b["verified_group2_value"]) <= 100):
            issues.append(f"{bid}: verified percentage outside 0..100")
        if not (0 <= float(b["synthetic_group1_value"]) <= 100 and 0 <= float(b["synthetic_group2_value"]) <= 100):
            issues.append(f"{bid}: synthetic percentage outside 0..100")
        if float(b["synthetic_group1_value"]) + float(b["synthetic_group2_value"]) > 100.0 + 1e-9:
            issues.append(f"{bid}: synthetic compared shares exceed a joint 100 percent")
        if b["verified_expected_gold"] == b["synthetic_expected_gold"]:
            issues.append(f"{bid}: synthetic evidence does not change the answer")
        for condition in CONDITIONS:
            prompt, mapping = render_prompt(b, condition)
            if set(mapping) != set("abc") or set(mapping.values()) != set("abc"):
                issues.append(f"{bid}/{condition}: choice mapping is not bijective")
            if expected_for(b, condition) not in "abc":
                issues.append(f"{bid}/{condition}: invalid expected answer")
            if condition == "synthetic_evidence":
                if "HYPOTHETICAL TABLE" not in prompt or "2015-16 Agriculture Census" in prompt:
                    issues.append(f"{bid}/{condition}: hypothetical prompt is ambiguous about provenance")
            if condition == "anonymised_verified_evidence":
                if b["group1"] in prompt or b["group2"] in prompt or "Group A" not in prompt or "Group B" not in prompt:
                    issues.append(f"{bid}/{condition}: entity anonymisation failed")
            if condition in ("verified_evidence", "anonymised_verified_evidence") and "source location:" not in prompt:
                issues.append(f"{bid}/{condition}: verified table lacks a source location")

    expected_main = len(C.AXES) * int(cfg["evidence_panel"]["bundles_per_axis"])
    expected_pilot = int(cfg["evidence_panel"]["pilot_bundles"])
    if sum(b.get("panel_role") == "main" for b in bundles) != expected_main:
        issues.append(f"main bundle count is not {expected_main}")
    if sum(b.get("panel_role") == "pilot" for b in bundles) != expected_pilot:
        issues.append(f"pilot bundle count is not {expected_pilot}")
    per = int(cfg["evidence_panel"]["bundles_per_axis"]) // 2
    for axis in C.AXES:
        for condition in ("equal", "diff"):
            if main_strata[f"{axis}/{condition}"] != per:
                issues.append(f"unbalanced main stratum {axis}/{condition}")
    return {"ok": not issues, "issues": issues, "main_strata": dict(sorted(main_strata.items()))}


def manual_check_rows(bundles: List[Dict]) -> List[Dict]:
    rows = []
    for b in bundles:
        for condition in CONDITIONS:
            prompt, display_to_canonical = render_prompt(b, condition)
            expected = expected_for(b, condition)
            expected_display = next(letter for letter, canonical in display_to_canonical.items() if canonical == expected)
            rows.append({
                "bundle_id": b["bundle_id"],
                "panel_role": b["panel_role"],
                "axis": b["axis"],
                "original_condition": b["original_condition"],
                "source_cell": b["source_cell"],
                "source_reference": b.get("source_reference", ""),
                "condition": condition,
                "prompt": prompt,
                "expected_canonical": expected,
                "expected_display_letter": expected_display,
                "prompt_and_gold_sha256": C.sha256_obj({"prompt": prompt, "expected": expected,
                                                         "expected_display_letter": expected_display}),
                "review_status_pass_or_fail": "",
                "reviewer_id": "",
                "review_date_utc": "",
                "review_notes": "",
            })
    return rows


def validate_manual_check_rows(bundles: List[Dict], supplied: List[Dict]) -> List[str]:
    expected = {(r["bundle_id"], r["condition"]): r for r in manual_check_rows(bundles)}
    by_key, problems = {}, []
    for row in supplied:
        key = (row.get("bundle_id", ""), row.get("condition", ""))
        if key in by_key:
            problems.append(f"duplicate worksheet row: {key}")
        by_key[key] = row
    if set(by_key) != set(expected):
        problems.append(f"worksheet keys differ: missing={len(set(expected)-set(by_key))}, unknown={len(set(by_key)-set(expected))}")
    for key in sorted(set(expected) & set(by_key)):
        got, exp = by_key[key], expected[key]
        if got.get("prompt_and_gold_sha256") != exp["prompt_and_gold_sha256"]:
            problems.append(f"{key}: prompt/gold hash changed")
        if got.get("review_status_pass_or_fail", "").strip().lower() != "pass":
            problems.append(f"{key}: review status is not pass")
        if not got.get("reviewer_id", "").strip() or not got.get("review_date_utc", "").strip():
            problems.append(f"{key}: reviewer_id/date missing")
    return problems


def import_manual_checks(cfg: Dict, completed_path: Path) -> Dict:
    """Validate a completed human worksheet and unlock only fully passed bundles."""
    out = C.output_dir(cfg) / "evidence_panel"
    bpath = out / "bundles.jsonl"
    if not bpath.exists():
        raise FileNotFoundError("build evidence-panel bundles before importing checks")
    bundles = C.read_jsonl(bpath)
    with open(completed_path, encoding="utf-8") as f:
        supplied = list(csv.DictReader(f))
    problems = validate_manual_check_rows(bundles, supplied)
    if problems:
        raise ValueError("manual check import rejected:\n" + "\n".join(problems[:30]))
    by_key = {(row["bundle_id"], row["condition"]): row for row in supplied}
    for b in bundles:
        checked = [by_key[(b["bundle_id"], c)] for c in CONDITIONS]
        b["manually_checked"] = True
        b["manual_check_notes"] = "; ".join(
            sorted({f"{r['reviewer_id']} on {r['review_date_utc']}" for r in checked})
        )
    C.write_jsonl(bpath, bundles)
    C.write_csv(out / "manual_checks_completed.csv", supplied)
    return {"status": "imported", "rows": len(supplied), "bundles_unlocked": len(bundles)}


# ----------------------------------------------------------------------------- gates

def preflight(cfg: Dict, bundles: List[Dict]) -> Tuple[bool, List[str]]:
    reasons = []
    if not cfg["gpu_enabled"]:
        reasons.append("gpu_enabled is false")
    if cfg["allow_paid_api"]:
        reasons.append("allow_paid_api must be false for this stage")
    if not bundles:
        reasons.append("no bundles")
    else:
        audit = automated_bundle_audit(bundles, cfg)
        if not audit["ok"] or not all(b.get("automated_checks_passed") for b in bundles):
            reasons.append(f"automated bundle audit failed: {audit['issues'][:5]}")
        if not all(b.get("manually_checked") for b in bundles):
            reasons.append(f"{sum(1 for b in bundles if not b.get('manually_checked'))} bundles not manually checked")
        elif not C.smoke(cfg):
            completed = C.output_dir(cfg) / "evidence_panel" / "manual_checks_completed.csv"
            if not completed.exists():
                reasons.append("manual-check flags lack an imported completed worksheet")
            else:
                with open(completed, encoding="utf-8") as f:
                    check_problems = validate_manual_check_rows(bundles, list(csv.DictReader(f)))
                if check_problems:
                    reasons.append(f"completed manual-check worksheet is invalid: {check_problems[:5]}")
    ep = cfg["evidence_panel"]
    from GPU_Run.common import model_registry
    for tier in ep["model_tiers"]:
        spec = model_registry.get_spec(tier)
        local = C.CODES_ROOT / "models" / spec.hf_id.replace("/", "__")
        if not any(local.glob("*.safetensors")) and not C.smoke(cfg):
            reasons.append(f"base weights missing for {tier} at {local}")   # smoke may pull from the Hub
        for method in ep["configurations"]:
            if method == "frozen_base":
                continue
            w = C.CODES_ROOT / "checkpoints" / tier / method / f"seed_{ep['seed']}"
            if not ((w / "final" / "adapter_model.safetensors").exists() or (w / "adapter_model.safetensors").exists()):
                reasons.append(f"adapter weights missing: {w}")
    return not reasons, reasons


# ----------------------------------------------------------------------------- inference

class Budget:
    def __init__(self, cap_minutes: float):
        self.cap = cap_minutes * 60.0; self.t0 = time.time()
    def elapsed(self) -> float: return time.time() - self.t0
    def remaining(self) -> float: return self.cap - self.elapsed()
    def exceeded(self) -> bool: return self.remaining() <= 0


def _load(tier: str, method: str, seed: int):
    import torch
    from GPU_Run.common import model_registry
    base, tok, meta = model_registry.load_model_and_tokenizer(tier)
    if method != "frozen_base":
        from peft import PeftModel
        w = C.CODES_ROOT / "checkpoints" / tier / method / f"seed_{seed}"
        w = w / "final" if (w / "final").exists() else w
        base = PeftModel.from_pretrained(base, str(w))
    base.eval()
    return base, tok


def _generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    import torch
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_panel(cfg: Dict, bundles: List[Dict], out: Path) -> Dict:
    ep = cfg["evidence_panel"]
    budget = Budget(cfg["gpu_total_deadline_minutes"])
    pilot = [b for b in bundles if b["panel_role"] == "pilot"]
    main_all = [b for b in bundles if b["panel_role"] == "main"]
    configs = [(t, m) for t in ep["model_tiers"] for m in ep["configurations"]]
    preds_path = out / "evidence_panel_predictions.jsonl"
    if preds_path.exists():
        raise FileExistsError(
            f"refusing to append to an existing final prediction file: {preds_path}; "
            "archive it and remove it deliberately before a new execution"
        )
    manifest = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": configs,
                "cap_minutes": cfg["gpu_total_deadline_minutes"], "pilot_bundle_ids": [b["bundle_id"] for b in pilot],
                "main_bundle_ids": [b["bundle_id"] for b in main_all], "pilot": [], "decision": None}

    # ---- pilot: measure load + per-bundle cost for every configuration
    load_times, per_bundle = {}, {}
    for tier, method in configs:
        if budget.exceeded():
            manifest["decision"] = "cap reached during pilot; P4 skipped"; break
        t = time.time(); model, tok = _load(tier, method, ep["seed"]); load_times[(tier, method)] = time.time() - t
        t = time.time(); n = 0
        for b in pilot:
            for cond in CONDITIONS:
                prompt, d2c = render_prompt(b, cond); raw = _generate(model, tok, prompt, ep["max_new_tokens"]); n += 1
                C.write_jsonl  # noqa (keeps the symbol referenced for readers)
                with open(preds_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"phase": "pilot", "tier": tier, "method": method, "seed": ep["seed"], "bundle_id": b["bundle_id"],
                                        "condition": cond, "is_hypothetical": cond == "synthetic_evidence",
                                        "raw_output": raw, "pred_canonical": parse_answer(raw, d2c), "expected": expected_for(b, cond)}) + "\n")
        per_bundle[(tier, method)] = (time.time() - t) / max(1, len(pilot))
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        manifest["pilot"].append({"tier": tier, "method": method, "load_s": round(load_times[(tier, method)], 1),
                                  "per_bundle_s": round(per_bundle[(tier, method)], 2)})

    # ---- choose the panel size BEFORE looking at any outcome
    if manifest["decision"] is None:
        def projected(nb: int) -> float:
            return ep["safety_factor"] * sum(load_times[c] + nb * per_bundle[c] for c in configs) + 60.0
        rem = budget.remaining()
        if projected(len(main_all)) <= rem:
            chosen, label = main_all, f"balanced fallback panel ({len(main_all)} bundles)"
        else:
            chosen, label = [], "balanced 48-bundle panel does not fit the remaining cap; P4 skipped"
        manifest["decision"] = label
        manifest["remaining_seconds_at_decision"] = round(rem, 1)

        # ---- main panel
        for tier, method in configs:
            if not chosen or budget.exceeded():
                break
            model, tok = _load(tier, method, ep["seed"])
            for b in chosen:
                if budget.exceeded():
                    manifest["decision"] += " | HARD STOP: cap reached mid-panel"; break
                for cond in CONDITIONS:
                    prompt, d2c = render_prompt(b, cond); raw = _generate(model, tok, prompt, ep["max_new_tokens"])
                    with open(preds_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"phase": "main", "tier": tier, "method": method, "seed": ep["seed"], "bundle_id": b["bundle_id"],
                                            "condition": cond, "is_hypothetical": cond == "synthetic_evidence",
                                            "raw_output": raw, "pred_canonical": parse_answer(raw, d2c), "expected": expected_for(b, cond)}) + "\n")
            del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    manifest["billable_minutes_measured"] = round(budget.elapsed() / 60.0, 2)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    C.write_json(out / "evidence_panel_run_manifest.json", manifest)
    return manifest


def score(out: Path, bundles: List[Dict]) -> List[Dict]:
    """Joint evidence-following correctness and the secondary endpoints, per configuration."""
    p = out / "evidence_panel_predictions.jsonl"
    if not p.exists():
        return []
    rows = [r for r in C.read_jsonl(p) if r["phase"] == "main"]
    by = defaultdict(dict)
    for r in rows:
        by[(r["tier"], r["method"], r["bundle_id"])][r["condition"]] = r
    res = defaultdict(lambda: defaultdict(int))
    for (tier, method, bid), conds in by.items():
        k = (tier, method); res[k]["bundles"] += 1
        ok = {c: (conds.get(c, {}).get("pred_canonical") == conds.get(c, {}).get("expected")) for c in CONDITIONS}
        inv = sum(1 for c in CONDITIONS if conds.get(c, {}).get("pred_canonical") is None)
        res[k]["invalid"] += inv
        for c in CONDITIONS:
            res[k][f"correct_{c}"] += ok[c]
        res[k]["joint_verified_and_synthetic"] += ok["verified_evidence"] and ok["synthetic_evidence"]
        res[k]["anonymised_consistent_and_correct"] += ok["anonymised_verified_evidence"] and ok["verified_evidence"]
    out_rows = []
    for (tier, method), d in sorted(res.items()):
        n = d["bundles"]
        out_rows.append({"tier": tier, "method": method, "bundles": n,
                         "joint_evidence_following_rate": round(d["joint_verified_and_synthetic"] / n, 4),
                         "no_evidence_acc": round(d["correct_no_evidence"] / n, 4), "verified_acc": round(d["correct_verified_evidence"] / n, 4),
                         "verified_minus_no_evidence": round((d["correct_verified_evidence"] - d["correct_no_evidence"]) / n, 4),
                         "synthetic_acc": round(d["correct_synthetic_evidence"] / n, 4),
                         "anonymised_consistent_and_correct_rate": round(d["anonymised_consistent_and_correct"] / n, 4),
                         "invalid_rate": round(d["invalid"] / (4 * n), 4)})
    return out_rows


def main(cfg: Dict) -> Dict:
    out = C.output_dir(cfg) / "evidence_panel"
    out.mkdir(parents=True, exist_ok=True)
    bpath = out / "bundles.jsonl"
    expected_total = len(C.AXES) * int(cfg["evidence_panel"]["bundles_per_axis"]) + int(cfg["evidence_panel"]["pilot_bundles"])
    if bpath.exists():
        bundles = C.read_jsonl(bpath)
        current = len(bundles) == expected_total and all(b.get("bundle_spec_version") == BUNDLE_SPEC_VERSION for b in bundles)
        if current:
            status = {"status": "loaded_existing", "bundles": len(bundles), "bundle_spec_version": BUNDLE_SPEC_VERSION}
        else:
            bundles, status = build_bundles(cfg)
            if bundles:
                C.write_jsonl(bpath, bundles)
    else:
        bundles, status = build_bundles(cfg)
        if bundles:
            C.write_jsonl(bpath, bundles)
    if bundles:
        # Exhaustive machine audit and a separate worksheet that only a real reviewer can sign.
        audit = automated_bundle_audit(bundles, cfg)
        C.write_json(out / "automated_bundle_audit.json", audit)
        worksheet = manual_check_rows(bundles)
        C.write_csv(out / "bundles_manual_check_worksheet.csv", worksheet)
        C.write_jsonl(out / "bundles_manual_check_worksheet.jsonl", worksheet)
    print(f"[evidence_panel] bundles: {status}")
    ok, reasons = preflight(cfg, bundles)
    if not ok:
        print("[evidence_panel] NOT RUN. Gates failed:"); [print("   -", r) for r in reasons]
        C.write_json(out / "preflight.json", {"ok": False, "reasons": reasons, "bundles": len(bundles)})
        return {"bundles": len(bundles), "run": False, "reasons": reasons}
    C.write_json(out / "preflight.json", {"ok": True, "bundles": len(bundles)})
    manifest = run_panel(cfg, bundles, out)
    C.write_csv(out / "evidence_panel_scores.csv", score(out, bundles))
    print(f"[evidence_panel] {manifest['decision']} | billable {manifest['billable_minutes_measured']} min")
    return {"bundles": len(bundles), "run": True, "decision": manifest["decision"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.CONFIG_PATH))
    ap.add_argument("--import-manual-checks", type=Path)
    args = ap.parse_args()
    config = C.load_config(Path(args.config))
    if args.import_manual_checks:
        print(json.dumps(import_manual_checks(config, args.import_manual_checks), indent=2))
    else:
        main(config)
