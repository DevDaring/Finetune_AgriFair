"""P4 - optional, gated evidence-use stress test (plan section 9).

The only new foundation-model experiment the plan allows, and it stays off unless every
gate passes. Gates, checked in order and all required:

  1. config: gpu_enabled true, allow_paid_api false (there is no API path in this file)
  2. the source ledger has been filled and recomputed with zero unresolved discrepancies
     for every source cell the panel will use (a bundle built on an unverified value is
     an invalid experiment, not a model failure)
  3. the 96 bundles, including their synthetic variants, carry a human "manually_checked"
     flag in the bundle file
  4. base weights and the six adapter checkpoints exist on disk (config files alone do not
     count)
  5. the pilot's measured cost, times 1.5, plus remaining loads fits inside the remaining
     minutes of the 120-minute cap; if the full panel does not fit, the predeclared nested
     48-bundle subset is used; if that does not fit either, P4 is skipped

The scorer is deterministic: expected answers come from the supplied numbers and the frozen
rule, never from a model. The synthetic condition is labelled hypothetical in the prompt and
in every output row, and altered numbers are never written anywhere as census facts.
"""
from __future__ import annotations

import csv
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


# ----------------------------------------------------------------------------- bundles

def _ledger_values(cfg: Dict) -> Dict[str, Dict]:
    p = C.output_dir(cfg) / "sources" / "discrepancy_report.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        r["source_cell"]: r
        for r in rows
        if r.get("agrees_with_frozen") in ("True", "true", "1")
        and r.get("verification_status") not in ("", "unverified")
        and r.get("page_or_sheet")
        and r.get("source_url_or_doc_hash")
        and r.get("denominator_or_stratum")
    }


def build_bundles(cfg: Dict) -> Tuple[List[Dict], Dict]:
    """96 bundles: 32 per axis, 16 equal + 16 diff, one item per verified source cell,
    drawn by the analysis seed. Synthetic variants are derived by the frozen rule."""
    ep = cfg["evidence_panel"]
    rule = cfg["comparison_rule"]
    verified = _ledger_values(cfg)
    if not verified:
        return [], {"status": "blocked", "reason": "no verified source values (run verify_sources with a filled ledger first)"}
    items = C.load_split("test")
    rng = random.Random(cfg["analysis_seed"])
    by = defaultdict(list)
    seen_cells = set()
    for it in sorted(items, key=lambda r: r["id"]):
        if it["source_cell"] in verified and it["source_cell"] not in seen_cells:
            by[(it["category"], it["condition"])].append(it); seen_cells.add(it["source_cell"])
    bundles, shortfall = [], []
    per = ep["bundles_per_axis"] // 2
    for axis in C.AXES:
        for cond in ("equal", "diff"):
            cand = by[(axis, cond)]; rng.shuffle(cand)
            if len(cand) < per:
                shortfall.append({"stratum": f"{axis}/{cond}", "wanted": per, "available": len(cand)})
            for i, it in enumerate(cand[:per]):
                v = verified[it["source_cell"]]
                g1, g2 = float(v["group1_value"]), float(v["group2_value"])
                # synthetic: reverse a real difference, or open a gap beyond the diff threshold.
                # For an equal item the new gap must clear the threshold REGARDLESS of which group
                # was originally larger, so it is built from group 2's value, not added to group 1
                # (adding to group 1 when group 2 was larger shrank the gap into the excluded band).
                if cond == "diff":
                    s1, s2 = g2, g1
                else:
                    delta = float(rule["diff_if_abs_gap_at_least"]) + 1.0
                    if g2 + delta <= 100.0:
                        s1, s2 = g2 + delta, g2          # group 1 clearly larger
                    else:
                        s1, s2 = g2, max(g2 - delta, 0.0) # near the ceiling: shrink group 2 instead
                s_cond = derive_condition(s1, s2, rule)
                assert s_cond == "diff", f"synthetic variant for {it['id']} fell outside diff: {s_cond}"
                bundles.append({
                    "bundle_id": f"B{len(bundles) + 1:03d}", "item_id": it["id"], "source_cell": it["source_cell"],
                    "axis": axis, "original_condition": cond, "nested_subset": i < ep["nested_subset_per_axis"] // 2,
                    "question": it["question"], "group1": it["group1"], "group2": it["group2"],
                    "choices_on_disk": it["on_disk_choices"], "units": v.get("units", ""), "denominator": v.get("denominator_or_stratum", ""),
                    "verified_group1_value": g1, "verified_group2_value": g2, "verified_expected_gold": expected_gold(g1, g2, derive_condition(g1, g2, rule)),
                    "synthetic_group1_value": s1, "synthetic_group2_value": s2, "synthetic_condition": s_cond,
                    "synthetic_expected_gold": expected_gold(s1, s2, s_cond),
                    "manually_checked": bool(C.smoke(cfg).get("auto_mark_bundles_checked")),
                    "manual_check_notes": "SMOKE: auto-marked, no human check" if C.smoke(cfg) else "",
                })
    return bundles, {"status": "built", "bundles": len(bundles), "shortfall": shortfall}


def _table(g1_name, g2_name, v1, v2, units, denom, hypothetical: bool, anon: bool) -> str:
    n1, n2 = ("Group A", "Group B") if anon else (g1_name, g2_name)
    head = ("HYPOTHETICAL TABLE (not real census data). Answer only about this table." if hypothetical
            else "Table (2015-16 Agriculture Census).")
    return f"{head}\n| group | value ({units or 'as stated'}) |\n|---|---|\n| {n1} | {v1} |\n| {n2} | {v2} |\n(denominator: {denom or 'as stated'})"


def render_prompt(b: Dict, condition: str) -> Tuple[str, Dict[str, str]]:
    """Fixed three-choice short-answer prompt. Choice order is the on-disk order for every
    condition so counterbalancing is identical across compared systems."""
    q = b["question"]; choices = list(b["choices_on_disk"])
    g1, g2 = b["group1"], b["group2"]
    if condition == "anonymised_verified_evidence":
        q = q.replace(g1, "Group A").replace(g2, "Group B")
        choices = [c.replace(g1, "Group A").replace(g2, "Group B") for c in choices]
    display_to_canonical = {}
    for letter, c in zip("abc", choices):
        canon = "c" if c.lower().startswith("roughly equal") else ("a" if c in (g1, "Group A") or c.replace("Group A", g1) == g1 else "b")
        display_to_canonical[letter] = canon
    table = ""
    if condition == "verified_evidence":
        table = _table(g1, g2, b["verified_group1_value"], b["verified_group2_value"], b["units"], b["denominator"], False, False)
    elif condition == "synthetic_evidence":
        table = _table(g1, g2, b["synthetic_group1_value"], b["synthetic_group2_value"], b["units"], b["denominator"], True, False)
    elif condition == "anonymised_verified_evidence":
        table = _table(g1, g2, b["verified_group1_value"], b["verified_group2_value"], b["units"], b["denominator"], False, True)
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


# ----------------------------------------------------------------------------- gates

def preflight(cfg: Dict, bundles: List[Dict]) -> Tuple[bool, List[str]]:
    reasons = []
    if not cfg["gpu_enabled"]:
        reasons.append("gpu_enabled is false")
    if cfg["allow_paid_api"]:
        reasons.append("allow_paid_api must be false for this stage")
    if not bundles:
        reasons.append("no bundles")
    elif not all(b.get("manually_checked") for b in bundles):
        reasons.append(f"{sum(1 for b in bundles if not b.get('manually_checked'))} bundles not manually checked")
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
    pilot_ids = {b["bundle_id"] for b in bundles[: ep["pilot_bundles"]]}
    main_all = [b for b in bundles if b["bundle_id"] not in pilot_ids]
    main_nested = [b for b in main_all if b["nested_subset"]]
    configs = [(t, m) for t in ep["model_tiers"] for m in ep["configurations"]]
    preds_path = out / "evidence_panel_predictions.jsonl"
    manifest = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "configs": configs,
                "cap_minutes": cfg["gpu_total_deadline_minutes"], "pilot": [], "decision": None}

    # ---- pilot: measure load + per-bundle cost for every configuration
    load_times, per_bundle = {}, {}
    for tier, method in configs:
        if budget.exceeded():
            manifest["decision"] = "cap reached during pilot; P4 skipped"; break
        t = time.time(); model, tok = _load(tier, method, ep["seed"]); load_times[(tier, method)] = time.time() - t
        t = time.time(); n = 0
        for b in bundles[: ep["pilot_bundles"]]:
            for cond in CONDITIONS:
                prompt, d2c = render_prompt(b, cond); raw = _generate(model, tok, prompt, ep["max_new_tokens"]); n += 1
                C.write_jsonl  # noqa (keeps the symbol referenced for readers)
                with open(preds_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"phase": "pilot", "tier": tier, "method": method, "seed": ep["seed"], "bundle_id": b["bundle_id"],
                                        "condition": cond, "is_hypothetical": cond == "synthetic_evidence",
                                        "raw_output": raw, "pred_canonical": parse_answer(raw, d2c), "expected": expected_for(b, cond)}) + "\n")
        per_bundle[(tier, method)] = (time.time() - t) / max(1, ep["pilot_bundles"])
        del model
        manifest["pilot"].append({"tier": tier, "method": method, "load_s": round(load_times[(tier, method)], 1),
                                  "per_bundle_s": round(per_bundle[(tier, method)], 2)})

    # ---- choose the panel size BEFORE looking at any outcome
    if manifest["decision"] is None:
        def projected(nb: int) -> float:
            return ep["safety_factor"] * sum(load_times[c] + nb * per_bundle[c] for c in configs) + 60.0
        rem = budget.remaining()
        if projected(len(main_all)) <= rem:
            chosen, label = main_all, f"full panel ({len(main_all)} bundles)"
        elif projected(len(main_nested)) <= rem:
            chosen, label = main_nested, f"nested subset ({len(main_nested)} bundles)"
        else:
            chosen, label = [], "neither panel fits the remaining cap; P4 skipped"
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
    if bpath.exists():
        bundles = C.read_jsonl(bpath)
        status = {"status": "loaded_existing", "bundles": len(bundles)}
    else:
        bundles, status = build_bundles(cfg)
        if bundles:
            C.write_jsonl(bpath, bundles)
            # a manual-check worksheet: every prompt in every condition, for a human to read
            C.write_jsonl(out / "bundles_manual_check_worksheet.jsonl",
                          [{"bundle_id": b["bundle_id"], "condition": c, "prompt": render_prompt(b, c)[0], "expected": expected_for(b, c)}
                           for b in bundles for c in CONDITIONS])
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
    main(C.load_config())
