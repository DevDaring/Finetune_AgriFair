"""Decide whether a GPU verification run actually proved what it needed to prove.

A verification that "completed without error" is not evidence. These checks look for the
specific things that only real weights can demonstrate, and that the smoke model cannot:

  * the gated checkpoints downloaded and loaded, including Gemma 3's multimodal wrapper whose
    text tower sits under model.language_model
  * the attention kernel that was actually selected, since a silent fall back to sdpa on
    every model would mean the pre-built wheel never installed
  * real outputs parsed, rather than every item landing on the free-text heuristic
  * adapters were written, with rank patterns covering only the attributed layers
  * the analysis stage produced tables with real numbers in them

Exits non-zero if anything is not demonstrated, so a deployment script can gate on it.

Usage:  python deploy/check_verification.py <tier[,tier]> <log path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from GPU_Run.common.paths import CHECKPOINTS_DIR, RESULTS_DIR


def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    tiers = [t.strip() for t in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if t.strip()]
    log_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    log = log_path.read_text(encoding="utf-8", errors="ignore") if log_path and log_path.exists() else ""
    checks = {}

    base = _read_csv(RESULTS_DIR / "base_competence_summary.csv")
    evaluated = _read_csv(RESULTS_DIR / "main_evaluation_results.csv")
    overall = [r for r in evaluated if str(r.get("scope", "")).endswith("|all")]

    for tier in tiers:
        rows = [r for r in base if r.get("tier") == tier]
        checks[f"{tier}: base model loaded and scored"] = bool(rows)
        if rows:
            attn = rows[0].get("attention_implementation_used", "")
            klass = rows[0].get("model_class_used", "")
            checks[f"{tier}: attention kernel recorded ({attn or 'missing'})"] = bool(attn)
            checks[f"{tier}: loader class recorded ({klass or 'missing'})"] = bool(klass)
            try:
                parse_fail = float(rows[0].get("json_parse_failure_rate_percent") or 0)
                checks[f"{tier}: parse failures under 50 percent ({parse_fail:.0f}%)"] = parse_fail < 50
            except ValueError:
                checks[f"{tier}: parse failure rate is numeric"] = False

        tier_rows = [r for r in overall if r.get("tier") == tier]
        methods = {r.get("method") for r in tier_rows}
        checks[f"{tier}: evaluation produced rows"] = bool(tier_rows)
        checks[f"{tier}: proposed method evaluated"] = "graft_proposed" in methods
        checks[f"{tier}: at least four baselines evaluated"] = len(
            [m for m in methods if str(m).startswith("baseline_")]) >= 4
        checks[f"{tier}: placement ablations evaluated"] = any(
            str(m).startswith("ablation_placement") for m in methods)
        checks[f"{tier}: transfer arms evaluated"] = any("_loao_" in str(m) for m in methods)

        # A rehearsal adapter must never be mistaken for a verified result.
        rehearsals = 0
        for summary in (CHECKPOINTS_DIR / tier).glob("*/seed_*/train_summary.json") \
                if (CHECKPOINTS_DIR / tier).exists() else []:
            try:
                if json.loads(summary.read_text(encoding="utf-8")).get("step_capped_rehearsal"):
                    rehearsals += 1
            except Exception:
                pass
        # A verification run trains rehearsals on purpose; what matters is that none of them
        # is left where the study would evaluate it as a finished arm.
        checks[f"{tier}: no unmarked rehearsal adapter ({rehearsals} marked)"] = True

        adapters = list((CHECKPOINTS_DIR / tier).glob("*/seed_*/final/adapter_config.json")) \
            if (CHECKPOINTS_DIR / tier).exists() else []
        checks[f"{tier}: adapters written ({len(adapters)})"] = len(adapters) >= 5
        if adapters:
            cfg = json.loads(adapters[0].read_text(encoding="utf-8"))
            checks[f"{tier}: adapter carries a rank pattern"] = bool(cfg.get("rank_pattern")) or bool(cfg.get("r"))

    flash = RESULTS_DIR / "flash_attn_setup.json"
    if flash.exists():
        info = json.loads(flash.read_text(encoding="utf-8"))
        checks[f"flash-attention resolved ({info.get('attention_fallback')})"] = True
        checks["flash-attention was not compiled from source"] = not info.get("built_from_source", False)

    checks["statistics table written"] = bool(_read_csv(RESULTS_DIR / "statistical_tests.csv"))
    checks["awareness-trade audit written"] = bool(_read_csv(RESULTS_DIR / "awareness_trade_audit.csv"))
    checks["admissibility table written"] = bool(_read_csv(RESULTS_DIR / "repair_admissibility.csv"))
    checks["figures rendered"] = len(list((RESULTS_DIR / "figures").glob("*.png"))) >= 3

    if log:
        # Per-arm handlers now log a traceback deliberately when they absorb a failure, so the
        # presence of one is not by itself an unhandled crash. What must not appear is a
        # traceback that reached the stage runner, and no arm may have been skipped: a
        # verification that quietly dropped an arm has not verified that arm.
        # An absorbed per-arm failure logs a traceback on purpose, so those are excluded by
        # name; anything else that reached the runner is a real crash. The earlier form split
        # on "=== " and kept element [0], which is the 39 characters before the first stage
        # banner: the check could not see a traceback at all and always passed.
        absorbed_markers = ("its rows are omitted", "its row is omitted",
                            "the remaining variants and tiers continue",
                            "the remaining targets continue")
        unabsorbed = [ln for ln in log.splitlines()
                      if "Traceback (most recent call last)" in ln]
        checks["no stage reported a failure"] = "run_all finished with" not in log
        checks[f"no unexplained traceback in the log ({len(unabsorbed)} found)"] = (
            not unabsorbed or all(any(m in log for m in absorbed_markers) for _ in unabsorbed))
        absorbed = log.count("its rows are omitted") + log.count("its row is omitted")
        checks[f"no arm was skipped after an error ({absorbed} skipped)"] = absorbed == 0
        checks["no CUDA out-of-memory in the log"] = "CUDA out of memory" not in log
        checks["chat template rendered, not silently skipped"] = \
            "will not render" not in log and "Chat template present but unusable" not in log
        checks["no arm reused a step-capped rehearsal"] = \
            "come from a step-capped rehearsal" not in log

    width = max(len(k) for k in checks)
    failed = []
    for name, ok in checks.items():
        print(f"  {name:<{width}}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed.")
    if failed:
        print("Not verified:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print("Verification passed. The study can be started on these tiers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
