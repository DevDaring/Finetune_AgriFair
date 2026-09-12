"""Orchestrate the final bounded audit round with resumable, hash-keyed stage caching.

    python Next_Run/run_final_audit.py                 # every stage, in order
    python Next_Run/run_final_audit.py --stage paired  # one stage
    python Next_Run/run_final_audit.py --force         # ignore caches

A stage is skipped only when the hash of (its inputs + the resolved config) matches the
hash recorded when it last completed. There is no automatic expensive fallback: if the
evidence panel's gates fail, the run reports that and moves on; nothing here will rent a
GPU, call an API, or train a model on its own. Each stage's provenance, runtime and
failure status are recorded, and a stage that raises leaves its predecessors' outputs in
place - resume is a re-run of the same command.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Next_Run import common as C  # noqa: E402

STAGES: List[tuple] = [
    ("inventory",       "Next_Run.inventory",       [("data", "*.jsonl"), ("results", "per_item_predictions_*.jsonl")]),
    ("verify_sources",  "Next_Run.verify_sources",  [("data", "*.jsonl")]),
    ("cpu_baselines",   "Next_Run.cpu_baselines",   [("data", "train_instances.jsonl"), ("data", "test_instances_frozen.jsonl")]),
    ("paired",          "Next_Run.paired_analysis", [("data", "test_instances_frozen.jsonl"), ("results", "per_item_predictions_*.jsonl")]),
    ("advice",          "Next_Run.advice_audit",    [("data", "agriadvice_pairs.jsonl"), ("results", "agriadvice_drift_*_seed*.jsonl")]),
    ("evidence_panel",  "Next_Run.evidence_panel",  [("data", "test_instances_frozen.jsonl")]),
    ("report",          "Next_Run.build_report",    []),
]


def _input_hash(cfg: Dict, globs) -> str:
    parts = {"config": C.sha256_obj(cfg)}
    for sub, pat in globs:
        for p in sorted((C.CODES_ROOT / sub).glob(pat)):
            parts[str(p.relative_to(C.CODES_ROOT))] = f"{p.stat().st_size}:{int(p.stat().st_mtime)}"
    return C.sha256_obj(parts)


def run(cfg: Dict, only: str | None, force: bool) -> int:
    root = C.output_dir(cfg)
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"stages": {}}
    C.write_json(root / "resolved_config.json", cfg)
    failures = 0
    for name, module, globs in STAGES:
        if only and name != only:
            continue
        h = _input_hash(cfg, globs)
        prev = state["stages"].get(name, {})
        # the report reads the other stages' outputs, which live outside the hashed inputs; it is
        # cheap, so it always rebuilds rather than risk a stale table
        if name != "report" and not force and prev.get("status") == "ok" and prev.get("input_hash") == h:
            print(f"[run] {name:15s} cached (inputs unchanged)"); continue
        t0 = time.time()
        rec = {"input_hash": h, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            mod = __import__(module, fromlist=["main"])
            result = mod.main(cfg)
            rec.update({"status": "ok", "result": result})
        except Exception as e:  # keep predecessors; record and continue so a report can still build
            failures += 1
            rec.update({"status": "failed", "error": C.scrub_secrets(f"{type(e).__name__}: {e}"),
                        "traceback": C.scrub_secrets(traceback.format_exc()[-2000:])})
            print(f"[run] {name:15s} FAILED: {rec['error']}")
        rec["runtime_s"] = round(time.time() - t0, 1)
        state["stages"][name] = rec
        C.write_json(state_path, state)
        if rec["status"] == "ok":
            print(f"[run] {name:15s} ok in {rec['runtime_s']}s")
    return failures


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[s[0] for s in STAGES])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=str(C.CONFIG_PATH))
    a = ap.parse_args()
    cfg = C.load_config(Path(a.config))
    for k in ("allow_training", "allow_new_attribution"):
        if cfg[k]:
            print(f"[run] refusing to start: {k} is true and Next_Run implements no such stage"); sys.exit(2)
    sys.exit(1 if run(cfg, a.stage, a.force) else 0)
