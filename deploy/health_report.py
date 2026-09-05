"""One health snapshot of a running study.

Answers the questions an hourly check actually needs to answer, which are about progress and
correctness rather than liveness. A process that is alive but has written nothing for an hour
is the failure worth catching, and "still running" would hide it.

Reports: GPU memory and utilisation, how many training arms and evaluation targets are done
against how many are expected, whether anything has been written recently, the parse-failure
and unresolved-answer rates so far, whether any arm has errored, and when the last snapshot
reached the artifacts branch.

Exit code is 0 when healthy, 1 when something needs attention, so a loop can alert on it.

Usage:  python deploy/health_report.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv

from GPU_Run.common import model_registry
from GPU_Run.common.paths import CHECKPOINTS_DIR, RESULTS_DIR

STALL_MINUTES = int(os.environ.get("HEALTH_STALL_MINUTES", "45"))


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            return "  gpu: nvidia-smi unavailable"
        lines = []
        for row in out.stdout.strip().splitlines():
            name, used, total, util, temp = [c.strip() for c in row.split(",")]
            lines.append(f"  gpu: {name}  {int(used) / 1024:.1f}/{int(total) / 1024:.0f} GiB  "
                         f"{util}% util  {temp}C")
        return "\n".join(lines)
    except Exception:
        return "  gpu: not visible"


def _newest_write() -> tuple:
    newest, newest_path = 0.0, ""
    for base in (RESULTS_DIR, CHECKPOINTS_DIR):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                m = p.stat().st_mtime
                if m > newest:
                    newest, newest_path = m, str(p.relative_to(base.parent))
    return newest, newest_path


def _rows(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    problems = []
    tiers = model_registry.active_tiers()
    print(f"  tiers: {', '.join(tiers)}")
    print(_gpu())

    # training progress
    expected_total = 0
    done_total = 0
    for tier in tiers:
        try:
            from CPU_Run.estimate_gpu_hours import count_arms

            expected = count_arms(tier)["training_runs"]
        except Exception:
            expected = 0
        done = len(list((CHECKPOINTS_DIR / tier).glob("*/seed_*/train_summary.json"))) \
            if (CHECKPOINTS_DIR / tier).exists() else 0
        expected_total += expected
        done_total += done
        pct = f"{100 * done / expected:.0f}%" if expected else "n/a"
        print(f"  training  {tier:<20s} {done}/{expected} arms ({pct})")
    if expected_total:
        print(f"  training  {'TOTAL':<20s} {done_total}/{expected_total} arms")

    # evaluation progress
    preds = list(RESULTS_DIR.glob("per_item_predictions_*.jsonl"))
    print(f"  evaluation targets with predictions: {len(preds)}")

    # correctness signals
    evaluated = [r for r in _rows(RESULTS_DIR / "main_evaluation_results.csv")
                 if str(r.get("scope", "")).endswith("|all")]
    if evaluated:
        def numeric(col):
            vals = []
            for r in evaluated:
                try:
                    vals.append(float(r[col]))
                except (KeyError, TypeError, ValueError):
                    pass
            return vals

        parse = numeric("json_parse_failure_rate_percent")
        unresolved = numeric("answers_left_unresolved")
        if parse:
            worst = max(parse)
            print(f"  parse failures: mean {sum(parse) / len(parse):.1f}%, worst {worst:.1f}%")
            if worst > 40:
                problems.append(f"a method has a {worst:.0f}% parse-failure rate")
        if unresolved and max(unresolved) > 0:
            print(f"  answers left unresolved after the judge: worst {max(unresolved):.0f}")

    # failed arms
    for name in ("train_graft_runs.json", "train_baselines_runs.json"):
        p = RESULTS_DIR / name
        if p.exists():
            try:
                runs = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            errors = [r for r in runs if "error" in r]
            if errors:
                problems.append(f"{len(errors)} arm(s) errored in {name}")
                for r in errors[:3]:
                    print(f"  ERROR {r.get('tier')}/{r.get('method')}: {str(r.get('error'))[:90]}")

    # stall detection
    newest, path = _newest_write()
    if newest:
        idle = (time.time() - newest) / 60
        print(f"  last write: {idle:.0f} min ago ({path})")
        if idle > STALL_MINUTES:
            problems.append(f"nothing written for {idle:.0f} minutes")
    else:
        problems.append("no results or checkpoints written yet")

    # snapshot freshness
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cr", "refs/heads/gpu-run-artifacts"],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            print(f"  last artifacts snapshot: {out.stdout.strip()}")
    except Exception:
        pass

    if problems:
        print("  STATUS: needs attention")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  STATUS: healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
