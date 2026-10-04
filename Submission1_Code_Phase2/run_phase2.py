"""One entry point for the Phase 2 repair package.

    python -m Submission1_Code_Phase2.run_phase2 --stage cpu     # everything that needs no GPU
    python -m Submission1_Code_Phase2.run_phase2 --stage packets # blinded packets for human readers
    python -m Submission1_Code_Phase2.run_phase2 --stage plan    # GPU sizing from the frozen designs
    python -m Submission1_Code_Phase2.run_phase2 --stage smoke   # whole path offline, no weights
    python -m Submission1_Code_Phase2.run_phase2 --stage pilot   # billable; needs gpu_enabled: true
    python -m Submission1_Code_Phase2.run_phase2 --stage main
    python -m Submission1_Code_Phase2.run_phase2 --stage analyse

Order for the resubmission: cpu -> packets (humans work) -> plan -> smoke -> pilot -> read the
pilot's projection -> main -> analyse. The pilot's timing projection is the gate for the main
run; if it does not fit the cap, reduce to the prespecified fallback design before running, never
after seeing outcomes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List

from Submission1_Code_Phase2 import common as C

STAGES = {
    "cpu": [("r1_template_audit", []), ("r1_fresh_panel", ["--build"]), ("r2_evidence", ["--build"]),
            ("r4_reanalysis", []), ("r0_constructs", ["--spec"])],
    "packets": [("r0_constructs", ["--packets"]), ("r1_fresh_panel", ["--checker-sheets"]),
                ("r3_advice", ["--template"])],
    "plan": [("gpu_plan", [])],
    "smoke": [("run_inference", ["--stage", "pilot", "--smoke"])],
    "pilot": [("run_inference", ["--stage", "pilot"])],
    "main": [("run_inference", ["--stage", "main"])],
    "analyse": [("analyse_followup", ["--stage", "main"]), ("r4_reanalysis", [])],
}


def run(module: str, args: List[str]) -> int:
    cmd = [sys.executable, "-m", f"Submission1_Code_Phase2.{module}"] + args
    print(f"\n>>> {' '.join(cmd[2:])}", flush=True)
    return subprocess.call(cmd, cwd=str(C.CODES_ROOT))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    a = ap.parse_args(argv)
    cfg = C.load_config()
    if a.stage in ("pilot", "main") and not cfg["gpu_enabled"]:
        raise SystemExit("gpu_enabled is false in Submission1_Code_Phase2/config.yaml. "
                         "Flip it deliberately, and check budget.tier before renting.")
    failures = []
    for module, args in STAGES[a.stage]:
        if run(module, args) != 0:
            failures.append(module)
    print("\n" + ("all stages completed" if not failures else f"failed: {failures}"))
    if a.stage == "pilot":
        print("Read predictions/pilot_manifest.json -> main_projection before starting the main run.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
