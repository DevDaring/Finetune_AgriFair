"""Budget sensitivity for rounds 2 and 3 (logged in the plan before running): re-run, with 256 new tokens,
every GPU system whose parse rate is below 0.95 in any round-2 or round-3 experiment (recall and
reasoning excluded). Runs on the GPU host after round 3.

    python -m Submission1_TMLR.retry_lowparse
"""
from __future__ import annotations

import collections
import subprocess
import sys

from Submission1_Code_Phase2 import common as C

OUT = C.CODES_ROOT / "results_submission1_tmlr"


def low_systems(stage: str):
    per = collections.defaultdict(lambda: [0, 0])
    for r in C.read_jsonl(OUT / f"gpu_{stage}_predictions.jsonl"):
        if r.get("experiment") in ("recall", "cot"):
            continue
        k = (r["system"], r["experiment"]); per[k][0] += bool(r["parse_ok"]); per[k][1] += 1
    return sorted({s for (s, e), (a, n) in per.items() if n and a / n < 0.95})


def main() -> None:
    plan = {}
    for stage in ("round2", "round3"):
        systems = low_systems(stage)
        plan[stage] = systems
        if not systems:
            continue
        fams = sorted({s.split("|")[0] for s in systems})
        cmd = [sys.executable, "-m", "Submission1_TMLR.run_gpu", "--attention", "sdpa", "--all-arms", "--sets", stage,
               "--stage-name", f"retry256_{stage}", "--max-new-tokens", "256", "--families", *fams, "--only-systems", *systems]
        print("running", " ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, check=True)
    C.write_json(OUT / "retry256_r23_plan.json", plan)
    print(plan)


if __name__ == "__main__":
    main()
