"""Local watchdog for the rented GPU (GPU run protocol: download, verify, then destroy).

    nohup python -m Submission1_TMLR.watch_vast <instance_id> > results_submission1_tmlr/watchdog.log 2>&1 &

Every five minutes it checks the instance for RUN_DONE or RUN_FAILED. On either, or once the
safety cap is reached, it downloads the outputs and logs and verifies the row counts against the
design. It destroys the instance only after the download, then confirms the instance is gone. The
verification result, not the session, decides what is reported; the API key is never printed.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CODES = Path(__file__).resolve().parents[1]
OUT = CODES / "results_submission1_tmlr"
MAX_HOURS = 6.0
EXPECTED_ROUND2 = {"gpu_logprobs_predictions.jsonl": 40 * 904, "gpu_round2_predictions.jsonl": 40 * 1374,
                   "gpu_round3_predictions.jsonl": 40 * 3406}
EXPECTED = {"gpu_main_predictions.jsonl": 32 * 786, "gpu_flash_predictions.jsonl": 24 * 786,
            "gpu_followups_predictions.jsonl": 32 * 930, "gpu_cot_predictions.jsonl": 8 * 310,
            "gpu_altered48_predictions.jsonl": 32 * 144, "gpu_draws_predictions.jsonl": 8 * 930,
            "gpu_retry256_predictions.jsonl": 2 * 786}


def key() -> str:
    for line in (CODES / ".env").read_text().splitlines():
        if line.strip().startswith("Vast_AI_PHD_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("Vast key not found")


def vast(*args: str) -> str:
    return subprocess.run(["vastai", *args, "--api-key", key()], capture_output=True, text=True, timeout=120).stdout


def api(method: str, iid: str) -> str:
    import urllib.request
    import urllib.error
    req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{iid}/", method=method,
                                 headers={"Authorization": "Bearer " + key()})
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            return f.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:                                               # network: keep polling
        return f"error {type(e).__name__} actual_status"


def ssh_target(iid: str):
    info = json.loads(vast("show", "instance", iid, "--raw") or "{}")
    return info.get("ssh_host"), info.get("ssh_port"), info.get("actual_status")


def remote(host, port, cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                           "-o", "ConnectTimeout=20", "-p", str(port), f"root@{host}", cmd],
                          capture_output=True, text=True, timeout=300)


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def download(host, port, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src in ("/workspace/Codes/results_submission1_tmlr/", "/workspace/Codes/logs/"):
        subprocess.run(["rsync", "-az", "-e", f"ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
                        f"root@{host}:{src}", str(dest / Path(src.rstrip('/')).name) + "/"],
                       capture_output=True, text=True, timeout=1800)


def verify(dest: Path) -> dict:
    res = {}
    expected = EXPECTED_ROUND2 if "--round2" in sys.argv else EXPECTED
    for name, n in expected.items():
        p = dest / "results_submission1_tmlr" / name
        rows = sum(1 for _ in p.open()) if p.exists() else 0
        res[name] = {"rows": rows, "expected": n, "complete": rows >= n}
    return res


def main() -> None:
    iid = sys.argv[1]
    start = time.time()
    dest = OUT / f"gpu_run_{iid}"
    while True:
        host, port, status = ssh_target(iid)
        hours = (time.time() - start) / 3600
        marker = ""
        if host and status == "running":
            r = remote(host, port, "ls /workspace/RUN_R4_DONE /workspace/RUN_R4_FAILED /workspace/RUN_R3_FAILED /workspace/RUN6_DONE /workspace/RUN6_FAILED /workspace/RUN5_FAILED /workspace/RUN4_FAILED /workspace/RUN3_FAILED /workspace/RUN2_FAILED /workspace/RUN_FAILED 2>/dev/null; tail -1 /workspace/Codes/logs/status.log 2>/dev/null")
            marker = r.stdout.strip()
        log(f"status={status} hours={hours:.2f} {marker.splitlines()[-1] if marker else ''}")
        finished = any(m in marker for m in ("RUN6_DONE", "RUN6_FAILED", "RUN5_FAILED", "RUN4_FAILED", "RUN3_FAILED", "RUN2_FAILED", "RUN_FAILED", "RUN_R4_DONE", "RUN_R4_FAILED", "RUN_R3_FAILED"))
        if finished or hours > MAX_HOURS:
            if host:
                download(host, port, dest)
            check = verify(dest)
            (dest / "verification.json").write_text(json.dumps(check, indent=1))
            log(f"downloaded to {dest}; verification {json.dumps(check)}")
            # REST, not the CLI: newer CLI versions ask for confirmation and silently do nothing here
            api("DELETE", iid)
            for _ in range(10):
                time.sleep(15)
                info = api("GET", iid)
                if '"instances": null' in info or "actual_status" not in info:
                    log(f"instance {iid} destroyed and confirmed gone")
                    (dest / "DESTROYED").write_text(datetime.now(timezone.utc).isoformat())
                    return
            log(f"WARNING: destroy of {iid} not confirmed; check the Vast console")
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
