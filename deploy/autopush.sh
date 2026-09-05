#!/usr/bin/env bash
# Push results, checkpoints and data to the artifacts branch every 30 minutes.
#
# The study writes incrementally, so a snapshot taken at any moment is a usable partial
# result. That is what makes an interruptible instance safe: the most a preemption can cost
# is the work done since the last snapshot.
#
# Snapshots go to an orphan branch through a separate git index, so they never touch the
# working tree, the staging area, or main. Those three directories are gitignored, so the
# snapshot force-adds them; without that it would push clean, empty commits forever.
#
# Usage:  nohup bash deploy/autopush.sh > results/autopush.log 2>&1 &
set -uo pipefail

CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INTERVAL_SECONDS="${AUTOPUSH_INTERVAL_SECONDS:-1800}"
cd "$CODE_DIR"

echo "autopush: every ${INTERVAL_SECONDS}s from $CODE_DIR"
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from GPU_Run.common.artifact_sync import snapshot_status
ok, why = snapshot_status()
print(f"autopush: preflight {'ready' if ok else 'NOT READY'} ({why})")
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  echo "autopush: refusing to start. Fix the GitHub token or the origin remote first." >&2
  exit 1
fi

while true; do
  STAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  python3 - "$STAMP" <<'PY'
import sys
sys.path.insert(0, ".")
from GPU_Run.common.artifact_sync import snapshot
stamp = sys.argv[1]
ok = snapshot(f"autosync {stamp}")
print(f"{stamp} snapshot {'ok' if ok else 'skipped or failed'}", flush=True)
PY
  sleep "$INTERVAL_SECONDS"
done
