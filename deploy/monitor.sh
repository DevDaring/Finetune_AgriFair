#!/usr/bin/env bash
# Hourly health check for a running study.
#
# Reports progress rather than liveness: a process that is alive but has produced nothing for
# an hour is the failure worth catching, and a plain "still running" would hide it.
#
# Usage:  nohup bash deploy/monitor.sh > results/monitor.log 2>&1 &
set -uo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"
INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-3600}"

while true; do
  echo "===== $(date -u '+%Y-%m-%dT%H:%M:%SZ') ====="
  python3 deploy/health_report.py
  sleep "$INTERVAL_SECONDS"
done
