#!/usr/bin/env bash
# Round 3 on Bedrock (Future_PLan.md "Round 3"): all 9 hosted models on the round-3 sets, then one repair
# pass for calls that failed after throttling (the runner skips rows that already succeeded).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
echo "== round3 $(date -u)"; python3 -m Submission1_TMLR.run_bedrock --models all --sets round3 --workers 6 || echo "FAILED: round3"
echo "== repair $(date -u)"; python3 -m Submission1_TMLR.run_bedrock --models all --sets round3 --workers 3 || echo "FAILED: repair"
echo "BEDROCK_R3_DONE $(date -u)"
