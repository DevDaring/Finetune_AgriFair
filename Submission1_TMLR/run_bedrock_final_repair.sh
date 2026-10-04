#!/usr/bin/env bash
# Final repair pass on Bedrock: fill the few calls that stayed throttled after the earlier repair passes.
# The runner skips every (prompt, system) that already has a successful row.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
for s in main followups cot altered48 round2; do
  echo "== original $s $(date -u)"; python3 -m Submission1_TMLR.run_bedrock --models original --sets $s --workers 1 || echo "FAILED: original $s"
  echo "== new $s $(date -u)"; python3 -m Submission1_TMLR.run_bedrock --models new --sets $s --workers 1 || echo "FAILED: new $s"
done
echo "BEDROCK_FINAL_REPAIR_DONE $(date -u)"
