#!/usr/bin/env bash
# P7 on Bedrock: the entity-renamed items for all 9 hosted models, then a repair pass.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
python3 -m Submission1_TMLR.run_bedrock --models all --sets renamed --workers 3 || echo "FAILED: renamed"
python3 -m Submission1_TMLR.run_bedrock --models all --sets renamed --workers 2 || echo "FAILED: repair"
echo "BEDROCK_RENAMED_DONE $(date -u)"
