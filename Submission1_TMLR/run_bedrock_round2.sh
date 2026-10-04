#!/usr/bin/env bash
# Round 2 on Bedrock (Future_PLan.md E3-E6): the seven new models on the full design, then every API
# model on the round-2 sets. Each step is resumable; the log is results_submission1_tmlr/bedrock_round2.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
run() { echo "== $* $(date -u)"; python3 -m Submission1_TMLR.run_bedrock "$@" || echo "FAILED: $*"; }
run --models new --sets main --workers 8
run --models new --sets followups --workers 8
run --models new --sets altered48 --workers 8
run --models new --sets cot --workers 6
run --models all --sets round2 --workers 8
run --models all --sets round2 --workers 3
run --models new --sets main --workers 3
run --models new --sets followups --workers 3
run --models new --sets altered48 --workers 3
run --models new --sets cot --workers 3
echo "BEDROCK_R2_DONE $(date -u)"
